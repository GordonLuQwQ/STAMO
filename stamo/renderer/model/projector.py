"""StaMo projectors with the MUSA attention adaptation retained."""

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor
from einops import rearrange

from stamo.renderer.utils.data import check_tensor


def _get(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class MLPProjector(nn.Module):
    """The progressive token compressor from StaMo."""

    def __init__(self, args, patches: int, channels: int) -> None:
        super().__init__()
        self.patches = patches
        self.channels = channels

        self.hidden_dim = int(args.projector.hidden_dim)
        self.cross_attention_dim = int(args.projector.cross_attention_dim)
        self.output_align_dim = int(args.projector.output_align_dim)
        self.num_token = int(args.projector.num_token)
        self.num_attn_layers = int(args.projector.num_attn_layers)
        self.num_attn_compress_layers = int(
            args.projector.num_attn_compress_layers
        )
        self.compress_dims = self._generate_compress_dims()

        self.compress_layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(
                self.compress_dims[:-1],
                self.compress_dims[1:],
            )
        )
        self.attn_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.channels,
                    num_attention_heads=8,
                    attention_head_dim=self.channels // 8,
                    dropout=0.1,
                    cross_attention_dim=self.channels,
                    upcast_attention=True,
                )
                for _ in range(self.num_attn_layers)
            ]
        )
        self.attn_compress_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=dim,
                    num_attention_heads=8,
                    attention_head_dim=self.cross_attention_dim // 8,
                    dropout=0.1,
                    cross_attention_dim=self.cross_attention_dim,
                    upcast_attention=True,
                )
                for dim in self.compress_dims
            ]
        )
        self.qkv_layer = nn.Linear(
            self.channels,
            self.hidden_dim + self.cross_attention_dim,
        )
        self.compress_align_mlp = nn.Linear(
            self.patches,
            self.compress_dims[0],
        )
        self.output_align_mlp = nn.Linear(self.hidden_dim, self.output_align_dim)

        for block in [*self.attn_layers, *self.attn_compress_layers]:
            block.attn1.set_processor(AttnProcessor())
            block.attn2.set_processor(AttnProcessor())

    def _generate_compress_dims(self):
        start_exp = self.patches.bit_length() - 1
        end_exp = self.num_token.bit_length() - 1
        exponents = np.linspace(
            start_exp,
            end_exp,
            self.num_attn_compress_layers,
        )
        exponents = np.ceil(exponents).astype(int)
        return [2**exponent for exponent in exponents]

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states = image_embeddings.clone()
        for block in self.attn_layers:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=image_embeddings,
            )
            check_tensor(hidden_states, "self-attn")

        hidden_states = self.qkv_layer(hidden_states)
        query = hidden_states[:, :, : self.hidden_dim]
        encoder_hidden_states = hidden_states[:, :, self.hidden_dim :]
        query = rearrange(query, "b s d -> b d s")
        hidden_states = self.compress_align_mlp(query)

        for projection, block in zip(self.compress_layers, self.attn_compress_layers):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = projection(hidden_states)
            check_tensor(hidden_states, "compressor")

        compressed_embeddings = rearrange(hidden_states, "b d s -> b s d")
        return self.output_align_mlp(compressed_embeddings)


class QformerBlock(nn.Module):
    """Query self-attention followed by cross-attention over DINO tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cross_dim: int,
    ) -> None:
        super().__init__()
        self.cross_attn = BasicTransformerBlock(
            dim=dim,
            num_attention_heads=num_heads,
            attention_head_dim=dim // num_heads,
            dropout=0.1,
            cross_attention_dim=cross_dim,
            upcast_attention=True,
            norm_elementwise_affine=False,
            norm_eps=1e-7,
        )
        self.cross_attn.attn1.set_processor(AttnProcessor())
        self.cross_attn.attn2.set_processor(AttnProcessor())

    def forward(
        self,
        query_tokens: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.cross_attn(
            hidden_states=query_tokens,
            encoder_hidden_states=image_embeddings,
        )


class QformerProjector(nn.Module):
    """Compress DINO patch tokens to a small FLUX.2 condition sequence."""

    def __init__(self, args, patches: int, channels: int) -> None:
        super().__init__()
        self.patches = patches
        self.channels = channels
        self.hidden_dim = int(args.projector.hidden_dim)
        self.output_align_dim = int(args.projector.output_align_dim)
        self.num_token = int(args.projector.num_token)
        self.num_attn_layers = int(args.projector.num_attn_layers)
        self.num_attention_heads = int(
            _get(args.projector, "num_attention_heads", 8)
        )

        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_token, self.hidden_dim)
        )
        self.qformer_layers = nn.ModuleList(
            [
                QformerBlock(
                    dim=self.hidden_dim,
                    num_heads=self.num_attention_heads,
                    cross_dim=self.channels,
                )
                for _ in range(self.num_attn_layers)
            ]
        )
        self.output_align_mlp = nn.Linear(self.hidden_dim, self.output_align_dim)
        self.norm = nn.LayerNorm(
            self.output_align_dim,
            eps=1e-6,
            elementwise_affine=False,
        )

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        query_tokens = self.query_tokens.expand(image_embeddings.shape[0], -1, -1)
        for index, block in enumerate(self.qformer_layers):
            query_tokens = block(query_tokens, image_embeddings)
            check_tensor(
                query_tokens,
                f"qformer{index}",
                check_bound=1e4,
                check_std=5e3,
            )
        return self.norm(self.output_align_mlp(query_tokens))


def build_projector(args, patches: int, channels: int) -> nn.Module:
    projector_type = str(_get(args.projector, "type", "mlp")).lower()
    if projector_type in {"mlp", "compress", "stamo"}:
        return MLPProjector(args, patches, channels)
    if projector_type in {"qformer", "q_former"}:
        return QformerProjector(args, patches, channels)
    raise ValueError(
        f"Unknown projector.type {projector_type!r}. Use 'mlp' or 'qformer'."
    )


Projector = MLPProjector
