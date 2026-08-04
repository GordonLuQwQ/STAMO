import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from einops import rearrange
from omegaconf import OmegaConf


class MixAttn(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, head) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=q_dim, num_heads=head, kdim=k_dim, vdim=v_dim, batch_first=True)


class Projector(nn.Module):
    """Original StaMo MLP / progressive compressor (token-length compression)."""

    def __init__(self, args, patches: int, channels: int) -> None:
        super().__init__()

        self.patches = patches
        self.channels = channels

        self.hidden_dim: int = args.projector.hidden_dim
        self.cross_attention_dim: int = args.projector.cross_attention_dim
        self.output_align_dim: int = args.projector.output_align_dim

        self.num_token: int = args.projector.num_token
        self.num_attn_layers: int = args.projector.num_attn_layers
        self.num_attn_compress_layers: int = args.projector.num_attn_compress_layers
        self.compress_dims = self._generate_compress_dims()

        self.compress_layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(self.compress_dims[:-1], self.compress_dims[1:])],
        )
        self.attn_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.channels,
                    num_attention_heads=8,
                    attention_head_dim=self.channels // 8,
                    dropout=0.1,
                    cross_attention_dim=self.channels,
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
                )
                for dim in self.compress_dims
            ]
        )
        self.qkv_layer = nn.Linear(self.channels, self.hidden_dim + self.cross_attention_dim)
        self.compress_align_mlp = nn.Linear(self.patches, self.compress_dims[0])
        self.output_align_mlp = nn.Linear(self.hidden_dim, self.output_align_dim)

    def _generate_compress_dims(self):
        start_exp = self.patches.bit_length() - 1
        end_exp = self.num_token.bit_length() - 1
        exps = np.linspace(start_exp, end_exp, self.num_attn_compress_layers)
        exps = np.ceil(exps).astype(int)
        dims = [2**e for e in exps]
        return dims

    def forward(self, image_embeddings: torch.Tensor):
        hidden_states = image_embeddings.clone()
        for transformer_block in self.attn_layers:
            hidden_states = transformer_block(
                hidden_states=hidden_states,
                encoder_hidden_states=image_embeddings,
            )

        hidden_states = self.qkv_layer(hidden_states)
        q = hidden_states[:, :, : self.hidden_dim]

        encoder_hidden_states = hidden_states[:, :, self.hidden_dim :]
        q = rearrange(q, "b s d -> b d s")
        hidden_states = self.compress_align_mlp(q)

        for compress_block, transformer_block in zip(self.compress_layers, self.attn_compress_layers):
            hidden_states = transformer_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = compress_block(hidden_states)

        compressed_embeds = rearrange(hidden_states, "b d s -> b s d")

        # Align feature dim to SD3 joint attention dim (e.g. 4096)
        compressed_embeds = self.output_align_mlp(compressed_embeds)

        return compressed_embeds


# Keep old name for checkpoints / imports that expect MLP compressor.
MLPProjector = Projector


class QformerBlock(nn.Module):
    """Q-Former block: self-attn on query tokens + cross-attn over image embeddings.

    Adapted from UniVAM:
    https://github.com/aim-uofa/STAIR/blob/main/third_party/UniVAM/src/univam/models/projector.py
    (image case: image_embeddings is [B, patches, C], not a video sequence).
    """

    def __init__(self, dim: int, num_heads: int, cross_dim: int) -> None:
        super().__init__()
        # Prefer UniVAM-style LayerNorm flags when available in this diffusers version.
        block_kwargs = dict(
            dim=dim,
            num_attention_heads=num_heads,
            attention_head_dim=dim // num_heads,
            dropout=0.1,
            cross_attention_dim=cross_dim,
        )
        try:
            self.cross_attn = BasicTransformerBlock(
                **block_kwargs,
                norm_elementwise_affine=False,
                norm_eps=1e-7,
            )
        except TypeError:
            self.cross_attn = BasicTransformerBlock(**block_kwargs)

    def forward(self, q_tokens: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        # BasicTransformerBlock already includes self-attn + cross-attn when
        # encoder_hidden_states is provided.
        q_tokens = self.cross_attn(
            hidden_states=q_tokens,
            encoder_hidden_states=image_embeddings,
        )
        return q_tokens


class QformerProjector(nn.Module):
    """Learnable query tokens + stacked Q-Former blocks -> fixed num_token output.

    Replaces StaMo's progressive length-compression with UniVAM-style Q-Former.
    Video temporal pooling is not needed here: StaMo feeds single-frame DINO patches.
    """

    def __init__(self, args, patches: int, channels: int) -> None:
        super().__init__()

        self.patches = patches
        self.channels = channels

        self.hidden_dim: int = args.projector.hidden_dim
        self.output_align_dim: int = args.projector.output_align_dim
        self.num_token: int = args.projector.num_token
        self.num_attn_layers: int = args.projector.num_attn_layers
        self.num_attention_heads: int = int(OmegaConf.select(args.projector, "num_attention_heads", default=8))

        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"projector.hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_token, self.hidden_dim) * 0.02)
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
        self.norm = nn.LayerNorm(self.output_align_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        # image_embeddings: [B, patches, C] from frozen DINOv3
        bsz = image_embeddings.size(0)
        q_tokens = self.query_tokens.expand(bsz, -1, -1)

        for block in self.qformer_layers:
            q_tokens = block(q_tokens, image_embeddings)

        compressed_embeds = self.output_align_mlp(q_tokens)
        compressed_embeds = self.norm(compressed_embeds)
        return compressed_embeds


def build_projector(args, patches: int, channels: int) -> nn.Module:
    """Build projector by args.projector.type: mlp (default) | qformer."""
    projector_type = OmegaConf.select(args.projector, "type", default="mlp")
    if projector_type is None:
        projector_type = "mlp"
    projector_type = str(projector_type).lower()

    if projector_type in ("mlp", "compress", "stamo"):
        return Projector(args, patches, channels)
    if projector_type in ("qformer", "q_former"):
        return QformerProjector(args, patches, channels)
    raise ValueError(f"Unknown projector.type '{projector_type}'. Use 'mlp' or 'qformer'.")


if __name__ == "__main__":
    from backbone import VisionBackbone
    from omegaconf import OmegaConf

    args = OmegaConf.load("./configs/debug.yaml")

    vision_backbone = VisionBackbone(
        model_name=args.vision_backbone.model_name,
        pretrained=args.vision_backbone.pretrained,
        local_ckpt=args.vision_backbone.local_ckpt,
    )

    model = build_projector(args, vision_backbone.patches, vision_backbone.channels)

    images = torch.randn((3, 224, 224)).unsqueeze(0)
    image_embeddings = vision_backbone(images)
    compressed_embeds = model(image_embeddings)

    total_params = sum(p.numel() for p in model.parameters())

    if hasattr(model, "compress_dims"):
        print(f"Token compression process: {model.compress_dims}")
    print(f"Projector: {type(model).__name__}")
    print(f"Total params: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"compressed_embeds.shape: {compressed_embeds.shape}")
