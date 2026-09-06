"""Architecture and backend contracts for the compact FLUX.2 Q-Former."""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "flux.yaml"
PROJECTOR = ROOT / "stamo" / "renderer" / "model" / "projector.py"
MUSA_SMOKE = ROOT / "scripts" / "qformer_musa_smoke_test.py"
VERIFIER = ROOT / "scripts" / "verify_flux2_klein_musa.sh"


def _projector_yaml() -> str:
    source = CONFIG.read_text(encoding="utf-8")
    return source[source.index("projector:\n") : source.index("\nrender_net:")]


def _yaml_scalar(source: str, key: str) -> str:
    match = re.search(rf"^  {re.escape(key)}:\s*(.+?)\s*$", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"Missing projector config key: {key}")
    return match.group(1).strip().strip('"')


class QformerV2SourceContractTests(unittest.TestCase):
    def test_production_dimensions_are_explicit(self):
        config = _projector_yaml()
        full_config = CONFIG.read_text(encoding="utf-8")
        task_name = re.search(
            r'(?m)^task_name:\s*["\']?([^"\'\s]+)',
            full_config,
        )
        self.assertIsNotNone(task_name)
        self.assertIn("qformer_v2", task_name.group(1))
        self.assertNotEqual(task_name.group(1), "egoverse_flux2_klein4b")
        self.assertRegex(full_config, r"(?m)^resume:\s*false\s*$")
        expected = {
            "architecture_version": "dino_flux2_qformer_v2_4l768",
            "input_token_count": "196",
            "input_dim": "768",
            "num_token": "4",
            "num_attn_layers": "4",
            "hidden_dim": "768",
            "num_attention_heads": "12",
            "attention_head_dim": "64",
            "attention_backend": "legacy_upcast",
            "attention_dropout": "0.0",
            "block_norm_eps": "1e-5",
            "input_norm_eps": "1e-6",
            "query_norm_eps": "1e-6",
            "output_norm_eps": "1e-6",
            "norm_elementwise_affine": "true",
            "input_norm_elementwise_affine": "true",
            "query_norm_elementwise_affine": "true",
            "output_norm_elementwise_affine": "true",
            "output_branch_dims": "[2560, 2560, 2560]",
            "output_align_dim": "7680",
        }
        for key, value in expected.items():
            self.assertEqual(_yaml_scalar(config, key), value, key)
        self.assertEqual(768 // 12, 64)
        self.assertEqual(sum((2560, 2560, 2560)), 7680)

    def test_qformer_explicitly_bypasses_fused_sdpa(self):
        source = PROJECTOR.read_text(encoding="utf-8")
        qformer_source = source[source.index("class QformerBlock") :]

        self.assertIn("from diffusers.models.attention_processor import AttnProcessor", source)
        self.assertIn("from diffusers.models.normalization import FP32LayerNorm", source)
        self.assertIn("upcast_attention=self.upcast_attention", qformer_source)
        self.assertIn("for attention in (self.cross_attn.attn1, self.cross_attn.attn2)", qformer_source)
        self.assertIn("attention.set_processor(AttnProcessor())", qformer_source)
        self.assertIn("type(attention.get_processor()) is not AttnProcessor", qformer_source)
        self.assertNotIn("AttnProcessor2_0()", qformer_source)
        self.assertNotIn("scaled_dot_product_attention(", qformer_source)
        self.assertNotIn("except TypeError", qformer_source)

    def test_fp32_norm_and_three_output_branches_are_structural(self):
        source = PROJECTOR.read_text(encoding="utf-8")
        qformer_source = source[source.index("class QformerBlock") :]

        self.assertIn('for norm_name in ("norm1", "norm2", "norm3")', qformer_source)
        self.assertIn("self.input_norm = FP32LayerNorm(", qformer_source)
        self.assertIn("self.query_norm = FP32LayerNorm(", qformer_source)
        self.assertIn("self.output_align_mlp = nn.ModuleList(", qformer_source)
        self.assertIn("self.output_branch_norms = nn.ModuleList(", qformer_source)
        self.assertIn("return torch.cat(output_branches, dim=-1)", qformer_source)
        self.assertIn("def architecture_contract(self) -> dict:", qformer_source)

    def test_musa_bf16_backward_smoke_is_wired_into_the_verifier(self):
        smoke = MUSA_SMOKE.read_text(encoding="utf-8")
        verifier = VERIFIER.read_text(encoding="utf-8")

        self.assertIn("EXPECTED_PARAMETER_COUNT = 53_165_568", smoke)
        self.assertIn("Q-Former unexpectedly called fused SDPA on MUSA", smoke)
        self.assertIn("for projection_name in (\"to_q\", \"to_k\", \"to_v\")", smoke)
        self.assertIn("loss.backward()", smoke)
        self.assertIn("dist.all_reduce(health, op=dist.ReduceOp.MIN)", smoke)
        self.assertIn("QFORMER_MUSA_SMOKE_PASS", smoke)
        self.assertIn("scripts/qformer_musa_smoke_test.py", verifier)
        self.assertIn("VERIFY_QFORMER_ITERATIONS", verifier)
        self.assertIn("sdpa_calls=0", verifier)


@unittest.skipUnless(
    importlib.util.find_spec("diffusers") is not None
    and importlib.util.find_spec("omegaconf") is not None,
    "diffusers and omegaconf are required for the runtime Q-Former test",
)
class QformerV2RuntimeTests(unittest.TestCase):
    def test_unversioned_sd3_qformer_preserves_legacy_v1_semantics(self):
        import torch
        from diffusers.models.normalization import FP32LayerNorm
        from omegaconf import OmegaConf

        from STAMO.stamo.renderer.model.projector2 import QformerProjector

        args = OmegaConf.create(
            {
                "projector": {
                    "num_token": 4,
                    "num_attn_layers": 2,
                    "hidden_dim": 16,
                    "num_attention_heads": 4,
                    "output_align_dim": 32,
                }
            }
        )
        model = QformerProjector(args, patches=5, channels=16)
        contract = model.architecture_contract()

        self.assertTrue(contract["legacy_v1"])
        self.assertEqual(contract["architecture_version"], "univam_qformer_legacy_v1")
        self.assertEqual(contract["attention_backend"], "univam_legacy_v1")
        self.assertEqual(contract["attention_dropout"], 0.1)
        self.assertEqual(contract["block_norm_eps"], 1e-7)
        self.assertFalse(contract["input_pre_norm"])
        self.assertFalse(contract["query_post_norm"])
        self.assertFalse(contract["fp32_layer_norm"])
        self.assertFalse(hasattr(model, "input_norm"))
        self.assertFalse(hasattr(model, "query_norm"))
        self.assertIs(type(model.norm), torch.nn.LayerNorm)
        self.assertNotIsInstance(model.qformer_layers[0].cross_attn.norm1, FP32LayerNorm)
        self.assertEqual(model.qformer_layers[0].cross_attn.norm1.eps, 1e-7)
        self.assertEqual(model.qformer_layers[0].cross_attn.attn1.to_out[1].p, 0.1)
        original_baddbmm = torch.baddbmm

        def stable_beta_zero_baddbmm(input_tensor, batch1, batch2, **kwargs):
            self.assertEqual(kwargs.get("beta", 1), 0)
            return torch.bmm(batch1, batch2) * kwargs.get("alpha", 1)

        torch.baddbmm = stable_beta_zero_baddbmm
        try:
            output = model(torch.randn(2, 5, 16))
        finally:
            torch.baddbmm = original_baddbmm
        self.assertEqual(tuple(output.shape), (2, 4, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_legacy_upcast_forward_backward_never_calls_sdpa(self):
        import torch
        import torch.nn.functional as functional
        from diffusers.models.attention_processor import AttnProcessor
        from diffusers.models.normalization import FP32LayerNorm
        from omegaconf import OmegaConf

        from STAMO.stamo.renderer.model.projector2 import QformerProjector

        args = OmegaConf.create(
            {
                "projector": {
                    "architecture_version": "qformer_test_v2",
                    "input_token_count": 5,
                    "input_dim": 16,
                    "num_token": 3,
                    "num_attn_layers": 2,
                    "hidden_dim": 16,
                    "num_attention_heads": 4,
                    "attention_head_dim": 4,
                    "attention_backend": "legacy_upcast",
                    "attention_dropout": 0.0,
                    "block_norm_eps": 1e-5,
                    "input_norm_eps": 1e-6,
                    "query_norm_eps": 1e-6,
                    "output_norm_eps": 1e-6,
                    "norm_elementwise_affine": False,
                    "input_norm_elementwise_affine": False,
                    "query_norm_elementwise_affine": False,
                    "output_norm_elementwise_affine": True,
                    "output_branch_dims": [12, 12, 12],
                    "output_align_dim": 36,
                }
            }
        )
        torch.manual_seed(7)
        model = QformerProjector(args, patches=5, channels=16)
        self.assertEqual(len(model.qformer_layers), 2)
        self.assertTrue(all(isinstance(norm, FP32LayerNorm) for norm in model.output_branch_norms))
        for block in model.qformer_layers:
            for attention in (block.cross_attn.attn1, block.cross_attn.attn2):
                self.assertIs(type(attention.get_processor()), AttnProcessor)
                self.assertTrue(attention.upcast_attention)

        original_sdpa = getattr(functional, "scaled_dot_product_attention", None)
        original_baddbmm = torch.baddbmm
        original_layer_norm = functional.layer_norm
        score_dtypes = []
        norm_input_dtypes = []

        def forbidden_sdpa(*args, **kwargs):
            raise AssertionError("Q-Former unexpectedly called fused SDPA")

        def traced_baddbmm(input_tensor, batch1, batch2, **kwargs):
            score_dtypes.append((input_tensor.dtype, batch1.dtype, batch2.dtype))
            # PyTorch 1.13 CPU can propagate uninitialized values from
            # Diffusers' beta=0 scratch tensor. Reproduce exactly the beta=0
            # math with bmm; the remote MUSA smoke exercises native baddbmm.
            self.assertEqual(kwargs.get("beta", 1), 0)
            return torch.bmm(batch1, batch2) * kwargs.get("alpha", 1)

        def traced_layer_norm(input_tensor, *args, **kwargs):
            norm_input_dtypes.append(input_tensor.dtype)
            return original_layer_norm(input_tensor, *args, **kwargs)

        functional.scaled_dot_product_attention = forbidden_sdpa
        functional.layer_norm = traced_layer_norm
        torch.baddbmm = traced_baddbmm
        try:
            features = torch.randn(2, 5, 16)
            output = model(features)
            self.assertEqual(tuple(output.shape), (2, 3, 36))
            self.assertTrue(torch.isfinite(output).all())
            target = torch.randn_like(output)
            (output.float() - target.float()).square().mean().backward()

            # Exercise the two explicit BF16 -> FP32 stability boundaries.
            bf16_norm_output = model.input_norm(
                torch.randn(2, 5, 16, dtype=torch.bfloat16)
            )
            self.assertIs(bf16_norm_output.dtype, torch.bfloat16)

            attention = model.qformer_layers[0].cross_attn.attn1
            attention_probs = attention.get_attention_scores(
                torch.randn(8, 3, 4, dtype=torch.bfloat16),
                torch.randn(8, 3, 4, dtype=torch.bfloat16),
            )
            self.assertTrue(torch.isfinite(attention_probs).all())
        finally:
            if original_sdpa is None:
                delattr(functional, "scaled_dot_product_attention")
            else:
                functional.scaled_dot_product_attention = original_sdpa
            torch.baddbmm = original_baddbmm
            functional.layer_norm = original_layer_norm

        self.assertEqual(len(score_dtypes), 5)
        self.assertTrue(
            all(
                dtype is torch.float32
                for dtypes in score_dtypes
                for dtype in dtypes
            )
        )
        self.assertTrue(norm_input_dtypes)
        self.assertEqual(set(norm_input_dtypes), {torch.float32})

        required_gradients = [
            model.query_tokens.grad,
            model.qformer_layers[0].cross_attn.attn1.to_q.weight.grad,
            model.qformer_layers[0].cross_attn.attn2.to_q.weight.grad,
            model.qformer_layers[-1].cross_attn.attn2.to_k.weight.grad,
            model.qformer_layers[-1].cross_attn.attn2.to_v.weight.grad,
            model.output_align_mlp[0].weight.grad,
            model.output_align_mlp[-1].weight.grad,
        ]
        for gradient in required_gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.float().abs().max()), 0.0)

        clone = QformerProjector(args, patches=5, channels=16)
        clone.load_state_dict(model.state_dict(), strict=True)
        original_state = model.state_dict()
        cloned_state = clone.state_dict()
        self.assertEqual(tuple(original_state), tuple(cloned_state))
        for name, value in original_state.items():
            self.assertTrue(
                torch.equal(value, cloned_state[name]),
                msg=f"state_dict tensor changed during strict round-trip: {name}",
            )


if __name__ == "__main__":
    unittest.main()
