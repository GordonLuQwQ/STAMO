"""CPU-only contract tests for the dependency-free FLUX.2 helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from stamo.renderer.model import flux2_utils


class Flux2Transformer2DModel:
    """Small stand-in whose class name matches the Diffusers contract."""

    def __init__(self, config):
        self.config = config


class AutoencoderKLFlux2:
    """Small stand-in whose class name matches the Diffusers contract."""

    def __init__(self, config, bn=object()):
        self.config = config
        self.bn = bn


def _transformer_config(**overrides):
    config = {
        "joint_attention_dim": 7_680,
        "in_channels": 128,
        "num_layers": 5,
        "num_single_layers": 20,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "patch_size": 1,
        "axes_dims_rope": (32, 32, 32, 32),
        "guidance_embeds": False,
    }
    config.update(overrides)
    return config


def _vae_config(**overrides):
    config = {
        "latent_channels": 32,
        "patch_size": (2, 2),
        "batch_norm_eps": 1e-4,
    }
    config.update(overrides)
    return config


class Flux2TensorHelperTests(unittest.TestCase):
    def test_patchify_pack_unpack_unpatchify_roundtrip(self):
        raw = torch.arange(2 * 32 * 4 * 6, dtype=torch.float32).reshape(
            2,
            32,
            4,
            6,
        )

        patchified = flux2_utils.patchify_latents(raw)
        packed = flux2_utils.pack_latents(patchified)
        unpacked = flux2_utils.unpack_latents(
            packed,
            latent_height=patchified.shape[-2],
            latent_width=patchified.shape[-1],
        )
        restored = flux2_utils.unpatchify_latents(unpacked)

        self.assertEqual(tuple(patchified.shape), (2, 128, 2, 3))
        self.assertEqual(tuple(packed.shape), (2, 6, 128))
        self.assertTrue(torch.equal(unpacked, patchified))
        self.assertTrue(torch.equal(restored, raw))

    def test_vae_bn_normalize_roundtrip(self):
        mean = torch.tensor([0.0, 1.0, -2.0, 0.5], dtype=torch.float64)
        variance = torch.tensor([1.0, 4.0, 9.0, 0.25], dtype=torch.float64)
        vae = AutoencoderKLFlux2(
            _vae_config(batch_norm_eps=1e-4),
            bn=SimpleNamespace(running_mean=mean, running_var=variance),
        )
        latents = torch.arange(16, dtype=torch.float64).reshape(1, 4, 2, 2)

        normalized = flux2_utils.normalize_vae_latents(vae, latents)
        expected = (latents - mean.reshape(1, 4, 1, 1)) / torch.sqrt(
            variance.reshape(1, 4, 1, 1) + 1e-4
        )
        restored = flux2_utils.denormalize_vae_latents(vae, normalized)

        torch.testing.assert_close(normalized, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(restored, latents, rtol=1e-12, atol=1e-12)

    def test_four_axis_image_and_text_ids_match_golden_values(self):
        latents = torch.zeros(2, 128, 2, 3)
        image_ids = flux2_utils.prepare_latent_ids(latents)
        expected_image_ids = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 2, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 2, 0],
            ],
            dtype=torch.int64,
        )

        self.assertEqual(tuple(image_ids.shape), (2, 6, 4))
        self.assertTrue(torch.equal(image_ids[0], expected_image_ids))
        self.assertTrue(torch.equal(image_ids[1], expected_image_ids))

        embeddings = torch.zeros(2, 3, 7)
        text_ids = flux2_utils.prepare_text_ids(embeddings)
        expected_text_ids = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 2],
            ],
            dtype=torch.int64,
        )

        self.assertEqual(tuple(text_ids.shape), (2, 3, 4))
        self.assertTrue(torch.equal(text_ids[0], expected_text_ids))
        self.assertTrue(torch.equal(text_ids[1], expected_text_ids))


class Flux2ArchitectureContractTests(unittest.TestCase):
    def test_exact_klein_base_4b_contract_is_accepted(self):
        transformer = Flux2Transformer2DModel(_transformer_config())
        vae = AutoencoderKLFlux2(_vae_config())

        self.assertIsNone(
            flux2_utils.validate_flux2_klein_4b_contract(
                transformer,
                vae,
                projector_output_dim=7_680,
            )
        )

    def test_non_4b_dimensions_are_rejected_field_by_field(self):
        transformer_mutations = {
            "joint_attention_dim": 12_288,
            "in_channels": 64,
            "num_layers": 8,
            "num_single_layers": 24,
            "attention_head_dim": 64,
            "num_attention_heads": 32,
            "patch_size": 2,
            "axes_dims_rope": (16, 56, 56),
        }
        for field, wrong_value in transformer_mutations.items():
            with self.subTest(field=field):
                transformer = Flux2Transformer2DModel(
                    _transformer_config(**{field: wrong_value})
                )
                vae = AutoencoderKLFlux2(_vae_config())
                with self.assertRaisesRegex(ValueError, field):
                    flux2_utils.validate_flux2_klein_4b_contract(
                        transformer,
                        vae,
                        projector_output_dim=7_680,
                    )

        with self.assertRaisesRegex(ValueError, "vae.latent_channels"):
            flux2_utils.validate_flux2_klein_4b_contract(
                Flux2Transformer2DModel(_transformer_config()),
                AutoencoderKLFlux2(_vae_config(latent_channels=16)),
                projector_output_dim=7_680,
            )
        with self.assertRaisesRegex(ValueError, "vae.patch_size"):
            flux2_utils.validate_flux2_klein_4b_contract(
                Flux2Transformer2DModel(_transformer_config()),
                AutoencoderKLFlux2(_vae_config(patch_size=(1, 1))),
                projector_output_dim=7_680,
            )
        with self.assertRaisesRegex(ValueError, "projector.output_align_dim"):
            flux2_utils.validate_flux2_klein_4b_contract(
                Flux2Transformer2DModel(_transformer_config()),
                AutoencoderKLFlux2(_vae_config()),
                projector_output_dim=4_096,
            )

    def test_distilled_guidance_and_missing_vae_bn_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "guidance_embeds=False"):
            flux2_utils.validate_flux2_klein_4b_contract(
                Flux2Transformer2DModel(
                    _transformer_config(guidance_embeds=True)
                ),
                AutoencoderKLFlux2(_vae_config()),
                projector_output_dim=7_680,
            )

        vae_without_bn = AutoencoderKLFlux2(_vae_config())
        del vae_without_bn.bn
        with self.assertRaisesRegex(ValueError, "BN latent statistics"):
            flux2_utils.validate_flux2_klein_4b_contract(
                Flux2Transformer2DModel(_transformer_config()),
                vae_without_bn,
                projector_output_dim=7_680,
            )

    def test_wrong_diffusers_model_classes_are_rejected(self):
        class WrongTransformer:
            config = _transformer_config()

        class WrongVae:
            config = _vae_config()
            bn = object()

        with self.assertRaisesRegex(TypeError, "Flux2Transformer2DModel"):
            flux2_utils.validate_flux2_klein_4b_contract(
                WrongTransformer(),
                AutoencoderKLFlux2(_vae_config()),
                projector_output_dim=7_680,
            )
        with self.assertRaisesRegex(TypeError, "AutoencoderKLFlux2"):
            flux2_utils.validate_flux2_klein_4b_contract(
                Flux2Transformer2DModel(_transformer_config()),
                WrongVae(),
                projector_output_dim=7_680,
            )


class Flux2LoraAndScheduleTests(unittest.TestCase):
    def test_scheduler_contract_accepts_base_and_rejects_wrong_sigma_behavior(self):
        base_config = {
            "num_train_timesteps": 1_000,
            "use_dynamic_shifting": True,
            "time_shift_type": "exponential",
            "stochastic_sampling": False,
            "invert_sigmas": False,
        }
        self.assertIsNone(
            flux2_utils.validate_flux2_scheduler_contract(
                SimpleNamespace(config=base_config)
            )
        )
        for key, wrong_value in {
            "num_train_timesteps": 100,
            "use_dynamic_shifting": False,
            "time_shift_type": "linear",
            "stochastic_sampling": True,
            "invert_sigmas": True,
        }.items():
            with self.subTest(key=key):
                config = dict(base_config)
                config[key] = wrong_value
                with self.assertRaisesRegex(ValueError, key):
                    flux2_utils.validate_flux2_scheduler_contract(
                        SimpleNamespace(config=config)
                    )

    def test_lora_targets_are_derived_from_twenty_single_blocks(self):
        transformer = SimpleNamespace(
            single_transformer_blocks=torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(20)]
            )
        )

        targets = flux2_utils.resolve_flux2_lora_targets(transformer)
        expected = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"]
        expected.extend(
            f"single_transformer_blocks.{index}.attn.to_out"
            for index in range(20)
        )

        self.assertEqual(targets, expected)
        self.assertEqual(len(targets), 25)
        self.assertEqual(len(set(targets)), len(targets))
        self.assertIn("single_transformer_blocks.19.attn.to_out", targets)
        self.assertNotIn("single_transformer_blocks.20.attn.to_out", targets)
        self.assertNotIn("single_transformer_blocks.23.attn.to_out", targets)

    def test_empirical_mu_matches_reference_points_and_boundary(self):
        cases = (
            (196, 4, 1.9604794624599917),
            (1_024, 4, 2.0306897079499455),
            (1_024, 50, 1.7019562073086316),
            (4_300, 200, 1.18452766),
            (4_301, 4, 1.18469693),
            (5_000, 4, 1.30301666),
            (5_000, 200, 1.30301666),
        )
        for image_seq_len, num_steps, expected in cases:
            with self.subTest(
                image_seq_len=image_seq_len,
                num_steps=num_steps,
            ):
                self.assertAlmostEqual(
                    flux2_utils.compute_empirical_mu(
                        image_seq_len,
                        num_steps,
                    ),
                    expected,
                    places=12,
                )


if __name__ == "__main__":
    unittest.main()
