"""Static integration gates for the active FLUX.2 Klein training path."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "stamo" / "renderer" / "model" / "renderer.py"
PROJECTOR = ROOT / "stamo" / "renderer" / "model" / "projector.py"
TRAINER = ROOT / "stamo" / "renderer" / "trainer.py"
ENTRYPOINT = ROOT / "train_renderer.py"
CONFIG = ROOT / "configs" / "flux.yaml"
VERIFIER = ROOT / "scripts" / "verify_flux2_klein_musa.sh"
CONFIG_MAKER = ROOT / "scripts" / "make_flux2_musa_smoke_config.py"
GENERATION_VERIFIER = ROOT / "scripts" / "verify_flux2_portable_generation.py"
MCCL_SMOKE = ROOT / "scripts" / "mccl_smoke_test.py"
VAE_SMOKE = ROOT / "scripts" / "flux2_vae_musa_smoke_test.py"
DATALOADER_SMOKE = ROOT / "scripts" / "dataloader_spawn_smoke_test.py"
METADATA_INDEX = ROOT / "stamo" / "renderer" / "utils" / "metadata_index.py"
VALIDATE_ENTRYPOINT = ROOT / "validate_renderer.py"
PROJECT = ROOT / "pyproject.toml"


def _load_checkpoint_metadata_selector():
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAINER))
    fields_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_FLUX2_DEEPSPEED_METADATA_FIELDS"
            for target in node.targets
        )
    )
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_select_stamo_deepspeed_checkpoint_metadata"
    )
    module = ast.Module(body=[fields_assignment, selector], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": object, "Dict": dict}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return (
        namespace["_FLUX2_DEEPSPEED_METADATA_FIELDS"],
        namespace["_select_stamo_deepspeed_checkpoint_metadata"],
    )


def _load_batchnorm_tracking_buffer_selector():
    source = RENDERER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RENDERER))
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_batchnorm_tracking_buffer_keys"
    )
    module = ast.Module(body=[selector], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"nn": nn}
    exec(compile(module, str(RENDERER), "exec"), namespace)
    return namespace["_batchnorm_tracking_buffer_keys"]


def _load_optimizer_group_builder():
    source = ENTRYPOINT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENTRYPOINT))
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_optimizer_groups"
    )
    module = ast.Module(body=[builder], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(ENTRYPOINT), "exec"), namespace)
    return namespace["build_optimizer_groups"]


def _load_deepspeed_microbatch_method():
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAINER))
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Trainer"
    )
    method = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "deepspeed_train_microbatch"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace["deepspeed_train_microbatch"]


CHECKPOINT_METADATA_FIELDS, select_checkpoint_metadata = (
    _load_checkpoint_metadata_selector()
)
batchnorm_tracking_buffer_keys = _load_batchnorm_tracking_buffer_selector()
build_optimizer_groups = _load_optimizer_group_builder()
deepspeed_train_microbatch = _load_deepspeed_microbatch_method()


class Flux2SourceContractTests(unittest.TestCase):
    def test_deepspeed_overflow_cannot_be_counted_as_an_optimizer_update(self):
        class Engine:
            def __init__(self, outcome):
                self.global_steps = 10
                self.skipped_steps = 3
                self._step_applied = False
                self.outcome = outcome

            def backward(self, loss):
                del loss

            def step(self):
                if self.outcome == "accumulate":
                    return
                self.global_steps += 1
                if self.outcome == "overflow":
                    self.skipped_steps += 1
                    self._step_applied = False
                else:
                    self._step_applied = True

        class Harness:
            deepspeed_train_microbatch = deepspeed_train_microbatch
            deepspeed_zero_stage = 2
            rank = 0

            def __init__(self, outcome):
                self.model = Engine(outcome)
                self._nonfinite_optimizer_baseline_checked = False

            def _prepare_nonfinite_gradient_probe(self, optimizer_step):
                del optimizer_step
                return False

            def _diagnose_replay_optimizer_state(self, **kwargs):
                del kwargs

            def _diagnose_replay_gradients(self, **kwargs):
                del kwargs

            def _trace_phase(self, *args, **kwargs):
                del args, kwargs

            def _validate_verifier_gradients(self):
                return None

            def _diagnostic_synchronize(self):
                return None

            def _distributed_consensus_or_raise(self, error, *, context):
                del context
                if error is not None:
                    raise error

        self.assertEqual(Harness("normal").deepspeed_train_microbatch(None), (True, 11))
        self.assertEqual(Harness("accumulate").deepspeed_train_microbatch(None), (False, 10))
        with self.assertRaisesRegex(FloatingPointError, "skipped_steps 3->4"):
            Harness("overflow").deepspeed_train_microbatch(None)

    def test_lora_dense_io_uses_its_own_conservative_learning_rate(self):
        class Config(dict):
            __getattr__ = dict.__getitem__

        class Args:
            train = Config(
                learning_rate=1e-5,
                lora_learning_rate=2e-5,
                transformer_io_learning_rate=1e-6,
                decay=1e-3,
            )

        class Model(nn.Module):
            transformer_training_mode = "lora"

            def __init__(self):
                super().__init__()
                self.DiT = nn.Module()
                self.DiT.lora_A = nn.Parameter(torch.ones(2, 2))
                self.DiT.context_embedder = nn.Linear(2, 2, bias=False)
                self.projector = nn.Linear(2, 2, bias=False)

        model = Model()
        groups = build_optimizer_groups(model, Args())
        lr_by_parameter = {
            id(parameter): float(group["lr"])
            for group in groups
            for parameter in group["params"]
        }
        self.assertEqual(lr_by_parameter[id(model.DiT.lora_A)], 2e-5)
        self.assertEqual(lr_by_parameter[id(model.DiT.context_embedder.weight)], 1e-6)
        self.assertEqual(lr_by_parameter[id(model.projector.weight)], 1e-5)

    def test_active_python_entrypoints_parse(self):
        for path in (
            RENDERER,
            TRAINER,
            ENTRYPOINT,
            CONFIG_MAKER,
            GENERATION_VERIFIER,
            MCCL_SMOKE,
            VAE_SMOKE,
            DATALOADER_SMOKE,
            METADATA_INDEX,
            VALIDATE_ENTRYPOINT,
        ):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_renderer_uses_exact_flux2_classes_and_fp32_training_timesteps(self):
        source = RENDERER.read_text(encoding="utf-8")
        self.assertIn("Flux2Transformer2DModel.from_pretrained", source)
        self.assertIn("AutoencoderKLFlux2.from_pretrained", source)
        self.assertIn("timestep=timesteps / 1000", source)
        self.assertNotIn("timesteps.to(packed_noisy_latents.dtype) / 1000", source)
        self.assertIn('self.model_variant = "base-4b"', source)
        self.assertIn('model_index.get("is_distilled") is True', source)

    def test_musa_uses_stable_legacy_attention_for_vae_and_qformer(self):
        renderer = RENDERER.read_text(encoding="utf-8")
        projector = PROJECTOR.read_text(encoding="utf-8")
        config = CONFIG.read_text(encoding="utf-8")
        trainer = TRAINER.read_text(encoding="utf-8")

        self.assertIn('musa_vae_attention_backend: "legacy"', config)
        self.assertIn('attention_backend: "legacy_upcast"', config)
        self.assertIn("self.vae.set_default_attn_processor()", renderer)
        self.assertIn('processor_types != {"AttnProcessor"}', renderer)
        self.assertIn("attention.set_processor(AttnProcessor())", projector)
        self.assertIn("upcast_attention=self.upcast_attention", projector)
        self.assertNotIn("self.DiT.set_default_attn_processor()", renderer)
        self.assertNotIn("self.projector.set_default_attn_processor()", renderer)
        self.assertIn('"musa_vae_attention_backend":', trainer)
        self.assertIn("STAMO_VAE_ATTENTION_BACKEND=", trainer)

    def test_verifier_phase_trace_brackets_loss_device_sync_and_consensus(self):
        source = TRAINER.read_text(encoding="utf-8")
        validation_source = source[
            source.index("    def _validate_verifier_loss") :
            source.index("    def _validate_verifier_gradients")
        ]
        validation_markers = (
            'self._trace_phase("BEFORE_LOSS_FINITE_CHECK")',
            '"AFTER_LOSS_FINITE_CHECK"',
            'self._trace_phase("BEFORE_LOSS_CONSENSUS")',
            'self._trace_phase("AFTER_LOSS_CONSENSUS")',
        )
        offsets = [validation_source.index(marker) for marker in validation_markers]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(
            source.count('self._trace_phase("AFTER_MODEL_FORWARD_RETURN")'),
            2,
        )

    def test_production_nonfinite_guards_precede_optimizer_updates(self):
        trainer = TRAINER.read_text(encoding="utf-8")
        entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

        loss_validation = trainer[
            trainer.index("    def _validate_verifier_loss") :
            trainer.index("    def _validate_verifier_gradients")
        ]
        self.assertIn("self.abort_on_nonfinite_loss", loss_validation)
        self.assertIn("_distributed_consensus_or_raise", loss_validation)
        loss_call = trainer.index(
            "self._validate_verifier_loss(\n"
            "                        outputs[\"loss\"],"
        )
        update_call = trainer.index(
            "optimizer_updated, engine_step = self.deepspeed_train_microbatch("
        )
        self.assertLess(loss_call, update_call)

        contract_source = trainer[
            trainer.index("self.training_contract.update(") :
            trainer.index("parameter_names_by_id =")
        ]
        self.assertNotIn('"abort_on_nonfinite_loss"', contract_source)
        self.assertIn('"check_grad_overflow": self.deepspeed_bf16', trainer)
        self.assertIn("STAMO_BF16_GRAD_OVERFLOW_GUARD=1", trainer)
        self.assertIn("skipped_steps_before", trainer)
        self.assertIn("_step_applied", trainer)
        self.assertIn("transformer_io_learning_rate", entrypoint)
        self.assertIn('or "lora_" in name', entrypoint)
        self.assertIn("STAMO_OPTIMIZER_GROUP", entrypoint)

    def test_nan_replay_separates_forward_backward_reduction_and_optimizer(self):
        trainer = TRAINER.read_text(encoding="utf-8")
        renderer = RENDERER.read_text(encoding="utf-8")

        self.assertLess(
            trainer.index("self._install_nonfinite_gradient_hooks()"),
            trainer.index(
                "self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize("
            ),
        )
        replay_source = trainer[
            trainer.index("    def _diagnose_replay_gradients") :
            trainer.index("    def _diagnose_replay_optimizer_state")
        ]
        self.assertIn("get_lp_grad_fragment", replay_source)
        self.assertIn("_index_in_param_group", replay_source)
        self.assertNotIn("params_in_partition", replay_source)
        self.assertIn("model_backward_before_gradient_reduction", replay_source)
        self.assertIn("zero2_gradient_reduction_or_mccl", replay_source)
        self.assertIn("global_failure_flags", replay_source)
        self.assertIn("DIST.ReduceOp.MAX", replay_source)

        optimizer_start = trainer.index(
            "    def _diagnose_replay_optimizer_state"
        )
        optimizer_source = trainer[
            optimizer_start : trainer.index("    def close", optimizer_start)
        ]
        self.assertIn("single_partition_of_fp32_groups", optimizer_source)
        self.assertIn('("exp_avg", "exp_avg_sq")', optimizer_source)
        self.assertNotIn("state_dict()", optimizer_source)

        self.assertIn('"_nonfinite_backward_diagnostics"', renderer)
        self.assertIn('"packed_model_pred_grad"', renderer)
        self.assertIn('"condition_embeddings_grad"', renderer)
        self.assertIn('"residual_fp32"', renderer)

        microbatch_source = trainer[
            trainer.index("    def deepspeed_train_microbatch") :
            trainer.index("    def reduce_mean")
        ]
        self.assertIn('phase="post_engine_step"', microbatch_source)
        self.assertNotIn(
            "replay_probe_active and optimizer_boundary and step_applied",
            microbatch_source,
        )

    def test_training_entrypoint_rejects_unsupported_deepspeed_versions(self):
        source = ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn('Version("0.17.2") <= deepspeed_version < Version("0.18")', source)
        self.assertIn("verify_rank_config_identity(args)", source)
        self.assertIn("STAMO_RANK_CONFIG_PASS", source)
        self.assertIn("distributed_startup_call(", source)

    def test_portable_checkpoint_binds_to_frozen_base_fingerprints(self):
        source = RENDERER.read_text(encoding="utf-8")
        trainer = TRAINER.read_text(encoding="utf-8")
        self.assertIn("sampled_tree_fingerprint", source)
        self.assertIn("sampled_file_fingerprint", source)
        self.assertIn('"checkpoint_version": 3', source)
        self.assertGreaterEqual(source.count('"base_model_fingerprints"'), 2)
        self.assertGreaterEqual(source.count('"projector_contract"'), 2)
        self.assertIn('"projector_contract": dict(projector_contract)', trainer)

    def test_portable_load_preserves_excluded_batchnorm_tracking_buffers(self):
        module = nn.Module()
        module.vae = nn.Module()
        module.vae.bn = nn.BatchNorm2d(4)
        module.projector = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))

        self.assertEqual(
            batchnorm_tracking_buffer_keys(module),
            {"vae.bn.num_batches_tracked"},
        )

        current_state = module.state_dict()
        preserved = batchnorm_tracking_buffer_keys(module)
        partial_state = {key: current_state[key] for key in preserved}
        incompatible = module.load_state_dict(partial_state, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertEqual(
            set(incompatible.missing_keys),
            set(current_state) - preserved,
        )

        source = RENDERER.read_text(encoding="utf-8")
        self.assertIn("preserved_batchnorm_buffers", source)
        self.assertIn("checkpoint_model_for_load.update", source)
        self.assertIn(
            "expected_reported_missing = allowed_missing - preserved_batchnorm_buffers",
            source,
        )

    def test_trainer_enforces_zero2_and_bounds_startup_collectives(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn("FLUX.2 Klein 4B uses ZeRO stage 2", source)
        self.assertIn('DeepSpeedEngine._broadcast_model = _skip_redundant_model_broadcast', source)
        self.assertIn('DeepSpeedEngine._broadcast_model = original_broadcast_model', source)
        self.assertIn("self.verify_model_checksum()", source)
        self.assertIn('tag="startup-training-contract"', source)
        self.assertIn('load_kwargs["tag"] = selected_tag', source)
        self.assertIn("STAMO_PARAMETER_AUDIT_PASS", source)
        self.assertIn("averaged_gradients", source)
        self.assertIn("source=zero2_fp32_master", source)
        self.assertIn("transformer_io_master_changed=1", source)
        self.assertIn("_zero2_fp32_fragment_record", source)
        self.assertIn("allow_extra_fields=True", source)
        self.assertIn("if client_metadata != checkpoint_manifest", source)
        self.assertNotIn("if client_state != checkpoint_manifest", source)
        for contract_key in (
            "deepspeed_skip_model_broadcast",
            "verify_step_numerics",
            "verify_parameter_updates",
            "enable_checkpointing",
            "enable_eval",
        ):
            self.assertIn(f'"{contract_key}":', source)

    def test_deepspeed_internal_client_state_fields_do_not_break_resume(self):
        manifest = {
            field: index
            for index, field in enumerate(sorted(CHECKPOINT_METADATA_FIELDS))
        }
        client_state = {
            **manifest,
            "buffer_names": ["vae.running_mean"],
            "param_shapes": object(),
            "frozen_param_shapes": None,
            "shared_params": {},
            "frozen_param_fragments": None,
            "global_samples": 8,
            "ds_config": {"zero_optimization": {"stage": 2}},
            "ds_version": "0.17.2",
        }

        selected = select_checkpoint_metadata(
            client_state,
            source="DeepSpeed client_state",
            allow_extra_fields=True,
        )
        self.assertEqual(selected, manifest)

        with self.assertRaisesRegex(RuntimeError, "unexpected fields"):
            select_checkpoint_metadata(
                client_state,
                source="checkpoint manifest",
                allow_extra_fields=False,
            )

        incomplete = dict(manifest)
        incomplete.pop("global_step")
        with self.assertRaisesRegex(RuntimeError, "missing.*global_step"):
            select_checkpoint_metadata(
                incomplete,
                source="DeepSpeed client_state",
                allow_extra_fields=True,
            )

        tampered = dict(client_state)
        tampered["global_step"] = int(manifest["global_step"]) + 1
        selected_tampered = select_checkpoint_metadata(
            tampered,
            source="DeepSpeed client_state",
            allow_extra_fields=True,
        )
        self.assertNotEqual(selected_tampered, manifest)

    def test_portable_validation_does_not_turn_into_a_training_launch(self):
        trainer = TRAINER.read_text(encoding="utf-8")
        validator = VALIDATE_ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn("transformer_training_run = self.do_train and self.train_transformer", trainer)
        self.assertIn("requires_dataset_identity = self.do_train or self.use_deepspeed", trainer)
        self.assertIn("args.do_train = False", validator)
        self.assertIn("without --deepspeed", validator)

    def test_production_yaml_is_lora_bf16_serial_zero2(self):
        source = CONFIG.read_text(encoding="utf-8")
        expected_patterns = (
            r'(?m)^\s*training_mode:\s*["\']?lora["\']?\s*$',
            r'(?m)^\s*mixed_precision:\s*["\']?bf16["\']?\s*$',
            r"(?m)^\s*deepspeed_zero_stage:\s*2\s*$",
            r"(?m)^\s*deepspeed_overlap_comm:\s*false\s*$",
            r"(?m)^\s*deepspeed_skip_model_broadcast:\s*true\s*$",
            r"(?m)^\s*abort_on_nonfinite_loss:\s*true\s*$",
            r"(?m)^\s*learning_rate:\s*1e-5\s*$",
            r"(?m)^\s*lora_learning_rate:\s*2e-5\s*$",
            r"(?m)^\s*transformer_io_learning_rate:\s*1e-6\s*$",
            r"(?m)^\s*local_batch_size:\s*16\s*$",
            r"(?m)^\s*num_workers:\s*10\s*$",
            r'(?m)^\s*worker_start_method:\s*["\']spawn["\']\s*$',
            r"(?m)^\s*prefetch_factor:\s*1\s*$",
            r"(?m)^\s*eval_num_workers:\s*1\s*$",
        )
        for pattern in expected_patterns:
            self.assertRegex(source, pattern)
        self.assertNotRegex(source, r"(?m)^\s*deepspeed_zero_stage:\s*3\s*$")

    def test_verifier_is_flux2_zero2_specific_and_can_test_resume(self):
        verifier = VERIFIER.read_text(encoding="utf-8")
        maker = CONFIG_MAKER.read_text(encoding="utf-8")
        self.assertIn("make_flux2_musa_smoke_config.py", verifier)
        self.assertIn("STAMO_TRAINING_COMPLETE", verifier)
        self.assertIn("VERIFY_CHECKPOINT_RESUME", verifier)
        self.assertIn("MASTER_PORT > 65534", verifier)
        self.assertIn('PYTHONPATH="${REPO_ROOT}/tests:${REPO_ROOT}', verifier)
        self.assertNotIn("tests.test_fingerprint", verifier)
        self.assertIn("test_deepspeed_zero3_compat", verifier)
        self.assertIn("Required FLUX.2 deployment file is missing or empty", verifier)
        self.assertIn("stamo_checkpoint_manifest.json", verifier)
        self.assertIn("verify_flux2_portable_generation.py", verifier)
        self.assertIn("STAMO_PORTABLE_GENERATION_PASS", verifier)
        self.assertIn("Unsupported MUSA FlashAttention fallback was still used", verifier)
        self.assertIn("scripts/mccl_smoke_test.py", verifier)
        self.assertIn("scripts/flux2_vae_musa_smoke_test.py", verifier)
        self.assertIn("scripts/dataloader_spawn_smoke_test.py", verifier)
        self.assertIn("STAMO_DATALOADER_SPAWN_SMOKE_PASS", DATALOADER_SMOKE.read_text(encoding="utf-8"))
        self.assertIn("source=zero2_fp32_master", verifier)
        self.assertIn('importlib.metadata.version("deepspeed")', verifier)
        self.assertIn('VERIFY_NUM_WORKERS="${VERIFY_NUM_WORKERS:-10}"', verifier)
        self.assertIn("STAMO_CPU_DATALOADER_WORKER_READY", verifier)
        self.assertIn("STAMO_DATALOADER_FULL_POOL_PASS", verifier)
        self.assertIn("unique_workers={len(expected)}", verifier)
        self.assertIn("timeout --signal=TERM --kill-after=30s", verifier)
        self.assertNotIn("verify_zero3_musa.sh", verifier)
        self.assertIn("config.train.deepspeed_zero_stage = 2", maker)
        self.assertIn("config.data.num_workers = cli.num_workers", maker)
        self.assertIn('config.data.worker_start_method = "spawn"', maker)
        self.assertIn("config.train.verify_parameter_updates = True", maker)
        self.assertNotIn("deepspeed_zero_stage = 3", maker)

    def test_mccl_preflight_covers_training_contract_dtypes(self):
        source = MCCL_SMOKE.read_text(encoding="utf-8")
        for operation in (
            "all_reduce_int32_min",
            "all_reduce_int32_max",
            "all_reduce_fp32_min",
            "all_reduce_fp32_sum",
            "all_gather_fp32",
        ):
            self.assertIn(operation, source)
        self.assertIn("dtype=torch.int32", source)
        self.assertIn("dtype=torch.float32", source)

    def test_vae_preflight_exercises_every_rank_without_sdpa_fallback(self):
        source = VAE_SMOKE.read_text(encoding="utf-8")
        verifier = VERIFIER.read_text(encoding="utf-8")
        self.assertIn("vae.set_default_attn_processor()", source)
        self.assertIn('processor_types != {"AttnProcessor"}', source)
        self.assertIn("musa.synchronize()", source)
        self.assertIn("dist.all_reduce(health, op=dist.ReduceOp.MIN)", source)
        self.assertIn("VAE_MUSA_BEGIN rank=", source)
        self.assertIn("VAE_MUSA_COMPUTE_END rank=", source)
        self.assertIn("VAE_MUSA_END rank=", source)
        self.assertIn("VAE_MUSA_SMOKE_PASS", source)
        self.assertIn("VERIFY_VAE_ITERATIONS", verifier)
        self.assertIn("Unsupported MUSA VAE FlashAttention fallback", verifier)
        self.assertIn("EXPECTED_MUSA_VISIBLE_DEVICES", verifier)
        self.assertIn("therefore requires", verifier)
        self.assertIn("MUSA_VISIBLE_DEVICES=${EXPECTED_MUSA_VISIBLE_DEVICES}", verifier)
        self.assertIn("image_size != 224", source)
        self.assertRegex(
            verifier,
            r'--master_port="\$\{MCCL_MASTER_PORT\}"\s*\\\n\s*scripts/mccl_smoke_test\.py',
        )
        self.assertRegex(
            verifier,
            r'--master_port="\$\{VAE_MASTER_PORT\}"\s*\\\n\s*scripts/flux2_vae_musa_smoke_test\.py',
        )

    def test_dependency_floor_matches_flux2_support(self):
        source = PROJECT.read_text(encoding="utf-8")
        self.assertIn('"diffusers==0.39.0"', source)
        self.assertIn('"peft>=0.17.0,<1.0"', source)
        self.assertIn('"deepspeed>=0.17.2,<0.18"', source)


if __name__ == "__main__":
    unittest.main()
