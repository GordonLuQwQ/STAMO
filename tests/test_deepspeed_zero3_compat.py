"""Logic tests for STAMO's process-local DeepSpeed ZeRO-3 backport."""

import ast
import importlib.util
import inspect
import os
import re
import runpy
import sys
import tempfile
import types
import unittest
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "stamo" / "renderer" / "trainer.py"
TRAIN_ENTRYPOINT_PATH = ROOT / "train_renderer.py"
MCCL_SMOKE_PATH = ROOT / "scripts" / "mccl_smoke_test.py"
PRODUCTION_CONFIG_PATH = ROOT / "configs" / "flux.yaml"
VERIFIER_PATH = ROOT / "scripts" / "verify_zero3_musa.sh"
PERF_CONFIG_PATH = ROOT / "scripts" / "make_musa_perf_config.py"
PERF_RUNNER_PATH = ROOT / "scripts" / "run_musa_perf_trial.sh"
OPTIMIZER_PATH = ROOT / "stamo" / "renderer" / "utils" / "optim.py"
DATA_UTILS_PATH = ROOT / "stamo" / "renderer" / "utils" / "data.py"
DEVICE_UTILS_PATH = ROOT / "stamo" / "renderer" / "utils" / "device.py"
METHOD_NAME = (
    "_DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads"
)


def _load_helpers():
    """Load only the pure-Python helpers without importing the training stack."""
    source = TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAINER_PATH))
    wanted = {
        "_ensure_deepspeed_zero3_ipg_bucket_reset_compatibility",
        "_assert_deepspeed_zero3_ipg_buckets_cleared",
    }
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in helper_nodes} != wanted:
        raise AssertionError("Could not find the ZeRO-3 compatibility helpers")

    module = ast.Module(body=helper_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "inspect": inspect,
        "re": re,
        "wraps": wraps,
    }
    exec(compile(module, str(TRAINER_PATH), "exec"), namespace)
    return (
        namespace[
            "_ensure_deepspeed_zero3_ipg_bucket_reset_compatibility"
        ],
        namespace["_assert_deepspeed_zero3_ipg_buckets_cleared"],
    )


ensure_bucket_reset, assert_buckets_cleared = _load_helpers()


def _load_bounded_leaf_helper():
    source = TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAINER_PATH))
    helper_names = {
        "_parameterized_hook_site_count",
        "_bounded_zero3_leaf_candidates",
    }
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    if {node.name for node in helpers} != helper_names:
        raise AssertionError("Could not find bounded ZeRO-3 leaf helpers")
    module = ast.Module(body=helpers, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(TRAINER_PATH), "exec"), namespace)
    return namespace["_bounded_zero3_leaf_candidates"]


bounded_leaf_candidates = _load_bounded_leaf_helper()


def _load_nonnegative_float_helper():
    source = TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAINER_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_as_nonnegative_float"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": __import__("math")}
    exec(compile(module, str(TRAINER_PATH), "exec"), namespace)
    return namespace["_as_nonnegative_float"]


as_nonnegative_float = _load_nonnegative_float_helper()


def _load_get_optimizer():
    source = OPTIMIZER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OPTIMIZER_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_optimizer"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "optim": torch.optim}
    exec(compile(module, str(OPTIMIZER_PATH), "exec"), namespace)
    return namespace["get_optimizer"]


get_optimizer_helper = _load_get_optimizer()


class Bucket:
    def __init__(self, params=None, elements=0):
        self.params = list(params or [])
        self.elements = int(elements)


class FixedZeroOptimizer:
    def fixed_reduce(self, communication_data_type, safe_mode=False):
        del safe_mode
        bucket = self.ipg_buckets[communication_data_type]
        params_in_bucket = bucket.params
        params_in_bucket.clear()
        bucket.elements = 0


setattr(FixedZeroOptimizer, METHOD_NAME, FixedZeroOptimizer.fixed_reduce)


def _make_optimizer_class(reduce_method):
    optimizer_class = type("DeepSpeedZeroOptimizer_Stage3", (), {})
    setattr(optimizer_class, METHOD_NAME, reduce_method)
    return optimizer_class


class DeepSpeedZero3BucketResetTests(unittest.TestCase):
    def test_memory_admission_float_rejects_nonfinite_values(self):
        self.assertEqual(as_nonnegative_float(0, "value"), 0.0)
        self.assertEqual(as_nonnegative_float("1.5", "value"), 1.5)
        for value in (-1, float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "finite non-negative"):
                as_nonnegative_float(value, "value")

    def test_deployment_sources_are_not_concatenated(self):
        trainer_source = TRAINER_PATH.read_text(encoding="utf-8")
        trainer_tree = ast.parse(trainer_source, filename=str(TRAINER_PATH))
        entrypoint_source = TRAIN_ENTRYPOINT_PATH.read_text(encoding="utf-8")
        entrypoint_tree = ast.parse(
            entrypoint_source,
            filename=str(TRAIN_ENTRYPOINT_PATH),
        )

        definitions = (
            (trainer_tree, ast.ClassDef, "Trainer"),
            (
                trainer_tree,
                ast.FunctionDef,
                "_ensure_deepspeed_zero3_ipg_bucket_reset_compatibility",
            ),
            (
                trainer_tree,
                ast.FunctionDef,
                "_assert_deepspeed_zero3_ipg_buckets_cleared",
            ),
            (entrypoint_tree, ast.FunctionDef, "main"),
        )
        for tree, node_type, name in definitions:
            count = sum(
                isinstance(node, node_type) and node.name == name
                for node in tree.body
            )
            self.assertEqual(
                count,
                1,
                f"expected one top-level {name}; source may be concatenated",
            )

    def test_buggy_0172_method_is_patched_and_resets_counter(self):
        def buggy_reduce(self, communication_data_type, safe_mode=False):
            del safe_mode
            self.ipg_buckets[communication_data_type].params.clear()
            return "reduced"

        optimizer_class = _make_optimizer_class(buggy_reduce)
        self.assertTrue(ensure_bucket_reset(optimizer_class))

        optimizer = optimizer_class()
        optimizer.ipg_buckets = {
            "bf16": Bucket(params=[object(), object()], elements=4096)
        }
        result = getattr(optimizer, METHOD_NAME)("bf16")

        self.assertEqual(result, "reduced")
        self.assertEqual(optimizer.ipg_buckets["bf16"].params, [])
        self.assertEqual(optimizer.ipg_buckets["bf16"].elements, 0)
        self.assertEqual(optimizer._stamo_zero3_ipg_bucket_reset_count, 1)
        assert_buckets_cleared(optimizer)

    def test_patch_installation_is_idempotent(self):
        def buggy_reduce(self, communication_data_type, safe_mode=False):
            del safe_mode
            self.ipg_buckets[communication_data_type].params.clear()

        optimizer_class = _make_optimizer_class(buggy_reduce)
        self.assertTrue(ensure_bucket_reset(optimizer_class))
        installed_method = getattr(optimizer_class, METHOD_NAME)

        self.assertFalse(ensure_bucket_reset(optimizer_class))
        self.assertIs(getattr(optimizer_class, METHOD_NAME), installed_method)

    def test_upstream_0173_fix_is_not_wrapped(self):
        original_method = getattr(FixedZeroOptimizer, METHOD_NAME)

        self.assertFalse(ensure_bucket_reset(FixedZeroOptimizer))
        self.assertIs(getattr(FixedZeroOptimizer, METHOD_NAME), original_method)

    def test_original_exception_is_preserved_without_resetting_state(self):
        def failing_reduce(self, communication_data_type, safe_mode=False):
            del self, communication_data_type, safe_mode
            raise RuntimeError("collective failed")

        optimizer_class = _make_optimizer_class(failing_reduce)
        self.assertTrue(ensure_bucket_reset(optimizer_class))
        optimizer = optimizer_class()
        optimizer.ipg_buckets = {
            "bf16": Bucket(params=[object()], elements=1024)
        }

        with self.assertRaisesRegex(RuntimeError, "collective failed"):
            getattr(optimizer, METHOD_NAME)("bf16")
        self.assertEqual(len(optimizer.ipg_buckets["bf16"].params), 1)
        self.assertEqual(optimizer.ipg_buckets["bf16"].elements, 1024)

    def test_changed_private_lifecycle_fails_fast(self):
        def incompatible_reduce(
            self,
            communication_data_type,
            safe_mode=False,
        ):
            del self, communication_data_type, safe_mode

        optimizer_class = _make_optimizer_class(incompatible_reduce)
        self.assertTrue(ensure_bucket_reset(optimizer_class))
        optimizer = optimizer_class()
        optimizer.ipg_buckets = {
            "bf16": Bucket(params=[object()], elements=1024)
        }

        with self.assertRaisesRegex(RuntimeError, "without clearing"):
            getattr(optimizer, METHOD_NAME)("bf16")
        self.assertEqual(optimizer.ipg_buckets["bf16"].elements, 1024)

    def test_missing_private_method_fails_before_training(self):
        class UnsupportedZeroOptimizer:
            pass

        with self.assertRaisesRegex(RuntimeError, "missing"):
            ensure_bucket_reset(UnsupportedZeroOptimizer)

    def test_stale_bucket_detector_reports_0172_state(self):
        class Optimizer:
            ipg_buckets = {
                "bf16": Bucket(params=[], elements=50_000_000),
                "fp32": Bucket(),
            }

        with self.assertRaisesRegex(RuntimeError, "#7418"):
            assert_buckets_cleared(Optimizer())

    def test_post_backward_detector_rejects_nonempty_bucket(self):
        class Optimizer:
            ipg_buckets = {
                "bf16": Bucket(params=[object()], elements=1024),
            }

        with self.assertRaisesRegex(RuntimeError, "without fully clearing"):
            assert_buckets_cleared(Optimizer())

    def test_empty_bucket_call_does_not_claim_a_real_reduction(self):
        def buggy_reduce(self, communication_data_type, safe_mode=False):
            del safe_mode
            self.ipg_buckets[communication_data_type].params.clear()

        optimizer_class = _make_optimizer_class(buggy_reduce)
        self.assertTrue(ensure_bucket_reset(optimizer_class))
        optimizer = optimizer_class()
        optimizer.ipg_buckets = {"bf16": Bucket()}

        getattr(optimizer, METHOD_NAME)("bf16")

        self.assertEqual(
            getattr(optimizer, "_stamo_zero3_ipg_bucket_reset_count", 0),
            0,
        )
        with self.assertRaisesRegex(RuntimeError, "did not process any non-empty"):
            assert_buckets_cleared(optimizer)

    def test_patch_prevents_one_collective_per_parameter_degradation(self):
        def buggy_reduce(self, communication_data_type, safe_mode=False):
            del safe_mode
            self.collective_count += 1
            self.ipg_buckets[communication_data_type].params.clear()

        def drive_backward(optimizer):
            bucket = optimizer.ipg_buckets["bf16"]
            reduce_method = getattr(optimizer, METHOD_NAME)
            for _ in range(100):
                parameter_elements = 10
                if (
                    bucket.elements + parameter_elements > 100
                    and bucket.elements > 0
                ):
                    reduce_method("bf16")
                bucket.params.append(object())
                bucket.elements += parameter_elements
            reduce_method("bf16")

        unpatched_class = _make_optimizer_class(buggy_reduce)
        unpatched = unpatched_class()
        unpatched.collective_count = 0
        unpatched.ipg_buckets = {"bf16": Bucket()}
        drive_backward(unpatched)
        self.assertEqual(unpatched.collective_count, 91)

        patched_class = _make_optimizer_class(buggy_reduce)
        self.assertTrue(ensure_bucket_reset(patched_class))
        patched = patched_class()
        patched.collective_count = 0
        patched.ipg_buckets = {"bf16": Bucket()}
        drive_backward(patched)
        self.assertEqual(patched.collective_count, 10)
        self.assertEqual(patched.ipg_buckets["bf16"].elements, 0)

    def test_trainer_applies_patch_to_every_zero3_run(self):
        source = TRAINER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAINER_PATH))
        trainer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Trainer"
        )
        prepare_method = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "prepare_dist_model"
        )
        use_deepspeed_branch = next(
            node for node in prepare_method.body if isinstance(node, ast.If)
        )

        zero3_branches = []
        for node in use_deepspeed_branch.body:
            if not isinstance(node, ast.If):
                continue
            test_source = ast.get_source_segment(source, node.test) or ""
            if (
                "self.deepspeed_zero_stage == 3" in test_source
                and "use_all_reduce_for_fetch_params" not in test_source
            ):
                zero3_branches.append(node)

        self.assertEqual(len(zero3_branches), 1)
        calls = {
            call.func.id
            for call in ast.walk(zero3_branches[0])
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        self.assertIn(
            "_ensure_deepspeed_zero3_ipg_bucket_reset_compatibility",
            calls,
        )
        self.assertNotIn(
            "_ensure_deepspeed_singleton_fetch_all_reduce_compatibility",
            source,
        )

    def test_checkpointing_switch_guards_periodic_and_final_saves(self):
        source = TRAINER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAINER_PATH))
        trainer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Trainer"
        )
        train_method = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "train_eval_by_iter"
        )

        guarded_save_conditions = []
        for node in ast.walk(train_method):
            if not isinstance(node, ast.If):
                continue
            directly_saves = any(
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "save_checkpoint"
                for statement in node.body
            )
            if directly_saves:
                guarded_save_conditions.append(ast.unparse(node.test))

        self.assertEqual(len(guarded_save_conditions), 2)
        for condition in guarded_save_conditions:
            self.assertIn("self.enable_checkpointing", condition)

    def test_training_entrypoint_supports_the_isolated_smoke_contract(self):
        source = TRAIN_ENTRYPOINT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAIN_ENTRYPOINT_PATH))
        main_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        called_attributes = {
            call.func.attr
            for call in ast.walk(main_function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
        }

        self.assertIn('args.train.get("enable_eval", True)', source)
        self.assertIn("configured_num_iters", source)
        self.assertIn("train_eval_by_iter", called_attributes)
        self.assertIn("close", called_attributes)
        self.assertIn("dist.destroy_process_group()", source)

    def test_mccl_blocking_wait_is_configured_before_process_group_init(self):
        source = TRAIN_ENTRYPOINT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAIN_ENTRYPOINT_PATH))
        top_level_calls = [
            node.value.func.id
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ]

        self.assertIn("_configure_musa_mccl_blocking_wait", top_level_calls)
        self.assertIn("_initialize_launcher_mccl", top_level_calls)
        self.assertLess(
            top_level_calls.index("_configure_musa_mccl_blocking_wait"),
            top_level_calls.index("_initialize_launcher_mccl"),
        )
        self.assertIn(
            'os.environ.setdefault("TORCH_MCCL_BLOCKING_WAIT", "1")',
            source,
        )

    def test_dataloader_spawn_child_skips_musa_binding_and_mccl_init(self):
        source = TRAIN_ENTRYPOINT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAIN_ENTRYPOINT_PATH))
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {"_bind_launcher_musa_device", "_initialize_launcher_mccl"}
        ]
        self.assertEqual(len(helpers), 2)

        # Execute only the two helpers in the exact module identity Python's
        # multiprocessing spawn machinery gives to a re-executed entrypoint.
        # No torch/os/dist stubs are supplied intentionally: reaching any
        # launcher logic would fail this test immediately with NameError.
        module = ast.Module(body=helpers, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"__name__": "__mp_main__"}
        exec(
            compile(module, str(TRAIN_ENTRYPOINT_PATH), "exec"),
            namespace,
        )
        namespace["_bind_launcher_musa_device"]()
        namespace["_initialize_launcher_mccl"]()

    def test_dataloader_spawn_child_isolated_before_project_imports(self):
        source = TRAIN_ENTRYPOINT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAIN_ENTRYPOINT_PATH))
        environment_keys = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_DATALOADER_DISTRIBUTED_ENV_KEYS"
                for target in node.targets
            )
        )
        isolate = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_isolate_dataloader_spawn_worker_environment"
        )
        module = ast.Module(body=[environment_keys, isolate], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"os": os}
        exec(
            compile(module, str(TRAIN_ENTRYPOINT_PATH), "exec"),
            namespace,
        )

        inherited = {
            "RANK": "3",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "3",
            "LOCAL_WORLD_SIZE": "8",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "52458",
            "GROUP_RANK": "0",
            "ROLE_RANK": "3",
            "ROLE_WORLD_SIZE": "8",
            "DS_ACCELERATOR": "musa",
        }
        with patch.dict(os.environ, inherited, clear=True):
            namespace["_isolate_dataloader_spawn_worker_environment"]()
            for key in namespace["_DATALOADER_DISTRIBUTED_ENV_KEYS"]:
                self.assertNotIn(key, os.environ)
            self.assertEqual(os.environ["STAMO_DATALOADER_PARENT_RANK"], "3")
            self.assertEqual(os.environ["STAMO_DATALOADER_SPAWN_CHILD"], "1")
            self.assertEqual(os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")
            self.assertNotIn("DS_ACCELERATOR", os.environ)
            self.assertEqual(
                os.environ["STAMO_DATALOADER_PARENT_ACCELERATOR"],
                "musa",
            )
            for thread_variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                self.assertEqual(os.environ[thread_variable], "1")

        unconditional_project_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and str(node.module).startswith("stamo.")
        ]
        self.assertEqual(unconditional_project_imports, [])
        guarded_project_imports = [
            child
            for node in tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "__name__ != '__mp_main__'"
            for child in node.body
            if isinstance(child, ast.ImportFrom)
            and str(child.module).startswith("stamo.")
        ]
        self.assertGreaterEqual(len(guarded_project_imports), 5)

    def test_production_loader_uses_ten_cpu_only_spawn_workers(self):
        config_source = PRODUCTION_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertRegex(config_source, r"(?m)^\s*num_workers:\s*10\s*$")
        self.assertRegex(
            config_source,
            r'(?m)^\s*worker_start_method:\s*["\']spawn["\']\s*$',
        )
        self.assertRegex(
            config_source,
            r"(?m)^\s*persistent_workers:\s*true\s*$",
        )
        self.assertRegex(
            config_source,
            r"(?m)^\s*loader_timeout_seconds:\s*120\s*$",
        )
        self.assertRegex(config_source, r"(?m)^\s*prefetch_factor:\s*1\s*$")
        self.assertRegex(config_source, r"(?m)^\s*eval_num_workers:\s*1\s*$")
        self.assertRegex(
            config_source,
            r"(?m)^\s*eval_persistent_workers:\s*false\s*$",
        )

        entrypoint_source = TRAIN_ENTRYPOINT_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"',
            entrypoint_source,
        )
        entrypoint_tree = ast.parse(
            entrypoint_source,
            filename=str(TRAIN_ENTRYPOINT_PATH),
        )
        torch_import = next(
            node
            for node in entrypoint_tree.body
            if isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        )
        spawn_bootstrap = next(
            node
            for node in entrypoint_tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "__name__ == '__mp_main__'"
            and any(
                isinstance(child, ast.Expr)
                and isinstance(child.value, ast.Call)
                and isinstance(child.value.func, ast.Name)
                and child.value.func.id
                == "_isolate_dataloader_spawn_worker_environment"
                for child in node.body
            )
        )
        self.assertLess(
            spawn_bootstrap.lineno,
            torch_import.lineno,
        )

        data_tree = ast.parse(
            DATA_UTILS_PATH.read_text(encoding="utf-8"),
            filename=str(DATA_UTILS_PATH),
        )
        eager_device_imports = [
            node
            for node in data_tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "stamo.renderer.utils.device"
        ]
        self.assertEqual(eager_device_imports, [])
        self.assertIn(
            "JsonlImagePathCollection",
            DATA_UTILS_PATH.read_text(encoding="utf-8"),
        )
        loader_options = next(
            node
            for node in data_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_loader_worker_options"
        )
        worker_initializer_assignments = [
            node
            for node in ast.walk(loader_options)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "options['worker_init_fn']"
            and ast.unparse(node.value) == "_initialize_cpu_dataloader_worker"
        ]
        self.assertEqual(len(worker_initializer_assignments), 1)
        self.assertIn(
            "MUSA DataLoader workers require worker_start_method='spawn'",
            ast.unparse(loader_options),
        )

        device_source = DEVICE_UTILS_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'os.environ.get("STAMO_DATALOADER_SPAWN_CHILD") == "1"',
            device_source,
        )

        verifier_source = (
            ROOT / "scripts" / "verify_flux2_klein_musa.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("STAMO_DATALOADER_FULL_POOL_PASS", verifier_source)
        self.assertIn("for parent_rank in range(world_size)", verifier_source)
        self.assertIn("for worker_id in range(workers)", verifier_source)
        self.assertIn("DataLoader worker PIDs are not unique", verifier_source)

        launcher_source = (
            ROOT
            / "scripts"
            / "train_egoverse_4token_dinov3_qformer.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'DATALOADER_PREFLIGHT_TIMEOUT_SECONDS="${DATALOADER_PREFLIGHT_TIMEOUT_SECONDS:-1800}"',
            launcher_source,
        )
        self.assertIn("timeout --signal=TERM --kill-after=30s", launcher_source)

    def test_cpu_loader_worker_rejects_musa_or_distributed_state(self):
        source = DATA_UTILS_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(DATA_UTILS_PATH))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_initialize_cpu_dataloader_worker"
        )
        module = ast.Module(body=[helper], type_ignores=[])
        ast.fix_missing_locations(module)

        class TorchStub:
            @staticmethod
            def set_num_threads(value):
                if value != 1:
                    raise AssertionError(value)

            @staticmethod
            def initial_seed():
                return 2**32 + 17

        class DistStub:
            initialized = False

            @classmethod
            def is_initialized(cls):
                return cls.initialized

        random_stub = types.SimpleNamespace(seed=lambda value: None)
        modules = {}
        namespace = {
            "os": os,
            "sys": types.SimpleNamespace(modules=modules),
            "torch": TorchStub,
            "dist": DistStub,
            "random": random_stub,
            "get_worker_info": lambda: types.SimpleNamespace(
                id=0,
                num_workers=10,
            ),
        }
        exec(compile(module, str(DATA_UTILS_PATH), "exec"), namespace)
        initialize = namespace["_initialize_cpu_dataloader_worker"]
        isolated_environment = {
            "STAMO_DATALOADER_SPAWN_CHILD": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        }
        with patch.dict(os.environ, isolated_environment, clear=True):
            initialize(0)
            modules["torch_musa"] = object()
            with self.assertRaisesRegex(RuntimeError, "imported torch_musa"):
                initialize(0)
            modules.clear()
            DistStub.initialized = True
            with self.assertRaisesRegex(RuntimeError, "distributed process group"):
                initialize(0)

    def test_musa_loader_options_require_spawn_and_bound_prefetch(self):
        source = DATA_UTILS_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(DATA_UTILS_PATH))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_loader_worker_options"
        )
        module = ast.Module(body=[helper], type_ignores=[])
        ast.fix_missing_locations(module)
        initializer = object()
        namespace = {
            "os": os,
            "_initialize_cpu_dataloader_worker": initializer,
        }
        exec(compile(module, str(DATA_UTILS_PATH), "exec"), namespace)
        options_for = namespace["_loader_worker_options"]
        with patch.dict(os.environ, {"DS_ACCELERATOR": "musa"}, clear=True):
            options = options_for(
                10,
                loader_timeout_seconds=120,
                persistent_workers=True,
                worker_start_method="spawn",
                prefetch_factor=1,
            )
            self.assertEqual(options["num_workers"], 10)
            self.assertEqual(options["timeout"], 120.0)
            self.assertTrue(options["persistent_workers"])
            self.assertEqual(options["multiprocessing_context"], "spawn")
            self.assertEqual(options["prefetch_factor"], 1)
            self.assertIs(options["worker_init_fn"], initializer)
            with self.assertRaisesRegex(ValueError, "require.*spawn"):
                options_for(10, worker_start_method="fork")

    def test_mccl_blocking_wait_default_and_unsafe_override(self):
        source = TRAIN_ENTRYPOINT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAIN_ENTRYPOINT_PATH))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_configure_musa_mccl_blocking_wait"
        )
        module = ast.Module(body=[helper], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"os": os}
        exec(
            compile(module, str(TRAIN_ENTRYPOINT_PATH), "exec"),
            namespace,
        )
        configure = namespace["_configure_musa_mccl_blocking_wait"]

        with patch.dict(os.environ, {"DS_ACCELERATOR": "musa"}, clear=True):
            configure()
            self.assertEqual(os.environ["TORCH_MCCL_BLOCKING_WAIT"], "1")

        unsafe_environment = {
            "DS_ACCELERATOR": "musa",
            "TORCH_MCCL_BLOCKING_WAIT": "0",
        }
        with patch.dict(os.environ, unsafe_environment, clear=True):
            with self.assertRaisesRegex(
                EnvironmentError,
                "TORCH_MCCL_BLOCKING_WAIT must remain enabled",
            ):
                configure()

    def test_mccl_smoke_requires_world_size_and_uses_large_broadcast(self):
        source = MCCL_SMOKE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(MCCL_SMOKE_PATH))
        broadcast_calls = [
            call
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "dist"
            and call.func.attr == "broadcast"
        ]

        self.assertIn('"--expected-world-size"', source)
        self.assertEqual(len(broadcast_calls), 1)
        self.assertIsInstance(broadcast_calls[0].args[0], ast.Name)
        self.assertEqual(broadcast_calls[0].args[0].id, "broadcast_payload")
        self.assertIn("broadcast_payload = torch.empty_like(reduce_payload)", source)

    def test_flux2_production_profile_uses_serial_zero2_without_stage3_fetch(self):
        config_source = PRODUCTION_CONFIG_PATH.read_text(encoding="utf-8")
        verifier_source = VERIFIER_PATH.read_text(encoding="utf-8")

        prefetch_assignments = re.findall(
            r"(?m)^\s{2}deepspeed_stage3_prefetch_bucket_size:\s*(\d+)\s*$",
            config_source,
        )
        self.assertEqual(prefetch_assignments, [])
        self.assertRegex(
            config_source,
            r"(?m)^  deepspeed_zero_stage:\s*2\s*$",
        )
        self.assertRegex(
            config_source,
            r"(?m)^  deepspeed_overlap_comm:\s*false\s*$",
        )
        self.assertRegex(
            config_source,
            r"(?m)^\s{2}hang_dump_after_seconds:\s*0\s*$",
        )
        self.assertIn(
            "config.train.hang_dump_after_seconds = 0",
            verifier_source,
        )
        self.assertNotIn(
            "config.train.hang_dump_after_seconds = 60",
            verifier_source,
        )
        self.assertIn(
            'TRAIN_TIMEOUT_SECONDS="${TRAIN_TIMEOUT_SECONDS:-7200}"',
            verifier_source,
        )
        self.assertNotIn(
            'TRAIN_TIMEOUT_SECONDS="${TRAIN_TIMEOUT_SECONDS:-2400}"',
            verifier_source,
        )
        self.assertIn("--numel 9437184", verifier_source)
        self.assertNotIn("9440256", verifier_source)

    def test_bounded_leaf_selector_coalesces_only_composite_modules(self):
        class TinyComposite(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = torch.nn.Linear(4, 4, bias=False)
                self.right = torch.nn.Linear(4, 4, bias=False)

        class OversizedRoot(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = TinyComposite()
                self.second = TinyComposite()

        root = OversizedRoot()
        selected = bounded_leaf_candidates(root, "root", 1, 40)
        self.assertEqual(
            [(name, numel) for name, _, numel in selected],
            [("root.first", 32), ("root.second", 32)],
        )
        self.assertTrue(all(module is not root for _, module, _ in selected))
        self.assertEqual(
            bounded_leaf_candidates(torch.nn.Linear(4, 4), "linear", 1, 40),
            [],
        )

        class OneHookWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, bias=False)

        self.assertEqual(
            bounded_leaf_candidates(OneHookWrapper(), "wrapper", 1, 40),
            [],
        )

        container = torch.nn.ModuleList([TinyComposite(), TinyComposite()])
        container_selected = bounded_leaf_candidates(container, "blocks", 1, 40)
        self.assertEqual(
            [name for name, _, _ in container_selected],
            ["blocks.0", "blocks.1"],
        )

    def test_leaf_registration_wraps_deepspeed_initialize_in_correct_order(self):
        source = TRAINER_PATH.read_text(encoding="utf-8")
        prepare_index = source.index("    def prepare_dist_model(")
        configure_index = source.index(
            "self._configure_zero3_leaf_modules()", prepare_index
        )
        initialize_index = source.index("deepspeed.initialize(", prepare_index)
        validate_index = source.index(
            "self._validate_zero3_leaf_modules()", prepare_index
        )
        self.assertLess(configure_index, initialize_index)
        self.assertLess(initialize_index, validate_index)
        self.assertIn("model.vae.encoder", source)
        self.assertNotIn('(\"vae\", model.vae)', source)
        self.assertIn("deepspeed_zero3_leaf_max_numel", source)

    def test_full_zero2_is_admitted_but_lora_can_checkpoint_and_resume(self):
        source = TRAINER_PATH.read_text(encoding="utf-8")
        prepare_index = source.index("    def prepare_dist_model(")
        admission_index = source.index(
            "self._validate_zero2_memory_admission()", prepare_index
        )
        initialize_index = source.index("deepspeed.initialize(", prepare_index)
        self.assertLess(admission_index, initialize_index)
        self.assertIn("deepspeed_allow_zero2_full_transformer", source)
        self.assertIn("not self.full_transformer_trainable", source)
        self.assertIn("musa.mem_get_info", source)
        self.assertIn("DIST.all_reduce(local_flags, op=DIST.ReduceOp.MIN)", source)
        self.assertIn("stamo_flux2_klein_deepspeed_v1", source)
        self.assertIn("frozen_parameters_excluded", source)
        self.assertNotIn("The full-FLUX ZeRO-2 path is admission-only", source)
        self.assertIn("_host_memory_info_bytes()", source)
        self.assertIn("18 * trainable_numel", source)
        self.assertIn("Use training_mode=lora", source)

    def test_performance_profiles_are_isolated_and_keep_serial_comm(self):
        source = PERF_CONFIG_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(PERF_CONFIG_PATH))
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"PROFILE_OVERRIDES", "SERIAL_BASELINE"}
        }
        profiles = assignments["PROFILE_OVERRIDES"]
        baseline = assignments["SERIAL_BASELINE"]
        self.assertTrue(
            {
                "zero3_baseline",
                "zero3_frozen_leaf",
                "zero3_condition_leaf",
                "zero3_safe_combined_leaf",
                "zero3_double_ff_leaf",
                "zero3_combined_leaf",
                "zero3_bucket75",
                "zero3_persist_small",
                "zero3_reuse1500",
                "zero2_admission",
                "musa_fused_adamw",
            }.issubset(profiles)
        )
        self.assertFalse(baseline["train.deepspeed_overlap_comm"])
        self.assertEqual(
            baseline["train.deepspeed_stage3_prefetch_bucket_size"],
            0,
        )
        self.assertFalse(
            baseline["train.deepspeed_stage3_use_all_reduce_for_fetch_params"]
        )
        self.assertFalse(baseline["train.deepspeed_allow_zero2_full_transformer"])
        self.assertEqual(
            baseline["train.deepspeed_stage3_model_persistence_threshold"],
            100_000_000,
        )
        self.assertFalse(baseline["train.deepspeed_zero3_leaf_condition_modules"])
        self.assertEqual(
            baseline["train.deepspeed_zero2_min_device_headroom_gib"],
            36.0,
        )
        self.assertEqual(
            profiles["zero3_double_ff_leaf"][
                "train.deepspeed_zero3_leaf_max_numel"
            ],
            80_000_000,
        )
        self.assertEqual(
            profiles["zero3_double_ff_leaf"][
                "train.deepspeed_zero3_leaf_flux_blocks"
            ],
            "double_ff",
        )
        self.assertEqual(
            profiles["zero3_double_ff_leaf"][
                "train.deepspeed_zero3_expected_double_blocks"
            ],
            19,
        )
        self.assertNotIn(
            "train.deepspeed_zero3_leaf_frozen_modules",
            profiles["zero3_double_ff_leaf"],
        )
        self.assertTrue(
            profiles["zero3_combined_leaf"][
                "train.deepspeed_zero3_leaf_frozen_modules"
            ]
        )
        self.assertTrue(
            profiles["zero3_combined_leaf"][
                "train.deepspeed_zero3_leaf_condition_modules"
            ]
        )
        self.assertEqual(
            profiles["zero3_condition_leaf"][
                "train.deepspeed_zero3_expected_qformer_blocks"
            ],
            4,
        )
        self.assertTrue(
            profiles["zero3_safe_combined_leaf"][
                "train.deepspeed_zero3_leaf_frozen_modules"
            ]
        )
        self.assertTrue(
            profiles["zero3_safe_combined_leaf"][
                "train.deepspeed_zero3_leaf_condition_modules"
            ]
        )
        self.assertEqual(
            profiles["zero3_safe_combined_leaf"][
                "train.deepspeed_zero3_expected_qformer_blocks"
            ],
            4,
        )
        self.assertEqual(
            profiles["zero3_combined_leaf"][
                "train.deepspeed_zero3_leaf_flux_blocks"
            ],
            "double_ff",
        )
        self.assertEqual(
            profiles["zero3_combined_leaf"][
                "train.deepspeed_zero3_leaf_max_numel"
            ],
            80_000_000,
        )
        self.assertEqual(
            profiles["zero3_persist_small"][
                "train.deepspeed_stage3_param_persistence_threshold"
            ],
            1_000_000,
        )
        self.assertEqual(
            profiles["zero3_persist_small"][
                "train.deepspeed_stage3_model_persistence_threshold"
            ],
            200_000_000,
        )
        self.assertNotIn("zero3_single_leaf", profiles)
        self.assertNotIn("zero3_all_leaf", profiles)
        self.assertIn("Refusing to overwrite the base/production config", source)
        self.assertIn("config.resume = False", source)
        self.assertIn("config.train.enable_checkpointing = False", source)
        self.assertIn("config.train.enable_eval = False", source)

    @unittest.skipUnless(
        importlib.util.find_spec("omegaconf"),
        "OmegaConf is validated in the remote training environment",
    )
    def test_persistence_profile_generation_is_bounded_and_non_mutating(self):
        namespace = runpy.run_path(str(PERF_CONFIG_PATH))
        base_before = PRODUCTION_CONFIG_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            output = temporary_root / "trial.yaml"
            artifact_root = temporary_root / "artifacts"
            manifest = namespace["build_config"](
                types.SimpleNamespace(
                    base_config=str(PRODUCTION_CONFIG_PATH),
                    output=str(output),
                    artifact_root=str(artifact_root),
                    profile="zero3_persist_small",
                    steps=20,
                    timeout_seconds=180,
                    task_name="unit_persistence_profile",
                )
            )
            generated = namespace["OmegaConf"].load(output)
            self.assertEqual(
                generated.train.deepspeed_stage3_param_persistence_threshold,
                1_000_000,
            )
            self.assertEqual(
                generated.train.deepspeed_stage3_model_persistence_threshold,
                200_000_000,
            )
            self.assertFalse(generated.train.deepspeed_overlap_comm)
            self.assertFalse(generated.train.enable_checkpointing)
            self.assertEqual(manifest["profile"], "zero3_persist_small")
            self.assertTrue((artifact_root / "config_manifest.json").is_file())
        self.assertEqual(PRODUCTION_CONFIG_PATH.read_bytes(), base_before)

    def test_performance_runner_bounds_hangs_and_requires_all_rank_progress(self):
        source = PERF_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn('export TORCH_MCCL_BLOCKING_WAIT="1"', source)
        self.assertIn("timeout --signal=TERM --kill-after=60s", source)
        self.assertIn("expected {world_size} rank traces", source)
        self.assertIn("did not complete exact steps 1..{steps}", source)
        self.assertIn("Base config unchanged", source)
        self.assertIn("MCCL_NUMEL=80000000", source)
        self.assertIn('"${PROFILE}" == "zero3_combined_leaf"', source)
        self.assertIn('PERF_WARMUP_STEPS="${PERF_WARMUP_STEPS:-1}"', source)
        self.assertIn(
            'window["step"] - window["window_steps"] >= warmup_steps',
            source,
        )
        self.assertIn("every rank stopped before DeepSpeed allocation", source)
        self.assertIn("PROFILE}" + '" != "zero2_admission"', source)
        self.assertIn("unittest discover -s tests -v", source)
        self.assertIn("STAMO_PERF_WINDOW", source)
        self.assertIn("median_synchronized_seconds_per_step", source)
        self.assertIn("STAMO_ZERO3_DOUBLE_FF_LEAVES=38", source)
        self.assertIn("STAMO_ZERO3_QFORMER_LEAVES=4", source)
        self.assertIn("STAMO_PERSISTENCE_STATS", source)
        self.assertIn("numel > 200_000_000", source)
        self.assertNotIn("unittest tests.test_deepspeed", source)
        self.assertNotIn("export MCCL_IB_DISABLE", source)
        self.assertIn('MCCL_P2P_DISABLE=${MCCL_P2P_DISABLE:-<unset>}', source)
        self.assertIn('MCCL_SHM_DISABLE=${MCCL_SHM_DISABLE:-<unset>}', source)

    def test_trainable_leaf_targets_only_positionally_called_flux_feedforwards(self):
        source = TRAINER_PATH.read_text(encoding="utf-8")
        projector_source = (
            ROOT / "stamo" / "renderer" / "model" / "projector.py"
        ).read_text(encoding="utf-8")
        self.assertIn('for attribute in ("ff", "ff_context")', source)
        self.assertIn('flux_block_mode == "double_ff"', source)
        self.assertIn("forward_pre_hook without", source)
        self.assertIn("Attention must never be", source)
        self.assertIn('block.__class__.__name__ != "FluxTransformerBlock"', source)
        self.assertIn('module.__class__.__name__ != "FeedForward"', source)
        configure_source = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef)
                and node.name == "_configure_zero3_leaf_modules"
            ),
        )
        self.assertIsNotNone(configure_source)
        self.assertNotIn("single_transformer_blocks", configure_source)
        self.assertNotIn('(\"attn\",', configure_source)
        self.assertIn('module.__class__.__name__ != "QformerBlock"', source)
        self.assertIn("q_tokens = block(q_tokens, image_embeddings)", projector_source)
        self.assertNotIn("projector.cross_attn", configure_source)

    def test_performance_marker_reuses_the_existing_loss_synchronization(self):
        source = TRAINER_PATH.read_text(encoding="utf-8")
        marker_index = source.index('"STAMO_PERF_WINDOW "')
        loss_sync_index = source.rfind(".cpu()", 0, marker_index)
        elapsed_index = source.rfind("elapsed = max(", 0, marker_index)
        self.assertGreater(loss_sync_index, 0)
        self.assertGreater(elapsed_index, loss_sync_index)
        self.assertNotIn(
            "_diagnostic_synchronize()",
            source[elapsed_index:marker_index],
        )

    def test_fused_adamw_requires_explicit_profile_and_deepspeed_opt_in(self):
        optimizer_source = OPTIMIZER_PATH.read_text(encoding="utf-8")
        entrypoint_source = TRAIN_ENTRYPOINT_PATH.read_text(encoding="utf-8")
        trainer_source = TRAINER_PATH.read_text(encoding="utf-8")
        config_source = PRODUCTION_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn('opt_type == "musa_fused_adamw"', optimizer_source)
        self.assertIn("from torch_musa.optim import FusedAdamW", optimizer_source)
        self.assertIn('args.train.get("optimizer_type", "adamw")', entrypoint_source)
        self.assertIn("STAMO_OPTIMIZER=", entrypoint_source)
        self.assertIn('ds_config["zero_allow_untested_optimizer"] = True', trainer_source)
        self.assertIn("STAMO_FUSED_ADAMW_OPT_IN=1", trainer_source)
        self.assertIn("musa_fused_adamw requires", trainer_source)
        self.assertIn('deepspeed_offload_optimizer_device != "none"', trainer_source)
        self.assertNotRegex(
            config_source,
            r'(?m)^  optimizer_type:\s*"?musa_fused_adamw"?\s*$',
        )

    def test_fused_adamw_lazy_import_preserves_explicit_hyperparameters(self):
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        baseline = get_optimizer_helper(
            [parameter],
            opt_type="adamw",
            lr=3e-5,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=1e-3,
            amsgrad=False,
            maximize=False,
            capturable=False,
        )
        self.assertIs(type(baseline), torch.optim.AdamW)

        class FakeFusedAdamW(torch.optim.AdamW):
            pass

        FakeFusedAdamW.__module__ = "torch_musa.optim.fused_adamw"
        torch_musa_module = types.ModuleType("torch_musa")
        torch_musa_optim_module = types.ModuleType("torch_musa.optim")
        torch_musa_optim_module.FusedAdamW = FakeFusedAdamW
        torch_musa_module.optim = torch_musa_optim_module
        with patch.dict(
            sys.modules,
            {
                "torch_musa": torch_musa_module,
                "torch_musa.optim": torch_musa_optim_module,
            },
        ):
            fused = get_optimizer_helper(
                [parameter],
                opt_type="musa_fused_adamw",
                lr=3e-5,
                betas=(0.9, 0.98),
                eps=1e-8,
                weight_decay=1e-3,
                amsgrad=False,
                maximize=False,
                capturable=False,
            )
        self.assertIs(type(fused), FakeFusedAdamW)
        group = fused.param_groups[0]
        self.assertEqual(group["lr"], 3e-5)
        self.assertEqual(group["betas"], (0.9, 0.98))
        self.assertEqual(group["eps"], 1e-8)
        self.assertEqual(group["weight_decay"], 1e-3)
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer type"):
            get_optimizer_helper([parameter], opt_type="fused_adamw")

    def test_flux2_production_config_keeps_unadmitted_optimizations_disabled(self):
        source = PRODUCTION_CONFIG_PATH.read_text(encoding="utf-8")
        safe_if_present = {
            "deepspeed_allow_zero2_full_transformer": "false",
            "deepspeed_zero3_leaf_frozen_modules": "false",
            "deepspeed_zero3_leaf_condition_modules": "false",
            "deepspeed_zero3_leaf_flux_blocks": '"none"',
            "deepspeed_zero3_expected_double_blocks": "0",
            "deepspeed_zero3_expected_qformer_blocks": "0",
            "phase_trace": "false",
        }
        for key, value in safe_if_present.items():
            matches = re.findall(
                rf"(?m)^  {re.escape(key)}:\s*(\S+)\s*$",
                source,
            )
            if matches:
                self.assertEqual(matches, [value], key)
        headroom = re.findall(
            r"(?m)^  deepspeed_zero2_min_device_headroom_gib:\s*([0-9.]+)\s*$",
            source,
        )
        if headroom:
            self.assertEqual(headroom, ["32.0"])
        self.assertRegex(
            source,
            r'(?m)^    training_mode:\s*"lora"\s*$',
        )
        self.assertRegex(
            source,
            r"(?m)^  deepspeed_zero_stage:\s*2\s*$",
        )


if __name__ == "__main__":
    unittest.main()
