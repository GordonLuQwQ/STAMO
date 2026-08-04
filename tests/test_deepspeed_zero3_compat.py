"""Logic tests for STAMO's process-local DeepSpeed ZeRO-3 backport."""

import ast
import inspect
import re
import unittest
from functools import wraps
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "stamo" / "renderer" / "trainer.py"
TRAIN_ENTRYPOINT_PATH = ROOT / "train_renderer.py"
MCCL_SMOKE_PATH = ROOT / "scripts" / "mccl_smoke_test.py"
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


if __name__ == "__main__":
    unittest.main()
