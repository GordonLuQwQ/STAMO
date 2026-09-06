"""CPU logic tests for the verification-only ZeRO-2 FP32 master audit."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "stamo" / "renderer" / "trainer.py"


def load_audit_helper_class():
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAINER))
    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Trainer"
    )
    wanted = {
        "_zero2_fp32_fragment",
        "_zero2_fp32_fragment_record",
    }
    methods = [
        copy.deepcopy(node)
        for node in trainer_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    helper_class = ast.ClassDef(
        name="AuditHelpers",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[helper_class], type_ignores=[]))
    namespace = {"hashlib": hashlib, "json": json, "torch": torch}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace["AuditHelpers"]


class _Address:
    def __init__(self, start, numel):
        self.start = start
        self.numel = numel


class _Mapping:
    def __init__(self, fragment, start=7):
        self.fragment = fragment
        self.address = _Address(start, fragment.numel())

    def get_hp_fragment(self):
        return self.fragment

    def get_hp_fragment_address(self):
        return self.address


class Zero2Fp32AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_audit_helper_class()

    def test_fp32_digest_detects_update_hidden_by_bf16_rounding(self):
        visible = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
        master = torch.tensor([1.0, 2.0], dtype=torch.float32)
        visible._hp_mapping = _Mapping(master)

        before_visible = visible.detach().clone()
        before = self.helpers._zero2_fp32_fragment_record("DiT.lora_B", visible)
        master[0] += 1e-5
        after = self.helpers._zero2_fp32_fragment_record("DiT.lora_B", visible)

        self.assertTrue(torch.equal(before_visible, visible.detach()))
        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertEqual(before["fragment_start"], after["fragment_start"])
        self.assertTrue(before["finite"] and after["finite"])

    def test_nonfinite_and_missing_mapping_fail_closed(self):
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.assertIsNone(
            self.helpers._zero2_fp32_fragment_record("projector.weight", parameter)
        )

        master = torch.tensor([1.0, float("nan")], dtype=torch.float32)
        parameter._hp_mapping = _Mapping(master)
        record = self.helpers._zero2_fp32_fragment_record(
            "projector.weight",
            parameter,
        )
        self.assertFalse(record["finite"])

    def test_non_fp32_master_is_rejected(self):
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        parameter._hp_mapping = _Mapping(torch.ones(2, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "expected FP32"):
            self.helpers._zero2_fp32_fragment_record("projector.weight", parameter)


if __name__ == "__main__":
    unittest.main()
