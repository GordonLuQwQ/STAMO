#!/usr/bin/env python3
"""Exercise ten spawn workers without importing or touching torch_musa."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


# This program is intentionally CPU-only.  The production entrypoint performs
# the same assignment only while it is replayed as a DataLoader spawn child.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ.pop("DS_ACCELERATOR", None)
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, get_worker_info

from stamo.renderer.utils.metadata_index import JsonlImagePathIndex


def install_cpu_musa_storage_predicate():
    """Satisfy the musified reducer without loading the MUSA runtime."""
    storage_type = getattr(torch, "UntypedStorage", None)
    if storage_type is None:
        raise RuntimeError("torch.UntypedStorage is unavailable.")
    if not hasattr(storage_type, "is_musa"):
        storage_type.is_musa = property(
            lambda storage: getattr(storage.device, "type", None) == "musa"
        )
    if bool(storage_type(0).is_musa):
        raise RuntimeError("A CPU probe storage was classified as MUSA.")


install_cpu_musa_storage_predicate()


class IndexedPathProbeDataset(Dataset):
    def __init__(self, metadata_path):
        self.paths = JsonlImagePathIndex(metadata_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        worker = get_worker_info()
        if worker is None:
            raise RuntimeError("The probe sample was loaded in the parent process.")
        return {
            "worker_id": int(worker.id),
            "pid": int(os.getpid()),
            "path_length": len(self.paths[index]),
        }


def initialize_worker(worker_id):
    torch.set_num_threads(1)
    install_cpu_musa_storage_predicate()
    if "torch_musa" in sys.modules:
        raise RuntimeError(
            f"CPU probe worker {worker_id} unexpectedly imported torch_musa."
        )
    if dist.is_initialized():
        raise RuntimeError(
            f"CPU probe worker {worker_id} initialized a process group."
        )


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--metadata-path")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    return parser.parse_args()


def resolve_training_manifest(config_path):
    from omegaconf import OmegaConf

    config_path = os.path.abspath(str(config_path))
    config = OmegaConf.load(config_path)
    dataset_config_path = os.path.abspath(
        os.path.expanduser(str(config.data.train_json_path))
    )
    with open(dataset_config_path, "r", encoding="utf-8") as stream:
        dataset_config = json.load(stream)
    datasets = dataset_config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(
            f"Dataset config has no non-empty datasets list: {dataset_config_path}"
        )
    metadata_path = os.path.expanduser(str(datasets[0]))
    if not os.path.isabs(metadata_path):
        metadata_path = os.path.join(
            os.path.dirname(dataset_config_path),
            metadata_path,
        )
    return os.path.abspath(metadata_path)


def main():
    cli = parse_args()
    if cli.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if cli.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if "torch_musa" in sys.modules:
        raise RuntimeError("The CPU-only probe parent imported torch_musa.")

    metadata_path = (
        os.path.abspath(os.path.expanduser(str(cli.metadata_path)))
        if cli.metadata_path
        else resolve_training_manifest(cli.config)
    )
    dataset = IndexedPathProbeDataset(metadata_path)
    payload_bytes = len(pickle.dumps(dataset))
    if payload_bytes >= 1024 * 1024:
        raise RuntimeError(
            "Spawn dataset payload is too large: "
            f"{payload_bytes} bytes for {len(dataset)} records."
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cli.num_workers,
        timeout=cli.timeout_seconds,
        worker_init_fn=initialize_worker,
        multiprocessing_context="spawn",
        prefetch_factor=1,
        persistent_workers=True,
    )
    iterator = iter(loader)
    worker_ids = set()
    worker_pids = set()
    try:
        for _ in range(cli.num_workers * 4):
            batch = next(iterator)
            worker_ids.add(int(batch["worker_id"].item()))
            worker_pids.add(int(batch["pid"].item()))
            if len(worker_ids) == cli.num_workers:
                break
        expected_ids = set(range(cli.num_workers))
        if worker_ids != expected_ids:
            raise RuntimeError(
                f"Only observed worker ids {sorted(worker_ids)}; "
                f"expected {sorted(expected_ids)}."
            )
        if len(worker_pids) != cli.num_workers:
            raise RuntimeError(
                f"Expected {cli.num_workers} worker PIDs, got {worker_pids}."
            )
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()

    print(
        "STAMO_DATALOADER_SPAWN_SMOKE_PASS "
        f"workers={cli.num_workers} unique_pids={len(worker_pids)} "
        f"records={len(dataset)} pickle_bytes={payload_bytes} "
        "autoload=0 musa_imported=0 distributed=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
