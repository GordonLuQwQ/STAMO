import json
import math
import os
import random
import re
import time
from itertools import islice

import jsonlines
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

from stamo.renderer.utils.device import get_accelerator_device
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def complex_to_device(complex, device, non_blocking=False):
    if complex is None:
        return complex
    if isinstance(complex, torch.Tensor):
        return complex.to(device, non_blocking=non_blocking)
    elif isinstance(complex, dict):
        return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
    elif isinstance(complex, list) or isinstance(complex, tuple):
        return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
    elif (
        isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
    ):
        return complex
    else:
        raise ValueError("Unsupported complex", complex)


def fp32_to_fp16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.half)
    elif isinstance(batch, list):
        new_batch = [fp32_to_fp16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_fp16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def fp32_to_bf16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.bfloat16)
    elif isinstance(batch, list):
        new_batch = [fp32_to_bf16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_bf16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def move_to_cuda(batch, device=None):
    device = device or get_accelerator_device()
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t, device=device) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def collate_fn(inputs):
    images = torch.stack([input["image"] for input in inputs])
    batch = {"images": images}
    paths = [input.get("path") for input in inputs]
    if all(path is not None for path in paths):
        batch["paths"] = paths
    return batch


def get_loader_info(dataset, epochs, bsz, gradient_accumulate_steps=1):
    dataset_len = len(dataset.dataset) if hasattr(dataset, "dataset") else len(dataset)
    images_per_gpu = bsz
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    images_per_batch = bsz * world_size * gradient_accumulate_steps
    iter_per_ep = math.ceil(dataset_len / images_per_batch) if dataset_len else 0
    num_iters = iter_per_ep * epochs
    loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
    return loader_info


class ImageData(Dataset):
    def __init__(
        self,
        metadata_path,
        flip_p,
        img_size: int = 224,
        max_read_attempts: int = 8,
        seed: int = 0,
        read_trace_dir=None,
        read_trace_samples: int = 0,
    ):
        self.flip_p = flip_p
        self.max_read_attempts = int(max_read_attempts)
        self.seed = int(seed)
        self.read_trace_dir = (
            os.path.abspath(os.path.expanduser(str(read_trace_dir)))
            if read_trace_dir
            else None
        )
        self.read_trace_samples = max(0, int(read_trace_samples))
        self._read_trace_count = 0
        self._read_trace_path = None
        if self.max_read_attempts <= 0:
            raise ValueError("max_read_attempts must be positive")

        self.metadata_source = os.path.abspath(str(metadata_path))
        self.metadata = []
        self._append_metadata(metadata_path)

        self.length = len(self.metadata)
        if self.length == 0:
            raise ValueError(f"No images were found in metadata file {metadata_path!r}.")

        overwatch.info(f"{self.length} data loaded from {metadata_path}")

        self.transforms = T.Compose(
            [
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )

    def preprocess_train(self, image):
        if torch.rand(1) < self.flip_p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = self.transforms(image)
        return image

    def _append_metadata(self, metadata_path):
        metadata_path = os.path.abspath(str(metadata_path))
        metadata_dir = os.path.dirname(metadata_path)
        with open(metadata_path, "r", encoding="utf8") as f:
            for line_number, item in enumerate(jsonlines.Reader(f), start=1):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{metadata_path}:{line_number} must contain a JSON object."
                    )
                image_path = item.get("image")
                if not isinstance(image_path, str) or not image_path.strip():
                    raise ValueError(
                        f"{metadata_path}:{line_number} has no non-empty 'image' path."
                    )
                image_path = os.path.expanduser(image_path.strip())
                if not os.path.isabs(image_path):
                    image_path = os.path.abspath(os.path.join(metadata_dir, image_path))
                self.metadata.append(image_path)

    def add(self, metadata_path):
        self._append_metadata(metadata_path)
        self.length = len(self.metadata)
        overwatch.info(f"{self.length} data loaded from {metadata_path}")

    @staticmethod
    def _eval_task_from_image_path(image_path):
        """Return the task in ``.../<split>/<task>/<episode>/<frame>``."""
        normalized_path = str(image_path).replace("\\", "/").rstrip("/")
        path_parts = [part for part in normalized_path.split("/") if part]
        split_names = {"eval", "validation", "val", "test"}
        split_indices = [
            index
            for index, part in enumerate(path_parts)
            if part.lower() in split_names
        ]
        if not split_indices:
            raise ValueError(
                "Cannot infer an evaluation task from image path "
                f"{image_path!r}; expected an eval/validation/test split "
                "component before <task>/<episode>/<frame>."
            )
        split_index = split_indices[-1]
        if split_index + 3 >= len(path_parts):
            raise ValueError(
                "Cannot infer an evaluation task from image path "
                f"{image_path!r}; expected "
                ".../<split>/<task>/<episode>/<frame>."
            )
        return path_parts[split_index + 1]

    def select_eval_tasks(self, mode, fast_num_tasks=100, seed=0):
        """Select one deterministic global task subset before rank sharding.

        ``test_mode`` retains every task in the manifest. ``fast_mode`` uses
        a rank-independent local RNG to choose tasks, then retains every image
        belonging to those tasks. The existing DistributedSampler can therefore
        shard exactly the same filtered dataset on every rank.
        """
        mode = str(mode).strip().lower()
        if mode not in {"test_mode", "fast_mode"}:
            raise ValueError(
                "data.eval_mode must be 'test_mode' or 'fast_mode', "
                f"got {mode!r}."
            )

        task_by_path = {
            image_path: self._eval_task_from_image_path(image_path)
            for image_path in self.metadata
        }
        available_tasks = sorted(set(task_by_path.values()))
        if not available_tasks:
            raise ValueError("The evaluation manifest contains no tasks.")

        if mode == "fast_mode":
            fast_num_tasks = int(fast_num_tasks)
            if fast_num_tasks <= 0:
                raise ValueError("data.fast_eval_num_tasks must be positive.")
            if fast_num_tasks > len(available_tasks):
                raise ValueError(
                    "data.fast_eval_num_tasks exceeds the available task count: "
                    f"requested={fast_num_tasks}, available={len(available_tasks)}."
                )
            selected_tasks = set(
                random.Random(int(seed)).sample(
                    available_tasks,
                    fast_num_tasks,
                )
            )
            self.metadata = [
                image_path
                for image_path in self.metadata
                if task_by_path[image_path] in selected_tasks
            ]
        else:
            selected_tasks = set(available_tasks)

        self.length = len(self.metadata)
        if self.length == 0:
            raise ValueError(
                f"Evaluation mode {mode!r} selected no images."
            )

        self.eval_mode = mode
        self.eval_available_task_count = len(available_tasks)
        self.eval_selected_tasks = tuple(sorted(selected_tasks))
        self.eval_task_seed = int(seed)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            overwatch.info(
                f"Evaluation {mode}: selected {len(selected_tasks)}/"
                f"{len(available_tasks)} tasks and {self.length} images "
                f"with seed={int(seed)}."
            )

    def __len__(self):
        return self.length

    def _retry_index(self, original_idx, attempt, attempted_indices):
        """Choose a deterministic, previously untried fallback sample."""
        candidate = (
            int(original_idx)
            + int(attempt) * 104729
            + self.seed * 1009
        ) % self.length
        while candidate in attempted_indices and len(attempted_indices) < self.length:
            candidate = (candidate + 1) % self.length
        return candidate

    def _trace_read(self, enabled, event, **details):
        if not enabled or self.read_trace_dir is None:
            return
        if self._read_trace_path is None:
            os.makedirs(self.read_trace_dir, exist_ok=True)
            rank = dist.get_rank() if dist.is_initialized() else 0
            metadata_name = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                os.path.basename(getattr(self, "metadata_source", "images")),
            )
            self._read_trace_path = os.path.join(
                self.read_trace_dir,
                f"data_{metadata_name}_rank{rank:02d}_pid{os.getpid()}.jsonl",
            )
        record = {
            "wall_time_ns": time.time_ns(),
            "pid": os.getpid(),
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "event": str(event),
        }
        record.update(details)
        # Only the first configured samples are traced, so opening per event is
        # cheap and avoids leaking a file descriptor into DataLoader workers.
        with open(self._read_trace_path, "a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
            trace_file.flush()

    def __getitem__(self, idx):
        original_idx = int(idx)
        trace_this_read = self._read_trace_count < self.read_trace_samples
        self._read_trace_count += 1
        candidate_idx = original_idx
        attempted_indices = set()
        failures = []
        total_attempts = min(self.max_read_attempts, self.length)

        for attempt in range(total_attempts):
            attempted_indices.add(candidate_idx)
            path = self.metadata[candidate_idx]
            self._trace_read(
                trace_this_read,
                "READ_BEGIN",
                requested_index=original_idx,
                candidate_index=candidate_idx,
                attempt=attempt + 1,
                path=path,
            )
            try:
                with Image.open(path) as opened_image:
                    opened_image.load()
                    image = opened_image.convert("RGB")
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                self._trace_read(
                    trace_this_read,
                    "READ_ERROR",
                    requested_index=original_idx,
                    candidate_index=candidate_idx,
                    attempt=attempt + 1,
                    path=path,
                    error=repr(exc),
                )
                failures.append((path, exc))
                if attempt + 1 < total_attempts:
                    candidate_idx = self._retry_index(
                        original_idx,
                        attempt + 1,
                        attempted_indices,
                    )
                continue

            self._trace_read(
                trace_this_read,
                "OPEN_OK",
                requested_index=original_idx,
                candidate_index=candidate_idx,
                attempt=attempt + 1,
                path=path,
            )
            # Keep transform errors visible. They are programming/configuration
            # errors, not corrupt-image errors that should silently change data.
            image = self.preprocess_train(image)
            self._trace_read(
                trace_this_read,
                "TRANSFORM_OK",
                requested_index=original_idx,
                candidate_index=candidate_idx,
                attempt=attempt + 1,
                path=path,
            )
            return {"image": image, "path": path}

        failure_summary = "; ".join(
            f"{path!r}: {type(exc).__name__}: {exc}"
            for path, exc in failures[-4:]
        )
        last_error = failures[-1][1] if failures else None
        raise RuntimeError(
            f"Failed to read an image after {total_attempts} bounded attempts "
            f"(requested index={original_idx}, rank="
            f"{dist.get_rank() if dist.is_initialized() else 0}). "
            f"Last failures: {failure_summary}"
        ) from last_error


class InfiniteDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
        """
        无限循环分布式采样器。
        :param dataset: 数据集
        :param num_replicas: 总共的设备数量
        :param rank: 当前设备的 rank
        :param shuffle: 是否随机打乱
        """
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=int(seed),
        )
        self._epoch = 0
        self._offset = 0

    def set_start_sample(self, consumed_local_samples: int) -> None:
        consumed_local_samples = int(consumed_local_samples)
        if consumed_local_samples < 0:
            raise ValueError("consumed_local_samples must be non-negative")
        self._epoch, self._offset = divmod(
            consumed_local_samples,
            self.num_samples,
        )

    def __iter__(self):
        """
        无限循环返回索引。
        """
        epoch = self._epoch
        offset = self._offset
        while True:
            self.set_epoch(epoch)
            indices = super().__iter__()
            if offset:
                indices = islice(indices, offset, None)
            yield from indices
            epoch += 1
            offset = 0
            self._epoch = epoch
            self._offset = 0

    def __len__(self):
        return self.num_samples


class InfiniteMultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True, seed=0):
        """
        多任务批量采样器，支持 Lightning 的分布式模式。
        :param datasets: 多个数据集的列表
        :param batch_size: 每个 batch 的大小
        :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_datasets = len(self.datasets)
        self.samples_per_dataset = sample_per_dataset
        # self.remaining_samples = batch_size % self.num_datasets
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]

        self.cumulative_sizes = [0] + self.dataset_lengths

        for i in range(1, len(self.cumulative_sizes)):
            self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

        self.cur_idx = 0

        # 为每个数据集创建无限采样器
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.samplers = [
            InfiniteDistributedSampler(
                dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=shuffle,
                seed=int(seed) + dataset_index,
            )
            for dataset_index, dataset in enumerate(datasets)
        ]
        self.iterators = [iter(sampler) for sampler in self.samplers]

    def __iter__(self):
        """
        无限生成每个 batch 的样本索引。
        """
        while True:
            batch = []
            for i in range(len(self.iterators)):
                iterator = self.iterators[i]
                for _ in range(self.samples_per_dataset[i]):
                    batch.append(next(iterator) + self.cumulative_sizes[i])
            yield batch

    def __len__(self):
        return sum(self.dataset_lengths)


class FiniteMultiTaskBatchSampler(BatchSampler):
    def __init__(
        self,
        datasets,
        batch_size,
        sample_per_dataset,
        drop_last=False,
        shuffle=True,
        seed=0,
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.samples_per_dataset = sample_per_dataset
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = [0] + self.dataset_lengths

        for i in range(1, len(self.cumulative_sizes)):
            self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

        self.drop_last = drop_last
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

        # 初始化标准分布式采样器
        self.samplers = [
            DistributedSampler(
                dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=shuffle,
                seed=int(seed) + dataset_index,
            )
            for dataset_index, dataset in enumerate(datasets)
        ]
        self.iterators = [iter(sampler) for sampler in self.samplers]
        # 计算每个数据集还剩多少样本
        self.remaining_samples = [len(sampler) for sampler in self.samplers]

    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]
        remaining_samples = self.remaining_samples.copy()

        while sum(remaining_samples) > 0:
            batch = []
            for i, iterator in enumerate(iterators):
                num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
                for _ in range(num_samples):
                    try:
                        idx = next(iterator)
                        batch.append(idx + self.cumulative_sizes[i])
                        remaining_samples[i] -= 1
                    except StopIteration:
                        remaining_samples[i] = 0
                        break  # 当前dataset采样完毕

            if len(batch) == 0:
                break

            # 根据drop_last判断batch大小
            if self.drop_last and len(batch) < self.batch_size:
                break

            yield batch

    def __len__(self):
        # 总的batch数量（近似值）
        total_samples = sum(self.dataset_lengths)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, index):
        cumulative_sizes = 0
        for dataset, length in zip(self.datasets, self.dataset_lengths):
            if index < cumulative_sizes + length:
                return dataset[index - cumulative_sizes]
            cumulative_sizes += length
        raise IndexError("Index out of range")


def _loader_worker_options(
    num_workers,
    loader_timeout_seconds=0,
    persistent_workers=False,
    worker_start_method=None,
    read_trace_dir=None,
    read_trace_samples=0,
):
    num_workers = int(num_workers)
    loader_timeout_seconds = float(loader_timeout_seconds)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if loader_timeout_seconds < 0:
        raise ValueError("loader_timeout_seconds must be non-negative")

    options = {
        "num_workers": num_workers,
        "timeout": loader_timeout_seconds if num_workers > 0 else 0,
        "persistent_workers": bool(persistent_workers) if num_workers > 0 else False,
    }
    worker_start_method = str(worker_start_method or "").strip().lower()
    if num_workers > 0 and worker_start_method:
        if worker_start_method not in {"fork", "spawn", "forkserver"}:
            raise ValueError(
                "worker_start_method must be fork, spawn, forkserver, or empty"
            )
        options["multiprocessing_context"] = worker_start_method
    return options


def load_unsampler_datasets_from_json(
    json_path,
    flip_p,
    img_size,
    local_batch_size,
    num_workers=8,
    is_infinite=True,
    shuffle=True,
    drop_last=False,
    max_read_attempts=8,
    seed=0,
    loader_timeout_seconds=0,
    persistent_workers=False,
    worker_start_method=None,
    read_trace_dir=None,
    read_trace_samples=0,
    eval_mode=None,
    fast_eval_num_tasks=100,
    fast_eval_task_seed=0,
):
    with open(json_path, "r") as f:
        config = json.load(f)

    dataset_paths = config["datasets"]
    if not dataset_paths:
        raise ValueError(f"Dataset config {json_path!r} contains no datasets.")
    dataset_path = os.path.join(os.path.dirname(json_path), dataset_paths[0])
    dataset = ImageData(
        dataset_path,
        flip_p=flip_p,
        img_size=img_size,
        max_read_attempts=max_read_attempts,
        seed=seed,
        read_trace_dir=read_trace_dir,
        read_trace_samples=read_trace_samples,
    )

    for dataset_path in dataset_paths[1:]:
        dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
        dataset.add(dataset_path)

    if eval_mode is not None:
        dataset.select_eval_tasks(
            mode=eval_mode,
            fast_num_tasks=fast_eval_num_tasks,
            seed=fast_eval_task_seed,
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    num_replicas = dist.get_world_size() if dist.is_initialized() else 1
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(seed) + rank)
    worker_options = _loader_worker_options(
        num_workers,
        loader_timeout_seconds=loader_timeout_seconds,
        persistent_workers=persistent_workers,
        worker_start_method=worker_start_method,
    )

    if is_infinite:
        sampler = InfiniteDistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=drop_last,
            generator=loader_generator,
            **worker_options,
        )
    else:
        # Original non-distributed eval loader:
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=local_batch_size,
        #     num_workers=num_workers,
        #     collate_fn=collate_fn,
        #     shuffle=shuffle,
        #     drop_last=drop_last,
        # )

        # Distributed eval loader:
        # each rank receives a different shard of the finite eval dataset.
        sampler = None
        if num_replicas > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=int(seed),
            )

        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            shuffle=False if sampler is not None else shuffle,
            drop_last=drop_last,
            generator=loader_generator,
            **worker_options,
        )
    return dataloader


def load_multi_datasets_form_json(
    json_path,
    flip_p,
    img_size,
    local_batch_size,
    num_workers=8,
    is_infinite=True,
    shuffle=True,
    drop_last=False,
    make_single_dataset=False,
    max_read_attempts=8,
    seed=0,
    loader_timeout_seconds=0,
    persistent_workers=False,
    worker_start_method=None,
    read_trace_dir=None,
    read_trace_samples=0,
    eval_mode=None,
    fast_eval_num_tasks=100,
    fast_eval_task_seed=0,
):
    if make_single_dataset:
        return load_unsampler_datasets_from_json(
            json_path=json_path,
            flip_p=flip_p,
            img_size=img_size,
            local_batch_size=local_batch_size,
            num_workers=num_workers,
            is_infinite=is_infinite,
            shuffle=shuffle,
            drop_last=drop_last,
            max_read_attempts=max_read_attempts,
            seed=seed,
            loader_timeout_seconds=loader_timeout_seconds,
            persistent_workers=persistent_workers,
            worker_start_method=worker_start_method,
            read_trace_dir=read_trace_dir,
            read_trace_samples=read_trace_samples,
            eval_mode=eval_mode,
            fast_eval_num_tasks=fast_eval_num_tasks,
            fast_eval_task_seed=fast_eval_task_seed,
        )
    if eval_mode is not None:
        raise ValueError(
            "Task-based evaluation modes require make_single_dataset=True."
        )
    with open(json_path, "r") as f:
        config = json.load(f)

    dataset_paths = config["datasets"]
    ratios = config["ratios"]
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

    datasets = []

    for dataset_path in dataset_paths:
        dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
        datasets.append(
            ImageData(
                dataset_path,
                flip_p=flip_p,
                img_size=img_size,
                max_read_attempts=max_read_attempts,
                seed=int(seed) + len(datasets),
                read_trace_dir=read_trace_dir,
                read_trace_samples=read_trace_samples,
            )
        )

    sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

    total = sum(sample_per_dataset)
    if total < local_batch_size:
        sample_per_dataset[-1] += local_batch_size - total
    elif total > local_batch_size:
        sample_per_dataset[-1] -= total - local_batch_size

    wrapped_dataset = MultiDatasetWrapper(datasets)

    if is_infinite:
        batch_sampler = InfiniteMultiTaskBatchSampler(
            datasets,
            local_batch_size,
            sample_per_dataset=sample_per_dataset,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        batch_sampler = FiniteMultiTaskBatchSampler(
            datasets,
            local_batch_size,
            sample_per_dataset=sample_per_dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(seed) + rank)
    dataloader = DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        generator=loader_generator,
        **_loader_worker_options(
            num_workers,
            loader_timeout_seconds=loader_timeout_seconds,
            persistent_workers=persistent_workers,
            worker_start_method=worker_start_method,
        ),
    )

    return dataloader


if __name__ == "__main__":
    import torchvision.transforms as T

    dataset = ImageData(metadata_path="jsons/eval_VLA.jsonl", flip_p=0.5)

    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
    data = next(iter(dataloader))
    print("image shape:", data["images"].shape)

    dataloader = load_multi_datasets_form_json(
        json_path="jsons/eval_VLA.json",
        flip_p=0,
        img_size=224,
        local_batch_size=4,
        num_workers=0,
        is_infinite=False,
        shuffle=False,
        drop_last=False,
        make_single_dataset=True,
    )

    data = next(iter(dataloader))
    print("image shape:", data["images"].shape)
