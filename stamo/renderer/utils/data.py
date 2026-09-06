# import json
# import math
# import os
# import random

# import jsonlines
# import torch
# import torch.distributed as dist
# import torchvision.transforms as T
# from PIL import Image
# from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

# from stamo.renderer.utils.device import get_accelerator_device  # edit for musa
# from stamo.renderer.utils.overwatch import initialize_overwatch


# overwatch = initialize_overwatch(__name__)


# def complex_to_device(complex, device, non_blocking=False):
#     if complex is None:
#         return complex
#     if isinstance(complex, torch.Tensor):
#         return complex.to(device, non_blocking=non_blocking)
#     elif isinstance(complex, dict):
#         return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
#     elif isinstance(complex, list) or isinstance(complex, tuple):
#         return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
#     elif (
#         isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
#     ):
#         return complex
#     else:
#         raise ValueError("Unsupported complex", complex)


# def fp32_to_fp16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.half)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_fp16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_fp16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def fp32_to_bf16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.bfloat16)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_bf16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_bf16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def move_to_cuda(batch, device=None):  # edit for musa
#     device = device or get_accelerator_device()
#     if isinstance(batch, torch.Tensor):
#         return batch.to(device=device, non_blocking=True)
#     elif isinstance(batch, list):
#         new_batch = [move_to_cuda(t, device=device) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def collate_fn(inputs):
#     images = torch.stack([input["image"] for input in inputs])
#     return {"images": images}


# def get_loader_info(dataset, epochs, bsz):
#     images_per_gpu = bsz
#     images_per_batch = bsz * overwatch.world_size()
#     iter_per_ep = len(dataset) // overwatch.world_size()
#     num_iters = iter_per_ep * epochs
#     loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
#     return loader_info


# class ImageData(Dataset):
#     def __init__(self, metadata_path, flip_p, img_size: int = 224):
#         self.flip_p = flip_p

#         self.metadata = []
#         with open(metadata_path, "r+", encoding="utf8") as f:
#             for item in jsonlines.Reader(f):
#                 self.metadata.append(item["image"])

#         self.length = len(self.metadata)

#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#         self.transforms = T.Compose(
#             [
#                 T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
#                 T.ToTensor(),
#             ]
#         )

#     def preprocess_train(self, image):
#         if torch.rand(1) < self.flip_p:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#         image = image.convert("RGB")
#         image = self.transforms(image)
#         return image

#     def add(self, metadata_path):
#         with open(metadata_path, "r+", encoding="utf8") as f:
#             for item in jsonlines.Reader(f):
#                 self.metadata.append(item["image"])
#         self.length = len(self.metadata)
#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         while True:
#             try:
#                 image = Image.open(self.metadata[idx])
#                 image = self.preprocess_train(image)
#                 inputs = {"image": image}
#                 break
#             except Exception:
#                 overwatch.warning(f"read {self.metadata[idx]} error")
#                 idx = random.randint(0, self.length - 1)
#         return inputs


# class InfiniteDistributedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
#         """
#         无限循环分布式采样器。
#         :param dataset: 数据集
#         :param num_replicas: 总共的设备数量
#         :param rank: 当前设备的 rank
#         :param shuffle: 是否随机打乱
#         """
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
#         self._epoch = 0

#     def __iter__(self):
#         """
#         无限循环返回索引。
#         """
#         while True:
#             self.set_epoch(self._epoch)
#             self._epoch += 1
#             indices = super().__iter__()
#             yield from indices

#     def __len__(self):
#         return len(self.dataset)


# class InfiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True):
#         """
#         多任务批量采样器，支持 Lightning 的分布式模式。
#         :param datasets: 多个数据集的列表
#         :param batch_size: 每个 batch 的大小
#         :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
#         """
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.num_datasets = len(self.datasets)
#         self.samples_per_dataset = sample_per_dataset
#         # self.remaining_samples = batch_size % self.num_datasets
#         self.dataset_lengths = [len(dataset) for dataset in self.datasets]

#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.cur_idx = 0

#         # 为每个数据集创建无限采样器
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#         self.samplers = [
#             InfiniteDistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle)
#             for dataset in datasets
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         """
#         无限生成每个 batch 的样本索引。
#         """
#         while True:
#             batch = []
#             for i in range(len(self.iterators)):
#                 iterator = self.iterators[i]
#                 for _ in range(self.samples_per_dataset[i]):
#                     batch.append(next(iterator) + self.cumulative_sizes[i])
#             yield batch

#     def __len__(self):
#         return sum(self.dataset_lengths)


# class FiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(self, datasets, batch_size, sample_per_dataset, drop_last=False, shuffle=True):
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.samples_per_dataset = sample_per_dataset
#         self.dataset_lengths = [len(dataset) for dataset in datasets]
#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.drop_last = drop_last
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

#         # 初始化标准分布式采样器
#         self.samplers = [
#             DistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle)
#             for dataset in datasets
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]
#         # 计算每个数据集还剩多少样本
#         self.remaining_samples = [len(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         iterators = [iter(sampler) for sampler in self.samplers]
#         remaining_samples = self.remaining_samples.copy()

#         while sum(remaining_samples) > 0:
#             batch = []
#             for i, iterator in enumerate(iterators):
#                 num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
#                 for _ in range(num_samples):
#                     try:
#                         idx = next(iterator)
#                         batch.append(idx + self.cumulative_sizes[i])
#                         remaining_samples[i] -= 1
#                     except StopIteration:
#                         remaining_samples[i] = 0
#                         break  # 当前dataset采样完毕

#             if len(batch) == 0:
#                 break

#             # 根据drop_last判断batch大小
#             if self.drop_last and len(batch) < self.batch_size:
#                 break

#             yield batch

#     def __len__(self):
#         # 总的batch数量（近似值）
#         total_samples = sum(self.dataset_lengths)
#         if self.drop_last:
#             return total_samples // self.batch_size
#         else:
#             return (total_samples + self.batch_size - 1) // self.batch_size


# class MultiDatasetWrapper(Dataset):
#     def __init__(self, datasets):
#         self.datasets = datasets
#         self.dataset_lengths = [len(ds) for ds in datasets]

#     def __len__(self):
#         return sum(self.dataset_lengths)

#     def __getitem__(self, index):
#         cumulative_sizes = 0
#         for dataset, length in zip(self.datasets, self.dataset_lengths):
#             if index < cumulative_sizes + length:
#                 return dataset[index - cumulative_sizes]
#             cumulative_sizes += length
#         raise IndexError("Index out of range")


# def load_unsampler_datasets_from_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
# ):
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     dataset_path = os.path.join(os.path.dirname(json_path), dataset_paths[0])
#     dataset = ImageData(dataset_path, flip_p=flip_p, img_size=img_size)

#     for dataset_path in dataset_paths[1:]:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         dataset.add(dataset_path)

#     rank = dist.get_rank() if dist.is_initialized() else 0
#     num_replicas = dist.get_world_size() if dist.is_initialized() else 1

#     if is_infinite:
#         sampler = InfiniteDistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             num_workers=num_workers,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             drop_last=drop_last,
#         )
#     else:
#         # Original non-distributed eval loader:
#         # dataloader = DataLoader(
#         #     dataset,
#         #     batch_size=local_batch_size,
#         #     num_workers=num_workers,
#         #     collate_fn=collate_fn,
#         #     shuffle=shuffle,
#         #     drop_last=drop_last,
#         # )

#         # Distributed eval loader:
#         # each rank receives a different shard of the finite eval dataset.
#         sampler = None
#         if num_replicas > 1:
#             sampler = DistributedSampler(
#                 dataset,
#                 num_replicas=num_replicas,
#                 rank=rank,
#                 shuffle=shuffle,
#                 drop_last=drop_last,
#             )

#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             num_workers=num_workers,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             shuffle=False if sampler is not None else shuffle,
#             drop_last=drop_last,
#         )
#     return dataloader


# def load_multi_datasets_form_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     make_single_dataset=False,
# ):
#     if make_single_dataset:
#         return load_unsampler_datasets_from_json(
#             json_path,
#             flip_p,
#             img_size,
#             local_batch_size,
#             num_workers,
#             is_infinite,
#             shuffle,
#             drop_last,
#         )
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     ratios = config["ratios"]
#     assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
#     assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

#     datasets = []

#     for dataset_path in dataset_paths:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         datasets.append(ImageData(dataset_path, flip_p=flip_p, img_size=img_size))

#     sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

#     total = sum(sample_per_dataset)
#     if total < local_batch_size:
#         sample_per_dataset[-1] += local_batch_size - total
#     elif total > local_batch_size:
#         sample_per_dataset[-1] -= total - local_batch_size

#     wrapped_dataset = MultiDatasetWrapper(datasets)

#     if is_infinite:
#         batch_sampler = InfiniteMultiTaskBatchSampler(
#             datasets, local_batch_size, sample_per_dataset=sample_per_dataset, shuffle=shuffle
#         )
#     else:
#         batch_sampler = FiniteMultiTaskBatchSampler(
#             datasets, local_batch_size, sample_per_dataset=sample_per_dataset, shuffle=shuffle, drop_last=drop_last
#         )

#     dataloader = DataLoader(
#         wrapped_dataset,
#         num_workers=num_workers,
#         batch_sampler=batch_sampler,
#         collate_fn=collate_fn,
#     )

#     return dataloader


# if __name__ == "__main__":
#     import torchvision.transforms as T

#     dataset = ImageData(metadata_path="jsons/eval_VLA.jsonl", flip_p=0.5)

#     dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)

#     dataloader = load_multi_datasets_form_json(
#         json_path="jsons/eval_VLA.json",
#         flip_p=0,
#         img_size=224,
#         local_batch_size=4,
#         num_workers=0,
#         is_infinite=False,
#         shuffle=False,
#         drop_last=False,
#         make_single_dataset=True,
#     )

#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)


# import json
# import math
# import os
# import random

# import jsonlines
# import torch
# import torch.distributed as dist
# import torchvision.transforms as T
# from PIL import Image
# from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

# from stamo.renderer.utils.device import get_accelerator_device
# from stamo.renderer.utils.overwatch import initialize_overwatch


# overwatch = initialize_overwatch(__name__)


# def complex_to_device(complex, device, non_blocking=False):
#     if complex is None:
#         return complex
#     if isinstance(complex, torch.Tensor):
#         return complex.to(device, non_blocking=non_blocking)
#     elif isinstance(complex, dict):
#         return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
#     elif isinstance(complex, list) or isinstance(complex, tuple):
#         return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
#     elif (
#         isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
#     ):
#         return complex
#     else:
#         raise ValueError("Unsupported complex", complex)


# def fp32_to_fp16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.half)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_fp16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_fp16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def fp32_to_bf16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.bfloat16)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_bf16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_bf16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def move_to_cuda(batch, device=None):
#     device = device or get_accelerator_device()
#     if isinstance(batch, torch.Tensor):
#         return batch.to(device=device, non_blocking=True)
#     elif isinstance(batch, list):
#         new_batch = [move_to_cuda(t, device=device) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def collate_fn(inputs):
#     images = torch.stack([input["image"] for input in inputs])
#     return {"images": images}


# def get_loader_info(dataset, epochs, bsz, gradient_accumulate_steps=1):
#     dataset_len = len(dataset.dataset) if hasattr(dataset, "dataset") else len(dataset)
#     images_per_gpu = bsz
#     images_per_batch = bsz * overwatch.world_size() * gradient_accumulate_steps
#     iter_per_ep = dataset_len // images_per_batch
#     num_iters = iter_per_ep * epochs
#     loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
#     return loader_info


# class ImageData(Dataset):
#     def __init__(self, metadata_path, flip_p, img_size: int = 224):
#         self.flip_p = flip_p

#         self.metadata = []
#         with open(metadata_path, "r", encoding="utf8") as f:
#             for item in jsonlines.Reader(f):
#                 self.metadata.append(item["image"])

#         self.length = len(self.metadata)

#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#         self.transforms = T.Compose(
#             [
#                 T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
#                 T.ToTensor(),
#             ]
#         )

#     def preprocess_train(self, image):
#         if torch.rand(1) < self.flip_p:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#         image = image.convert("RGB")
#         image = self.transforms(image)
#         return image

#     def add(self, metadata_path):
#         with open(metadata_path, "r", encoding="utf8") as f:
#             for item in jsonlines.Reader(f):
#                 self.metadata.append(item["image"])
#         self.length = len(self.metadata)
#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         while True:
#             try:
#                 image = Image.open(self.metadata[idx])
#                 image = self.preprocess_train(image)
#                 inputs = {"image": image}
#                 break
#             except Exception:
#                 overwatch.warning(f"read {self.metadata[idx]} error")
#                 idx = random.randint(0, self.length - 1)
#         return inputs


# class InfiniteDistributedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
#         """
#         无限循环分布式采样器。
#         :param dataset: 数据集
#         :param num_replicas: 总共的设备数量
#         :param rank: 当前设备的 rank
#         :param shuffle: 是否随机打乱
#         """
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
#         self._epoch = 0

#     def __iter__(self):
#         """
#         无限循环返回索引。
#         """
#         while True:
#             self.set_epoch(self._epoch)
#             self._epoch += 1
#             indices = super().__iter__()
#             yield from indices

#     def __len__(self):
#         return len(self.dataset)


# class InfiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True):
#         """
#         多任务批量采样器，支持 Lightning 的分布式模式。
#         :param datasets: 多个数据集的列表
#         :param batch_size: 每个 batch 的大小
#         :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
#         """
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.num_datasets = len(self.datasets)
#         self.samples_per_dataset = sample_per_dataset
#         # self.remaining_samples = batch_size % self.num_datasets
#         self.dataset_lengths = [len(dataset) for dataset in self.datasets]

#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.cur_idx = 0

#         # 为每个数据集创建无限采样器
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#         self.samplers = [
#             InfiniteDistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle)
#             for dataset in datasets
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         """
#         无限生成每个 batch 的样本索引。
#         """
#         while True:
#             batch = []
#             for i in range(len(self.iterators)):
#                 iterator = self.iterators[i]
#                 for _ in range(self.samples_per_dataset[i]):
#                     batch.append(next(iterator) + self.cumulative_sizes[i])
#             yield batch

#     def __len__(self):
#         return sum(self.dataset_lengths)


# class FiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(self, datasets, batch_size, sample_per_dataset, drop_last=False, shuffle=True):
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.samples_per_dataset = sample_per_dataset
#         self.dataset_lengths = [len(dataset) for dataset in datasets]
#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.drop_last = drop_last
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

#         # 初始化标准分布式采样器
#         self.samplers = [
#             DistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle)
#             for dataset in datasets
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]
#         # 计算每个数据集还剩多少样本
#         self.remaining_samples = [len(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         iterators = [iter(sampler) for sampler in self.samplers]
#         remaining_samples = self.remaining_samples.copy()

#         while sum(remaining_samples) > 0:
#             batch = []
#             for i, iterator in enumerate(iterators):
#                 num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
#                 for _ in range(num_samples):
#                     try:
#                         idx = next(iterator)
#                         batch.append(idx + self.cumulative_sizes[i])
#                         remaining_samples[i] -= 1
#                     except StopIteration:
#                         remaining_samples[i] = 0
#                         break  # 当前dataset采样完毕

#             if len(batch) == 0:
#                 break

#             # 根据drop_last判断batch大小
#             if self.drop_last and len(batch) < self.batch_size:
#                 break

#             yield batch

#     def __len__(self):
#         # 总的batch数量（近似值）
#         total_samples = sum(self.dataset_lengths)
#         if self.drop_last:
#             return total_samples // self.batch_size
#         else:
#             return (total_samples + self.batch_size - 1) // self.batch_size


# class MultiDatasetWrapper(Dataset):
#     def __init__(self, datasets):
#         self.datasets = datasets
#         self.dataset_lengths = [len(ds) for ds in datasets]

#     def __len__(self):
#         return sum(self.dataset_lengths)

#     def __getitem__(self, index):
#         cumulative_sizes = 0
#         for dataset, length in zip(self.datasets, self.dataset_lengths):
#             if index < cumulative_sizes + length:
#                 return dataset[index - cumulative_sizes]
#             cumulative_sizes += length
#         raise IndexError("Index out of range")


# def load_unsampler_datasets_from_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
# ):
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     dataset_path = os.path.join(os.path.dirname(json_path), dataset_paths[0])
#     dataset = ImageData(dataset_path, flip_p=flip_p, img_size=img_size)

#     for dataset_path in dataset_paths[1:]:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         dataset.add(dataset_path)

#     rank = dist.get_rank() if dist.is_initialized() else 0
#     num_replicas = dist.get_world_size() if dist.is_initialized() else 1

#     if is_infinite:
#         sampler = InfiniteDistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             num_workers=num_workers,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             drop_last=drop_last,
#         )
#     else:
#         # Original non-distributed eval loader:
#         # dataloader = DataLoader(
#         #     dataset,
#         #     batch_size=local_batch_size,
#         #     num_workers=num_workers,
#         #     collate_fn=collate_fn,
#         #     shuffle=shuffle,
#         #     drop_last=drop_last,
#         # )

#         # Distributed eval loader:
#         # each rank receives a different shard of the finite eval dataset.
#         sampler = None
#         if num_replicas > 1:
#             sampler = DistributedSampler(
#                 dataset,
#                 num_replicas=num_replicas,
#                 rank=rank,
#                 shuffle=shuffle,
#                 drop_last=drop_last,
#             )

#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             num_workers=num_workers,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             shuffle=False if sampler is not None else shuffle,
#             drop_last=drop_last,
#         )
#     return dataloader


# def load_multi_datasets_form_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     make_single_dataset=False,
# ):
#     if make_single_dataset:
#         return load_unsampler_datasets_from_json(
#             json_path,
#             flip_p,
#             img_size,
#             local_batch_size,
#             num_workers,
#             is_infinite,
#             shuffle,
#             drop_last,
#         )
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     ratios = config["ratios"]
#     assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
#     assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

#     datasets = []

#     for dataset_path in dataset_paths:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         datasets.append(ImageData(dataset_path, flip_p=flip_p, img_size=img_size))

#     sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

#     total = sum(sample_per_dataset)
#     if total < local_batch_size:
#         sample_per_dataset[-1] += local_batch_size - total
#     elif total > local_batch_size:
#         sample_per_dataset[-1] -= total - local_batch_size

#     wrapped_dataset = MultiDatasetWrapper(datasets)

#     if is_infinite:
#         batch_sampler = InfiniteMultiTaskBatchSampler(
#             datasets, local_batch_size, sample_per_dataset=sample_per_dataset, shuffle=shuffle
#         )
#     else:
#         batch_sampler = FiniteMultiTaskBatchSampler(
#             datasets, local_batch_size, sample_per_dataset=sample_per_dataset, shuffle=shuffle, drop_last=drop_last
#         )

#     dataloader = DataLoader(
#         wrapped_dataset,
#         num_workers=num_workers,
#         batch_sampler=batch_sampler,
#         collate_fn=collate_fn,
#     )

#     return dataloader


# if __name__ == "__main__":
#     import torchvision.transforms as T

#     dataset = ImageData(metadata_path="jsons/eval_VLA.jsonl", flip_p=0.5)

#     dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)

#     dataloader = load_multi_datasets_form_json(
#         json_path="jsons/eval_VLA.json",
#         flip_p=0,
#         img_size=224,
#         local_batch_size=4,
#         num_workers=0,
#         is_infinite=False,
#         shuffle=False,
#         drop_last=False,
#         make_single_dataset=True,
#     )

#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)






# import json
# import math
# import os
# import re
# import time
# from itertools import islice

# import jsonlines
# import torch
# import torch.distributed as dist
# import torchvision.transforms as T
# from PIL import Image, UnidentifiedImageError
# from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

# from stamo.renderer.utils.device import get_accelerator_device
# from stamo.renderer.utils.overwatch import initialize_overwatch


# overwatch = initialize_overwatch(__name__)


# def complex_to_device(complex, device, non_blocking=False):
#     if complex is None:
#         return complex
#     if isinstance(complex, torch.Tensor):
#         return complex.to(device, non_blocking=non_blocking)
#     elif isinstance(complex, dict):
#         return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
#     elif isinstance(complex, list) or isinstance(complex, tuple):
#         return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
#     elif (
#         isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
#     ):
#         return complex
#     else:
#         raise ValueError("Unsupported complex", complex)


# def fp32_to_fp16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.half)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_fp16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_fp16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def fp32_to_bf16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.bfloat16)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_bf16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_bf16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def move_to_cuda(batch, device=None):
#     device = device or get_accelerator_device()
#     if isinstance(batch, torch.Tensor):
#         return batch.to(device=device, non_blocking=True)
#     elif isinstance(batch, list):
#         new_batch = [move_to_cuda(t, device=device) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def collate_fn(inputs):
#     images = torch.stack([input["image"] for input in inputs])
#     batch = {"images": images}
#     paths = [input.get("path") for input in inputs]
#     if all(path is not None for path in paths):
#         batch["paths"] = paths
#     return batch


# def get_loader_info(dataset, epochs, bsz, gradient_accumulate_steps=1):
#     dataset_len = len(dataset.dataset) if hasattr(dataset, "dataset") else len(dataset)
#     images_per_gpu = bsz
#     world_size = dist.get_world_size() if dist.is_initialized() else 1
#     images_per_batch = bsz * world_size * gradient_accumulate_steps
#     iter_per_ep = math.ceil(dataset_len / images_per_batch) if dataset_len else 0
#     num_iters = iter_per_ep * epochs
#     loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
#     return loader_info


# class ImageData(Dataset):
#     def __init__(
#         self,
#         metadata_path,
#         flip_p,
#         img_size: int = 224,
#         max_read_attempts: int = 8,
#         seed: int = 0,
#         read_trace_dir=None,
#         read_trace_samples: int = 0,
#     ):
#         self.flip_p = flip_p
#         self.max_read_attempts = int(max_read_attempts)
#         self.seed = int(seed)
#         self.read_trace_dir = (
#             os.path.abspath(os.path.expanduser(str(read_trace_dir)))
#             if read_trace_dir
#             else None
#         )
#         self.read_trace_samples = max(0, int(read_trace_samples))
#         self._read_trace_count = 0
#         self._read_trace_path = None
#         if self.max_read_attempts <= 0:
#             raise ValueError("max_read_attempts must be positive")

#         self.metadata_source = os.path.abspath(str(metadata_path))
#         self.metadata = []
#         self._append_metadata(metadata_path)

#         self.length = len(self.metadata)
#         if self.length == 0:
#             raise ValueError(f"No images were found in metadata file {metadata_path!r}.")

#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#         self.transforms = T.Compose(
#             [
#                 T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
#                 T.ToTensor(),
#             ]
#         )

#     def preprocess_train(self, image):
#         if torch.rand(1) < self.flip_p:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#         image = self.transforms(image)
#         return image

#     def _append_metadata(self, metadata_path):
#         metadata_path = os.path.abspath(str(metadata_path))
#         metadata_dir = os.path.dirname(metadata_path)
#         with open(metadata_path, "r", encoding="utf8") as f:
#             for line_number, item in enumerate(jsonlines.Reader(f), start=1):
#                 if not isinstance(item, dict):
#                     raise ValueError(
#                         f"{metadata_path}:{line_number} must contain a JSON object."
#                     )
#                 image_path = item.get("image")
#                 if not isinstance(image_path, str) or not image_path.strip():
#                     raise ValueError(
#                         f"{metadata_path}:{line_number} has no non-empty 'image' path."
#                     )
#                 image_path = os.path.expanduser(image_path.strip())
#                 if not os.path.isabs(image_path):
#                     image_path = os.path.abspath(os.path.join(metadata_dir, image_path))
#                 self.metadata.append(image_path)

#     def add(self, metadata_path):
#         self._append_metadata(metadata_path)
#         self.length = len(self.metadata)
#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#     def __len__(self):
#         return self.length

#     def _retry_index(self, original_idx, attempt, attempted_indices):
#         """Choose a deterministic, previously untried fallback sample."""
#         candidate = (
#             int(original_idx)
#             + int(attempt) * 104729
#             + self.seed * 1009
#         ) % self.length
#         while candidate in attempted_indices and len(attempted_indices) < self.length:
#             candidate = (candidate + 1) % self.length
#         return candidate

#     def _trace_read(self, enabled, event, **details):
#         if not enabled or self.read_trace_dir is None:
#             return
#         if self._read_trace_path is None:
#             os.makedirs(self.read_trace_dir, exist_ok=True)
#             rank = dist.get_rank() if dist.is_initialized() else 0
#             metadata_name = re.sub(
#                 r"[^A-Za-z0-9_.-]+",
#                 "_",
#                 os.path.basename(getattr(self, "metadata_source", "images")),
#             )
#             self._read_trace_path = os.path.join(
#                 self.read_trace_dir,
#                 f"data_{metadata_name}_rank{rank:02d}_pid{os.getpid()}.jsonl",
#             )
#         record = {
#             "wall_time_ns": time.time_ns(),
#             "pid": os.getpid(),
#             "rank": dist.get_rank() if dist.is_initialized() else 0,
#             "event": str(event),
#         }
#         record.update(details)
#         # Only the first configured samples are traced, so opening per event is
#         # cheap and avoids leaking a file descriptor into DataLoader workers.
#         with open(self._read_trace_path, "a", encoding="utf-8") as trace_file:
#             trace_file.write(
#                 json.dumps(record, ensure_ascii=False, default=str) + "\n"
#             )
#             trace_file.flush()

#     def __getitem__(self, idx):
#         original_idx = int(idx)
#         trace_this_read = self._read_trace_count < self.read_trace_samples
#         self._read_trace_count += 1
#         candidate_idx = original_idx
#         attempted_indices = set()
#         failures = []
#         total_attempts = min(self.max_read_attempts, self.length)

#         for attempt in range(total_attempts):
#             attempted_indices.add(candidate_idx)
#             path = self.metadata[candidate_idx]
#             self._trace_read(
#                 trace_this_read,
#                 "READ_BEGIN",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             try:
#                 with Image.open(path) as opened_image:
#                     opened_image.load()
#                     image = opened_image.convert("RGB")
#             except (OSError, ValueError, UnidentifiedImageError) as exc:
#                 self._trace_read(
#                     trace_this_read,
#                     "READ_ERROR",
#                     requested_index=original_idx,
#                     candidate_index=candidate_idx,
#                     attempt=attempt + 1,
#                     path=path,
#                     error=repr(exc),
#                 )
#                 failures.append((path, exc))
#                 if attempt + 1 < total_attempts:
#                     candidate_idx = self._retry_index(
#                         original_idx,
#                         attempt + 1,
#                         attempted_indices,
#                     )
#                 continue

#             self._trace_read(
#                 trace_this_read,
#                 "OPEN_OK",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             # Keep transform errors visible. They are programming/configuration
#             # errors, not corrupt-image errors that should silently change data.
#             image = self.preprocess_train(image)
#             self._trace_read(
#                 trace_this_read,
#                 "TRANSFORM_OK",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             return {"image": image, "path": path}

#         failure_summary = "; ".join(
#             f"{path!r}: {type(exc).__name__}: {exc}"
#             for path, exc in failures[-4:]
#         )
#         last_error = failures[-1][1] if failures else None
#         raise RuntimeError(
#             f"Failed to read an image after {total_attempts} bounded attempts "
#             f"(requested index={original_idx}, rank="
#             f"{dist.get_rank() if dist.is_initialized() else 0}). "
#             f"Last failures: {failure_summary}"
#         ) from last_error


# class InfiniteDistributedSampler(DistributedSampler):
#     def __init__(
#         self,
#         dataset,
#         num_replicas=None,
#         rank=None,
#         shuffle=True,
#         seed=0,
#     ):
#         """
#         无限循环分布式采样器。
#         :param dataset: 数据集
#         :param num_replicas: 总共的设备数量
#         :param rank: 当前设备的 rank
#         :param shuffle: 是否随机打乱
#         """
#         super().__init__(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#             seed=int(seed),
#         )
#         self._epoch = 0
#         self._offset = 0

#     def set_start_sample(self, consumed_local_samples: int) -> None:
#         consumed_local_samples = int(consumed_local_samples)
#         if consumed_local_samples < 0:
#             raise ValueError("consumed_local_samples must be non-negative")
#         self._epoch, self._offset = divmod(
#             consumed_local_samples,
#             self.num_samples,
#         )

#     def __iter__(self):
#         """
#         无限循环返回索引。
#         """
#         epoch = self._epoch
#         offset = self._offset
#         while True:
#             self.set_epoch(epoch)
#             indices = super().__iter__()
#             if offset:
#                 indices = islice(indices, offset, None)
#             yield from indices
#             epoch += 1
#             offset = 0
#             self._epoch = epoch
#             self._offset = 0

#     def __len__(self):
#         return self.num_samples


# class InfiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True, seed=0):
#         """
#         多任务批量采样器，支持 Lightning 的分布式模式。
#         :param datasets: 多个数据集的列表
#         :param batch_size: 每个 batch 的大小
#         :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
#         """
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.num_datasets = len(self.datasets)
#         self.samples_per_dataset = sample_per_dataset
#         # self.remaining_samples = batch_size % self.num_datasets
#         self.dataset_lengths = [len(dataset) for dataset in self.datasets]

#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.cur_idx = 0

#         # 为每个数据集创建无限采样器
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#         self.samplers = [
#             InfiniteDistributedSampler(
#                 dataset,
#                 num_replicas=self.num_replicas,
#                 rank=self.rank,
#                 shuffle=shuffle,
#                 seed=int(seed) + dataset_index,
#             )
#             for dataset_index, dataset in enumerate(datasets)
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         """
#         无限生成每个 batch 的样本索引。
#         """
#         while True:
#             batch = []
#             for i in range(len(self.iterators)):
#                 iterator = self.iterators[i]
#                 for _ in range(self.samples_per_dataset[i]):
#                     batch.append(next(iterator) + self.cumulative_sizes[i])
#             yield batch

#     def __len__(self):
#         return sum(self.dataset_lengths)


# class FiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(
#         self,
#         datasets,
#         batch_size,
#         sample_per_dataset,
#         drop_last=False,
#         shuffle=True,
#         seed=0,
#     ):
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.samples_per_dataset = sample_per_dataset
#         self.dataset_lengths = [len(dataset) for dataset in datasets]
#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.drop_last = drop_last
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

#         # 初始化标准分布式采样器
#         self.samplers = [
#             DistributedSampler(
#                 dataset,
#                 num_replicas=self.num_replicas,
#                 rank=self.rank,
#                 shuffle=shuffle,
#                 seed=int(seed) + dataset_index,
#             )
#             for dataset_index, dataset in enumerate(datasets)
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]
#         # 计算每个数据集还剩多少样本
#         self.remaining_samples = [len(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         iterators = [iter(sampler) for sampler in self.samplers]
#         remaining_samples = self.remaining_samples.copy()

#         while sum(remaining_samples) > 0:
#             batch = []
#             for i, iterator in enumerate(iterators):
#                 num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
#                 for _ in range(num_samples):
#                     try:
#                         idx = next(iterator)
#                         batch.append(idx + self.cumulative_sizes[i])
#                         remaining_samples[i] -= 1
#                     except StopIteration:
#                         remaining_samples[i] = 0
#                         break  # 当前dataset采样完毕

#             if len(batch) == 0:
#                 break

#             # 根据drop_last判断batch大小
#             if self.drop_last and len(batch) < self.batch_size:
#                 break

#             yield batch

#     def __len__(self):
#         # 总的batch数量（近似值）
#         total_samples = sum(self.dataset_lengths)
#         if self.drop_last:
#             return total_samples // self.batch_size
#         else:
#             return (total_samples + self.batch_size - 1) // self.batch_size


# class MultiDatasetWrapper(Dataset):
#     def __init__(self, datasets):
#         self.datasets = datasets
#         self.dataset_lengths = [len(ds) for ds in datasets]

#     def __len__(self):
#         return sum(self.dataset_lengths)

#     def __getitem__(self, index):
#         cumulative_sizes = 0
#         for dataset, length in zip(self.datasets, self.dataset_lengths):
#             if index < cumulative_sizes + length:
#                 return dataset[index - cumulative_sizes]
#             cumulative_sizes += length
#         raise IndexError("Index out of range")


# def _loader_worker_options(
#     num_workers,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     read_trace_dir=None,
#     read_trace_samples=0,
# ):
#     num_workers = int(num_workers)
#     loader_timeout_seconds = float(loader_timeout_seconds)
#     if num_workers < 0:
#         raise ValueError("num_workers must be non-negative")
#     if loader_timeout_seconds < 0:
#         raise ValueError("loader_timeout_seconds must be non-negative")

#     options = {
#         "num_workers": num_workers,
#         "timeout": loader_timeout_seconds if num_workers > 0 else 0,
#         "persistent_workers": bool(persistent_workers) if num_workers > 0 else False,
#     }
#     worker_start_method = str(worker_start_method or "").strip().lower()
#     if num_workers > 0 and worker_start_method:
#         if worker_start_method not in {"fork", "spawn", "forkserver"}:
#             raise ValueError(
#                 "worker_start_method must be fork, spawn, forkserver, or empty"
#             )
#         options["multiprocessing_context"] = worker_start_method
#     return options


# def load_unsampler_datasets_from_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     max_read_attempts=8,
#     seed=0,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     read_trace_dir=None,
#     read_trace_samples=0,
# ):
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     if not dataset_paths:
#         raise ValueError(f"Dataset config {json_path!r} contains no datasets.")
#     dataset_path = os.path.join(os.path.dirname(json_path), dataset_paths[0])
#     dataset = ImageData(
#         dataset_path,
#         flip_p=flip_p,
#         img_size=img_size,
#         max_read_attempts=max_read_attempts,
#         seed=seed,
#         read_trace_dir=read_trace_dir,
#         read_trace_samples=read_trace_samples,
#     )

#     for dataset_path in dataset_paths[1:]:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         dataset.add(dataset_path)

#     rank = dist.get_rank() if dist.is_initialized() else 0
#     num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#     loader_generator = torch.Generator()
#     loader_generator.manual_seed(int(seed) + rank)
#     worker_options = _loader_worker_options(
#         num_workers,
#         loader_timeout_seconds=loader_timeout_seconds,
#         persistent_workers=persistent_workers,
#         worker_start_method=worker_start_method,
#     )

#     if is_infinite:
#         sampler = InfiniteDistributedSampler(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#             seed=seed,
#         )
#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             drop_last=drop_last,
#             generator=loader_generator,
#             **worker_options,
#         )
#     else:
#         # Original non-distributed eval loader:
#         # dataloader = DataLoader(
#         #     dataset,
#         #     batch_size=local_batch_size,
#         #     num_workers=num_workers,
#         #     collate_fn=collate_fn,
#         #     shuffle=shuffle,
#         #     drop_last=drop_last,
#         # )

#         # Distributed eval loader:
#         # each rank receives a different shard of the finite eval dataset.
#         sampler = None
#         if num_replicas > 1:
#             sampler = DistributedSampler(
#                 dataset,
#                 num_replicas=num_replicas,
#                 rank=rank,
#                 shuffle=shuffle,
#                 drop_last=drop_last,
#                 seed=int(seed),
#             )

#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             shuffle=False if sampler is not None else shuffle,
#             drop_last=drop_last,
#             generator=loader_generator,
#             **worker_options,
#         )
#     return dataloader


# def load_multi_datasets_form_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     make_single_dataset=False,
#     max_read_attempts=8,
#     seed=0,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     read_trace_dir=None,
#     read_trace_samples=0,
# ):
#     if make_single_dataset:
#         return load_unsampler_datasets_from_json(
#             json_path=json_path,
#             flip_p=flip_p,
#             img_size=img_size,
#             local_batch_size=local_batch_size,
#             num_workers=num_workers,
#             is_infinite=is_infinite,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             max_read_attempts=max_read_attempts,
#             seed=seed,
#             loader_timeout_seconds=loader_timeout_seconds,
#             persistent_workers=persistent_workers,
#             worker_start_method=worker_start_method,
#             read_trace_dir=read_trace_dir,
#             read_trace_samples=read_trace_samples,
#         )
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     ratios = config["ratios"]
#     assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
#     assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

#     datasets = []

#     for dataset_path in dataset_paths:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         datasets.append(
#             ImageData(
#                 dataset_path,
#                 flip_p=flip_p,
#                 img_size=img_size,
#                 max_read_attempts=max_read_attempts,
#                 seed=int(seed) + len(datasets),
#                 read_trace_dir=read_trace_dir,
#                 read_trace_samples=read_trace_samples,
#             )
#         )

#     sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

#     total = sum(sample_per_dataset)
#     if total < local_batch_size:
#         sample_per_dataset[-1] += local_batch_size - total
#     elif total > local_batch_size:
#         sample_per_dataset[-1] -= total - local_batch_size

#     wrapped_dataset = MultiDatasetWrapper(datasets)

#     if is_infinite:
#         batch_sampler = InfiniteMultiTaskBatchSampler(
#             datasets,
#             local_batch_size,
#             sample_per_dataset=sample_per_dataset,
#             shuffle=shuffle,
#             seed=seed,
#         )
#     else:
#         batch_sampler = FiniteMultiTaskBatchSampler(
#             datasets,
#             local_batch_size,
#             sample_per_dataset=sample_per_dataset,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             seed=seed,
#         )

#     rank = dist.get_rank() if dist.is_initialized() else 0
#     loader_generator = torch.Generator()
#     loader_generator.manual_seed(int(seed) + rank)
#     dataloader = DataLoader(
#         wrapped_dataset,
#         batch_sampler=batch_sampler,
#         collate_fn=collate_fn,
#         generator=loader_generator,
#         **_loader_worker_options(
#             num_workers,
#             loader_timeout_seconds=loader_timeout_seconds,
#             persistent_workers=persistent_workers,
#             worker_start_method=worker_start_method,
#         ),
#     )

#     return dataloader


# if __name__ == "__main__":
#     import torchvision.transforms as T

#     dataset = ImageData(metadata_path="jsons/eval_VLA.jsonl", flip_p=0.5)

#     dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)

#     dataloader = load_multi_datasets_form_json(
#         json_path="jsons/eval_VLA.json",
#         flip_p=0,
#         img_size=224,
#         local_batch_size=4,
#         num_workers=0,
#         is_infinite=False,
#         shuffle=False,
#         drop_last=False,
#         make_single_dataset=True,
#     )

#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)



# import json
# import math
# import os
# import random
# import re
# import time
# from itertools import islice

# import jsonlines
# import torch
# import torch.distributed as dist
# import torchvision.transforms as T
# from PIL import Image, UnidentifiedImageError
# from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

# from stamo.renderer.utils.device import get_accelerator_device
# from stamo.renderer.utils.overwatch import initialize_overwatch


# overwatch = initialize_overwatch(__name__)


# def complex_to_device(complex, device, non_blocking=False):
#     if complex is None:
#         return complex
#     if isinstance(complex, torch.Tensor):
#         return complex.to(device, non_blocking=non_blocking)
#     elif isinstance(complex, dict):
#         return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
#     elif isinstance(complex, list) or isinstance(complex, tuple):
#         return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
#     elif (
#         isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
#     ):
#         return complex
#     else:
#         raise ValueError("Unsupported complex", complex)


# def fp32_to_fp16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.half)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_fp16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_fp16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def fp32_to_bf16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.bfloat16)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_bf16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_bf16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def move_to_cuda(batch, device=None):
#     device = device or get_accelerator_device()
#     if isinstance(batch, torch.Tensor):
#         return batch.to(device=device, non_blocking=True)
#     elif isinstance(batch, list):
#         new_batch = [move_to_cuda(t, device=device) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def collate_fn(inputs):
#     images = torch.stack([input["image"] for input in inputs])
#     batch = {"images": images}
#     paths = [input.get("path") for input in inputs]
#     if all(path is not None for path in paths):
#         batch["paths"] = paths
#     return batch


# def get_loader_info(dataset, epochs, bsz, gradient_accumulate_steps=1):
#     dataset_len = len(dataset.dataset) if hasattr(dataset, "dataset") else len(dataset)
#     images_per_gpu = bsz
#     world_size = dist.get_world_size() if dist.is_initialized() else 1
#     images_per_batch = bsz * world_size * gradient_accumulate_steps
#     iter_per_ep = math.ceil(dataset_len / images_per_batch) if dataset_len else 0
#     num_iters = iter_per_ep * epochs
#     loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
#     return loader_info


# class ImageData(Dataset):
#     def __init__(
#         self,
#         metadata_path,
#         flip_p,
#         img_size: int = 224,
#         max_read_attempts: int = 8,
#         seed: int = 0,
#         read_trace_dir=None,
#         read_trace_samples: int = 0,
#     ):
#         self.flip_p = flip_p
#         self.max_read_attempts = int(max_read_attempts)
#         self.seed = int(seed)
#         self.read_trace_dir = (
#             os.path.abspath(os.path.expanduser(str(read_trace_dir)))
#             if read_trace_dir
#             else None
#         )
#         self.read_trace_samples = max(0, int(read_trace_samples))
#         self._read_trace_count = 0
#         self._read_trace_path = None
#         if self.max_read_attempts <= 0:
#             raise ValueError("max_read_attempts must be positive")

#         self.metadata_source = os.path.abspath(str(metadata_path))
#         self.metadata = []
#         self._append_metadata(metadata_path)

#         self.length = len(self.metadata)
#         if self.length == 0:
#             raise ValueError(f"No images were found in metadata file {metadata_path!r}.")

#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#         self.transforms = T.Compose(
#             [
#                 T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
#                 T.ToTensor(),
#             ]
#         )

#     def preprocess_train(self, image):
#         if torch.rand(1) < self.flip_p:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#         image = self.transforms(image)
#         return image

#     def _append_metadata(self, metadata_path):
#         metadata_path = os.path.abspath(str(metadata_path))
#         metadata_dir = os.path.dirname(metadata_path)
#         with open(metadata_path, "r", encoding="utf8") as f:
#             for line_number, item in enumerate(jsonlines.Reader(f), start=1):
#                 if not isinstance(item, dict):
#                     raise ValueError(
#                         f"{metadata_path}:{line_number} must contain a JSON object."
#                     )
#                 image_path = item.get("image")
#                 if not isinstance(image_path, str) or not image_path.strip():
#                     raise ValueError(
#                         f"{metadata_path}:{line_number} has no non-empty 'image' path."
#                     )
#                 image_path = os.path.expanduser(image_path.strip())
#                 if not os.path.isabs(image_path):
#                     image_path = os.path.abspath(os.path.join(metadata_dir, image_path))
#                 self.metadata.append(image_path)

#     def add(self, metadata_path):
#         self._append_metadata(metadata_path)
#         self.length = len(self.metadata)
#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#     @staticmethod
#     def _eval_task_from_image_path(image_path):
#         """Return the task in ``.../<split>/<task>/<episode>/<frame>``."""
#         normalized_path = str(image_path).replace("\\", "/").rstrip("/")
#         path_parts = [part for part in normalized_path.split("/") if part]
#         split_names = {"eval", "validation", "val", "test"}
#         split_indices = [
#             index
#             for index, part in enumerate(path_parts)
#             if part.lower() in split_names
#         ]
#         if not split_indices:
#             raise ValueError(
#                 "Cannot infer an evaluation task from image path "
#                 f"{image_path!r}; expected an eval/validation/test split "
#                 "component before <task>/<episode>/<frame>."
#             )
#         split_index = split_indices[-1]
#         if split_index + 3 >= len(path_parts):
#             raise ValueError(
#                 "Cannot infer an evaluation task from image path "
#                 f"{image_path!r}; expected "
#                 ".../<split>/<task>/<episode>/<frame>."
#             )
#         return path_parts[split_index + 1]

#     def select_eval_tasks(self, mode, fast_num_tasks=100, seed=0):
#         """Select one deterministic global task subset before rank sharding.

#         ``test_mode`` retains every task in the manifest. ``fast_mode`` uses
#         a rank-independent local RNG to choose tasks, then retains every image
#         belonging to those tasks. The existing DistributedSampler can therefore
#         shard exactly the same filtered dataset on every rank.
#         """
#         mode = str(mode).strip().lower()
#         if mode not in {"test_mode", "fast_mode"}:
#             raise ValueError(
#                 "data.eval_mode must be 'test_mode' or 'fast_mode', "
#                 f"got {mode!r}."
#             )

#         task_by_path = {
#             image_path: self._eval_task_from_image_path(image_path)
#             for image_path in self.metadata
#         }
#         available_tasks = sorted(set(task_by_path.values()))
#         if not available_tasks:
#             raise ValueError("The evaluation manifest contains no tasks.")

#         if mode == "fast_mode":
#             fast_num_tasks = int(fast_num_tasks)
#             if fast_num_tasks <= 0:
#                 raise ValueError("data.fast_eval_num_tasks must be positive.")
#             if fast_num_tasks > len(available_tasks):
#                 raise ValueError(
#                     "data.fast_eval_num_tasks exceeds the available task count: "
#                     f"requested={fast_num_tasks}, available={len(available_tasks)}."
#                 )
#             selected_tasks = set(
#                 random.Random(int(seed)).sample(
#                     available_tasks,
#                     fast_num_tasks,
#                 )
#             )
#             self.metadata = [
#                 image_path
#                 for image_path in self.metadata
#                 if task_by_path[image_path] in selected_tasks
#             ]
#         else:
#             selected_tasks = set(available_tasks)

#         self.length = len(self.metadata)
#         if self.length == 0:
#             raise ValueError(
#                 f"Evaluation mode {mode!r} selected no images."
#             )

#         self.eval_mode = mode
#         self.eval_available_task_count = len(available_tasks)
#         self.eval_selected_tasks = tuple(sorted(selected_tasks))
#         self.eval_task_seed = int(seed)

#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0:
#             overwatch.info(
#                 f"Evaluation {mode}: selected {len(selected_tasks)}/"
#                 f"{len(available_tasks)} tasks and {self.length} images "
#                 f"with seed={int(seed)}."
#             )

#     def __len__(self):
#         return self.length

#     def _retry_index(self, original_idx, attempt, attempted_indices):
#         """Choose a deterministic, previously untried fallback sample."""
#         candidate = (
#             int(original_idx)
#             + int(attempt) * 104729
#             + self.seed * 1009
#         ) % self.length
#         while candidate in attempted_indices and len(attempted_indices) < self.length:
#             candidate = (candidate + 1) % self.length
#         return candidate

#     def _trace_read(self, enabled, event, **details):
#         if not enabled or self.read_trace_dir is None:
#             return
#         if self._read_trace_path is None:
#             os.makedirs(self.read_trace_dir, exist_ok=True)
#             rank = dist.get_rank() if dist.is_initialized() else 0
#             metadata_name = re.sub(
#                 r"[^A-Za-z0-9_.-]+",
#                 "_",
#                 os.path.basename(getattr(self, "metadata_source", "images")),
#             )
#             self._read_trace_path = os.path.join(
#                 self.read_trace_dir,
#                 f"data_{metadata_name}_rank{rank:02d}_pid{os.getpid()}.jsonl",
#             )
#         record = {
#             "wall_time_ns": time.time_ns(),
#             "pid": os.getpid(),
#             "rank": dist.get_rank() if dist.is_initialized() else 0,
#             "event": str(event),
#         }
#         record.update(details)
#         # Only the first configured samples are traced, so opening per event is
#         # cheap and avoids leaking a file descriptor into DataLoader workers.
#         with open(self._read_trace_path, "a", encoding="utf-8") as trace_file:
#             trace_file.write(
#                 json.dumps(record, ensure_ascii=False, default=str) + "\n"
#             )
#             trace_file.flush()

#     def __getitem__(self, idx):
#         original_idx = int(idx)
#         trace_this_read = self._read_trace_count < self.read_trace_samples
#         self._read_trace_count += 1
#         candidate_idx = original_idx
#         attempted_indices = set()
#         failures = []
#         total_attempts = min(self.max_read_attempts, self.length)

#         for attempt in range(total_attempts):
#             attempted_indices.add(candidate_idx)
#             path = self.metadata[candidate_idx]
#             self._trace_read(
#                 trace_this_read,
#                 "READ_BEGIN",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             try:
#                 with Image.open(path) as opened_image:
#                     opened_image.load()
#                     image = opened_image.convert("RGB")
#             except (OSError, ValueError, UnidentifiedImageError) as exc:
#                 self._trace_read(
#                     trace_this_read,
#                     "READ_ERROR",
#                     requested_index=original_idx,
#                     candidate_index=candidate_idx,
#                     attempt=attempt + 1,
#                     path=path,
#                     error=repr(exc),
#                 )
#                 failures.append((path, exc))
#                 if attempt + 1 < total_attempts:
#                     candidate_idx = self._retry_index(
#                         original_idx,
#                         attempt + 1,
#                         attempted_indices,
#                     )
#                 continue

#             self._trace_read(
#                 trace_this_read,
#                 "OPEN_OK",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             # Keep transform errors visible. They are programming/configuration
#             # errors, not corrupt-image errors that should silently change data.
#             image = self.preprocess_train(image)
#             self._trace_read(
#                 trace_this_read,
#                 "TRANSFORM_OK",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             return {"image": image, "path": path}

#         failure_summary = "; ".join(
#             f"{path!r}: {type(exc).__name__}: {exc}"
#             for path, exc in failures[-4:]
#         )
#         last_error = failures[-1][1] if failures else None
#         raise RuntimeError(
#             f"Failed to read an image after {total_attempts} bounded attempts "
#             f"(requested index={original_idx}, rank="
#             f"{dist.get_rank() if dist.is_initialized() else 0}). "
#             f"Last failures: {failure_summary}"
#         ) from last_error


# class InfiniteDistributedSampler(DistributedSampler):
#     def __init__(
#         self,
#         dataset,
#         num_replicas=None,
#         rank=None,
#         shuffle=True,
#         seed=0,
#     ):
#         """
#         无限循环分布式采样器。
#         :param dataset: 数据集
#         :param num_replicas: 总共的设备数量
#         :param rank: 当前设备的 rank
#         :param shuffle: 是否随机打乱
#         """
#         super().__init__(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#             seed=int(seed),
#         )
#         self._epoch = 0
#         self._offset = 0

#     def set_start_sample(self, consumed_local_samples: int) -> None:
#         consumed_local_samples = int(consumed_local_samples)
#         if consumed_local_samples < 0:
#             raise ValueError("consumed_local_samples must be non-negative")
#         self._epoch, self._offset = divmod(
#             consumed_local_samples,
#             self.num_samples,
#         )

#     def __iter__(self):
#         """
#         无限循环返回索引。
#         """
#         epoch = self._epoch
#         offset = self._offset
#         while True:
#             self.set_epoch(epoch)
#             indices = super().__iter__()
#             if offset:
#                 indices = islice(indices, offset, None)
#             yield from indices
#             epoch += 1
#             offset = 0
#             self._epoch = epoch
#             self._offset = 0

#     def __len__(self):
#         return self.num_samples


# class InfiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True, seed=0):
#         """
#         多任务批量采样器，支持 Lightning 的分布式模式。
#         :param datasets: 多个数据集的列表
#         :param batch_size: 每个 batch 的大小
#         :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
#         """
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.num_datasets = len(self.datasets)
#         self.samples_per_dataset = sample_per_dataset
#         # self.remaining_samples = batch_size % self.num_datasets
#         self.dataset_lengths = [len(dataset) for dataset in self.datasets]

#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.cur_idx = 0

#         # 为每个数据集创建无限采样器
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#         self.samplers = [
#             InfiniteDistributedSampler(
#                 dataset,
#                 num_replicas=self.num_replicas,
#                 rank=self.rank,
#                 shuffle=shuffle,
#                 seed=int(seed) + dataset_index,
#             )
#             for dataset_index, dataset in enumerate(datasets)
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         """
#         无限生成每个 batch 的样本索引。
#         """
#         while True:
#             batch = []
#             for i in range(len(self.iterators)):
#                 iterator = self.iterators[i]
#                 for _ in range(self.samples_per_dataset[i]):
#                     batch.append(next(iterator) + self.cumulative_sizes[i])
#             yield batch

#     def __len__(self):
#         return sum(self.dataset_lengths)


# class FiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(
#         self,
#         datasets,
#         batch_size,
#         sample_per_dataset,
#         drop_last=False,
#         shuffle=True,
#         seed=0,
#     ):
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.samples_per_dataset = sample_per_dataset
#         self.dataset_lengths = [len(dataset) for dataset in datasets]
#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.drop_last = drop_last
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

#         # 初始化标准分布式采样器
#         self.samplers = [
#             DistributedSampler(
#                 dataset,
#                 num_replicas=self.num_replicas,
#                 rank=self.rank,
#                 shuffle=shuffle,
#                 seed=int(seed) + dataset_index,
#             )
#             for dataset_index, dataset in enumerate(datasets)
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]
#         # 计算每个数据集还剩多少样本
#         self.remaining_samples = [len(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         iterators = [iter(sampler) for sampler in self.samplers]
#         remaining_samples = self.remaining_samples.copy()

#         while sum(remaining_samples) > 0:
#             batch = []
#             for i, iterator in enumerate(iterators):
#                 num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
#                 for _ in range(num_samples):
#                     try:
#                         idx = next(iterator)
#                         batch.append(idx + self.cumulative_sizes[i])
#                         remaining_samples[i] -= 1
#                     except StopIteration:
#                         remaining_samples[i] = 0
#                         break  # 当前dataset采样完毕

#             if len(batch) == 0:
#                 break

#             # 根据drop_last判断batch大小
#             if self.drop_last and len(batch) < self.batch_size:
#                 break

#             yield batch

#     def __len__(self):
#         # 总的batch数量（近似值）
#         total_samples = sum(self.dataset_lengths)
#         if self.drop_last:
#             return total_samples // self.batch_size
#         else:
#             return (total_samples + self.batch_size - 1) // self.batch_size


# class MultiDatasetWrapper(Dataset):
#     def __init__(self, datasets):
#         self.datasets = datasets
#         self.dataset_lengths = [len(ds) for ds in datasets]

#     def __len__(self):
#         return sum(self.dataset_lengths)

#     def __getitem__(self, index):
#         cumulative_sizes = 0
#         for dataset, length in zip(self.datasets, self.dataset_lengths):
#             if index < cumulative_sizes + length:
#                 return dataset[index - cumulative_sizes]
#             cumulative_sizes += length
#         raise IndexError("Index out of range")


# def _loader_worker_options(
#     num_workers,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     read_trace_dir=None,
#     read_trace_samples=0,
# ):
#     num_workers = int(num_workers)
#     loader_timeout_seconds = float(loader_timeout_seconds)
#     if num_workers < 0:
#         raise ValueError("num_workers must be non-negative")
#     if loader_timeout_seconds < 0:
#         raise ValueError("loader_timeout_seconds must be non-negative")

#     options = {
#         "num_workers": num_workers,
#         "timeout": loader_timeout_seconds if num_workers > 0 else 0,
#         "persistent_workers": bool(persistent_workers) if num_workers > 0 else False,
#     }
#     worker_start_method = str(worker_start_method or "").strip().lower()
#     if num_workers > 0 and worker_start_method:
#         if worker_start_method not in {"fork", "spawn", "forkserver"}:
#             raise ValueError(
#                 "worker_start_method must be fork, spawn, forkserver, or empty"
#             )
#         options["multiprocessing_context"] = worker_start_method
#     return options


# def load_unsampler_datasets_from_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     max_read_attempts=8,
#     seed=0,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     read_trace_dir=None,
#     read_trace_samples=0,
#     eval_mode=None,
#     fast_eval_num_tasks=100,
#     fast_eval_task_seed=0,
# ):
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     if not dataset_paths:
#         raise ValueError(f"Dataset config {json_path!r} contains no datasets.")
#     dataset_path = os.path.join(os.path.dirname(json_path), dataset_paths[0])
#     dataset = ImageData(
#         dataset_path,
#         flip_p=flip_p,
#         img_size=img_size,
#         max_read_attempts=max_read_attempts,
#         seed=seed,
#         read_trace_dir=read_trace_dir,
#         read_trace_samples=read_trace_samples,
#     )

#     for dataset_path in dataset_paths[1:]:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         dataset.add(dataset_path)

#     if eval_mode is not None:
#         dataset.select_eval_tasks(
#             mode=eval_mode,
#             fast_num_tasks=fast_eval_num_tasks,
#             seed=fast_eval_task_seed,
#         )

#     rank = dist.get_rank() if dist.is_initialized() else 0
#     num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#     loader_generator = torch.Generator()
#     loader_generator.manual_seed(int(seed) + rank)
#     worker_options = _loader_worker_options(
#         num_workers,
#         loader_timeout_seconds=loader_timeout_seconds,
#         persistent_workers=persistent_workers,
#         worker_start_method=worker_start_method,
#     )

#     if is_infinite:
#         sampler = InfiniteDistributedSampler(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#             seed=seed,
#         )
#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             drop_last=drop_last,
#             generator=loader_generator,
#             **worker_options,
#         )
#     else:
#         # Original non-distributed eval loader:
#         # dataloader = DataLoader(
#         #     dataset,
#         #     batch_size=local_batch_size,
#         #     num_workers=num_workers,
#         #     collate_fn=collate_fn,
#         #     shuffle=shuffle,
#         #     drop_last=drop_last,
#         # )

#         # Distributed eval loader:
#         # each rank receives a different shard of the finite eval dataset.
#         sampler = None
#         if num_replicas > 1:
#             sampler = DistributedSampler(
#                 dataset,
#                 num_replicas=num_replicas,
#                 rank=rank,
#                 shuffle=shuffle,
#                 drop_last=drop_last,
#                 seed=int(seed),
#             )

#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             shuffle=False if sampler is not None else shuffle,
#             drop_last=drop_last,
#             generator=loader_generator,
#             **worker_options,
#         )
#     return dataloader


# def load_multi_datasets_form_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     make_single_dataset=False,
#     max_read_attempts=8,
#     seed=0,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     read_trace_dir=None,
#     read_trace_samples=0,
#     eval_mode=None,
#     fast_eval_num_tasks=100,
#     fast_eval_task_seed=0,
# ):
#     if make_single_dataset:
#         return load_unsampler_datasets_from_json(
#             json_path=json_path,
#             flip_p=flip_p,
#             img_size=img_size,
#             local_batch_size=local_batch_size,
#             num_workers=num_workers,
#             is_infinite=is_infinite,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             max_read_attempts=max_read_attempts,
#             seed=seed,
#             loader_timeout_seconds=loader_timeout_seconds,
#             persistent_workers=persistent_workers,
#             worker_start_method=worker_start_method,
#             read_trace_dir=read_trace_dir,
#             read_trace_samples=read_trace_samples,
#             eval_mode=eval_mode,
#             fast_eval_num_tasks=fast_eval_num_tasks,
#             fast_eval_task_seed=fast_eval_task_seed,
#         )
#     if eval_mode is not None:
#         raise ValueError(
#             "Task-based evaluation modes require make_single_dataset=True."
#         )
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     ratios = config["ratios"]
#     assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
#     assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

#     datasets = []

#     for dataset_path in dataset_paths:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         datasets.append(
#             ImageData(
#                 dataset_path,
#                 flip_p=flip_p,
#                 img_size=img_size,
#                 max_read_attempts=max_read_attempts,
#                 seed=int(seed) + len(datasets),
#                 read_trace_dir=read_trace_dir,
#                 read_trace_samples=read_trace_samples,
#             )
#         )

#     sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

#     total = sum(sample_per_dataset)
#     if total < local_batch_size:
#         sample_per_dataset[-1] += local_batch_size - total
#     elif total > local_batch_size:
#         sample_per_dataset[-1] -= total - local_batch_size

#     wrapped_dataset = MultiDatasetWrapper(datasets)

#     if is_infinite:
#         batch_sampler = InfiniteMultiTaskBatchSampler(
#             datasets,
#             local_batch_size,
#             sample_per_dataset=sample_per_dataset,
#             shuffle=shuffle,
#             seed=seed,
#         )
#     else:
#         batch_sampler = FiniteMultiTaskBatchSampler(
#             datasets,
#             local_batch_size,
#             sample_per_dataset=sample_per_dataset,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             seed=seed,
#         )

#     rank = dist.get_rank() if dist.is_initialized() else 0
#     loader_generator = torch.Generator()
#     loader_generator.manual_seed(int(seed) + rank)
#     dataloader = DataLoader(
#         wrapped_dataset,
#         batch_sampler=batch_sampler,
#         collate_fn=collate_fn,
#         generator=loader_generator,
#         **_loader_worker_options(
#             num_workers,
#             loader_timeout_seconds=loader_timeout_seconds,
#             persistent_workers=persistent_workers,
#             worker_start_method=worker_start_method,
#         ),
#     )

#     return dataloader


# if __name__ == "__main__":
#     import torchvision.transforms as T

#     dataset = ImageData(metadata_path="jsons/eval_VLA.jsonl", flip_p=0.5)

#     dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)

#     dataloader = load_multi_datasets_form_json(
#         json_path="jsons/eval_VLA.json",
#         flip_p=0,
#         img_size=224,
#         local_batch_size=4,
#         num_workers=0,
#         is_infinite=False,
#         shuffle=False,
#         drop_last=False,
#         make_single_dataset=True,
#     )

#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)



#v1
# import json
# import math
# import os
# import random
# import re
# import sys
# import time

# import numpy as np
# import torch
# import torch.distributed as dist
# import torchvision.transforms as T
# from PIL import Image, UnidentifiedImageError
# from torch.utils.data import (
#     BatchSampler,
#     DataLoader,
#     Dataset,
#     DistributedSampler,
#     get_worker_info,
# )

# from stamo.renderer.utils.overwatch import initialize_overwatch
# from stamo.renderer.utils.metadata_index import JsonlImagePathCollection


# overwatch = initialize_overwatch(__name__)


# def _data_process_rank():
#     if dist.is_initialized():
#         return int(dist.get_rank())
#     return int(os.environ.get("STAMO_DATALOADER_PARENT_RANK", 0))


# def _initialize_cpu_dataloader_worker(worker_id):
#     """Seed a loader worker and assert that it never became a MUSA rank."""
#     torch.set_num_threads(1)
#     worker_seed = 33 + int(worker_id)
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

#     if os.environ.get("STAMO_DATALOADER_SPAWN_CHILD") != "1":
#         return
#     if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
#         raise RuntimeError(
#             "Spawned DataLoader worker did not disable accelerator autoload "
#             "before importing torch."
#         )
#     if "torch_musa" in sys.modules:
#         raise RuntimeError(
#             "Spawned DataLoader worker imported torch_musa; CPU workers must "
#             "not own MUSA contexts."
#         )
#     storage_type = getattr(torch, "UntypedStorage", None)
#     if storage_type is not None and not hasattr(storage_type, "is_musa"):
#         raise RuntimeError(
#             "Spawned DataLoader worker is missing the CPU is_musa storage "
#             "predicate required by the musified multiprocessing reducer."
#         )
#     unexpected_training_modules = {
#         module_name
#         for module_name in ("accelerate", "deepspeed")
#         if module_name in sys.modules
#     }
#     if unexpected_training_modules:
#         raise RuntimeError(
#             "Spawned DataLoader worker imported the training stack: "
#             f"{sorted(unexpected_training_modules)}"
#         )
#     if dist.is_initialized():
#         raise RuntimeError(
#             "Spawned DataLoader worker initialized a distributed process group."
#         )
#     worker_info = get_worker_info()
#     if worker_info is None or int(worker_info.id) != int(worker_id):
#         raise RuntimeError("DataLoader worker identity is unavailable or inconsistent.")
#     # Emit exactly one admission marker per worker.  The formal verifier uses
#     # the complete (parent_rank, worker_id, pid) set to prove that all 8 x 10
#     # production workers initialized, not merely worker 0 from each rank.
#     print(
#         "STAMO_CPU_DATALOADER_WORKER_READY "
#         f"parent_rank={os.environ.get('STAMO_DATALOADER_PARENT_RANK', '?')} "
#         f"worker_id={int(worker_id)} num_workers={int(worker_info.num_workers)} "
#         f"pid={os.getpid()} autoload=0 musa_imported=0 "
#         "training_stack_imported=0 distributed=0",
#         flush=True,
#     )


# def complex_to_device(complex, device, non_blocking=False):
#     if complex is None:
#         return complex
#     if isinstance(complex, torch.Tensor):
#         return complex.to(device, non_blocking=non_blocking)
#     elif isinstance(complex, dict):
#         return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
#     elif isinstance(complex, list) or isinstance(complex, tuple):
#         return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
#     elif (
#         isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
#     ):
#         return complex
#     else:
#         raise ValueError("Unsupported complex", complex)


# def fp32_to_fp16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.half)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_fp16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_fp16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def fp32_to_bf16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.bfloat16)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_bf16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_bf16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def move_to_cuda(batch, device=None):
#     if device is None:
#         # Keep accelerator discovery out of DataLoader worker module imports.
#         # Training ranks call this path with their MUSA runtime already bound.
#         from stamo.renderer.utils.device import get_accelerator_device

#         device = get_accelerator_device()
#     if isinstance(batch, torch.Tensor):
#         return batch.to(device=device, non_blocking=True)
#     elif isinstance(batch, list):
#         new_batch = [move_to_cuda(t, device=device) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def collate_fn(inputs):
#     images = torch.stack([input["image"] for input in inputs])
#     batch = {"images": images}
#     paths = [input.get("path") for input in inputs]
#     if all(path is not None for path in paths):
#         batch["paths"] = paths
#     return batch


# def get_loader_info(dataset, epochs, bsz, gradient_accumulate_steps=1):
#     dataset_len = len(dataset.dataset) if hasattr(dataset, "dataset") else len(dataset)
#     images_per_gpu = bsz
#     world_size = dist.get_world_size() if dist.is_initialized() else 1
#     images_per_batch = bsz * world_size * gradient_accumulate_steps
#     iter_per_ep = math.ceil(dataset_len / images_per_batch) if dataset_len else 0
#     num_iters = iter_per_ep * epochs
#     loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
#     return loader_info


# class ImageData(Dataset):
#     def __init__(
#         self,
#         metadata_path,
#         flip_p,
#         img_size: int = 224,
#         max_read_attempts: int = 8,
#         seed: int = 0,
#         read_trace_dir=None,
#         read_trace_samples: int = 0,
#     ):
#         self.flip_p = flip_p
#         self.max_read_attempts = int(max_read_attempts)
#         self.seed = int(seed)
#         self.read_trace_dir = (
#             os.path.abspath(os.path.expanduser(str(read_trace_dir)))
#             if read_trace_dir
#             else None
#         )
#         self.read_trace_samples = max(0, int(read_trace_samples))
#         self._read_trace_count = 0
#         self._read_trace_path = None
#         if self.max_read_attempts <= 0:
#             raise ValueError("max_read_attempts must be positive")

#         self.metadata_source = os.path.abspath(str(metadata_path))
#         self.metadata = JsonlImagePathCollection()
#         self._append_metadata(metadata_path)

#         self.length = len(self.metadata)
#         if self.length == 0:
#             raise ValueError(f"No images were found in metadata file {metadata_path!r}.")

#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#         self.transforms = T.Compose(
#             [
#                 T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
#                 T.ToTensor(),
#             ]
#         )

#     def preprocess_train(self, image):
#         if torch.rand(1) < self.flip_p:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#         image = self.transforms(image)
#         return image

#     def _append_metadata(self, metadata_path):
#         if not isinstance(self.metadata, JsonlImagePathCollection):
#             raise RuntimeError(
#                 "Cannot append a manifest after selecting an evaluation subset."
#             )
#         self.metadata.add(metadata_path)

#     def add(self, metadata_path):
#         self._append_metadata(metadata_path)
#         self.length = len(self.metadata)
#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#     @staticmethod
#     def _eval_task_from_image_path(image_path):
#         """Return the task in ``.../<split>/<task>/<episode>/<frame>``."""
#         normalized_path = str(image_path).replace("\\", "/").rstrip("/")
#         path_parts = [part for part in normalized_path.split("/") if part]
#         split_names = {"eval", "validation", "val", "test"}
#         split_indices = [
#             index
#             for index, part in enumerate(path_parts)
#             if part.lower() in split_names
#         ]
#         if not split_indices:
#             raise ValueError(
#                 "Cannot infer an evaluation task from image path "
#                 f"{image_path!r}; expected an eval/validation/test split "
#                 "component before <task>/<episode>/<frame>."
#             )
#         split_index = split_indices[-1]
#         if split_index + 3 >= len(path_parts):
#             raise ValueError(
#                 "Cannot infer an evaluation task from image path "
#                 f"{image_path!r}; expected "
#                 ".../<split>/<task>/<episode>/<frame>."
#             )
#         return path_parts[split_index + 1]

#     def select_eval_tasks(self, mode, fast_num_tasks=100, seed=0):
#         """Select one deterministic global task subset before rank sharding.

#         ``test_mode`` retains every task in the manifest. ``fast_mode`` uses
#         a rank-independent local RNG to choose tasks, then retains every image
#         belonging to those tasks. The existing DistributedSampler can therefore
#         shard exactly the same filtered dataset on every rank.
#         """
#         mode = str(mode).strip().lower()
#         if mode not in {"test_mode", "fast_mode"}:
#             raise ValueError(
#                 "data.eval_mode must be 'test_mode' or 'fast_mode', "
#                 f"got {mode!r}."
#             )

#         task_by_path = {
#             image_path: self._eval_task_from_image_path(image_path)
#             for image_path in self.metadata
#         }
#         available_tasks = sorted(set(task_by_path.values()))
#         if not available_tasks:
#             raise ValueError("The evaluation manifest contains no tasks.")

#         if mode == "fast_mode":
#             fast_num_tasks = int(fast_num_tasks)
#             if fast_num_tasks <= 0:
#                 raise ValueError("data.fast_eval_num_tasks must be positive.")
#             if fast_num_tasks > len(available_tasks):
#                 raise ValueError(
#                     "data.fast_eval_num_tasks exceeds the available task count: "
#                     f"requested={fast_num_tasks}, available={len(available_tasks)}."
#                 )
#             selected_tasks = set(
#                 random.Random(int(seed)).sample(
#                     available_tasks,
#                     fast_num_tasks,
#                 )
#             )
#             self.metadata = [
#                 image_path
#                 for image_path in self.metadata
#                 if task_by_path[image_path] in selected_tasks
#             ]
#         else:
#             selected_tasks = set(available_tasks)

#         self.length = len(self.metadata)
#         if self.length == 0:
#             raise ValueError(
#                 f"Evaluation mode {mode!r} selected no images."
#             )

#         self.eval_mode = mode
#         self.eval_available_task_count = len(available_tasks)
#         self.eval_selected_tasks = tuple(sorted(selected_tasks))
#         self.eval_task_seed = int(seed)

#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0:
#             overwatch.info(
#                 f"Evaluation {mode}: selected {len(selected_tasks)}/"
#                 f"{len(available_tasks)} tasks and {self.length} images "
#                 f"with seed={int(seed)}."
#             )

#     def __len__(self):
#         return self.length

#     def _retry_index(self, original_idx, attempt, attempted_indices):
#         """Choose a deterministic, previously untried fallback sample."""
#         candidate = (
#             int(original_idx)
#             + int(attempt) * 104729
#             + self.seed * 1009
#         ) % self.length
#         while candidate in attempted_indices and len(attempted_indices) < self.length:
#             candidate = (candidate + 1) % self.length
#         return candidate

#     def _trace_read(self, enabled, event, **details):
#         if not enabled or self.read_trace_dir is None:
#             return
#         if self._read_trace_path is None:
#             os.makedirs(self.read_trace_dir, exist_ok=True)
#             rank = _data_process_rank()
#             metadata_name = re.sub(
#                 r"[^A-Za-z0-9_.-]+",
#                 "_",
#                 os.path.basename(getattr(self, "metadata_source", "images")),
#             )
#             self._read_trace_path = os.path.join(
#                 self.read_trace_dir,
#                 f"data_{metadata_name}_rank{rank:02d}_pid{os.getpid()}.jsonl",
#             )
#         record = {
#             "wall_time_ns": time.time_ns(),
#             "pid": os.getpid(),
#             "rank": _data_process_rank(),
#             "event": str(event),
#         }
#         record.update(details)
#         # Only the first configured samples are traced, so opening per event is
#         # cheap and avoids leaking a file descriptor into DataLoader workers.
#         with open(self._read_trace_path, "a", encoding="utf-8") as trace_file:
#             trace_file.write(
#                 json.dumps(record, ensure_ascii=False, default=str) + "\n"
#             )
#             trace_file.flush()

#     def __getitem__(self, idx):
#         original_idx = int(idx)
#         trace_this_read = self._read_trace_count < self.read_trace_samples
#         self._read_trace_count += 1
#         candidate_idx = original_idx
#         attempted_indices = set()
#         failures = []
#         total_attempts = min(self.max_read_attempts, self.length)

#         for attempt in range(total_attempts):
#             attempted_indices.add(candidate_idx)
#             path = self.metadata[candidate_idx]
#             self._trace_read(
#                 trace_this_read,
#                 "READ_BEGIN",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             try:
#                 with Image.open(path) as opened_image:
#                     opened_image.load()
#                     image = opened_image.convert("RGB")
#             except (OSError, ValueError, UnidentifiedImageError) as exc:
#                 self._trace_read(
#                     trace_this_read,
#                     "READ_ERROR",
#                     requested_index=original_idx,
#                     candidate_index=candidate_idx,
#                     attempt=attempt + 1,
#                     path=path,
#                     error=repr(exc),
#                 )
#                 failures.append((path, exc))
#                 if attempt + 1 < total_attempts:
#                     candidate_idx = self._retry_index(
#                         original_idx,
#                         attempt + 1,
#                         attempted_indices,
#                     )
#                 continue

#             self._trace_read(
#                 trace_this_read,
#                 "OPEN_OK",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             # Keep transform errors visible. They are programming/configuration
#             # errors, not corrupt-image errors that should silently change data.
#             image = self.preprocess_train(image)
#             self._trace_read(
#                 trace_this_read,
#                 "TRANSFORM_OK",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             return {"image": image, "path": path}

#         failure_summary = "; ".join(
#             f"{path!r}: {type(exc).__name__}: {exc}"
#             for path, exc in failures[-4:]
#         )
#         last_error = failures[-1][1] if failures else None
#         raise RuntimeError(
#             f"Failed to read an image after {total_attempts} bounded attempts "
#             f"(requested index={original_idx}, rank={_data_process_rank()}). "
#             f"Last failures: {failure_summary}"
#         ) from last_error


# class InfiniteDistributedSampler(DistributedSampler):
#     def __init__(
#         self,
#         dataset,
#         num_replicas=None,
#         rank=None,
#         shuffle=True,
#     ):
#         """
#         无限循环分布式采样器。
#         :param dataset: 数据集
#         :param num_replicas: 总共的设备数量
#         :param rank: 当前设备的 rank
#         :param shuffle: 是否随机打乱
#         """
#         super().__init__(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#         )
#         self._epoch = 0

#     def __iter__(self):
#         """
#         无限循环返回索引。
#         """
#         while True:
#             self.set_epoch(self._epoch)
#             self._epoch += 1
#             indices = super().__iter__()
#             yield from indices

#     def __len__(self):
#         return len(self.dataset)


# class InfiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True):
#         """
#         多任务批量采样器，支持 Lightning 的分布式模式。
#         :param datasets: 多个数据集的列表
#         :param batch_size: 每个 batch 的大小
#         :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
#         """
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.num_datasets = len(self.datasets)
#         self.samples_per_dataset = sample_per_dataset
#         # self.remaining_samples = batch_size % self.num_datasets
#         self.dataset_lengths = [len(dataset) for dataset in self.datasets]

#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.cur_idx = 0

#         # 为每个数据集创建无限采样器
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#         self.samplers = [
#             InfiniteDistributedSampler(
#                 dataset,
#                 num_replicas=self.num_replicas,
#                 rank=self.rank,
#                 shuffle=shuffle,
#             )
#             for dataset in datasets
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         """
#         无限生成每个 batch 的样本索引。
#         """
#         while True:
#             batch = []
#             for i in range(len(self.iterators)):
#                 iterator = self.iterators[i]
#                 for _ in range(self.samples_per_dataset[i]):
#                     batch.append(next(iterator) + self.cumulative_sizes[i])
#             yield batch

#     def __len__(self):
#         return sum(self.dataset_lengths)


# class FiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(
#         self,
#         datasets,
#         batch_size,
#         sample_per_dataset,
#         drop_last=False,
#         shuffle=True,
#     ):
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.samples_per_dataset = sample_per_dataset
#         self.dataset_lengths = [len(dataset) for dataset in datasets]
#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.drop_last = drop_last
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

#         # 初始化标准分布式采样器
#         self.samplers = [
#             DistributedSampler(
#                 dataset,
#                 num_replicas=self.num_replicas,
#                 rank=self.rank,
#                 shuffle=shuffle,
#             )
#             for dataset in datasets
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]
#         # 计算每个数据集还剩多少样本
#         self.remaining_samples = [len(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         iterators = [iter(sampler) for sampler in self.samplers]
#         remaining_samples = self.remaining_samples.copy()

#         while sum(remaining_samples) > 0:
#             batch = []
#             for i, iterator in enumerate(iterators):
#                 num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
#                 for _ in range(num_samples):
#                     try:
#                         idx = next(iterator)
#                         batch.append(idx + self.cumulative_sizes[i])
#                         remaining_samples[i] -= 1
#                     except StopIteration:
#                         remaining_samples[i] = 0
#                         break  # 当前dataset采样完毕

#             if len(batch) == 0:
#                 break

#             # 根据drop_last判断batch大小
#             if self.drop_last and len(batch) < self.batch_size:
#                 break

#             yield batch

#     def __len__(self):
#         # 总的batch数量（近似值）
#         total_samples = sum(self.dataset_lengths)
#         if self.drop_last:
#             return total_samples // self.batch_size
#         else:
#             return (total_samples + self.batch_size - 1) // self.batch_size


# class MultiDatasetWrapper(Dataset):
#     def __init__(self, datasets):
#         self.datasets = datasets
#         self.dataset_lengths = [len(ds) for ds in datasets]

#     def __len__(self):
#         return sum(self.dataset_lengths)

#     def __getitem__(self, index):
#         cumulative_sizes = 0
#         for dataset, length in zip(self.datasets, self.dataset_lengths):
#             if index < cumulative_sizes + length:
#                 return dataset[index - cumulative_sizes]
#             cumulative_sizes += length
#         raise IndexError("Index out of range")


# def _loader_worker_options(
#     num_workers,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     prefetch_factor=2,
#     read_trace_dir=None,
#     read_trace_samples=0,
# ):
#     num_workers = int(num_workers)
#     loader_timeout_seconds = float(loader_timeout_seconds)
#     if num_workers < 0:
#         raise ValueError("num_workers must be non-negative")
#     if loader_timeout_seconds < 0:
#         raise ValueError("loader_timeout_seconds must be non-negative")
#     prefetch_factor = int(prefetch_factor)
#     if num_workers > 0 and prefetch_factor <= 0:
#         raise ValueError("prefetch_factor must be positive when num_workers > 0")

#     options = {
#         "num_workers": num_workers,
#         "timeout": loader_timeout_seconds if num_workers > 0 else 0,
#         "persistent_workers": bool(persistent_workers) if num_workers > 0 else False,
#     }
#     if num_workers > 0:
#         options["worker_init_fn"] = _initialize_cpu_dataloader_worker
#         options["prefetch_factor"] = prefetch_factor
#     worker_start_method = str(worker_start_method or "").strip().lower()
#     if (
#         num_workers > 0
#         and os.environ.get("DS_ACCELERATOR", "").strip().lower() == "musa"
#         and worker_start_method != "spawn"
#     ):
#         raise ValueError(
#             "MUSA DataLoader workers require worker_start_method='spawn'; "
#             "fork/forkserver can inherit an initialized MUSA/MCCL runtime."
#         )
#     if num_workers > 0 and worker_start_method:
#         if worker_start_method not in {"fork", "spawn", "forkserver"}:
#             raise ValueError(
#                 "worker_start_method must be fork, spawn, forkserver, or empty"
#             )
#         options["multiprocessing_context"] = worker_start_method
#     return options


# def load_unsampler_datasets_from_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     max_read_attempts=8,
#     seed=0,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     prefetch_factor=2,
#     read_trace_dir=None,
#     read_trace_samples=0,
#     eval_mode=None,
#     fast_eval_num_tasks=100,
#     fast_eval_task_seed=0,
# ):
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     if not dataset_paths:
#         raise ValueError(f"Dataset config {json_path!r} contains no datasets.")
#     dataset_path = os.path.join(os.path.dirname(json_path), dataset_paths[0])
#     dataset = ImageData(
#         dataset_path,
#         flip_p=flip_p,
#         img_size=img_size,
#         max_read_attempts=max_read_attempts,
#         seed=seed,
#         read_trace_dir=read_trace_dir,
#         read_trace_samples=read_trace_samples,
#     )

#     for dataset_path in dataset_paths[1:]:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         dataset.add(dataset_path)

#     if eval_mode is not None:
#         dataset.select_eval_tasks(
#             mode=eval_mode,
#             fast_num_tasks=fast_eval_num_tasks,
#             seed=fast_eval_task_seed,
#         )

#     rank = dist.get_rank() if dist.is_initialized() else 0
#     num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#     worker_options = _loader_worker_options(
#         num_workers,
#         loader_timeout_seconds=loader_timeout_seconds,
#         persistent_workers=persistent_workers,
#         worker_start_method=worker_start_method,
#         prefetch_factor=prefetch_factor,
#     )

#     if is_infinite:
#         sampler = InfiniteDistributedSampler(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#         )
#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             drop_last=drop_last,
#             **worker_options,
#         )
#     else:
#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             collate_fn=collate_fn,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             **worker_options,
#         )
#     return dataloader


# def load_multi_datasets_form_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     make_single_dataset=False,
#     max_read_attempts=8,
#     seed=0,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     prefetch_factor=2,
#     read_trace_dir=None,
#     read_trace_samples=0,
#     eval_mode=None,
#     fast_eval_num_tasks=100,
#     fast_eval_task_seed=0,
# ):
#     if make_single_dataset:
#         return load_unsampler_datasets_from_json(
#             json_path=json_path,
#             flip_p=flip_p,
#             img_size=img_size,
#             local_batch_size=local_batch_size,
#             num_workers=num_workers,
#             is_infinite=is_infinite,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             max_read_attempts=max_read_attempts,
#             seed=seed,
#             loader_timeout_seconds=loader_timeout_seconds,
#             persistent_workers=persistent_workers,
#             worker_start_method=worker_start_method,
#             prefetch_factor=prefetch_factor,
#             read_trace_dir=read_trace_dir,
#             read_trace_samples=read_trace_samples,
#             eval_mode=eval_mode,
#             fast_eval_num_tasks=fast_eval_num_tasks,
#             fast_eval_task_seed=fast_eval_task_seed,
#         )
#     if eval_mode is not None:
#         raise ValueError(
#             "Task-based evaluation modes require make_single_dataset=True."
#         )
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     ratios = config["ratios"]
#     assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
#     assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

#     datasets = []

#     for dataset_path in dataset_paths:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         datasets.append(
#             ImageData(
#                 dataset_path,
#                 flip_p=flip_p,
#                 img_size=img_size,
#                 max_read_attempts=max_read_attempts,
#                 seed=int(seed) + len(datasets),
#                 read_trace_dir=read_trace_dir,
#                 read_trace_samples=read_trace_samples,
#             )
#         )

#     sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

#     total = sum(sample_per_dataset)
#     if total < local_batch_size:
#         sample_per_dataset[-1] += local_batch_size - total
#     elif total > local_batch_size:
#         sample_per_dataset[-1] -= total - local_batch_size

#     wrapped_dataset = MultiDatasetWrapper(datasets)

#     if is_infinite:
#         batch_sampler = InfiniteMultiTaskBatchSampler(
#             datasets,
#             local_batch_size,
#             sample_per_dataset=sample_per_dataset,
#             shuffle=shuffle,
#         )
#     else:
#         batch_sampler = FiniteMultiTaskBatchSampler(
#             datasets,
#             local_batch_size,
#             sample_per_dataset=sample_per_dataset,
#             shuffle=shuffle,
#             drop_last=drop_last,
#         )

#     dataloader = DataLoader(
#         wrapped_dataset,
#         batch_sampler=batch_sampler,
#         collate_fn=collate_fn,
#         **_loader_worker_options(
#             num_workers,
#             loader_timeout_seconds=loader_timeout_seconds,
#             persistent_workers=persistent_workers,
#             worker_start_method=worker_start_method,
#             prefetch_factor=prefetch_factor,
#         ),
#     )

#     return dataloader


# def check_tensor(obj, name, check_bound=1e4, check_std=1e3, _visited=None, force_output=False):
#     if not int(os.environ.get("CHECK_TENSOR", 0)):
#         return
#     if _visited is None:
#         _visited = set()
#     if id(obj) in _visited:
#         return False
#     _visited.add(id(obj))

#     # list / tuple
#     if isinstance(obj, (list, tuple)):
#         problem = False
#         for i, v in enumerate(obj):
#             if check_tensor(v, f"{name}[{i}]", _visited=_visited):
#                 problem = True
#         return problem

#     # dict
#     if isinstance(obj, dict):
#         problem = False
#         for k, v in obj.items():
#             if check_tensor(v, f"{name}['{k}']", _visited=_visited):
#                 problem = True
#         return problem

#     # not tensor
#     if not isinstance(obj, torch.Tensor):
#         return False

#     t = obj
#     problem_found = False

#     has_nan = torch.isnan(t).any().item()
#     has_inf = torch.isinf(t).any().item()

#     if has_nan:
#         overwatch.error(f"<{name}> 检测到 NaN")
#         problem_found = True
#     if has_inf:
#         overwatch.error(f"<{name}> 检测到 Inf")
#         problem_found = True

#     try:
#         minv = t.min().item()
#         maxv = t.max().item()
#         meanv = t.mean().item()
#         stdv = t.std().item() if t.numel() > 1 else 0.0
#     except Exception as e:
#         overwatch.error(f"<{name}> 无法计算统计值: {e}")
#         return True

#     if abs(maxv) > float(check_bound) or abs(minv) > float(check_bound):
#         overwatch.error(f"<{name}> 数值过大 (|value| > {check_bound}), 可能导致梯度爆炸")
#         problem_found = True

#     if stdv > float(check_std):
#         overwatch.error(f"<{name}> 标准差过大 (std={stdv}), 数值不稳定")
#         problem_found = True

#     if t.numel() > 1 and stdv < 1e-12:
#         overwatch.error(f"<{name}> 标准差过小 (=0), 张量可能塌陷/全常数")
#         problem_found = True

#     if torch.all(t == 0):
#         overwatch.error(f"<{name}> 张量全为 0")
#         problem_found = True

#     if t.numel() > 1 and torch.all(t == t.flatten()[0]):
#         overwatch.error(f"<{name}> 张量全为常数")
#         problem_found = True

#     if t.requires_grad and t.is_leaf and t.grad is not None:
#         g = t.grad
#         if torch.isnan(g).any():
#             overwatch.error(f"<{name}> 梯度存在 NaN")
#             problem_found = True
#         if torch.isinf(g).any():
#             overwatch.error(f"<{name}> 梯度存在 Inf")
#             problem_found = True
#         if g.abs().max().item() > 1e6:
#             overwatch.error(f"<{name}> 梯度爆炸 (grad > 1e6)")
#             problem_found = True

#     # if problem_found:
#     if force_output or problem_found:
#         overwatch.error(f"<{name}> 基本信息: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
#         overwatch.error(f"<{name}> 数值统计: min={minv}, max={maxv}, mean={meanv}, std={stdv}")

#     return problem_found


# if __name__ == "__main__":
#     import torchvision.transforms as T

#     dataset = ImageData(metadata_path="jsons/eval_VLA.jsonl", flip_p=0.5)

#     dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)

#     dataloader = load_multi_datasets_form_json(
#         json_path="jsons/eval_VLA.json",
#         flip_p=0,
#         img_size=224,
#         local_batch_size=4,
#         num_workers=0,
#         is_infinite=False,
#         shuffle=False,
#         drop_last=False,
#         make_single_dataset=True,
#     )

#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)


# import json
# import math
# import os
# import random
# import re
# import sys
# import time

# import numpy as np
# import torch
# import torch.distributed as dist
# import torchvision.transforms as T
# from PIL import Image, UnidentifiedImageError
# from torch.utils.data import (
#     BatchSampler,
#     DataLoader,
#     Dataset,
#     DistributedSampler,
#     get_worker_info,
# )

# from stamo.renderer.utils.overwatch import initialize_overwatch
# from stamo.renderer.utils.metadata_index import JsonlImagePathCollection


# overwatch = initialize_overwatch(__name__)


# def _data_process_rank():
#     if dist.is_initialized():
#         return int(dist.get_rank())
#     return int(os.environ.get("STAMO_DATALOADER_PARENT_RANK", 0))


# def _initialize_cpu_dataloader_worker(worker_id):
#     """Seed a loader worker and assert that it never became a MUSA rank."""
#     torch.set_num_threads(1)
#     worker_seed = 33 + int(worker_id)
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

#     if os.environ.get("STAMO_DATALOADER_SPAWN_CHILD") != "1":
#         return
#     if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
#         raise RuntimeError(
#             "Spawned DataLoader worker did not disable accelerator autoload "
#             "before importing torch."
#         )
#     if "torch_musa" in sys.modules:
#         raise RuntimeError(
#             "Spawned DataLoader worker imported torch_musa; CPU workers must "
#             "not own MUSA contexts."
#         )
#     storage_type = getattr(torch, "UntypedStorage", None)
#     if storage_type is not None and not hasattr(storage_type, "is_musa"):
#         raise RuntimeError(
#             "Spawned DataLoader worker is missing the CPU is_musa storage "
#             "predicate required by the musified multiprocessing reducer."
#         )
#     unexpected_training_modules = {
#         module_name
#         for module_name in ("accelerate", "deepspeed")
#         if module_name in sys.modules
#     }
#     if unexpected_training_modules:
#         raise RuntimeError(
#             "Spawned DataLoader worker imported the training stack: "
#             f"{sorted(unexpected_training_modules)}"
#         )
#     if dist.is_initialized():
#         raise RuntimeError(
#             "Spawned DataLoader worker initialized a distributed process group."
#         )
#     worker_info = get_worker_info()
#     if worker_info is None or int(worker_info.id) != int(worker_id):
#         raise RuntimeError("DataLoader worker identity is unavailable or inconsistent.")
#     # Emit exactly one admission marker per worker.  The formal verifier uses
#     # the complete (parent_rank, worker_id, pid) set to prove that all 8 x 10
#     # production workers initialized, not merely worker 0 from each rank.
#     print(
#         "STAMO_CPU_DATALOADER_WORKER_READY "
#         f"parent_rank={os.environ.get('STAMO_DATALOADER_PARENT_RANK', '?')} "
#         f"worker_id={int(worker_id)} num_workers={int(worker_info.num_workers)} "
#         f"pid={os.getpid()} autoload=0 musa_imported=0 "
#         "training_stack_imported=0 distributed=0",
#         flush=True,
#     )


# def complex_to_device(complex, device, non_blocking=False):
#     if complex is None:
#         return complex
#     if isinstance(complex, torch.Tensor):
#         return complex.to(device, non_blocking=non_blocking)
#     elif isinstance(complex, dict):
#         return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
#     elif isinstance(complex, list) or isinstance(complex, tuple):
#         return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
#     elif (
#         isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
#     ):
#         return complex
#     else:
#         raise ValueError("Unsupported complex", complex)


# def fp32_to_fp16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.half)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_fp16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_fp16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def fp32_to_bf16(batch):
#     # deepspeed does not auto cast inputs.
#     if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
#         return batch.to(dtype=torch.bfloat16)
#     elif isinstance(batch, list):
#         new_batch = [fp32_to_bf16(t) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(fp32_to_bf16(t) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def move_to_cuda(batch, device=None):
#     if device is None:
#         # Keep accelerator discovery out of DataLoader worker module imports.
#         # Training ranks call this path with their MUSA runtime already bound.
#         from stamo.renderer.utils.device import get_accelerator_device

#         device = get_accelerator_device()
#     if isinstance(batch, torch.Tensor):
#         return batch.to(device=device, non_blocking=True)
#     elif isinstance(batch, list):
#         new_batch = [move_to_cuda(t, device=device) for t in batch]
#     elif isinstance(batch, tuple):
#         new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
#     elif isinstance(batch, dict):
#         new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
#     else:
#         return batch
#     return new_batch


# def collate_fn(inputs):
#     images = torch.stack([input["image"] for input in inputs])
#     batch = {"images": images}
#     paths = [input.get("path") for input in inputs]
#     if all(path is not None for path in paths):
#         batch["paths"] = paths
#     return batch


# def get_loader_info(dataset, epochs, bsz, gradient_accumulate_steps=1):
#     dataset_len = len(dataset.dataset) if hasattr(dataset, "dataset") else len(dataset)
#     images_per_gpu = bsz
#     world_size = dist.get_world_size() if dist.is_initialized() else 1
#     images_per_batch = bsz * world_size * gradient_accumulate_steps
#     iter_per_ep = math.ceil(dataset_len / images_per_batch) if dataset_len else 0
#     num_iters = iter_per_ep * epochs
#     loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
#     return loader_info


# class ImageData(Dataset):
#     def __init__(
#         self,
#         metadata_path,
#         flip_p,
#         img_size: int = 224,
#         max_read_attempts: int = 8,
#         seed: int = 0,
#         read_trace_dir=None,
#         read_trace_samples: int = 0,
#     ):
#         self.flip_p = flip_p
#         self.max_read_attempts = int(max_read_attempts)
#         self.seed = int(seed)
#         self.read_trace_dir = (
#             os.path.abspath(os.path.expanduser(str(read_trace_dir)))
#             if read_trace_dir
#             else None
#         )
#         self.read_trace_samples = max(0, int(read_trace_samples))
#         self._read_trace_count = 0
#         self._read_trace_path = None
#         if self.max_read_attempts <= 0:
#             raise ValueError("max_read_attempts must be positive")

#         self.metadata_source = os.path.abspath(str(metadata_path))
#         self.metadata = JsonlImagePathCollection()
#         self._append_metadata(metadata_path)

#         self.length = len(self.metadata)
#         if self.length == 0:
#             raise ValueError(f"No images were found in metadata file {metadata_path!r}.")

#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#         self.transforms = T.Compose(
#             [
#                 T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
#                 T.ToTensor(),
#             ]
#         )

#     def preprocess_train(self, image):
#         if torch.rand(1) < self.flip_p:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#         image = self.transforms(image)
#         return image

#     def _append_metadata(self, metadata_path):
#         if not isinstance(self.metadata, JsonlImagePathCollection):
#             raise RuntimeError(
#                 "Cannot append a manifest after selecting an evaluation subset."
#             )
#         self.metadata.add(metadata_path)

#     def add(self, metadata_path):
#         self._append_metadata(metadata_path)
#         self.length = len(self.metadata)
#         overwatch.info(f"{self.length} data loaded from {metadata_path}")

#     @staticmethod
#     def _eval_task_from_image_path(image_path):
#         """Return the task in ``.../<split>/<task>/<episode>/<frame>``."""
#         normalized_path = str(image_path).replace("\\", "/").rstrip("/")
#         path_parts = [part for part in normalized_path.split("/") if part]
#         split_names = {"eval", "validation", "val", "test"}
#         split_indices = [
#             index
#             for index, part in enumerate(path_parts)
#             if part.lower() in split_names
#         ]
#         if not split_indices:
#             raise ValueError(
#                 "Cannot infer an evaluation task from image path "
#                 f"{image_path!r}; expected an eval/validation/test split "
#                 "component before <task>/<episode>/<frame>."
#             )
#         split_index = split_indices[-1]
#         if split_index + 3 >= len(path_parts):
#             raise ValueError(
#                 "Cannot infer an evaluation task from image path "
#                 f"{image_path!r}; expected "
#                 ".../<split>/<task>/<episode>/<frame>."
#             )
#         return path_parts[split_index + 1]

#     def select_eval_tasks(self, mode, fast_num_tasks=100, seed=0):
#         """Select one deterministic global task subset before rank sharding.

#         ``test_mode`` retains every task in the manifest. ``fast_mode`` uses
#         a rank-independent local RNG to choose tasks, then retains every image
#         belonging to those tasks. The existing DistributedSampler can therefore
#         shard exactly the same filtered dataset on every rank.
#         """
#         mode = str(mode).strip().lower()
#         if mode not in {"test_mode", "fast_mode"}:
#             raise ValueError(
#                 "data.eval_mode must be 'test_mode' or 'fast_mode', "
#                 f"got {mode!r}."
#             )

#         task_by_path = {
#             image_path: self._eval_task_from_image_path(image_path)
#             for image_path in self.metadata
#         }
#         available_tasks = sorted(set(task_by_path.values()))
#         if not available_tasks:
#             raise ValueError("The evaluation manifest contains no tasks.")

#         if mode == "fast_mode":
#             fast_num_tasks = int(fast_num_tasks)
#             if fast_num_tasks <= 0:
#                 raise ValueError("data.fast_eval_num_tasks must be positive.")
#             if fast_num_tasks > len(available_tasks):
#                 raise ValueError(
#                     "data.fast_eval_num_tasks exceeds the available task count: "
#                     f"requested={fast_num_tasks}, available={len(available_tasks)}."
#                 )
#             selected_tasks = set(
#                 random.Random(int(seed)).sample(
#                     available_tasks,
#                     fast_num_tasks,
#                 )
#             )
#             self.metadata = [
#                 image_path
#                 for image_path in self.metadata
#                 if task_by_path[image_path] in selected_tasks
#             ]
#         else:
#             selected_tasks = set(available_tasks)

#         self.length = len(self.metadata)
#         if self.length == 0:
#             raise ValueError(
#                 f"Evaluation mode {mode!r} selected no images."
#             )

#         self.eval_mode = mode
#         self.eval_available_task_count = len(available_tasks)
#         self.eval_selected_tasks = tuple(sorted(selected_tasks))
#         self.eval_task_seed = int(seed)

#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0:
#             overwatch.info(
#                 f"Evaluation {mode}: selected {len(selected_tasks)}/"
#                 f"{len(available_tasks)} tasks and {self.length} images "
#                 f"with seed={int(seed)}."
#             )

#     def __len__(self):
#         return self.length

#     def _retry_index(self, original_idx, attempt, attempted_indices):
#         """Choose a deterministic, previously untried fallback sample."""
#         candidate = (
#             int(original_idx)
#             + int(attempt) * 104729
#             + self.seed * 1009
#         ) % self.length
#         while candidate in attempted_indices and len(attempted_indices) < self.length:
#             candidate = (candidate + 1) % self.length
#         return candidate

#     def _trace_read(self, enabled, event, **details):
#         if not enabled or self.read_trace_dir is None:
#             return
#         if self._read_trace_path is None:
#             os.makedirs(self.read_trace_dir, exist_ok=True)
#             rank = _data_process_rank()
#             metadata_name = re.sub(
#                 r"[^A-Za-z0-9_.-]+",
#                 "_",
#                 os.path.basename(getattr(self, "metadata_source", "images")),
#             )
#             self._read_trace_path = os.path.join(
#                 self.read_trace_dir,
#                 f"data_{metadata_name}_rank{rank:02d}_pid{os.getpid()}.jsonl",
#             )
#         record = {
#             "wall_time_ns": time.time_ns(),
#             "pid": os.getpid(),
#             "rank": _data_process_rank(),
#             "event": str(event),
#         }
#         record.update(details)
#         # Only the first configured samples are traced, so opening per event is
#         # cheap and avoids leaking a file descriptor into DataLoader workers.
#         with open(self._read_trace_path, "a", encoding="utf-8") as trace_file:
#             trace_file.write(
#                 json.dumps(record, ensure_ascii=False, default=str) + "\n"
#             )
#             trace_file.flush()

#     def __getitem__(self, idx):
#         original_idx = int(idx)
#         trace_this_read = self._read_trace_count < self.read_trace_samples
#         self._read_trace_count += 1
#         candidate_idx = original_idx
#         attempted_indices = set()
#         failures = []
#         total_attempts = min(self.max_read_attempts, self.length)

#         for attempt in range(total_attempts):
#             attempted_indices.add(candidate_idx)
#             path = self.metadata[candidate_idx]
#             self._trace_read(
#                 trace_this_read,
#                 "READ_BEGIN",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             try:
#                 with Image.open(path) as opened_image:
#                     opened_image.load()
#                     image = opened_image.convert("RGB")
#             except (OSError, ValueError, UnidentifiedImageError) as exc:
#                 self._trace_read(
#                     trace_this_read,
#                     "READ_ERROR",
#                     requested_index=original_idx,
#                     candidate_index=candidate_idx,
#                     attempt=attempt + 1,
#                     path=path,
#                     error=repr(exc),
#                 )
#                 failures.append((path, exc))
#                 if attempt + 1 < total_attempts:
#                     candidate_idx = self._retry_index(
#                         original_idx,
#                         attempt + 1,
#                         attempted_indices,
#                     )
#                 continue

#             self._trace_read(
#                 trace_this_read,
#                 "OPEN_OK",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             # Keep transform errors visible. They are programming/configuration
#             # errors, not corrupt-image errors that should silently change data.
#             image = self.preprocess_train(image)
#             self._trace_read(
#                 trace_this_read,
#                 "TRANSFORM_OK",
#                 requested_index=original_idx,
#                 candidate_index=candidate_idx,
#                 attempt=attempt + 1,
#                 path=path,
#             )
#             return {"image": image, "path": path}

#         failure_summary = "; ".join(
#             f"{path!r}: {type(exc).__name__}: {exc}"
#             for path, exc in failures[-4:]
#         )
#         last_error = failures[-1][1] if failures else None
#         raise RuntimeError(
#             f"Failed to read an image after {total_attempts} bounded attempts "
#             f"(requested index={original_idx}, rank={_data_process_rank()}). "
#             f"Last failures: {failure_summary}"
#         ) from last_error


# class InfiniteDistributedSampler(DistributedSampler):
#     def __init__(
#         self,
#         dataset,
#         num_replicas=None,
#         rank=None,
#         shuffle=True,
#     ):
#         """
#         无限循环分布式采样器。
#         :param dataset: 数据集
#         :param num_replicas: 总共的设备数量
#         :param rank: 当前设备的 rank
#         :param shuffle: 是否随机打乱
#         """
#         super().__init__(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#         )
#         self._epoch = 0

#     def __iter__(self):
#         """
#         无限循环返回索引。
#         """
#         while True:
#             self.set_epoch(self._epoch)
#             self._epoch += 1
#             indices = super().__iter__()
#             yield from indices

#     def __len__(self):
#         return len(self.dataset)


# class InfiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True):
#         """
#         多任务批量采样器，支持 Lightning 的分布式模式。
#         :param datasets: 多个数据集的列表
#         :param batch_size: 每个 batch 的大小
#         :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
#         """
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.num_datasets = len(self.datasets)
#         self.samples_per_dataset = sample_per_dataset
#         # self.remaining_samples = batch_size % self.num_datasets
#         self.dataset_lengths = [len(dataset) for dataset in self.datasets]

#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.cur_idx = 0

#         # 为每个数据集创建无限采样器
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
#         self.samplers = [
#             InfiniteDistributedSampler(
#                 dataset,
#                 num_replicas=self.num_replicas,
#                 rank=self.rank,
#                 shuffle=shuffle,
#             )
#             for dataset in datasets
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         """
#         无限生成每个 batch 的样本索引。
#         """
#         while True:
#             batch = []
#             for i in range(len(self.iterators)):
#                 iterator = self.iterators[i]
#                 for _ in range(self.samples_per_dataset[i]):
#                     batch.append(next(iterator) + self.cumulative_sizes[i])
#             yield batch

#     def __len__(self):
#         return sum(self.dataset_lengths)


# class FiniteMultiTaskBatchSampler(BatchSampler):
#     def __init__(
#         self,
#         datasets,
#         batch_size,
#         sample_per_dataset,
#         drop_last=False,
#         shuffle=True,
#     ):
#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.samples_per_dataset = sample_per_dataset
#         self.dataset_lengths = [len(dataset) for dataset in datasets]
#         self.cumulative_sizes = [0] + self.dataset_lengths

#         for i in range(1, len(self.cumulative_sizes)):
#             self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

#         self.drop_last = drop_last
#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

#         # 初始化标准分布式采样器
#         self.samplers = [
#             DistributedSampler(
#                 dataset,
#                 num_replicas=self.num_replicas,
#                 rank=self.rank,
#                 shuffle=shuffle,
#             )
#             for dataset in datasets
#         ]
#         self.iterators = [iter(sampler) for sampler in self.samplers]
#         # 计算每个数据集还剩多少样本
#         self.remaining_samples = [len(sampler) for sampler in self.samplers]

#     def __iter__(self):
#         iterators = [iter(sampler) for sampler in self.samplers]
#         remaining_samples = self.remaining_samples.copy()

#         while sum(remaining_samples) > 0:
#             batch = []
#             for i, iterator in enumerate(iterators):
#                 num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
#                 for _ in range(num_samples):
#                     try:
#                         idx = next(iterator)
#                         batch.append(idx + self.cumulative_sizes[i])
#                         remaining_samples[i] -= 1
#                     except StopIteration:
#                         remaining_samples[i] = 0
#                         break  # 当前dataset采样完毕

#             if len(batch) == 0:
#                 break

#             # 根据drop_last判断batch大小
#             if self.drop_last and len(batch) < self.batch_size:
#                 break

#             yield batch

#     def __len__(self):
#         # 总的batch数量（近似值）
#         total_samples = sum(self.dataset_lengths)
#         if self.drop_last:
#             return total_samples // self.batch_size
#         else:
#             return (total_samples + self.batch_size - 1) // self.batch_size


# class MultiDatasetWrapper(Dataset):
#     def __init__(self, datasets):
#         self.datasets = datasets
#         self.dataset_lengths = [len(ds) for ds in datasets]

#     def __len__(self):
#         return sum(self.dataset_lengths)

#     def __getitem__(self, index):
#         cumulative_sizes = 0
#         for dataset, length in zip(self.datasets, self.dataset_lengths):
#             if index < cumulative_sizes + length:
#                 return dataset[index - cumulative_sizes]
#             cumulative_sizes += length
#         raise IndexError("Index out of range")


# def _loader_worker_options(
#     num_workers,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     prefetch_factor=2,
#     read_trace_dir=None,
#     read_trace_samples=0,
# ):
#     num_workers = int(num_workers)
#     loader_timeout_seconds = float(loader_timeout_seconds)
#     if num_workers < 0:
#         raise ValueError("num_workers must be non-negative")
#     if loader_timeout_seconds < 0:
#         raise ValueError("loader_timeout_seconds must be non-negative")
#     prefetch_factor = int(prefetch_factor)
#     if num_workers > 0 and prefetch_factor <= 0:
#         raise ValueError("prefetch_factor must be positive when num_workers > 0")

#     options = {
#         "num_workers": num_workers,
#         "timeout": loader_timeout_seconds if num_workers > 0 else 0,
#         "persistent_workers": bool(persistent_workers) if num_workers > 0 else False,
#     }
#     if num_workers > 0:
#         options["worker_init_fn"] = _initialize_cpu_dataloader_worker
#         options["prefetch_factor"] = prefetch_factor
#     worker_start_method = str(worker_start_method or "").strip().lower()
#     if (
#         num_workers > 0
#         and os.environ.get("DS_ACCELERATOR", "").strip().lower() == "musa"
#         and worker_start_method != "spawn"
#     ):
#         raise ValueError(
#             "MUSA DataLoader workers require worker_start_method='spawn'; "
#             "fork/forkserver can inherit an initialized MUSA/MCCL runtime."
#         )
#     if num_workers > 0 and worker_start_method:
#         if worker_start_method not in {"fork", "spawn", "forkserver"}:
#             raise ValueError(
#                 "worker_start_method must be fork, spawn, forkserver, or empty"
#             )
#         options["multiprocessing_context"] = worker_start_method
#     return options


# def load_unsampler_datasets_from_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     max_read_attempts=8,
#     seed=0,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     prefetch_factor=2,
#     read_trace_dir=None,
#     read_trace_samples=0,
#     eval_mode=None,
#     fast_eval_num_tasks=100,
#     fast_eval_task_seed=0,
# ):
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     if not dataset_paths:
#         raise ValueError(f"Dataset config {json_path!r} contains no datasets.")
#     dataset_path = os.path.join(os.path.dirname(json_path), dataset_paths[0])
#     dataset = ImageData(
#         dataset_path,
#         flip_p=flip_p,
#         img_size=img_size,
#         max_read_attempts=max_read_attempts,
#         seed=seed,
#         read_trace_dir=read_trace_dir,
#         read_trace_samples=read_trace_samples,
#     )

#     for dataset_path in dataset_paths[1:]:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         dataset.add(dataset_path)

#     if eval_mode is not None:
#         dataset.select_eval_tasks(
#             mode=eval_mode,
#             fast_num_tasks=fast_eval_num_tasks,
#             seed=fast_eval_task_seed,
#         )

#     # Accelerate sets RANK/WORLD_SIZE before this loader is built.  Use those
#     # values as a fallback because the process group may not be initialized yet.
#     rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
#     num_replicas = (
#         dist.get_world_size()
#         if dist.is_initialized()
#         else int(os.environ.get("WORLD_SIZE", "1"))
#     )
#     worker_options = _loader_worker_options(
#         num_workers,
#         loader_timeout_seconds=loader_timeout_seconds,
#         persistent_workers=persistent_workers,
#         worker_start_method=worker_start_method,
#         prefetch_factor=prefetch_factor,
#     )

#     if is_infinite:
#         sampler = InfiniteDistributedSampler(
#             dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#         )
#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             drop_last=drop_last,
#             **worker_options,
#         )
#     else:
#         sampler = None
#         if num_replicas > 1:
#             sampler = DistributedSampler(
#                 dataset,
#                 num_replicas=num_replicas,
#                 rank=rank,
#                 shuffle=False,
#                 drop_last=drop_last,
#             )
#         dataloader = DataLoader(
#             dataset,
#             batch_size=local_batch_size,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             shuffle=shuffle if sampler is None else False,
#             drop_last=drop_last,
#             **worker_options,
#         )
#     return dataloader


# def load_multi_datasets_form_json(
#     json_path,
#     flip_p,
#     img_size,
#     local_batch_size,
#     num_workers=8,
#     is_infinite=True,
#     shuffle=True,
#     drop_last=False,
#     make_single_dataset=False,
#     max_read_attempts=8,
#     seed=0,
#     loader_timeout_seconds=0,
#     persistent_workers=False,
#     worker_start_method=None,
#     prefetch_factor=2,
#     read_trace_dir=None,
#     read_trace_samples=0,
#     eval_mode=None,
#     fast_eval_num_tasks=100,
#     fast_eval_task_seed=0,
# ):
#     if make_single_dataset:
#         return load_unsampler_datasets_from_json(
#             json_path=json_path,
#             flip_p=flip_p,
#             img_size=img_size,
#             local_batch_size=local_batch_size,
#             num_workers=num_workers,
#             is_infinite=is_infinite,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             max_read_attempts=max_read_attempts,
#             seed=seed,
#             loader_timeout_seconds=loader_timeout_seconds,
#             persistent_workers=persistent_workers,
#             worker_start_method=worker_start_method,
#             prefetch_factor=prefetch_factor,
#             read_trace_dir=read_trace_dir,
#             read_trace_samples=read_trace_samples,
#             eval_mode=eval_mode,
#             fast_eval_num_tasks=fast_eval_num_tasks,
#             fast_eval_task_seed=fast_eval_task_seed,
#         )
#     if eval_mode is not None:
#         raise ValueError(
#             "Task-based evaluation modes require make_single_dataset=True."
#         )
#     with open(json_path, "r") as f:
#         config = json.load(f)

#     dataset_paths = config["datasets"]
#     ratios = config["ratios"]
#     assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
#     assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

#     datasets = []

#     for dataset_path in dataset_paths:
#         dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
#         datasets.append(
#             ImageData(
#                 dataset_path,
#                 flip_p=flip_p,
#                 img_size=img_size,
#                 max_read_attempts=max_read_attempts,
#                 seed=int(seed) + len(datasets),
#                 read_trace_dir=read_trace_dir,
#                 read_trace_samples=read_trace_samples,
#             )
#         )

#     sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

#     total = sum(sample_per_dataset)
#     if total < local_batch_size:
#         sample_per_dataset[-1] += local_batch_size - total
#     elif total > local_batch_size:
#         sample_per_dataset[-1] -= total - local_batch_size

#     wrapped_dataset = MultiDatasetWrapper(datasets)

#     if is_infinite:
#         batch_sampler = InfiniteMultiTaskBatchSampler(
#             datasets,
#             local_batch_size,
#             sample_per_dataset=sample_per_dataset,
#             shuffle=shuffle,
#         )
#     else:
#         batch_sampler = FiniteMultiTaskBatchSampler(
#             datasets,
#             local_batch_size,
#             sample_per_dataset=sample_per_dataset,
#             shuffle=shuffle,
#             drop_last=drop_last,
#         )

#     dataloader = DataLoader(
#         wrapped_dataset,
#         batch_sampler=batch_sampler,
#         collate_fn=collate_fn,
#         **_loader_worker_options(
#             num_workers,
#             loader_timeout_seconds=loader_timeout_seconds,
#             persistent_workers=persistent_workers,
#             worker_start_method=worker_start_method,
#             prefetch_factor=prefetch_factor,
#         ),
#     )

#     return dataloader


# def check_tensor(obj, name, check_bound=1e4, check_std=1e3, _visited=None, force_output=False):
#     if not int(os.environ.get("CHECK_TENSOR", 0)):
#         return
#     if _visited is None:
#         _visited = set()
#     if id(obj) in _visited:
#         return False
#     _visited.add(id(obj))

#     # list / tuple
#     if isinstance(obj, (list, tuple)):
#         problem = False
#         for i, v in enumerate(obj):
#             if check_tensor(v, f"{name}[{i}]", _visited=_visited):
#                 problem = True
#         return problem

#     # dict
#     if isinstance(obj, dict):
#         problem = False
#         for k, v in obj.items():
#             if check_tensor(v, f"{name}['{k}']", _visited=_visited):
#                 problem = True
#         return problem

#     # not tensor
#     if not isinstance(obj, torch.Tensor):
#         return False

#     t = obj
#     problem_found = False

#     has_nan = torch.isnan(t).any().item()
#     has_inf = torch.isinf(t).any().item()

#     if has_nan:
#         overwatch.error(f"<{name}> 检测到 NaN")
#         problem_found = True
#     if has_inf:
#         overwatch.error(f"<{name}> 检测到 Inf")
#         problem_found = True

#     try:
#         minv = t.min().item()
#         maxv = t.max().item()
#         meanv = t.mean().item()
#         stdv = t.std().item() if t.numel() > 1 else 0.0
#     except Exception as e:
#         overwatch.error(f"<{name}> 无法计算统计值: {e}")
#         return True

#     if abs(maxv) > float(check_bound) or abs(minv) > float(check_bound):
#         overwatch.error(f"<{name}> 数值过大 (|value| > {check_bound}), 可能导致梯度爆炸")
#         problem_found = True

#     if stdv > float(check_std):
#         overwatch.error(f"<{name}> 标准差过大 (std={stdv}), 数值不稳定")
#         problem_found = True

#     if t.numel() > 1 and stdv < 1e-12:
#         overwatch.error(f"<{name}> 标准差过小 (=0), 张量可能塌陷/全常数")
#         problem_found = True

#     if torch.all(t == 0):
#         overwatch.error(f"<{name}> 张量全为 0")
#         problem_found = True

#     if t.numel() > 1 and torch.all(t == t.flatten()[0]):
#         overwatch.error(f"<{name}> 张量全为常数")
#         problem_found = True

#     if t.requires_grad and t.is_leaf and t.grad is not None:
#         g = t.grad
#         if torch.isnan(g).any():
#             overwatch.error(f"<{name}> 梯度存在 NaN")
#             problem_found = True
#         if torch.isinf(g).any():
#             overwatch.error(f"<{name}> 梯度存在 Inf")
#             problem_found = True
#         if g.abs().max().item() > 1e6:
#             overwatch.error(f"<{name}> 梯度爆炸 (grad > 1e6)")
#             problem_found = True

#     # if problem_found:
#     if force_output or problem_found:
#         overwatch.error(f"<{name}> 基本信息: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
#         overwatch.error(f"<{name}> 数值统计: min={minv}, max={maxv}, mean={meanv}, std={stdv}")

#     return problem_found


# if __name__ == "__main__":
#     import torchvision.transforms as T

#     dataset = ImageData(metadata_path="jsons/eval_VLA.jsonl", flip_p=0.5)

#     dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)

#     dataloader = load_multi_datasets_form_json(
#         json_path="jsons/eval_VLA.json",
#         flip_p=0,
#         img_size=224,
#         local_batch_size=4,
#         num_workers=0,
#         is_infinite=False,
#         shuffle=False,
#         drop_last=False,
#         make_single_dataset=True,
#     )

#     data = next(iter(dataloader))
#     print("image shape:", data["images"].shape)

"""Spawn-safe EgoVerse image and hand-joint data loading."""

from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    get_worker_info,
)

from stamo.renderer.utils.metadata_index import JsonlImagePathCollection
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def _data_process_rank() -> int:
    if dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("STAMO_DATALOADER_PARENT_RANK", 0))


def _initialize_cpu_dataloader_worker(worker_id: int) -> None:
    """Seed and verify one spawn-based, CPU-only DataLoader worker."""
    torch.set_num_threads(1)
    worker_seed = 33 + int(worker_id)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    if os.environ.get("STAMO_DATALOADER_SPAWN_CHILD") != "1":
        return
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise RuntimeError(
            "Spawned DataLoader worker did not disable accelerator autoload "
            "before importing torch."
        )
    if "torch_musa" in sys.modules:
        raise RuntimeError(
            "Spawned DataLoader worker imported torch_musa; CPU workers must "
            "not own MUSA contexts."
        )
    storage_type = getattr(torch, "UntypedStorage", None)
    if storage_type is not None and not hasattr(storage_type, "is_musa"):
        raise RuntimeError(
            "Spawned DataLoader worker is missing the CPU is_musa storage "
            "predicate required by the musified multiprocessing reducer."
        )
    unexpected_training_modules = {
        module_name
        for module_name in ("accelerate", "deepspeed")
        if module_name in sys.modules
    }
    if unexpected_training_modules:
        raise RuntimeError(
            "Spawned DataLoader worker imported the training stack: "
            f"{sorted(unexpected_training_modules)}"
        )
    if dist.is_initialized():
        raise RuntimeError(
            "Spawned DataLoader worker initialized a distributed process group."
        )
    worker_info = get_worker_info()
    if worker_info is None or int(worker_info.id) != int(worker_id):
        raise RuntimeError(
            "DataLoader worker identity is unavailable or inconsistent."
        )
    print(
        "STAMO_CPU_DATALOADER_WORKER_READY "
        f"parent_rank={os.environ.get('STAMO_DATALOADER_PARENT_RANK', '?')} "
        f"worker_id={int(worker_id)} "
        f"num_workers={int(worker_info.num_workers)} "
        f"pid={os.getpid()} autoload=0 musa_imported=0 "
        "training_stack_imported=0 distributed=0",
        flush=True,
    )


def fp32_to_fp16(batch):
    """Recursively convert FP32 tensors to FP16 for DeepSpeed inputs."""
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.float16)
    if isinstance(batch, list):
        return [fp32_to_fp16(item) for item in batch]
    if isinstance(batch, tuple):
        return tuple(fp32_to_fp16(item) for item in batch)
    if isinstance(batch, dict):
        return {key: fp32_to_fp16(item) for key, item in batch.items()}
    return batch


def fp32_to_bf16(batch):
    """Recursively convert FP32 tensors to BF16 for DeepSpeed inputs."""
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.bfloat16)
    if isinstance(batch, list):
        return [fp32_to_bf16(item) for item in batch]
    if isinstance(batch, tuple):
        return tuple(fp32_to_bf16(item) for item in batch)
    if isinstance(batch, dict):
        return {key: fp32_to_bf16(item) for key, item in batch.items()}
    return batch


def move_to_cuda(batch, device=None):
    """Move a nested training batch to the selected MUSA/CUDA device."""
    if device is None:
        from stamo.renderer.utils.device import get_accelerator_device

        device = get_accelerator_device()
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, list):
        return [move_to_cuda(item, device=device) for item in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_cuda(item, device=device) for item in batch)
    if isinstance(batch, dict):
        return {
            key: move_to_cuda(item, device=device)
            for key, item in batch.items()
        }
    return batch


def get_loader_info(dataset, epochs, bsz, gradient_accumulate_steps=1):
    """Return per-device batch, global batch, steps/epoch and total steps."""
    dataset_len = len(dataset.dataset) if hasattr(dataset, "dataset") else len(dataset)
    images_per_gpu = int(bsz)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    images_per_batch = int(bsz) * world_size * int(gradient_accumulate_steps)
    iter_per_ep = (
        math.ceil(dataset_len / images_per_batch) if dataset_len else 0
    )
    num_iters = iter_per_ep * int(epochs)
    return images_per_gpu, images_per_batch, iter_per_ep, num_iters


def _write_gaussian_patch(
    target: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float,
) -> None:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    height, width = target.shape
    x0 = max(0, int(math.floor(center_x)) - radius)
    x1 = min(width, int(math.floor(center_x)) + radius + 1)
    y0 = max(0, int(math.floor(center_y)) - radius)
    y1 = min(height, int(math.floor(center_y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    patch = np.exp(-distance_sq / (2.0 * sigma * sigma))
    target[y0:y1, x0:x1] = patch


def render_hand_joints(
    pose_uvz: np.ndarray,
    image_size: int,
    sigma: float,
    horizontal_flip: bool = False,
    flip_swap_hands: bool = True,
) -> torch.Tensor:
    """Render 42 independent joint Gaussians on one full image canvas.

    The output is ``[42,image_size,image_size]``. Channels 0..20 are the left
    hand and channels 21..41 are the right hand. No bone, capsule, hand crop,
    depth raster, or max-reduction across different joints is constructed.
    """

    pose = np.nan_to_num(
        np.asarray(pose_uvz, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).copy()
    if pose.shape != (2, 21, 3):
        raise ValueError(
            f"pose sidecar row must have shape (2,21,3), got {pose.shape}"
        )

    if horizontal_flip:
        pose[..., 0] = 1.0 - pose[..., 0]
        if flip_swap_hands:
            pose = pose[::-1].copy()

    output = np.zeros((42, image_size, image_size), dtype=np.float32)
    for hand_index in range(2):
        # Sidecar v5 stores q=(source_pixel+0.5)/source_size.  This recovers
        # the exact pixel-centre coordinate after resizing to image_size.
        xy = pose[hand_index, :, :2] * float(image_size) - 0.5
        for joint_index in range(21):
            point = xy[joint_index]
            _write_gaussian_patch(
                output[hand_index * 21 + joint_index],
                point[0],
                point[1],
                sigma,
            )

    return torch.from_numpy(output)


class ImageData(Dataset):
    """Image dataset with a line-aligned, memory-mapped pose sidecar."""

    def __init__(
        self,
        metadata_path,
        flip_p,
        pose_sidecar_path,
        img_size: int = 224,
        pose_flip_swap_hands: bool = True,
        pose_verify_manifest: bool = True,
        max_read_attempts: int = 8,
        seed: int = 0,
        read_trace_dir=None,
        read_trace_samples: int = 0,
    ) -> None:
        self.pose_sidecar_path = os.path.abspath(
            os.path.expanduser(str(pose_sidecar_path))
        )
        self.pose_flip_swap_hands = bool(pose_flip_swap_hands)
        self.pose_verify_manifest = bool(pose_verify_manifest)
        self._pose_array = None
        self._pose_rows: Optional[np.ndarray] = None
        self.flip_p = float(flip_p)
        self.max_read_attempts = int(max_read_attempts)
        self.seed = int(seed)
        self.read_trace_dir = (
            os.path.abspath(os.path.expanduser(str(read_trace_dir)))
            if read_trace_dir
            else None
        )
        self.read_trace_samples = max(0, int(read_trace_samples))
        if self.max_read_attempts <= 0:
            raise ValueError("max_read_attempts must be positive")

        self._read_trace_count = 0
        self._read_trace_path = None
        self.metadata_source = os.path.abspath(
            os.path.expanduser(str(metadata_path))
        )
        self.metadata = JsonlImagePathCollection()
        self.metadata.add(self.metadata_source)
        self.length = len(self.metadata)
        if self.length == 0:
            raise ValueError(
                f"No images were found in metadata file {metadata_path!r}."
            )
        overwatch.info(f"{self.length} data loaded from {metadata_path}")
        self.transforms = T.Compose(
            [
                T.Resize(
                    (int(img_size), int(img_size)),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
            ]
        )

        if not os.path.isfile(self.pose_sidecar_path):
            raise FileNotFoundError(
                f"Pose sidecar does not exist: {self.pose_sidecar_path}"
            )
        probe = np.load(self.pose_sidecar_path, mmap_mode="r")
        try:
            if probe.shape != (self.length, 2, 21, 3):
                raise ValueError(
                    "Pose sidecar must be line-aligned with the image JSONL: "
                    f"pose_shape={probe.shape}, image_rows={self.length}, "
                    f"path={self.pose_sidecar_path}"
                )
        finally:
            del probe
        if self.pose_verify_manifest:
            self._verify_pose_summary(metadata_path)

    @staticmethod
    def _eval_task_from_image_path(image_path):
        """Return the task in ``.../<split>/<task>/<episode>/<frame>``."""
        normalized_path = str(image_path).replace("\\", "/").rstrip("/")
        path_parts = [part for part in normalized_path.split("/") if part]
        split_names = {"eval", "validation", "val", "test", "train"}
        split_indices = [
            index
            for index, part in enumerate(path_parts)
            if part.lower() in split_names
        ]
        if not split_indices:
            raise ValueError(
                "Cannot infer a task from image path "
                f"{image_path!r}; expected a split component before "
                "<task>/<episode>/<frame>."
            )
        split_index = split_indices[-1]
        if split_index + 3 >= len(path_parts):
            raise ValueError(
                "Cannot infer a task from image path "
                f"{image_path!r}; expected "
                ".../<split>/<task>/<episode>/<frame>."
            )
        return path_parts[split_index + 1]

    def __len__(self):
        return self.length

    def _retry_index(self, original_idx, attempt, attempted_indices):
        """Choose a deterministic, previously untried fallback sample."""
        candidate = (
            int(original_idx)
            + int(attempt) * 104729
            + self.seed * 1009
        ) % self.length
        while (
            candidate in attempted_indices
            and len(attempted_indices) < self.length
        ):
            candidate = (candidate + 1) % self.length
        return candidate

    def _trace_read(self, enabled, event, **details):
        if not enabled or self.read_trace_dir is None:
            return
        if self._read_trace_path is None:
            os.makedirs(self.read_trace_dir, exist_ok=True)
            metadata_name = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                os.path.basename(self.metadata_source),
            )
            self._read_trace_path = os.path.join(
                self.read_trace_dir,
                f"data_{metadata_name}_rank{_data_process_rank():02d}_"
                f"pid{os.getpid()}.jsonl",
            )
        record = {
            "wall_time_ns": time.time_ns(),
            "pid": os.getpid(),
            "rank": _data_process_rank(),
            "event": str(event),
        }
        record.update(details)
        with open(self._read_trace_path, "a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
            stream.flush()

    def _verify_pose_summary(self, metadata_path) -> None:
        """Fail before training when a sidecar belongs to another JSONL."""

        summary_pattern = (
            os.path.splitext(self.pose_sidecar_path)[0] + ".summary*.json"
        )
        summary_matches = sorted(glob.glob(summary_pattern))
        if not summary_matches:
            raise FileNotFoundError(
                "Pose manifest verification is enabled but the builder "
                f"summary is missing: {summary_pattern}"
            )
        if len(summary_matches) != 1:
            raise RuntimeError(
                "Pose manifest verification found multiple summaries for "
                f"one sidecar: {summary_matches}"
            )
        summary_path = summary_matches[0]
        with open(summary_path, "r", encoding="utf-8") as stream:
            summary = json.load(stream)
        summary_format = str(summary.get("format", ""))
        if not (
            summary_format.startswith("stamo-egoverse-pose-")
            and summary_format.endswith("-v5-pixel-centre")
        ):
            raise ValueError(
                f"Unsupported pose summary format in {summary_path}: "
                f"{summary_format!r}"
            )

        manifest_path = os.path.realpath(
            os.path.abspath(os.path.expanduser(str(metadata_path)))
        )
        recorded_path = os.path.realpath(str(summary.get("manifest", "")))
        stat = os.stat(manifest_path)
        expected = {
            "manifest": recorded_path,
            "manifest_size_bytes": int(summary.get("manifest_size_bytes", -1)),
            "manifest_mtime_ns": int(summary.get("manifest_mtime_ns", -1)),
            "manifest_rows": int(summary.get("manifest_rows", -1)),
        }
        actual = {
            "manifest": manifest_path,
            "manifest_size_bytes": int(stat.st_size),
            "manifest_mtime_ns": int(stat.st_mtime_ns),
            "manifest_rows": int(self.length),
        }
        if actual != expected:
            raise ValueError(
                "Pose sidecar/JSONL identity mismatch. Rebuild the sidecar "
                f"for this exact manifest. expected={expected}, actual={actual}"
            )

    def add(self, metadata_path):
        raise RuntimeError(
            "The hand-pose loader currently requires exactly one JSONL and one "
            "line-aligned pose sidecar."
        )

    def _get_pose_array(self):
        if self._pose_array is None:
            self._pose_array = np.load(self.pose_sidecar_path, mmap_mode="r")
        return self._pose_array

    def _pose_row(self, candidate_index: int) -> np.ndarray:
        pose_index = (
            int(candidate_index)
            if self._pose_rows is None
            else int(self._pose_rows[candidate_index])
        )
        return np.asarray(self._get_pose_array()[pose_index], dtype=np.float32)

    def select_eval_tasks(self, mode, fast_num_tasks=100, seed=0):
        mode = str(mode).strip().lower()
        if mode not in {"test_mode", "fast_mode"}:
            raise ValueError(
                "data.eval_mode must be 'test_mode' or 'fast_mode', "
                f"got {mode!r}."
            )

        rows_and_paths = list(enumerate(self.metadata))
        task_by_path = {
            path: self._eval_task_from_image_path(path)
            for _, path in rows_and_paths
        }
        available_tasks = sorted(set(task_by_path.values()))
        if mode == "fast_mode":
            count = int(fast_num_tasks)
            if count <= 0 or count > len(available_tasks):
                raise ValueError(
                    "Invalid fast_eval_num_tasks: "
                    f"requested={count}, available={len(available_tasks)}"
                )
            selected_tasks = set(
                random.Random(int(seed)).sample(available_tasks, count)
            )
        else:
            selected_tasks = set(available_tasks)

        selected = [
            (row, path)
            for row, path in rows_and_paths
            if task_by_path[path] in selected_tasks
        ]
        if not selected:
            raise ValueError(f"Evaluation mode {mode!r} selected no images.")
        self.metadata = [path for _, path in selected]
        self._pose_rows = np.asarray([row for row, _ in selected], dtype=np.int64)
        self.length = len(selected)
        self.eval_mode = mode
        self.eval_available_task_count = len(available_tasks)
        self.eval_selected_tasks = tuple(sorted(selected_tasks))
        self.eval_task_seed = int(seed)
        if _data_process_rank() == 0:
            overwatch.info(
                f"Evaluation {mode}: selected {len(selected_tasks)}/"
                f"{len(available_tasks)} tasks and {self.length} images "
                f"with seed={int(seed)}."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # A numpy memmap must be reopened independently by each spawn worker.
        state["_pose_array"] = None
        return state

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
            pose = self._pose_row(candidate_idx)
            horizontal_flip = bool(torch.rand(()) < self.flip_p)
            if horizontal_flip:
                transpose_enum = getattr(Image, "Transpose", Image)
                image = image.transpose(transpose_enum.FLIP_LEFT_RIGHT)
            # Keep the UVZ joints aligned with the transformed RGB image.
            pose_model = np.nan_to_num(
                np.asarray(pose, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).copy()
            if horizontal_flip:
                pose_model[..., 0] = 1.0 - pose_model[..., 0]
                if self.pose_flip_swap_hands:
                    pose_model = pose_model[::-1].copy()
            image = self.transforms(image)
            self._trace_read(
                trace_this_read,
                "TRANSFORM_OK",
                requested_index=original_idx,
                candidate_index=candidate_idx,
                attempt=attempt + 1,
                path=path,
            )
            return {
                "image": image,
                "pose_uvz": torch.from_numpy(pose_model),
                "path": path,
            }

        failure_summary = "; ".join(
            f"{path!r}: {type(exc).__name__}: {exc}"
            for path, exc in failures[-4:]
        )
        last_error = failures[-1][1] if failures else None
        raise RuntimeError(
            f"Failed to read an image after {total_attempts} bounded attempts "
            f"(requested index={original_idx}, rank={_data_process_rank()}). "
            f"Last failures: {failure_summary}"
        ) from last_error


def collate_fn(inputs):
    batch = {
        "images": torch.stack([item["image"] for item in inputs]),
        "pose_uvz": torch.stack([item["pose_uvz"] for item in inputs]),
    }
    paths = [item.get("path") for item in inputs]
    if all(path is not None for path in paths):
        batch["paths"] = paths
    return batch


class InfiniteDistributedSampler(DistributedSampler):
    """Repeat deterministic distributed epochs without materializing indices."""

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self._epoch = 0

    def __iter__(self):
        while True:
            self.set_epoch(self._epoch)
            self._epoch += 1
            yield from super().__iter__()

    def __len__(self):
        return len(self.dataset)


class InfiniteMultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True):
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
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_sizes = [0] + self.dataset_lengths
        for i in range(1, len(self.cumulative_sizes)):
            self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

        self.cur_idx = 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.samplers = [
            InfiniteDistributedSampler(
                dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=shuffle,
            )
            for dataset in datasets
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
        self.samplers = [
            DistributedSampler(
                dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=shuffle,
            )
            for dataset in datasets
        ]
        self.iterators = [iter(sampler) for sampler in self.samplers]
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
                        break

            if len(batch) == 0:
                break

            if self.drop_last and len(batch) < self.batch_size:
                break

            yield batch

    def __len__(self):
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
    prefetch_factor=2,
):
    """Build validated DataLoader worker options for CPU/MUSA execution."""
    num_workers = int(num_workers)
    loader_timeout_seconds = float(loader_timeout_seconds)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if loader_timeout_seconds < 0:
        raise ValueError("loader_timeout_seconds must be non-negative")
    prefetch_factor = int(prefetch_factor)
    if num_workers > 0 and prefetch_factor <= 0:
        raise ValueError(
            "prefetch_factor must be positive when num_workers > 0"
        )

    options = {
        "num_workers": num_workers,
        "timeout": loader_timeout_seconds if num_workers > 0 else 0,
        "persistent_workers": (
            bool(persistent_workers) if num_workers > 0 else False
        ),
    }
    if num_workers > 0:
        options["worker_init_fn"] = _initialize_cpu_dataloader_worker
        options["prefetch_factor"] = prefetch_factor

    worker_start_method = str(worker_start_method or "").strip().lower()
    if (
        num_workers > 0
        and os.environ.get("DS_ACCELERATOR", "").strip().lower() == "musa"
        and worker_start_method != "spawn"
    ):
        raise ValueError(
            "MUSA DataLoader workers require worker_start_method='spawn'; "
            "fork/forkserver can inherit an initialized MUSA/MCCL runtime."
        )
    if num_workers > 0 and worker_start_method:
        if worker_start_method not in {"fork", "spawn", "forkserver"}:
            raise ValueError(
                "worker_start_method must be fork, spawn, forkserver, or empty"
            )
        options["multiprocessing_context"] = worker_start_method
    return options


def _resolve_dataset_paths(json_path, pose_sidecar_path):
    """Resolve line-aligned JSONL/pose pairs from one wrapper JSON."""
    with open(json_path, "r", encoding="utf-8") as stream:
        wrapper = json.load(stream)

    dataset_paths = list(wrapper.get("datasets", []))
    if not dataset_paths:
        raise ValueError(f"Dataset config {json_path!r} contains no datasets.")

    sidecar_value = pose_sidecar_path
    if isinstance(sidecar_value, (str, os.PathLike)):
        pose_sidecar_paths = [str(sidecar_value)]
    elif sidecar_value is not None:
        pose_sidecar_paths = [str(path) for path in sidecar_value]
    else:
        pose_sidecar_paths = []
    if len(pose_sidecar_paths) != len(dataset_paths):
        raise ValueError(
            "Every dataset JSONL needs one line-aligned pose sidecar: "
            f"datasets={len(dataset_paths)}, "
            f"pose_sidecars={len(pose_sidecar_paths)}."
        )

    wrapper_dir = os.path.dirname(os.path.abspath(json_path))

    def resolve(path):
        path = os.path.expanduser(str(path))
        if not os.path.isabs(path):
            path = os.path.join(wrapper_dir, path)
        return os.path.abspath(path)

    return (
        wrapper,
        [resolve(path) for path in dataset_paths],
        [resolve(path) for path in pose_sidecar_paths],
    )


def _build_datasets(
    dataset_paths,
    pose_sidecar_paths,
    flip_p,
    img_size,
    pose_flip_swap_hands,
    pose_verify_manifest,
    max_read_attempts,
    seed,
    read_trace_dir,
    read_trace_samples,
):
    datasets = []
    for dataset_index, (dataset_path, sidecar_path) in enumerate(
        zip(dataset_paths, pose_sidecar_paths)
    ):
        datasets.append(
            ImageData(
                metadata_path=dataset_path,
                flip_p=flip_p,
                pose_sidecar_path=sidecar_path,
                img_size=img_size,
                pose_flip_swap_hands=pose_flip_swap_hands,
                pose_verify_manifest=pose_verify_manifest,
                max_read_attempts=max_read_attempts,
                seed=int(seed) + dataset_index,
                read_trace_dir=read_trace_dir,
                read_trace_samples=read_trace_samples,
            )
        )
    return datasets


def load_unsampler_datasets_from_json(
    json_path,
    flip_p,
    img_size,
    local_batch_size,
    pose_sidecar_path=None,
    num_workers=8,
    is_infinite=True,
    shuffle=True,
    drop_last=False,
    max_read_attempts=8,
    seed=0,
    loader_timeout_seconds=0,
    persistent_workers=False,
    worker_start_method=None,
    prefetch_factor=2,
    read_trace_dir=None,
    read_trace_samples=0,
    eval_mode=None,
    fast_eval_num_tasks=100,
    fast_eval_task_seed=0,
    pose_flip_swap_hands=True,
    pose_verify_manifest=True,
):
    """Merge datasets uniformly, without ratio-based task batch sampling."""
    _, dataset_paths, sidecar_paths = _resolve_dataset_paths(
        json_path,
        pose_sidecar_path,
    )
    datasets = _build_datasets(
        dataset_paths=dataset_paths,
        pose_sidecar_paths=sidecar_paths,
        flip_p=flip_p,
        img_size=img_size,
        pose_flip_swap_hands=pose_flip_swap_hands,
        pose_verify_manifest=pose_verify_manifest,
        max_read_attempts=max_read_attempts,
        seed=seed,
        read_trace_dir=read_trace_dir,
        read_trace_samples=read_trace_samples,
    )
    dataset = datasets[0] if len(datasets) == 1 else MultiDatasetWrapper(datasets)
    if eval_mode is not None:
        dataset.select_eval_tasks(
            mode=eval_mode,
            fast_num_tasks=fast_eval_num_tasks,
            seed=fast_eval_task_seed,
        )

    rank = dist.get_rank() if dist.is_initialized() else int(
        os.environ.get("RANK", "0")
    )
    world_size = dist.get_world_size() if dist.is_initialized() else int(
        os.environ.get("WORLD_SIZE", "1")
    )
    worker_options = _loader_worker_options(
        num_workers,
        loader_timeout_seconds=loader_timeout_seconds,
        persistent_workers=persistent_workers,
        worker_start_method=worker_start_method,
        prefetch_factor=prefetch_factor,
    )

    if is_infinite:
        sampler = InfiniteDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
    else:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=drop_last,
            )
            if world_size > 1
            else None
        )

    return DataLoader(
        dataset,
        batch_size=local_batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        **worker_options,
    )


def load_multi_datasets_form_json(
    json_path,
    flip_p,
    img_size,
    local_batch_size,
    pose_sidecar_path=None,
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
    prefetch_factor=2,
    read_trace_dir=None,
    read_trace_samples=0,
    eval_mode=None,
    fast_eval_num_tasks=100,
    fast_eval_task_seed=0,
    pose_flip_swap_hands=True,
    pose_verify_manifest=True,
):
    if make_single_dataset:
        return load_unsampler_datasets_from_json(
            json_path=json_path,
            flip_p=flip_p,
            img_size=img_size,
            local_batch_size=local_batch_size,
            pose_sidecar_path=pose_sidecar_path,
            num_workers=num_workers,
            is_infinite=is_infinite,
            shuffle=shuffle,
            drop_last=drop_last,
            max_read_attempts=max_read_attempts,
            seed=seed,
            loader_timeout_seconds=loader_timeout_seconds,
            persistent_workers=persistent_workers,
            worker_start_method=worker_start_method,
            prefetch_factor=prefetch_factor,
            read_trace_dir=read_trace_dir,
            read_trace_samples=read_trace_samples,
            eval_mode=eval_mode,
            fast_eval_num_tasks=fast_eval_num_tasks,
            fast_eval_task_seed=fast_eval_task_seed,
            pose_flip_swap_hands=pose_flip_swap_hands,
            pose_verify_manifest=pose_verify_manifest,
        )
    if eval_mode is not None:
        raise ValueError(
            "Task-based evaluation modes require make_single_dataset=True."
        )

    wrapper, dataset_paths, sidecar_paths = _resolve_dataset_paths(
        json_path,
        pose_sidecar_path,
    )
    ratios = wrapper["ratios"]
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    assert len(ratios) == len(dataset_paths), (
        "Each dataset must have a corresponding ratio"
    )

    datasets = _build_datasets(
        dataset_paths=dataset_paths,
        pose_sidecar_paths=sidecar_paths,
        flip_p=flip_p,
        img_size=img_size,
        pose_flip_swap_hands=pose_flip_swap_hands,
        pose_verify_manifest=pose_verify_manifest,
        max_read_attempts=max_read_attempts,
        seed=seed,
        read_trace_dir=read_trace_dir,
        read_trace_samples=read_trace_samples,
    )
    sample_per_dataset = [
        max(1, math.floor(ratio * local_batch_size)) for ratio in ratios
    ]
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
        )
    else:
        batch_sampler = FiniteMultiTaskBatchSampler(
            datasets,
            local_batch_size,
            sample_per_dataset=sample_per_dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    return DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        **_loader_worker_options(
            num_workers,
            loader_timeout_seconds=loader_timeout_seconds,
            persistent_workers=persistent_workers,
            worker_start_method=worker_start_method,
            prefetch_factor=prefetch_factor,
        ),
    )


def check_tensor(
    obj,
    name,
    check_bound=1e4,
    check_std=1e3,
    _visited=None,
    force_output=False,
):
    """Recursively report non-finite, extreme or collapsed tensors."""
    if not int(os.environ.get("CHECK_TENSOR", 1)):
        return False
    if _visited is None:
        _visited = set()
    if id(obj) in _visited:
        return False
    _visited.add(id(obj))

    if isinstance(obj, (list, tuple)):
        problem_found = False
        for index, value in enumerate(obj):
            if check_tensor(
                value,
                f"{name}[{index}]",
                check_bound=check_bound,
                check_std=check_std,
                _visited=_visited,
                force_output=force_output,
            ):
                problem_found = True
        return problem_found
    if isinstance(obj, dict):
        problem_found = False
        for key, value in obj.items():
            if check_tensor(
                value,
                f"{name}[{key!r}]",
                check_bound=check_bound,
                check_std=check_std,
                _visited=_visited,
                force_output=force_output,
            ):
                problem_found = True
        return problem_found
    if not isinstance(obj, torch.Tensor):
        return False

    tensor = obj
    problem_found = False
    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    if has_nan:
        overwatch.error(f"<{name}> contains NaN")
        problem_found = True
    if has_inf:
        overwatch.error(f"<{name}> contains Inf")
        problem_found = True

    try:
        min_value = tensor.min().item()
        max_value = tensor.max().item()
        mean_value = tensor.mean().item()
        std_value = tensor.std().item() if tensor.numel() > 1 else 0.0
    except Exception as exc:
        overwatch.error(f"<{name}> cannot compute statistics: {exc}")
        return True

    if (
        abs(max_value) > float(check_bound)
        or abs(min_value) > float(check_bound)
    ):
        overwatch.error(
            f"<{name}> has |value| > {float(check_bound):g}"
        )
        problem_found = True
    if std_value > float(check_std):
        overwatch.error(
            f"<{name}> has unstable standard deviation: {std_value}"
        )
        problem_found = True
    if tensor.numel() > 1 and std_value < 1e-12:
        overwatch.error(f"<{name}> has collapsed standard deviation")
        problem_found = True
    if bool(torch.all(tensor == 0).item()):
        overwatch.error(f"<{name}> is entirely zero")
        problem_found = True
    if tensor.numel() > 1 and bool(
        torch.all(tensor == tensor.flatten()[0]).item()
    ):
        overwatch.error(f"<{name}> is entirely constant")
        problem_found = True

    if tensor.requires_grad and tensor.is_leaf and tensor.grad is not None:
        gradient = tensor.grad
        if bool(torch.isnan(gradient).any().item()):
            overwatch.error(f"<{name}> gradient contains NaN")
            problem_found = True
        if bool(torch.isinf(gradient).any().item()):
            overwatch.error(f"<{name}> gradient contains Inf")
            problem_found = True
        if gradient.numel() and gradient.abs().max().item() > 1e6:
            overwatch.error(f"<{name}> gradient magnitude exceeds 1e6")
            problem_found = True

    if force_output or problem_found:
        overwatch.error(
            f"<{name}> shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device}"
        )
        overwatch.error(
            f"<{name}> min={min_value} max={max_value} "
            f"mean={mean_value} std={std_value}"
        )
    return problem_found
