# import torch
# import torch.nn as nn
# from torch import optim


# class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         max_iter,
#         min_lr=3e-5,
#         warmup_ratio=0.1,
#         last_epoch=-1,
#     ):
#         self.max_iter = max_iter
#         self.min_lr = min_lr
#         self.warmup_ratio = warmup_ratio
#         self.warmup_iters = int(warmup_ratio * max_iter)
#         super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

#     def get_lr_factor(self):
#         tot_step = self.max_iter
#         warmup_step = self.warmup_iters
#         step = self.last_epoch
#         if step < warmup_step:
#             return max(0, step / warmup_step)
#         elif step > tot_step:
#             step = tot_step
#         return max(0, (tot_step - step) / (tot_step - warmup_step))

#     def get_lr(self):
#         warmup_factor = self.get_lr_factor()
#         return [max(self.min_lr, base_lr * warmup_factor) for base_lr in self.base_lrs]


# class WarmupLinearConstantLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         max_iter,
#         min_lr=1e-8,
#         warmup_ratio=0.1,
#         last_epoch=-1,
#     ):
#         self.max_iter = max_iter
#         self.min_lr = min_lr
#         self.warmup_ratio = warmup_ratio
#         self.warmup_iters = int(warmup_ratio * max_iter)
#         super(WarmupLinearConstantLR, self).__init__(optimizer, last_epoch)

#     def get_lr_factor(self):
#         warmup_step = self.warmup_iters
#         step = self.last_epoch
#         if step < warmup_step:
#             return max(0, step / warmup_step)
#         elif step >= warmup_step:
#             return 1.0  # constant base lr

#     def get_lr(self):
#         warmup_factor = self.get_lr_factor()
#         return [max(self.min_lr, base_lr * warmup_factor) for base_lr in self.base_lrs]


# class DiffusionLoss(nn.Module):
#     def __init__(self, reduction: str = "mean"):
#         super().__init__()
#         assert reduction in ["mean", "sum", "none"], f"Invalid reduction type: {reduction}"
#         self.reduction = reduction

#     def forward(self, weighting: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss = weighting.float() * (model_pred.float() - target.float()) ** 2
#         loss = torch.mean(loss.reshape(target.shape[0], -1), dim=1)

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss


# def get_criterion(loss_type="l2", reduction="mean") -> nn.Module:
#     loss_type = loss_type.lower()

#     if loss_type == "l1":
#         return nn.L1Loss(reduction=reduction)
#     elif loss_type == "l2":
#         return nn.MSELoss(reduction=reduction)
#     elif loss_type == "cross_entropy":
#         return nn.CrossEntropyLoss(reduction=reduction)
#     elif loss_type == "diffusion":
#         return DiffusionLoss(reduction=reduction)
#     else:
#         raise ValueError(f"Unsupported loss_type: {loss_type}")


# def get_optimizer(params, opt_type="adamw", lr=1e-3, weight_decay=0.01, **kwargs) -> torch.optim.Optimizer:
#     opt_type = opt_type.lower()

#     if opt_type == "sgd":
#         optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     elif opt_type == "adam":
#         optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     elif opt_type == "adamw":
#         optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     else:
#         raise ValueError(f"Unsupported optimizer type: {opt_type}")

#     return optimizer
# import torch
# import torch.nn as nn
# from torch import optim


# class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         max_iter,
#         min_lr=0.0,
#         warmup_ratio=0.1,
#         last_epoch=-1,
#     ):
#         if int(max_iter) <= 0:
#             raise ValueError("max_iter must be positive")
#         if not 0 <= float(warmup_ratio) < 1:
#             raise ValueError("warmup_ratio must be in [0, 1)")
#         if float(min_lr) < 0:
#             raise ValueError("min_lr must be non-negative")
#         self.max_iter = int(max_iter)
#         self.min_lr = float(min_lr)
#         self.warmup_ratio = float(warmup_ratio)
#         self.warmup_iters = int(self.warmup_ratio * self.max_iter)
#         super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

#     def get_lr_factor(self):
#         step = min(max(int(self.last_epoch), 0), self.max_iter)
#         if self.warmup_iters > 0 and step < self.warmup_iters:
#             # _LRScheduler evaluates once during construction. Using step + 1
#             # avoids a zero-learning-rate first optimizer update.
#             return (step + 1) / self.warmup_iters
#         decay_steps = self.max_iter - self.warmup_iters
#         if decay_steps <= 0:
#             return 1.0
#         return max(0.0, (self.max_iter - step) / decay_steps)

#     def get_lr(self):
#         factor = self.get_lr_factor()
#         return [
#             self.min_lr + (base_lr - self.min_lr) * factor
#             for base_lr in self.base_lrs
#         ]


# class WarmupLinearConstantLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         max_iter,
#         min_lr=1e-8,
#         warmup_ratio=0.1,
#         last_epoch=-1,
#     ):
#         if int(max_iter) <= 0:
#             raise ValueError("max_iter must be positive")
#         if not 0 <= float(warmup_ratio) < 1:
#             raise ValueError("warmup_ratio must be in [0, 1)")
#         if float(min_lr) < 0:
#             raise ValueError("min_lr must be non-negative")
#         self.max_iter = int(max_iter)
#         self.min_lr = float(min_lr)
#         self.warmup_ratio = float(warmup_ratio)
#         self.warmup_iters = int(self.warmup_ratio * self.max_iter)
#         super(WarmupLinearConstantLR, self).__init__(optimizer, last_epoch)

#     def get_lr_factor(self):
#         step = max(int(self.last_epoch), 0)
#         if self.warmup_iters > 0 and step < self.warmup_iters:
#             return (step + 1) / self.warmup_iters
#         return 1.0

#     def get_lr(self):
#         factor = self.get_lr_factor()
#         return [
#             self.min_lr + (base_lr - self.min_lr) * factor
#             for base_lr in self.base_lrs
#         ]


# class DiffusionLoss(nn.Module):
#     def __init__(self, reduction: str = "mean"):
#         super().__init__()
#         assert reduction in ["mean", "sum", "none"], f"Invalid reduction type: {reduction}"
#         self.reduction = reduction

#     def forward(self, weighting: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss = weighting.float() * (model_pred.float() - target.float()) ** 2
#         loss = torch.mean(loss.reshape(target.shape[0], -1), dim=1)

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss


# def get_criterion(loss_type="l2", reduction="mean") -> nn.Module:
#     loss_type = loss_type.lower()

#     if loss_type == "l1":
#         return nn.L1Loss(reduction=reduction)
#     elif loss_type == "l2":
#         return nn.MSELoss(reduction=reduction)
#     elif loss_type == "cross_entropy":
#         return nn.CrossEntropyLoss(reduction=reduction)
#     elif loss_type == "diffusion":
#         return DiffusionLoss(reduction=reduction)
#     else:
#         raise ValueError(f"Unsupported loss_type: {loss_type}")


# def get_optimizer(params, opt_type="adamw", lr=1e-3, weight_decay=0.01, **kwargs) -> torch.optim.Optimizer:
#     opt_type = opt_type.lower()

#     if opt_type == "sgd":
#         optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     elif opt_type == "adam":
#         optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     elif opt_type == "adamw":
#         optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     else:
#         raise ValueError(f"Unsupported optimizer type: {opt_type}")

#     return optimizer


# import torch
# import torch.nn as nn
# from torch import optim


# class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         max_iter,
#         min_lr=0.0,
#         warmup_ratio=0.1,
#         last_epoch=-1,
#     ):
#         if int(max_iter) <= 0:
#             raise ValueError("max_iter must be positive")
#         if not 0 <= float(warmup_ratio) < 1:
#             raise ValueError("warmup_ratio must be in [0, 1)")
#         if float(min_lr) < 0:
#             raise ValueError("min_lr must be non-negative")
#         self.max_iter = int(max_iter)
#         self.min_lr = float(min_lr)
#         self.warmup_ratio = float(warmup_ratio)
#         self.warmup_iters = int(self.warmup_ratio * self.max_iter)
#         super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

#     def get_lr_factor(self):
#         step = min(max(int(self.last_epoch), 0), self.max_iter)
#         if self.warmup_iters > 0 and step < self.warmup_iters:
#             # _LRScheduler evaluates once during construction. Using step + 1
#             # avoids a zero-learning-rate first optimizer update.
#             return (step + 1) / self.warmup_iters
#         decay_steps = self.max_iter - self.warmup_iters
#         if decay_steps <= 0:
#             return 1.0
#         return max(0.0, (self.max_iter - step) / decay_steps)

#     def get_lr(self):
#         factor = self.get_lr_factor()
#         return [
#             self.min_lr + (base_lr - self.min_lr) * factor
#             for base_lr in self.base_lrs
#         ]


# class WarmupLinearConstantLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         max_iter,
#         min_lr=1e-8,
#         warmup_ratio=0.1,
#         last_epoch=-1,
#     ):
#         if int(max_iter) <= 0:
#             raise ValueError("max_iter must be positive")
#         if not 0 <= float(warmup_ratio) < 1:
#             raise ValueError("warmup_ratio must be in [0, 1)")
#         if float(min_lr) < 0:
#             raise ValueError("min_lr must be non-negative")
#         self.max_iter = int(max_iter)
#         self.min_lr = float(min_lr)
#         self.warmup_ratio = float(warmup_ratio)
#         self.warmup_iters = int(self.warmup_ratio * self.max_iter)
#         super(WarmupLinearConstantLR, self).__init__(optimizer, last_epoch)

#     def get_lr_factor(self):
#         step = max(int(self.last_epoch), 0)
#         if self.warmup_iters > 0 and step < self.warmup_iters:
#             return (step + 1) / self.warmup_iters
#         return 1.0

#     def get_lr(self):
#         factor = self.get_lr_factor()
#         return [
#             self.min_lr + (base_lr - self.min_lr) * factor
#             for base_lr in self.base_lrs
#         ]


# class DiffusionLoss(nn.Module):
#     def __init__(self, reduction: str = "mean"):
#         super().__init__()
#         assert reduction in ["mean", "sum", "none"], f"Invalid reduction type: {reduction}"
#         self.reduction = reduction

#     def forward(self, weighting: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss = weighting.float() * (model_pred.float() - target.float()) ** 2
#         loss = torch.mean(loss.reshape(target.shape[0], -1), dim=1)

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss


# def get_criterion(loss_type="l2", reduction="mean") -> nn.Module:
#     loss_type = loss_type.lower()

#     if loss_type == "l1":
#         return nn.L1Loss(reduction=reduction)
#     elif loss_type == "l2":
#         return nn.MSELoss(reduction=reduction)
#     elif loss_type == "cross_entropy":
#         return nn.CrossEntropyLoss(reduction=reduction)
#     elif loss_type == "diffusion":
#         return DiffusionLoss(reduction=reduction)
#     else:
#         raise ValueError(f"Unsupported loss_type: {loss_type}")


# def get_optimizer(params, opt_type="adamw", lr=1e-3, weight_decay=0.01, **kwargs) -> torch.optim.Optimizer:
#     opt_type = opt_type.lower()

#     if opt_type == "sgd":
#         optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     elif opt_type == "adam":
#         optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     elif opt_type == "adamw":
#         optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     else:
#         raise ValueError(f"Unsupported optimizer type: {opt_type}")

#     return optimizer


# import torch
# import torch.nn as nn
# from torch import optim


# class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         max_iter,
#         min_lr=0.0,
#         warmup_ratio=0.1,
#         last_epoch=-1,
#     ):
#         if int(max_iter) <= 0:
#             raise ValueError("max_iter must be positive")
#         if not 0 <= float(warmup_ratio) < 1:
#             raise ValueError("warmup_ratio must be in [0, 1)")
#         if float(min_lr) < 0:
#             raise ValueError("min_lr must be non-negative")
#         self.max_iter = int(max_iter)
#         self.min_lr = float(min_lr)
#         self.warmup_ratio = float(warmup_ratio)
#         self.warmup_iters = int(self.warmup_ratio * self.max_iter)
#         super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

#     def get_lr_factor(self):
#         step = min(max(int(self.last_epoch), 0), self.max_iter)
#         if self.warmup_iters > 0 and step < self.warmup_iters:
#             # _LRScheduler evaluates once during construction. Using step + 1
#             # avoids a zero-learning-rate first optimizer update.
#             return (step + 1) / self.warmup_iters
#         decay_steps = self.max_iter - self.warmup_iters
#         if decay_steps <= 0:
#             return 1.0
#         return max(0.0, (self.max_iter - step) / decay_steps)

#     def get_lr(self):
#         factor = self.get_lr_factor()
#         return [
#             self.min_lr + (base_lr - self.min_lr) * factor
#             for base_lr in self.base_lrs
#         ]


# class WarmupLinearConstantLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         max_iter,
#         min_lr=1e-8,
#         warmup_ratio=0.1,
#         last_epoch=-1,
#     ):
#         if int(max_iter) <= 0:
#             raise ValueError("max_iter must be positive")
#         if not 0 <= float(warmup_ratio) < 1:
#             raise ValueError("warmup_ratio must be in [0, 1)")
#         if float(min_lr) < 0:
#             raise ValueError("min_lr must be non-negative")
#         self.max_iter = int(max_iter)
#         self.min_lr = float(min_lr)
#         self.warmup_ratio = float(warmup_ratio)
#         self.warmup_iters = int(self.warmup_ratio * self.max_iter)
#         super(WarmupLinearConstantLR, self).__init__(optimizer, last_epoch)

#     def get_lr_factor(self):
#         warmup_step = self.warmup_iters
#         step = self.last_epoch
#         if step < warmup_step:
#             return max(0, step / warmup_step)
#         elif step >= warmup_step:
#             return 1.0  # constant base lr

#     def get_lr(self):
#         factor = self.get_lr_factor()
#         return [
#             self.min_lr + (base_lr - self.min_lr) * factor
#             for base_lr in self.base_lrs
#         ]


# class DiffusionLoss(nn.Module):
#     def __init__(self, reduction: str = "mean"):
#         super().__init__()
#         assert reduction in ["mean", "sum", "none"], f"Invalid reduction type: {reduction}"
#         self.reduction = reduction

#     def forward(self, weighting: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss = weighting.float() * (model_pred.float() - target.float()) ** 2
#         loss = torch.mean(loss.reshape(target.shape[0], -1), dim=1)

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss


# def get_criterion(loss_type="l2", reduction="mean") -> nn.Module:
#     loss_type = loss_type.lower()

#     if loss_type == "l1":
#         return nn.L1Loss(reduction=reduction)
#     elif loss_type == "l2":
#         return nn.MSELoss(reduction=reduction)
#     elif loss_type == "cross_entropy":
#         return nn.CrossEntropyLoss(reduction=reduction)
#     elif loss_type == "diffusion":
#         return DiffusionLoss(reduction=reduction)
#     else:
#         raise ValueError(f"Unsupported loss_type: {loss_type}")


# def get_optimizer(params, opt_type="adamw", lr=1e-3, weight_decay=0.01, **kwargs) -> torch.optim.Optimizer:
#     opt_type = opt_type.lower()

#     if opt_type == "sgd":
#         optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     elif opt_type == "adam":
#         optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     elif opt_type == "adamw":
#         optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
#     else:
#         raise ValueError(f"Unsupported optimizer type: {opt_type}")

#     return optimizer

import torch
import torch.nn as nn
from torch import optim


class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=3e-5,
        warmup_ratio=0.1,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = int(warmup_ratio * max_iter)
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr_factor(self):
        tot_step = self.max_iter
        warmup_step = self.warmup_iters
        step = self.last_epoch
        if step < warmup_step:
            return max(0, step / warmup_step)
        elif step > tot_step:
            step = tot_step
        return max(0, (tot_step - step) / (tot_step - warmup_step))

    def get_lr(self):
        warmup_factor = self.get_lr_factor()
        return [max(self.min_lr, base_lr * warmup_factor) for base_lr in self.base_lrs]


class WarmupLinearConstantLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=1e-8,
        warmup_ratio=0.1,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = int(warmup_ratio * max_iter)
        super(WarmupLinearConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr_factor(self):
        warmup_step = self.warmup_iters
        step = self.last_epoch
        if step < warmup_step:
            return max(0, step / warmup_step)
        elif step >= warmup_step:
            return 1.0  # constant base lr

    def get_lr(self):
        warmup_factor = self.get_lr_factor()
        return [max(self.min_lr, base_lr * warmup_factor) for base_lr in self.base_lrs]


class DiffusionLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ["mean", "sum", "none"], f"Invalid reduction type: {reduction}"
        self.reduction = reduction

    def forward(self, weighting: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = weighting.float() * (model_pred.float() - target.float()) ** 2
        loss = torch.mean(loss.reshape(target.shape[0], -1), dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_criterion(loss_type="l2", reduction="mean") -> nn.Module:
    loss_type = loss_type.lower()

    if loss_type == "l1":
        return nn.L1Loss(reduction=reduction)
    elif loss_type == "l2":
        return nn.MSELoss(reduction=reduction)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=reduction)
    elif loss_type == "diffusion":
        return DiffusionLoss(reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def get_optimizer(params, opt_type="adamw", lr=1e-3, weight_decay=0.01, **kwargs) -> torch.optim.Optimizer:
    opt_type = opt_type.lower()

    if opt_type == "sgd":
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif opt_type == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif opt_type == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    return optimizer
