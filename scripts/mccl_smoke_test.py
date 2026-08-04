"""MCCL collectives used by FLUX ZeRO-3, independent of model code."""

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

try:
    import torch_musa  # noqa: F401
except ImportError as exc:  # pragma: no cover - runs only on the MUSA host
    raise RuntimeError("torch_musa is required for this smoke test") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--numel",
        type=int,
        default=50_000_000,
        help="Global BF16 bucket size in elements; must divide world size.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=None,
        help="Fail if the launcher did not start the requested number of ranks.",
    )
    parser.add_argument(
        "--trace-operations",
        action="store_true",
        help="Print rank-0 begin/end markers around every collective.",
    )
    parser.add_argument("--local_rank", type=int, default=None)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    local_rank = (
        int(args.local_rank)
        if args.local_rank is not None
        else int(os.environ.get("LOCAL_RANK", 0))
    )
    if args.iterations <= 0 or args.numel <= 0 or args.log_interval <= 0:
        raise ValueError("iterations, numel, and log-interval must be positive")
    if args.expected_world_size is not None and args.expected_world_size <= 0:
        raise ValueError("expected-world-size must be positive")
    if args.timeout_seconds < 60:
        raise ValueError("timeout-seconds must be at least 60")

    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available():
        raise RuntimeError("MUSA is not available")
    musa.set_device(local_rank)
    device = torch.device("musa", local_rank)

    dist.init_process_group(
        backend="mccl",
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if (
        args.expected_world_size is not None
        and world_size != args.expected_world_size
    ):
        raise RuntimeError(
            f"expected world_size={args.expected_world_size}, got {world_size}"
        )
    if args.numel % world_size != 0:
        raise ValueError(
            f"numel={args.numel} must be divisible by world_size={world_size}"
        )
    local_numel = args.numel // world_size
    expected_sum = world_size * (world_size + 1) / 2
    reduce_payload = torch.empty(
        args.numel,
        device=device,
        dtype=torch.bfloat16,
    )
    gather_input = torch.empty(
        local_numel,
        device=device,
        dtype=torch.bfloat16,
    )
    gather_output = torch.empty(
        args.numel,
        device=device,
        dtype=torch.bfloat16,
    )
    scatter_input = torch.empty_like(gather_output)
    scatter_output = torch.empty_like(gather_input)
    broadcast_payload = torch.empty_like(reduce_payload)
    started = time.perf_counter()

    def trace_operation(phase, operation, iteration):
        if args.trace_operations and rank == 0:
            print(
                f"MCCL_{phase} iteration={iteration + 1}/{args.iterations} "
                f"operation={operation}",
                flush=True,
            )

    try:
        for iteration in range(args.iterations):
            reduce_payload.fill_(rank + 1)
            trace_operation("BEGIN", "all_reduce", iteration)
            dist.all_reduce(reduce_payload, op=dist.ReduceOp.SUM)
            trace_operation("END", "all_reduce", iteration)

            gather_input.fill_(rank + 1)
            trace_operation("BEGIN", "all_gather_into_tensor", iteration)
            dist.all_gather_into_tensor(gather_output, gather_input)
            trace_operation("END", "all_gather_into_tensor", iteration)

            scatter_input.fill_(rank + 1)
            trace_operation("BEGIN", "reduce_scatter_tensor", iteration)
            dist.reduce_scatter_tensor(
                scatter_output,
                scatter_input,
                op=dist.ReduceOp.SUM,
            )
            trace_operation("END", "reduce_scatter_tensor", iteration)

            expected_broadcast = float(iteration + 1)
            broadcast_payload.fill_(expected_broadcast if rank == 0 else -1)
            trace_operation("BEGIN", "broadcast", iteration)
            dist.broadcast(broadcast_payload, src=0)
            trace_operation("END", "broadcast", iteration)
            trace_operation("BEGIN", "musa_synchronize", iteration)
            musa.synchronize()
            trace_operation("END", "musa_synchronize", iteration)

            actual_sum = float(
                reduce_payload[0].float().cpu().item()
            )
            actual_scatter = float(
                scatter_output[0].float().cpu().item()
            )
            actual_broadcast_first = float(
                broadcast_payload[0].float().cpu().item()
            )
            actual_broadcast_last = float(
                broadcast_payload[-1].float().cpu().item()
            )
            gathered_markers = [
                float(
                    gather_output[source_rank * local_numel]
                    .float()
                    .cpu()
                    .item()
                )
                for source_rank in range(world_size)
            ]
            expected_markers = [
                float(source_rank + 1)
                for source_rank in range(world_size)
            ]
            if (
                actual_sum != expected_sum
                or actual_scatter != expected_sum
                or gathered_markers != expected_markers
                or actual_broadcast_first != expected_broadcast
                or actual_broadcast_last != expected_broadcast
            ):
                raise RuntimeError(
                    f"rank={rank} iteration={iteration} collective corruption: "
                    f"all_reduce={actual_sum}, "
                    f"all_gather={gathered_markers}, "
                    f"reduce_scatter={actual_scatter}, "
                    f"broadcast_first={actual_broadcast_first}, "
                    f"broadcast_last={actual_broadcast_last}"
                )

            if iteration % args.log_interval == 0 or iteration + 1 == args.iterations:
                if rank == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"MCCL iteration {iteration + 1}/{args.iterations}, "
                        f"{(iteration + 1) / elapsed:.2f} iterations/s",
                        flush=True,
                    )
    finally:
        # A barrier here would hide a failed/lagging rank.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
