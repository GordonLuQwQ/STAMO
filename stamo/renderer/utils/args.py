# import argparse

# from omegaconf import OmegaConf

# from stamo.renderer.utils.overwatch import initialize_overwatch


# overwatch = initialize_overwatch(__name__)


# def init_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--config_path", default=None, required=True, type=str)
#     parser.add_argument("--deepspeed", action="store_true")

#     args = parser.parse_args()

#     config = OmegaConf.load(args.config_path)

#     config.world_size = overwatch.world_size()
#     config.local_rank = overwatch.local_rank()
#     config.deepspeed = args.deepspeed

#     if args.deepspeed or config.fabric:
#         config.dist = True

#     return config
# import argparse
# import os

# import torch.distributed as dist
# from omegaconf import OmegaConf


# def init_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--config_path", default=None, required=True, type=str)
#     parser.add_argument("--deepspeed", action="store_true")
#     parser.add_argument(
#         "--local_rank",
#         type=int,
#         default=int(os.environ.get("LOCAL_RANK", -1)),
#         help="Local process rank supplied by the DeepSpeed launcher.",
#     )

#     args = parser.parse_args()

#     config = OmegaConf.load(args.config_path)

#     config.world_size = dist.get_world_size() if dist.is_initialized() else 1
#     config.local_rank = (
#         int(args.local_rank)
#         if int(args.local_rank) >= 0
#         else int(os.environ.get("LOCAL_RANK", 0))
#     )
#     config.deepspeed = args.deepspeed

#     if args.deepspeed or config.fabric:
#         config.dist = True

#     return config



import argparse
import os

import torch.distributed as dist
from omegaconf import OmegaConf


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default=None, required=True, type=str)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
        help="Local process rank supplied by the DeepSpeed launcher.",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    config.world_size = dist.get_world_size() if dist.is_initialized() else 1
    config.local_rank = (
        int(args.local_rank)
        if int(args.local_rank) >= 0
        else int(os.environ.get("LOCAL_RANK", 0))
    )
    config.deepspeed = args.deepspeed

    if args.deepspeed or config.fabric:
        config.dist = True

    return config