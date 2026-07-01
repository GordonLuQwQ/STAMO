import argparse

from omegaconf import OmegaConf

from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default=None, required=True, type=str)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", default=None, type=int)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    config.world_size = overwatch.world_size()
    config.local_rank = args.local_rank if args.local_rank is not None else overwatch.local_rank()
    config.deepspeed = args.deepspeed

    if config.world_size > 1 or args.deepspeed or bool(config.get("fabric", False)):
        config.dist = True

    return config
