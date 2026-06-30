import random

import numpy as np
import torch

from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.trainer import Trainer
from stamo.renderer.utils.args import init_args
from stamo.renderer.utils.data import get_loader_info, load_multi_datasets_form_json
from stamo.renderer.utils.optim import WarmupLinearConstantLR, WarmupLinearLR, get_criterion, get_optimizer
from stamo.renderer.utils.overwatch import initialize_overwatch


torch.multiprocessing.set_sharing_strategy("file_system")

overwatch = initialize_overwatch(__name__)


def main(args):
    try:
        import torch_musa  # noqa: F401  # edit for musa
    except ImportError:
        pass

    random.seed(args.seed)  # edit for musa
    np.random.seed(args.seed)  # edit for musa
    torch.manual_seed(args.seed)  # edit for musa
    if hasattr(torch, "musa") and torch.musa.is_available():  # edit for musa
        torch.musa.manual_seed_all(args.seed)  # edit for musa
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # init models
    overwatch.info("Building models...")
    model = RenderNet(args)
    if args.do_train:
        overwatch.warning("Do training...")

        optimizer = get_optimizer(
            (p for p in model.parameters() if p.requires_grad),
            opt_type="AdamW",
            lr=args.train.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=args.train.decay,
        )

        criterion = get_criterion(
            loss_type="diffusion",
            reduction="mean",
        )

        if args.train.constant_lr:
            scheduler = WarmupLinearConstantLR(
                optimizer,
                max_iter=(args.train.num_iters // args.train.gradient_accumulate_steps) + 1,
                warmup_ratio=getattr(args, "warmup_ratio", 0.00001),
            )
        else:
            scheduler = WarmupLinearLR(
                optimizer,
                max_iter=10000,
                warmup_ratio=getattr(args, "warmup_ratio", 0.00001),
            )

        trainer = Trainer(args, model, criterion, optimizer, scheduler)
        trainer.setup_model_for_training()

        train_dataloader = load_multi_datasets_form_json(
            args.data.train_json_path,
            flip_p=args.data.flip_p,
            img_size=args.data.img_size,
            local_batch_size=args.train.local_batch_size,
            num_workers=args.data.num_workers,
            is_infinite=True,
            shuffle=True,
            make_single_dataset=True,
        )

        eval_dataloader = load_multi_datasets_form_json(
            args.data.eval_json_path,
            flip_p=0,
            img_size=args.data.img_size,
            local_batch_size=args.train.local_batch_size,
            num_workers=args.data.num_workers,
            is_infinite=False,
            shuffle=False,
            drop_last=False,
            make_single_dataset=True,
        )

        train_info = get_loader_info(train_dataloader, args.train.epochs, args.train.local_batch_size)
        _, images_per_batch, args.train.iter_per_ep, args.train.num_iters = train_info
        trainer.iter_per_ep = args.train.iter_per_ep
        trainer.num_iters = args.train.num_iters
        if not isinstance(trainer.global_step, int):
            trainer.global_step = 30000
        overwatch.info(f"Total batch size {images_per_batch}")
        overwatch.info(f"Total training steps {args.train.num_iters}")
        overwatch.info(f"Starting train iter: {trainer.global_step + 1}")
        overwatch.info(f"Training steps per epoch (accumulated) {args.train.iter_per_ep}")
        overwatch.info(f"Training dataloader length {len(train_dataloader)}")
        overwatch.info(f"Evaluation happens every {args.train.eval_step} steps")
        overwatch.info(f"Checkpoint saves every {args.train.save_step} steps")

        trainer.train_eval_by_iter(train_loader=train_dataloader, eval_loader=eval_dataloader)


if __name__ == "__main__":
    config = init_args()
    main(config)
