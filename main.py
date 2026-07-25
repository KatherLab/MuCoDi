#!/usr/bin/env python
from __future__ import annotations

import builtins
import datetime as dt
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from dataprep import loader
from dataprep.config import get_teacher_dims_str, load_profile
from models.factory import build_backbone
from models.student import MultiTeacherStudent
from utils import engine, options

warnings.filterwarnings("ignore", message="Overwriting .* in registry")


def init_wandb(args, run_name: str) -> None:
    if wandb is None:
        raise RuntimeError("Weights & Biases is not installed. Run `uv sync` first.")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mucodi"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name,
        resume="allow",
        config=vars(args),
        settings=wandb.Settings(quiet=True),
    )


def validate_args(args, ngpus_per_node: int) -> None:
    if not torch.cuda.is_available() or ngpus_per_node == 0:
        raise RuntimeError("MuCoDi pretraining requires at least one CUDA-capable GPU.")
    if args.world_size < 1:
        raise ValueError("--world-size must be at least 1.")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world-size.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.train_steps < 1:
        raise ValueError("--train-steps must be positive.")
    if args.warmup_steps < 0 or args.warmup_steps >= args.train_steps:
        raise ValueError("--warmup-steps must satisfy 0 <= warmup-steps < train-steps.")
    if args.save_every < 1:
        raise ValueError("--save-every must be positive.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    if args.print_freq < 1:
        raise ValueError("--print-freq must be positive.")
    if args.num_layers < 1:
        raise ValueError("--num-layers must be at least 1.")
    if args.moco_t <= 0:
        raise ValueError("--moco-t must be positive.")
    if not 0 < args.crop_min <= 1:
        raise ValueError("--crop-min must satisfy 0 < crop-min <= 1.")
    if args.start_step < 0:
        raise ValueError("--start-step cannot be negative.")

    total_processes = args.world_size * ngpus_per_node if args.multiprocessing_distributed else args.world_size
    if args.batch_size % total_processes != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by the total "
            f"number of processes ({total_processes})."
        )


def main() -> None:
    parser = options.get_args_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    profile = load_profile(args.dataset)
    args.shard_dir = profile["shard_dir"]
    if not args.teacher_dims:
        args.teacher_dims = get_teacher_dims_str(profile)
    options.parse_teacher_dims(args.teacher_dims)

    ngpus_per_node = torch.cuda.device_count()
    validate_args(args, ngpus_per_node)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        print(
            f"Launching {ngpus_per_node} processes per node for "
            f"{args.world_size} total processes."
        )
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if args.gpu is None:
            args.gpu = 0
        if not args.distributed:
            args.rank = 0
            args.world_size = 1
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu: int, ngpus_per_node: int, args) -> None:
    args.gpu = gpu

    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu

    if args.seed is not None:
        process_seed = args.seed + args.rank
        random.seed(process_seed)
        torch.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)
        cudnn.deterministic = True

    if args.rank != 0:
        builtins.print = lambda *args, **kwargs: None

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            device_id=torch.device(f"cuda:{args.gpu}"),
            timeout=dt.timedelta(minutes=30),
        )
        dist.barrier()
    else:
        torch.cuda.set_device(args.gpu)

    if args.batch_size % args.world_size != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by world size {args.world_size}."
        )

    global_batch_size = args.batch_size
    local_batch_size = global_batch_size // args.world_size
    args.batch_size = local_batch_size
    if args.multiprocessing_distributed:
        args.workers = (args.workers + ngpus_per_node - 1) // ngpus_per_node

    print(
        f"Learning rate: {args.lr}; global batch size: {global_batch_size}; "
        f"local batch size: {local_batch_size}"
    )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    args.amp_dtype = dtype_map[args.amp_dtype]
    if args.amp:
        print(f"AMP enabled with dtype={args.amp_dtype}")

    stamp = dt.datetime.now().strftime("%y%m%d-%H%M")
    run_name = os.environ.get("WANDB_NAME") or f"{args.arch}|{stamp}"

    if args.rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    backbone = build_backbone(args.arch, num_classes=0)

    # Remove a trailing post-BN activation that would clip feature values.
    for attr in ("norm_head", "bn2"):
        head = getattr(backbone, attr, None)
        if head is not None and hasattr(head, "act") and not isinstance(head.act, nn.Identity):
            head.act = nn.Identity()

    if os.environ.get("MUCODI_GRAD_CKPT", "").lower() in ("1", "true", "yes"):
        if hasattr(backbone, "grad_checkpointing"):
            backbone.grad_checkpointing = True
            print(f"[grad-checkpointing] enabled on {args.arch}")
        else:
            print(f"[grad-checkpointing] {args.arch} does not support this option")

    print(
        f"\n[Backbone] {args.arch} with "
        f"{sum(p.numel() for p in backbone.parameters()):,} parameters\n"
    )

    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        output = backbone(torch.randn(2, 3, 224, 224))
    backbone.train(was_training)
    if output.ndim != 2:
        raise ValueError(
            f"Backbone '{args.arch}' returned shape {tuple(output.shape)}; "
            "MuCoDi requires a pooled [batch, feature_dim] tensor."
        )
    student_dim = output.shape[1]

    teacher_dims_dict = options.parse_teacher_dims(args.teacher_dims)
    model = MultiTeacherStudent(
        backbone,
        feat_dim=student_dim,
        teacher_dims=teacher_dims_dict,
        hidden=student_dim,
        num_layers=args.num_layers,
    )

    args._wandb_ok = False
    if args.rank == 0 and args.wandb:
        try:
            init_wandb(args, run_name)
            wandb.watch(model, log="all", log_freq=100)
            args._wandb_ok = True
        except Exception as exc:
            print(
                f"[wandb] initialization failed "
                f"({type(exc).__name__}: {exc}); continuing without W&B"
            )

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
        args.start_step = int(checkpoint["step"]) + 1
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded checkpoint '{args.resume}' at step {checkpoint['step']}")

    cudnn.benchmark = args.seed is None
    train_loader = loader.get_wds_loader(args)

    engine.train_one_epoch(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        args=args,
        teacher_dims_dict=teacher_dims_dict,
        start_step=args.start_step,
    )

    if args.rank == 0 and args._wandb_ok:
        wandb.finish(quiet=True, exit_code=0)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
