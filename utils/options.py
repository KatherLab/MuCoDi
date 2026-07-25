import argparse


PAPER_ARCHITECTURES = (
    "repvit_m0_6",
    "repvit_m0_9",
    "repvit_m1_0",
    "repvit_m1_1",
    "repvit_m1_5",
    "repvit_m2_3",
    "mobileone_mu0",
    "mobileone_mu1",
    "mobileone_mu2",
    "mobileone_s0",
    "mobileone_s1",
    "mobileone_s2",
    "mobileone_s3",
    "mobileone_s4",
)


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="MuCoDi multi-teacher contrastive distillation pretraining"
    )

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="repvit_m1_5",
        help=(
            "student backbone (paper architectures: "
            + ", ".join(PAPER_ARCHITECTURES)
            + "; other timm models are also supported)"
        ),
    )
    model_group.add_argument(
        "--teacher-dims",
        type=str,
        default="",
        help=(
            "comma-separated teacher dimensions, for example "
            "'virchow2:1280,uni2:1536' (default: selected dataset profile)"
        ),
    )
    model_group.add_argument(
        "--dataset",
        type=str,
        default="default",
        help="dataset profile name from dataprep/datasets.yaml",
    )
    model_group.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="number of layers in each teacher-specific projection head",
    )
    model_group.add_argument(
        "--moco-t",
        default=0.2,
        type=float,
        help="softmax temperature (default: 0.2)",
    )

    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument(
        "--lr",
        "--learning-rate",
        default=0.02,
        type=float,
        metavar="LR",
        help="initial base learning rate (default: 0.02)",
        dest="lr",
    )
    opt_group.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-6,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-6)",
        dest="weight_decay",
    )
    opt_group.add_argument(
        "--train-steps",
        type=int,
        default=100000,
        help="total optimizer-step budget",
    )
    opt_group.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="manual starting step (normally inferred from --resume)",
    )
    opt_group.add_argument(
        "--warmup-steps",
        type=int,
        default=5000,
        help="number of linear warmup steps (approximately 5%% of --train-steps)",
    )
    opt_group.add_argument(
        "--clip-grad",
        default=1.0,
        type=float,
        help="gradient clipping maximum norm; set to 0 to disable (default: 1.0)",
    )
    opt_group.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatic mixed precision (enabled by default; disable with --no-amp)",
    )
    opt_group.add_argument(
        "--amp-dtype",
        default="bfloat16",
        type=str,
        choices=["bfloat16", "float16"],
        help="AMP dtype (default: bfloat16)",
    )

    sys_group = parser.add_argument_group("System & Data")
    sys_group.add_argument(
        "-j",
        "--workers",
        default=32,
        type=int,
        metavar="N",
        help=(
            "data-loading workers per node for --multiprocessing-distributed "
            "(divided across local GPUs), or per process otherwise (default: 32)"
        ),
    )
    sys_group.add_argument(
        "-b",
        "--batch-size",
        default=2048,
        type=int,
        metavar="N",
        help="global mini-batch size across all GPUs (default: 2048)",
    )
    sys_group.add_argument(
        "--crop-min",
        default=0.08,
        type=float,
        help="minimum random-resized-crop scale (default: 0.08)",
    )
    sys_group.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="checkpoint to resume from",
    )
    sys_group.add_argument(
        "--save-every",
        type=int,
        default=5000,
        help="save a checkpoint every N optimizer steps",
    )
    sys_group.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="checkpoint output directory",
    )
    sys_group.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="logging interval in optimizer steps (default: 10)",
    )
    sys_group.add_argument(
        "--seed",
        default=None,
        type=int,
        help="random seed",
    )
    sys_group.add_argument(
        "--shard-count",
        default=None,
        type=int,
        help="deterministically select only N training shards",
    )
    sys_group.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="GPU id for a non-spawned single-process run",
    )
    sys_group.add_argument(
        "--wandb",
        action="store_true",
        help="enable Weights & Biases logging (disabled by default)",
    )

    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes (default: 1)",
    )
    dist_group.add_argument(
        "--rank",
        default=0,
        type=int,
        help="node rank (default: 0)",
    )
    dist_group.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:10001",
        type=str,
        help="distributed initialization URL",
    )
    dist_group.add_argument(
        "--dist-backend",
        default="nccl",
        type=str,
        help="distributed backend (default: nccl)",
    )
    dist_group.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="launch one training process per local GPU",
    )

    return parser


def parse_teacher_dims(dims_str):
    """Parse ``name:dim,...`` into a dictionary."""
    dims = {}
    for pair in dims_str.split(","):
        if not pair.strip():
            continue
        try:
            key, value = pair.split(":", maxsplit=1)
            key = key.strip()
            if not key:
                raise ValueError
            dimension = int(value)
            if dimension <= 0:
                raise ValueError
            if key in dims:
                raise ValueError
            dims[key] = dimension
        except ValueError as exc:
            raise ValueError(
                f"Invalid teacher specification '{pair}'. Expected a unique name:positive_dimension pair."
            ) from exc
    if not dims:
        raise ValueError("At least one teacher dimension must be configured.")
    return dims
