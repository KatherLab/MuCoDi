import math
import os

import torch
import torch.distributed as dist


class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def display(self, step_idx):
        entries = [f"{self.prefix} [{step_idx}]"] + [str(meter) for meter in self.meters]
        print("\t".join(entries), flush=True)

    def display_summary(self):
        entries = ["Summary:"] + [meter.summary() for meter in self.meters]
        print(" ".join(entries), flush=True)


class ValueMeter:
    def __init__(self, name, fmt=":d"):
        self.name, self.fmt, self.val = name, fmt, 0

    def update(self, value):
        self.val = int(value)

    def __str__(self):
        return f"{self.name} {self.val}"

    def summary(self):
        return f"{self.name} {self.val}"


def gather_global_num_wsis(local_wsis):
    if not (dist.is_available() and dist.is_initialized()):
        return len(set(local_wsis))

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, list(local_wsis))
    return len({wsi for rank_wsis in gathered for wsi in rank_wsis})


def adjust_learning_rate_steps(optimizer, step, args):
    warmup_len = args.warmup_steps
    if warmup_len > 0 and step < warmup_len:
        lr = args.lr * (step / warmup_len)
    else:
        current_step = step - warmup_len
        decay_steps = max(1, args.train_steps - warmup_len)
        progress = min(1.0, max(0.0, current_step / decay_steps))
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(model, optimizer, step_idx, args):
    target_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        "step": step_idx,
        "arch": args.arch,
        "state_dict": target_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "teacher_dims": args.teacher_dims,
            "num_layers": args.num_layers,
            "moco_t": args.moco_t,
            "crop_min": args.crop_min,
            "amp": args.amp,
            "amp_dtype": str(args.amp_dtype),
        },
    }

    os.makedirs(args.save_dir, exist_ok=True)
    filename = os.path.join(args.save_dir, f"step_{step_idx:07d}.pth.tar")
    torch.save(checkpoint, filename)
    print(f"[checkpoint] saved {filename}", flush=True)
