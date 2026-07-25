import time
from contextlib import nullcontext

import torch

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from moco.loss import KDInfoNCELoss
from utils import utils


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    args,
    teacher_dims_dict,
    start_step=0,
):
    """Run the configured step budget, cycling over shards as necessary."""
    batch_time = utils.AverageMeter("Time", ":6.3f")
    data_time = utils.AverageMeter("Data", ":6.3f")
    learning_rates = utils.AverageMeter("LR", ":.4e")
    losses_meter = utils.AverageMeter("Loss", ":.4e")
    wsis_meter = utils.ValueMeter("WSIs", ":d")

    progress = utils.ProgressMeter(
        [batch_time, data_time, learning_rates, losses_meter, wsis_meter],
        prefix="Step",
    )

    amp_enabled = bool(args.amp)
    amp_dtype = args.amp_dtype
    use_grad_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    model.train()
    kd_criterion = KDInfoNCELoss(temperature=args.moco_t).cuda(args.gpu)

    end = time.time()
    step_idx = start_step
    total_steps = args.train_steps
    loader_iter = iter(train_loader)

    while step_idx < total_steps:
        try:
            im_q1, im_q2, feats_dict, metadata = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            try:
                im_q1, im_q2, feats_dict, metadata = next(loader_iter)
            except StopIteration as exc:
                raise RuntimeError("The WebDataset pipeline produced no batches.") from exc

        missing_teachers = set(teacher_dims_dict) - set(feats_dict)
        if missing_teachers:
            raise KeyError(
                "The current batch is missing teacher features for: "
                + ", ".join(sorted(missing_teachers))
            )

        global_num_wsis = utils.gather_global_num_wsis(metadata["wsis"])
        wsis_meter.update(global_num_wsis)
        data_time.update(time.time() - end)

        lr = utils.adjust_learning_rate_steps(optimizer, step_idx, args)
        learning_rates.update(lr)

        im_q1 = im_q1.cuda(args.gpu, non_blocking=True)
        im_q2 = im_q2.cuda(args.gpu, non_blocking=True)
        teacher_features_gpu = {
            name: feats_dict[name].cuda(args.gpu, non_blocking=True).float().detach()
            for name in teacher_dims_dict
        }

        gathered = {
            teacher_name: kd_criterion.gather_keys(features)
            for teacher_name, features in teacher_features_gpu.items()
        }

        optimizer.zero_grad(set_to_none=True)
        log_dict = {}

        no_sync = model.no_sync if hasattr(model, "no_sync") else nullcontext
        with no_sync():
            with torch.amp.autocast(
                "cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output_view1 = model(im_q1)
                view1_losses = []
                for teacher_name, (keys, labels) in gathered.items():
                    loss = kd_criterion.forward_single(
                        output_view1[teacher_name],
                        keys,
                        labels,
                    )
                    view1_losses.append(loss)
                    log_dict[f"train/loss_{teacher_name}_view1"] = loss.item()
                loss1 = torch.stack(view1_losses).sum()
            scaler.scale(loss1).backward()

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output_view2 = model(im_q2)
            view2_losses = []
            for teacher_name, (keys, labels) in gathered.items():
                loss = kd_criterion.forward_single(
                    output_view2[teacher_name],
                    keys,
                    labels,
                )
                view2_losses.append(loss)
                log_dict[f"train/loss_{teacher_name}_view2"] = loss.item()
            loss2 = torch.stack(view2_losses).sum()
        scaler.scale(loss2).backward()

        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

        scaler.step(optimizer)
        scaler.update()

        total_loss = loss1.item() + loss2.item()
        losses_meter.update(total_loss, im_q1.size(0))

        if args.rank == 0 and getattr(args, "_wandb_ok", False):
            if wandb is None:
                raise RuntimeError("W&B logging was enabled but wandb is unavailable.")
            wandb_log = {
                "step": step_idx,
                "train/loss_total": total_loss,
                "train/lr": float(lr),
                "timing/data_sec": data_time.val,
                "data/global_num_wsis": global_num_wsis,
            }
            wandb_log.update(log_dict)
            wandb.log(wandb_log, step=step_idx)

        batch_time.update(time.time() - end)
        end = time.time()

        if step_idx % args.print_freq == 0 and args.rank == 0:
            progress.display(step_idx)

        if (
            args.rank == 0
            and step_idx > start_step
            and step_idx % args.save_every == 0
        ):
            utils.save_checkpoint(model, optimizer, step_idx, args)

        step_idx += 1

    if args.rank == 0:
        last_completed_step = max(start_step - 1, step_idx - 1)
        utils.save_checkpoint(model, optimizer, last_completed_step, args)
        progress.display_summary()
