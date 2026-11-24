"""Phase 2 trainer with crash-safe resume, warm starts and rich logging."""

from __future__ import annotations

import copy
import csv
import hashlib
import math
import random
import signal
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.driving_dataset import DrivingSequenceDataset, driving_worker_init_fn
from src.data.labels import CAUSE_SITUATIONS
from src.eval.metrics import TURN_THRESHOLD
from src.models import VisionRWKVDriver
from src.training.checkpointing import CheckpointManager, capture_rng_state, restore_rng_state
from src.training.config import (
    DatasetConfig,
    SchedulerConfig,
    TrainingConfig,
    resume_conflicts,
    save_training_config,
    training_config_to_dict,
)
from src.training.plotting import plot_run_dashboard
from src.utils.device import describe_device, select_device
from src.utils.memory import loader_rss_mib
from src.utils.settings import setting

wandb: Any
try:
    import wandb as _wandb

    wandb = _wandb
except Exception:  # pragma: no cover - optional dependency at runtime
    wandb = None


def seed_everything(seed: int, deterministic: bool) -> None:
    """Set all RNG seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _format_float_tag(value: float) -> str:
    tag = f"{value:.0e}".replace("+", "")
    return tag.replace("e-0", "e-")


def build_run_name(cfg: TrainingConfig) -> str:
    """Build a parameter-rich run name for quick visual identification."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tcfg = cfg.model.tokenizer
    ecfg = cfg.model.encoder
    run = cfg.run
    dcfg = cfg.dataset
    ocfg = cfg.optimization
    return (
        f"{run.name_prefix}"
        f"_e{ecfg.n_layers}d{ecfg.embed_dim}"
        f"_s{dcfg.seq_len}b{dcfg.batch_size}"
        f"_lr{_format_float_tag(ocfg.learning_rate)}"
        f"_wd{_format_float_tag(ocfg.weight_decay)}"
        f"_res{tcfg.img_height}x{tcfg.img_width}"
        f"_seed{run.seed}_{ts}"
    )


def build_loader(dcfg: DatasetConfig, seed: int, split: str) -> DataLoader[dict[str, torch.Tensor]]:
    """The training loader (the dataset's sampler, augmentation) or the validation loader."""
    train = split == "train"
    ds = DrivingSequenceDataset(
        data_dir=dcfg.data_dir or setting("DATA_DIR"),
        seq_len=dcfg.seq_len,
        stride=dcfg.stride,
        fps=dcfg.fps,
        split=split,
        val_fraction=dcfg.val_fraction,
        seed=seed,
        augment=dcfg.augment_train and train,
        resize_height=dcfg.resize_height,
        resize_width=dcfg.resize_width,
    )
    workers = dcfg.num_workers if train else dcfg.num_workers // 2
    # Each worker prefetches 2 batches (the PyTorch default); 1 halves the loader's memory
    # at the cost of slightly lower GPU utilisation.
    return DataLoader(
        ds,
        batch_size=dcfg.batch_size,
        sampler=ds.get_sampler(weighted=dcfg.weighted_sampling) if train else None,
        num_workers=workers,
        pin_memory=dcfg.pin_memory,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=driving_worker_init_fn,
        drop_last=train,
    )


class NonFiniteTraining(RuntimeError):
    """Too many training steps in a row had a non-finite loss or gradient."""


class ControlError:
    """Mean absolute control error of moving and of turning frames, pooled over batches."""

    def __init__(self) -> None:
        self.moving_error, self.moving_frames = 0.0, 0
        self.turning_error, self.turning_frames = 0.0, 0

    def add(self, pred: torch.Tensor, target: torch.Tensor, frame_weight: torch.Tensor) -> None:
        """Score a batch; stationary, burn-in and excluded frames do not count."""
        error = (pred.float() - target).abs().mean(dim=-1)
        moving = frame_weight > DrivingSequenceDataset.STATIONARY_LOSS_WEIGHT
        turning = moving & (target[..., 0].abs() > TURN_THRESHOLD)
        self.moving_error += float(error[moving].sum())
        self.moving_frames += int(moving.sum())
        self.turning_error += float(error[turning].sum())
        self.turning_frames += int(turning.sum())

    def metrics(self) -> dict[str, float]:
        """Both errors and their mean, the score that selects checkpoints.

        Without turning frames the score is the moving error; without moving frames it is NaN.
        """
        nan = float("nan")
        moving = self.moving_error / self.moving_frames if self.moving_frames else nan
        turning = self.turning_error / self.turning_frames if self.turning_frames else nan
        return {
            "moving_control_mae": moving,
            "turn_control_mae": turning,
            "control_score": 0.5 * (moving + turning) if self.turning_frames else moving,
        }


def token_spread(tokens: torch.Tensor) -> float:
    """How much frame tokens vary across frames, relative to their size.

    Near zero when every frame gets the same token, i.e. the model no longer sees the images.
    """
    frames = tokens.detach().float().flatten(0, -2)
    return float(frames.std(dim=0).mean() / frames.abs().mean())


def learning_rate_factor(step: int, total_steps: int, cfg: SchedulerConfig) -> float:
    """Share of the configured learning rate for optimizer step `step` (counted from 0).

    The rate comes from the config and the step alone, so a resumed run uses what its config says.
    """
    if cfg.name == "constant":
        return 1.0
    warmup = max(0, cfg.warmup_steps)
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class StepTimes:
    """Mean seconds per step spent waiting for data and training, since the last report."""

    def __init__(self) -> None:
        self.waiting, self.training, self.steps = 0.0, 0.0, 0

    def add(self, waiting: float, training: float) -> None:
        self.waiting += waiting
        self.training += training
        self.steps += 1

    def report(self) -> dict[str, float]:
        steps = max(self.steps, 1)
        times = {"data_wait_s": self.waiting / steps, "train_step_s": self.training / steps}
        self.waiting, self.training, self.steps = 0.0, 0.0, 0
        return times


class CSVRunLogger:
    """CSV + TensorBoard (+ optional W&B) experiment logger."""

    METRIC_FIELDS = [
        "timestamp",
        "step",
        "epoch",
        "split",
        "total_loss",
        "control_loss",
        "speed_loss",
        "mode_loss",
        "waypoint_loss",
        "situation_loss",
        "distill_loss",
        "steer_mae",
        "accel_mae",
        "moving_control_mae",
        "turn_control_mae",
        "control_score",
        "token_spread",
        "lr",
        "grad_norm",
        "data_wait_s",
        "train_step_s",
        "rss_main_mib",
        "rss_worker_max_mib",
    ]

    EVENT_FIELDS = ["timestamp", "step", "epoch", "event_type", "detail", "value"]

    def __init__(self, cfg: TrainingConfig, run_dir: Path, run_name: str) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_name = run_name
        self.logs_dir = run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.logs_dir / "metrics.csv"
        self.events_path = self.logs_dir / "events.csv"

        self._metrics_fh = self.metrics_path.open("a", encoding="utf-8", newline="")
        self._events_fh = self.events_path.open("a", encoding="utf-8", newline="")
        self._metrics_writer = csv.DictWriter(self._metrics_fh, fieldnames=self.METRIC_FIELDS)
        self._events_writer = csv.DictWriter(self._events_fh, fieldnames=self.EVENT_FIELDS)

        if self.metrics_path.stat().st_size == 0:
            self._metrics_writer.writeheader()
        if self.events_path.stat().st_size == 0:
            self._events_writer.writeheader()

        self.tb_writer: SummaryWriter | None = None
        if cfg.logging.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.logs_dir / "tensorboard"))

        self._wandb_run = None
        if cfg.logging.use_wandb:
            if wandb is None:
                raise RuntimeError("logging.use_wandb=true but wandb is not installed")
            self._wandb_run = wandb.init(
                project=cfg.run.project,
                entity=cfg.run.entity,
                name=run_name,
                tags=cfg.run.tags,
                config=training_config_to_dict(cfg),
                dir=str(run_dir),
                resume="allow",
            )

    def log_metrics(self, split: str, step: int, epoch: int, metrics: dict[str, float]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        row = {
            "timestamp": now,
            "step": step,
            "epoch": epoch,
            "split": split,
            "total_loss": metrics.get("total_loss", float("nan")),
            "control_loss": metrics.get("control_loss", float("nan")),
            "speed_loss": metrics.get("speed_loss", float("nan")),
            "mode_loss": metrics.get("mode_loss", float("nan")),
            "waypoint_loss": metrics.get("waypoint_loss", float("nan")),
            "situation_loss": metrics.get("situation_loss", float("nan")),
            "distill_loss": metrics.get("distill_loss", float("nan")),
            "steer_mae": metrics.get("steer_mae", float("nan")),
            "accel_mae": metrics.get("accel_mae", float("nan")),
            "moving_control_mae": metrics.get("moving_control_mae", float("nan")),
            "turn_control_mae": metrics.get("turn_control_mae", float("nan")),
            "control_score": metrics.get("control_score", float("nan")),
            "token_spread": metrics.get("token_spread", float("nan")),
            "lr": metrics.get("lr", float("nan")),
            "grad_norm": metrics.get("grad_norm", float("nan")),
            "data_wait_s": metrics.get("data_wait_s", float("nan")),
            "train_step_s": metrics.get("train_step_s", float("nan")),
            "rss_main_mib": metrics.get("rss_main_mib", float("nan")),
            "rss_worker_max_mib": max(
                (v for k, v in metrics.items() if k.startswith("rss_worker")), default=float("nan")
            ),
        }
        self._metrics_writer.writerow(row)
        self._metrics_fh.flush()

        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"{split}/{key}", value, step)
            self.tb_writer.flush()

        if self._wandb_run is not None:
            payload = {f"{split}/{k}": v for k, v in metrics.items()}
            payload["epoch"] = epoch
            self._wandb_run.log(payload, step=step)

    def log_event(self, step: int, epoch: int, event_type: str, detail: str, value: str) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "epoch": epoch,
            "event_type": event_type,
            "detail": detail,
            "value": value,
        }
        self._events_writer.writerow(row)
        self._events_fh.flush()

        if self.tb_writer is not None:
            self.tb_writer.add_text("events", f"[{step}] {event_type} {detail}={value}", step)
            self.tb_writer.flush()

        if self._wandb_run is not None:
            self._wandb_run.log(
                {
                    "event/type": event_type,
                    "event/detail": detail,
                    "event/value": value,
                },
                step=step,
            )

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
        self._metrics_fh.close()
        self._events_fh.close()


class ModelEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                model_v = msd[k].detach()
                if not torch.is_floating_point(model_v):
                    ema_v.copy_(model_v)
                else:
                    ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)

    def state_dict(self) -> dict[str, Any]:
        return self.ema_model.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema_model.load_state_dict(state)

    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema_model.state_dict(), strict=True)
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=True)


class Trainer:
    """Main trainer implementing robust supervised sequence learning."""

    def __init__(self, cfg: TrainingConfig, run_name: str | None = None) -> None:
        self.cfg = cfg
        self.run_name = run_name or cfg.run.run_name or build_run_name(cfg)

        self.output_root = Path(cfg.run.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.output_root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Convenience pointer to the latest run.
        (self.output_root / "LAST_RUN").write_text(self.run_name + "\n", encoding="utf-8")

        self.checkpoints = CheckpointManager(self.run_dir, cfg.checkpoint)
        resume_path = self.checkpoints.resolve_resume_path(cfg.run.resume)
        resumed = None if resume_path is None else self.checkpoints.load(resume_path)
        if resumed is not None:
            conflicts = resume_conflicts(resumed["config"], cfg)
            if conflicts:
                raise ValueError(
                    f"Resuming {resume_path} would change the settings it trained with:\n  "
                    + "\n  ".join(conflicts)
                    + "\nStart a new run from its weights instead (run.init_from)."
                )

        save_training_config(cfg, self.run_dir / "config_resolved.yaml")
        self.logger = CSVRunLogger(cfg, self.run_dir, self.run_name)

        seed_everything(cfg.run.seed, cfg.run.deterministic)
        self.device = self._resolve_device(cfg.run.device)
        print(f"[trainer] device: {describe_device(self.device)}")

        self.model = VisionRWKVDriver(cfg.model)
        if cfg.model.tokenizer.pretrained:
            self.model.tokenizer.load_pretrained()
        if resumed is None and cfg.run.init_from:
            self._warm_start(Path(cfg.run.init_from))
        self.model.to(self.device)
        if cfg.optimization.compile_encoder and self.device.type == "cuda":
            self.model.encoder = torch.compile(  # type: ignore[assignment]
                self.model.encoder, mode="default"
            )
            print("[trainer] torch.compile applied to encoder (mode='default')")
        self.teacher: VisionRWKVDriver | None = self._build_teacher_if_needed()

        ocfg = cfg.optimization
        backbone = list(self.model.tokenizer.backbone.parameters())
        in_backbone = {id(p) for p in backbone}
        rest = [p for p in self.model.parameters() if id(p) not in in_backbone]
        groups = [(rest, 1.0), (backbone, ocfg.backbone_lr_scale)]
        self.optimizer = torch.optim.AdamW(
            [{"params": p, "lr": ocfg.learning_rate * s, "lr_scale": s} for p, s in groups],
            weight_decay=ocfg.weight_decay,
        )

        self.train_loader = build_loader(cfg.dataset, cfg.run.seed, "train")
        self.val_loader = build_loader(cfg.dataset, cfg.run.seed, "val")
        self.total_steps = max(1, cfg.optimization.epochs * len(self.train_loader))
        self.optimizer_steps = 0  # steps taken, skipped ones excluded: the schedule's position
        self.situation_pos_weight: torch.Tensor | None = None
        if cfg.model.situation_classes:
            # Rare causes count more: positives weigh sqrt(negatives / positives) of the training
            # split, so pedestrians (0.4% of v3 frames) weigh about 16.
            train = cast(DrivingSequenceDataset, self.train_loader.dataset)
            share = np.concatenate(train.situations)[:, CAUSE_SITUATIONS].any(-1).mean(0)
            share = share.clip(1e-4, 1 - 1e-4)
            self.situation_pos_weight = torch.tensor(
                np.sqrt((1 - share) / share), dtype=torch.float32, device=self.device
            )

        self.use_bf16 = cfg.optimization.amp and self.device.type == "cuda"
        self.nonfinite_in_a_row = 0

        self.ema: ModelEMA | None = None
        if cfg.optimization.ema_decay > 0.0:
            self.ema = ModelEMA(self.model, decay=cfg.optimization.ema_decay)

        self.global_step = 0
        self.current_epoch = 0
        self.resume_batch_in_epoch = 0
        self.best_metric: float | None = None
        self._stop_requested = False
        self._stop_reason = ""
        self._last_val_loss: float | None = None

        self._register_signal_handlers()
        if resume_path is not None and resumed is not None:
            self._resume(resume_path, resumed)

    def _resolve_device(self, device: str) -> torch.device:
        return select_device(device)

    def _build_teacher_if_needed(self) -> VisionRWKVDriver | None:
        dcfg = self.cfg.distillation
        if not dcfg.enabled:
            return None
        if not dcfg.teacher_checkpoint:
            raise ValueError("distillation.enabled=true requires distillation.teacher_checkpoint")

        teacher = VisionRWKVDriver(self.cfg.model).to(self.device)
        payload = torch.load(dcfg.teacher_checkpoint, map_location=self.device, weights_only=False)
        state_dict = (
            payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        )
        teacher.load_state_dict(state_dict, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def _register_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            self._stop_requested = True
            self._stop_reason = f"signal_{signum}"
            print("\n[trainer] Ctrl+C received — finishing step, saving checkpoint, then exiting.")

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _warm_start(self, path: Path) -> None:
        """Start this new run from another run's weights, with a new optimizer and schedule."""
        self.model.load_state_dict(self.checkpoints.load(path)["model"], strict=True)
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        self.logger.log_event(
            step=0, epoch=0, event_type="warm_start", detail=str(path), value=f"sha256:{digest}"
        )
        print(f"[trainer] warm start from {path} (sha256 {digest[:12]})")

    def _resume(self, path: Path, payload: dict[str, Any]) -> None:
        """Continue the run saved in `payload` exactly where it stopped."""
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.ema is not None and "ema" in payload:
            self.ema.load_state_dict(payload["ema"])

        # Checkpoints from before this counter kept it as the scheduler's position.
        steps = payload.get("optimizer_steps", payload.get("scheduler", {}).get("last_epoch", 0))
        self.optimizer_steps = int(steps)
        self.global_step = int(payload.get("global_step", 0))
        self.current_epoch = int(payload.get("epoch", 0))
        self.resume_batch_in_epoch = int(payload.get("batch_in_epoch", 0))
        self.best_metric = payload.get("best_metric", None)
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])

        if self.resume_batch_in_epoch >= len(self.train_loader):
            self.current_epoch += 1
            self.resume_batch_in_epoch = 0

        self.logger.log_event(
            step=self.global_step,
            epoch=self.current_epoch,
            event_type="resume",
            detail="checkpoint",
            value=str(path),
        )

    def _autocast(self) -> torch.amp.autocast:
        return torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16)

    def _masked_mean(self, values: torch.Tensor, frame_weight: torch.Tensor) -> torch.Tensor:
        denom = frame_weight.sum().clamp(min=1e-6)
        return (values * frame_weight).sum() / denom

    def _compute_losses(
        self, batch: dict[str, torch.Tensor], control_error: ControlError | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        images = batch["images"].to(self.device, non_blocking=True)
        nav = batch["nav_command"].to(self.device, non_blocking=True)
        speed = batch["speed"].to(self.device, non_blocking=True)
        target_point = batch["target_point"].to(self.device, non_blocking=True)
        target_controls = batch["controls"].to(self.device, non_blocking=True)

        pred_controls, _, aux = self.model(images, nav, speed, target_point)

        _bsz, _seq_len, _ = pred_controls.shape
        frame_weight = batch["loss_weight"].to(self.device, non_blocking=True).float()

        # Burn-in frames warm recurrent state but do not contribute to loss.
        burn_in = self.cfg.loss.burn_in
        if burn_in > 0:
            frame_weight = frame_weight.clone()
            frame_weight[:, :burn_in] = 0.0

        if control_error is not None:
            control_error.add(pred_controls.detach(), target_controls, frame_weight)

        control_mse = (pred_controls - target_controls).pow(2).mean(dim=-1)
        control_loss = self._masked_mean(control_mse, frame_weight)

        target_speed = batch["target_speed"].to(self.device, non_blocking=True).float()
        speed_pred = aux["speed"].squeeze(-1)
        speed_mse = (speed_pred - target_speed).pow(2)
        speed_loss = self._masked_mean(speed_mse, frame_weight)

        mode_target = batch["driving_mode"].to(self.device, non_blocking=True).long()
        mode_logits = aux["mode_logits"]
        valid = frame_weight > 0
        if valid.any():
            mode_loss = F.cross_entropy(mode_logits[valid], mode_target[valid])
        else:
            mode_loss = torch.zeros((), device=self.device)

        situation_loss = torch.zeros((), device=self.device)
        if self.situation_pos_weight is not None and valid.any():
            situation = batch["situation"].to(self.device, non_blocking=True)
            causes = situation[..., CAUSE_SITUATIONS].any(-1).float()
            situation_loss = F.binary_cross_entropy_with_logits(
                aux["situation_logits"][valid].float(),
                causes[valid],
                pos_weight=self.situation_pos_weight,
            )

        wp_target = batch["future_waypoints"].to(self.device, non_blocking=True).float()
        wp_mask = batch["waypoint_mask"].to(self.device, non_blocking=True).float()
        wp_l1 = (aux["future_waypoints"] - wp_target).abs().mean(dim=-1)
        waypoint_loss = self._masked_mean(wp_l1, frame_weight.unsqueeze(-1) * wp_mask)

        distill_loss = torch.zeros((), device=self.device)
        if self.teacher is not None:
            with torch.no_grad():
                teacher_controls, _, _ = self.teacher(images, nav, speed, target_point)
            distill_mse = (pred_controls - teacher_controls).pow(2).mean(dim=-1)
            distill_loss = self._masked_mean(distill_mse, frame_weight)

        lcfg = self.cfg.loss
        total_loss = (
            lcfg.control_weight * control_loss
            + lcfg.speed_weight * speed_loss
            + lcfg.mode_weight * mode_loss
            + lcfg.waypoint_weight * waypoint_loss
            + lcfg.situation_weight * situation_loss
            + self.cfg.distillation.weight * distill_loss
        )

        steer_mae = self._masked_mean(
            (pred_controls[..., 0] - target_controls[..., 0]).abs(), frame_weight
        )
        accel_mae = self._masked_mean(
            (pred_controls[..., 1] - target_controls[..., 1]).abs(), frame_weight
        )

        metrics = {
            "total_loss": float(total_loss.detach().item()),
            "control_loss": float(control_loss.detach().item()),
            "speed_loss": float(speed_loss.detach().item()),
            "mode_loss": float(mode_loss.detach().item()),
            "waypoint_loss": float(waypoint_loss.detach().item()),
            "situation_loss": float(situation_loss.detach().item()),
            "distill_loss": float(distill_loss.detach().item()),
            "steer_mae": float(steer_mae.detach().item()),
            "accel_mae": float(accel_mae.detach().item()),
        }
        return total_loss, metrics

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float] | None:
        """One optimizer step on `batch`; None if it was skipped for a non-finite value.

        A skipped step leaves the weights, optimizer, schedule and EMA as they were.
        """
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            total_loss, metrics = self._compute_losses(batch)

        bad_losses = [name for name, value in metrics.items() if not math.isfinite(value)]
        if bad_losses:
            self._skip_step(batch, ", ".join(bad_losses))
            return None

        total_loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        grad_norm = get_total_norm(grads)
        if not grad_norm.isfinite():
            named = self.model.named_parameters()
            bad = (n for n, p in named if p.grad is not None and not p.grad.isfinite().all())
            self._skip_step(batch, f"gradient of {next(bad, 'the model')}")
            return None
        clip_grads_with_norm_(
            self.model.parameters(), self.cfg.optimization.gradient_clip, grad_norm
        )

        self.nonfinite_in_a_row = 0
        factor = learning_rate_factor(self.optimizer_steps, self.total_steps, self.cfg.scheduler)
        for group in self.optimizer.param_groups:
            group["lr"] = self.cfg.optimization.learning_rate * group["lr_scale"] * factor
        self.optimizer.step()
        self.optimizer_steps += 1
        if self.ema is not None:
            self.ema.update(self.model)
        return metrics | {"lr": self._current_lr(), "grad_norm": float(grad_norm)}

    def _skip_step(self, batch: dict[str, torch.Tensor], problem: str) -> None:
        """Log a non-finite step with the clips that caused it; abort if they keep coming."""
        files = cast(DrivingSequenceDataset, self.train_loader.dataset).episode_files
        clips = " ".join(f"{files[ep].name}:{start}" for ep, start in batch["clip"].tolist())
        self.logger.log_event(
            step=self.global_step,
            epoch=self.current_epoch,
            event_type="nonfinite_step",
            detail=problem,
            value=clips,
        )
        tqdm.write(f"  [skip] step {self.global_step}: non-finite {problem} in {clips}")
        self.nonfinite_in_a_row += 1
        limit = self.cfg.optimization.max_nonfinite_steps
        if self.nonfinite_in_a_row >= limit:
            raise NonFiniteTraining(f"{limit} steps in a row were not finite, last: {problem}")

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _save_checkpoint(self, *, is_best: bool = False, tag: str | None = None) -> Path:
        payload: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "epoch": self.current_epoch,
            "batch_in_epoch": self.resume_batch_in_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "rng_state": capture_rng_state(),
            "config": training_config_to_dict(self.cfg),
            "run_name": self.run_name,
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        ckpt_path = self.checkpoints.save(
            payload,
            step=self.global_step,
            epoch=self.current_epoch,
            is_best=is_best,
            tag=tag,
        )
        self.logger.log_event(
            step=self.global_step,
            epoch=self.current_epoch,
            event_type="checkpoint",
            detail="save",
            value=str(ckpt_path.name),
        )
        return ckpt_path

    def _is_better(self, candidate: float) -> bool:
        if not math.isfinite(candidate):
            return False
        if self.best_metric is None:
            return True
        mode = self.cfg.checkpoint.monitor_mode
        if mode == "min":
            return candidate < self.best_metric
        if mode == "max":
            return candidate > self.best_metric
        raise ValueError(f"Unsupported checkpoint.monitor_mode: {mode}")

    def evaluate(self) -> dict[str, float]:
        model = self.model
        backup: dict[str, torch.Tensor] | None = None
        if self.ema is not None:
            backup = self.ema.apply_to(model)

        model.eval()
        agg: dict[str, float] = {
            "total_loss": 0.0,
            "control_loss": 0.0,
            "speed_loss": 0.0,
            "mode_loss": 0.0,
            "waypoint_loss": 0.0,
            "situation_loss": 0.0,
            "distill_loss": 0.0,
            "steer_mae": 0.0,
            "accel_mae": 0.0,
        }

        n_batches = 0
        max_batches = int(self.cfg.logging.val_max_batches)
        control_error = ControlError()
        spreads: list[float] = []
        hook = model.tokenizer.register_forward_hook(
            lambda _module, _inputs, tokens: spreads.append(token_spread(tokens))
        )
        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    with self._autocast():
                        _, metrics = self._compute_losses(batch, control_error)
                    for key, value in metrics.items():
                        agg[key] += value
                    n_batches += 1
                    if max_batches > 0 and n_batches >= max_batches:
                        break
        finally:
            hook.remove()

        if n_batches == 0:
            out = {k: float("nan") for k in agg}
        else:
            out = {k: v / n_batches for k, v in agg.items()}
        out |= control_error.metrics()
        out["token_spread"] = sum(spreads) / len(spreads) if spreads else float("nan")

        if backup is not None:
            ModelEMA.restore(model, backup)

        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def fit(self, max_epochs: int | None = None) -> Path:
        """Train and return the run directory."""
        self.model.train()
        target_epochs = max_epochs if max_epochs is not None else self.cfg.optimization.epochs

        try:
            resume_epoch = self.current_epoch
            steps_per_epoch = len(self.train_loader)
            total_steps = (target_epochs - self.current_epoch) * steps_per_epoch
            step_times = StepTimes()

            for epoch in range(self.current_epoch, target_epochs):
                self.current_epoch = epoch

                pbar = tqdm(
                    enumerate(self.train_loader),
                    total=steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{target_epochs}",
                    dynamic_ncols=True,
                    leave=True,
                )
                wait_start = time.time()
                for batch_idx, batch in pbar:
                    waited = time.time() - wait_start
                    if epoch == resume_epoch and batch_idx < self.resume_batch_in_epoch:
                        pbar.update()
                        wait_start = time.time()
                        continue

                    self.global_step += 1
                    self.resume_batch_in_epoch = batch_idx + 1

                    t_step_start = time.time()
                    metrics = self._train_step(batch)
                    step_secs = time.time() - t_step_start
                    step_times.add(waited, step_secs)

                    if metrics is not None:
                        steps_done = self.global_step - (resume_epoch * steps_per_epoch)
                        eta_secs = step_secs * (total_steps - steps_done)
                        postfix: dict[str, str] = {
                            "tr_loss": f"{metrics['total_loss']:.3f}",
                            "lr": f"{metrics['lr']:.1e}",
                            "gnorm": f"{metrics['grad_norm']:.2f}",
                            "sps": f"{1.0 / step_secs:.1f}",
                            "eta": f"{eta_secs / 60:.0f}m",
                        }
                        if self._last_val_loss is not None:
                            postfix["val_loss"] = f"{self._last_val_loss:.3f}"
                        pbar.set_postfix(postfix)

                    logging_step = self.global_step % max(1, self.cfg.logging.log_every_steps) == 0
                    if metrics is not None and logging_step:
                        metrics |= loader_rss_mib(self.train_loader) | step_times.report()
                        self.logger.log_metrics(
                            "train", self.global_step, self.current_epoch, metrics
                        )

                    if self.global_step % max(1, self.cfg.logging.val_every_steps) == 0:
                        pbar.write(f"  [val] step {self.global_step} — running validation...")
                        val_metrics = self.evaluate()
                        val_metrics["lr"] = self._current_lr()
                        val_metrics["grad_norm"] = float("nan")
                        self._last_val_loss = val_metrics["total_loss"]
                        self.logger.log_metrics(
                            "val", self.global_step, self.current_epoch, val_metrics
                        )
                        pbar.write(
                            f"  [val] step {self.global_step}  "
                            f"loss={val_metrics['total_loss']:.3f}  "
                            f"steer_mae={val_metrics['steer_mae']:.4f}  "
                            f"accel_mae={val_metrics['accel_mae']:.4f}  "
                            f"control_score={val_metrics['control_score']:.4f}  "
                            f"token_spread={val_metrics['token_spread']:.3f}"
                        )

                        monitor_name = self.cfg.checkpoint.monitor_metric.split("/", maxsplit=1)[-1]
                        monitor_value = float(val_metrics[monitor_name])
                        is_best = self._is_better(monitor_value)
                        if is_best:
                            self.best_metric = monitor_value
                        self._save_checkpoint(is_best=is_best, tag="val")

                    if self.global_step % max(1, self.cfg.checkpoint.save_every_steps) == 0:
                        self._save_checkpoint(tag="step")

                    if self._stop_requested:
                        pbar.close()
                        self._save_checkpoint(tag=self._stop_reason or "stop")
                        self.logger.log_event(
                            step=self.global_step,
                            epoch=self.current_epoch,
                            event_type="stop",
                            detail="requested",
                            value=self._stop_reason,
                        )
                        if self.cfg.logging.plot_on_finish:
                            plot_run_dashboard(self.run_dir)
                        return self.run_dir

                    wait_start = time.time()

                pbar.close()
                if (epoch + 1) % max(1, self.cfg.checkpoint.save_every_epochs) == 0:
                    self._save_checkpoint(tag="epoch")

                self.resume_batch_in_epoch = 0

            self._save_checkpoint(tag="final")

        except Exception as exc:
            self._save_checkpoint(tag="crash")
            self.logger.log_event(
                step=self.global_step,
                epoch=self.current_epoch,
                event_type="exception",
                detail=type(exc).__name__,
                value=str(exc),
            )
            raise
        finally:
            if self.cfg.logging.plot_on_finish:
                try:
                    plot_path = plot_run_dashboard(self.run_dir)
                    self.logger.log_event(
                        step=self.global_step,
                        epoch=self.current_epoch,
                        event_type="plot",
                        detail="dashboard",
                        value=str(plot_path),
                    )
                except Exception as plot_exc:
                    self.logger.log_event(
                        step=self.global_step,
                        epoch=self.current_epoch,
                        event_type="plot_error",
                        detail=type(plot_exc).__name__,
                        value=str(plot_exc),
                    )
            self.logger.close()

        return self.run_dir

    def export_dynamic_quantized_model(self) -> Path:
        """Export a post-training dynamic-int8 quantized model."""
        qcfg = self.cfg.quantization
        if not qcfg.export_dynamic_int8:
            raise ValueError(
                "quantization export requested but quantization.export_dynamic_int8 is false"
            )

        cpu_model = copy.deepcopy(self.model).to("cpu").eval()
        quantized = torch.quantization.quantize_dynamic(
            cpu_model, {torch.nn.Linear}, dtype=torch.qint8
        )

        out_path = self.run_dir / qcfg.export_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": quantized.state_dict(), "config": asdict(self.cfg)}, out_path)

        self.logger.log_event(
            step=self.global_step,
            epoch=self.current_epoch,
            event_type="quant_export",
            detail="dynamic_int8",
            value=str(out_path),
        )
        return out_path
