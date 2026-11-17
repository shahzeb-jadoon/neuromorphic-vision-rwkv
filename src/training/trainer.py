"""Phase 2 trainer with crash-safe resume, rich logging, and runtime overrides."""

from __future__ import annotations

import copy
import csv
import math
import random
import signal
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.driving_dataset import DrivingSequenceDataset
from src.models import VisionRWKVDriver
from src.training.checkpointing import CheckpointManager, capture_rng_state, restore_rng_state
from src.training.config import (
    TrainingConfig,
    apply_dotlist_overrides,
    save_training_config,
    training_config_to_dict,
)
from src.training.plotting import plot_run_dashboard
from src.utils.device import describe_device, select_device
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
        "distill_loss",
        "steer_mae",
        "accel_mae",
        "lr",
        "grad_norm",
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
            "distill_loss": metrics.get("distill_loss", float("nan")),
            "steer_mae": metrics.get("steer_mae", float("nan")),
            "accel_mae": metrics.get("accel_mae", float("nan")),
            "lr": metrics.get("lr", float("nan")),
            "grad_norm": metrics.get("grad_norm", float("nan")),
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

        save_training_config(cfg, self.run_dir / "config_resolved.yaml")

        seed_everything(cfg.run.seed, cfg.run.deterministic)
        self.device = self._resolve_device(cfg.run.device)
        print(f"[trainer] device: {describe_device(self.device)}")

        self.model = VisionRWKVDriver(cfg.model).to(self.device)
        if cfg.optimization.compile_encoder and self.device.type == "cuda":
            self.model.encoder = torch.compile(  # type: ignore[assignment]
                self.model.encoder, mode="default"
            )
            print("[trainer] torch.compile applied to encoder (mode='default')")
        self.teacher: VisionRWKVDriver | None = self._build_teacher_if_needed()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optimization.learning_rate,
            weight_decay=cfg.optimization.weight_decay,
        )

        self.train_loader, self.val_loader = self._build_dataloaders()
        total_steps = max(1, cfg.optimization.epochs * len(self.train_loader))
        self.scheduler = self._build_scheduler(total_steps)

        self.scaler = torch.amp.GradScaler(
            "cuda" if self.device.type == "cuda" else "cpu",
            enabled=cfg.optimization.amp and self.device.type == "cuda",
        )

        self.ema: ModelEMA | None = None
        if cfg.optimization.ema_decay > 0.0:
            self.ema = ModelEMA(self.model, decay=cfg.optimization.ema_decay)

        self.checkpoints = CheckpointManager(self.run_dir, cfg.checkpoint)
        self.logger = CSVRunLogger(cfg, self.run_dir, self.run_name)

        self.global_step = 0
        self.current_epoch = 0
        self.resume_batch_in_epoch = 0
        self.best_metric: float | None = None
        self._stop_requested = False
        self._stop_reason = ""
        self._last_override_version = -1
        self._last_val_loss: float | None = None

        self._register_signal_handlers()
        self._maybe_resume()

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

    def _build_dataloaders(
        self,
    ) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
        dcfg = self.cfg.dataset
        data_dir = dcfg.data_dir or setting("DATA_DIR")

        train_ds = DrivingSequenceDataset(
            data_dir=data_dir,
            seq_len=dcfg.seq_len,
            stride=dcfg.stride,
            split="train",
            val_fraction=dcfg.val_fraction,
            seed=self.cfg.run.seed,
            augment=dcfg.augment_train,
            resize_height=dcfg.resize_height,
            resize_width=dcfg.resize_width,
        )
        val_ds = DrivingSequenceDataset(
            data_dir=data_dir,
            seq_len=dcfg.seq_len,
            stride=dcfg.stride,
            split="val",
            val_fraction=dcfg.val_fraction,
            seed=self.cfg.run.seed,
            augment=False,
            resize_height=dcfg.resize_height,
            resize_width=dcfg.resize_width,
        )

        sampler = train_ds.get_sampler() if dcfg.weighted_sampling else None
        # prefetch_factor=2 is the PyTorch default (each worker prefetches 2 batches).
        # Lower to 1 to halve DataLoader RAM at the cost of slightly lower GPU utilisation.
        pf = 2 if dcfg.num_workers > 0 else None
        train_loader = DataLoader(
            train_ds,
            batch_size=dcfg.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=dcfg.num_workers,
            pin_memory=dcfg.pin_memory,
            persistent_workers=dcfg.num_workers > 0,
            prefetch_factor=pf,
            drop_last=True,
        )
        val_workers = max(0, dcfg.num_workers // 2)
        val_loader = DataLoader(
            val_ds,
            batch_size=dcfg.batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=dcfg.pin_memory,
            persistent_workers=val_workers > 0,
            prefetch_factor=2 if val_workers > 0 else None,
            drop_last=False,
        )
        return train_loader, val_loader

    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        scfg = self.cfg.scheduler
        warmup = max(0, int(scfg.warmup_steps))
        min_ratio = float(scfg.min_lr_ratio)

        def lr_lambda(step: int) -> float:
            if scfg.name == "constant":
                return 1.0
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)

            progress_denom = max(1, total_steps - warmup)
            progress = min(1.0, max(0.0, (step - warmup) / progress_denom))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        # LambdaLR calls .step() internally during __init__, which triggers a
        # PyTorch warning about step ordering. This is a known false positive —
        # our training loop calls optimizer.step() before scheduler.step().
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Detected call of")
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _register_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            self._stop_requested = True
            self._stop_reason = f"signal_{signum}"
            print("\n[trainer] Ctrl+C received — finishing step, saving checkpoint, then exiting.")

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _maybe_resume(self) -> None:
        resume_path = self.checkpoints.resolve_resume_path(self.cfg.run.resume)
        if resume_path is None:
            return

        payload = self.checkpoints.load(resume_path, map_location="cpu")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload and self.scaler.is_enabled():
            self.scaler.load_state_dict(payload["scaler"])
        if self.ema is not None and "ema" in payload:
            self.ema.load_state_dict(payload["ema"])

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
            value=str(resume_path),
        )

    def _poll_runtime_overrides(self) -> None:
        ocfg = self.cfg.runtime_overrides
        if not ocfg.enabled:
            return
        if self.global_step % max(1, ocfg.poll_every_steps) != 0:
            return

        override_path = self.run_dir / ocfg.file_name
        if not override_path.exists():
            return

        with override_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        version = int(payload.get("version", 0))
        if version <= self._last_override_version:
            return

        changes = payload.get("changes", {})
        if not isinstance(changes, dict):
            return

        apply_dotlist_overrides(self.cfg, changes)
        if "optimization.learning_rate" in changes:
            lr = float(changes["optimization.learning_rate"])
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        self._last_override_version = version
        for key, value in changes.items():
            self.logger.log_event(
                step=self.global_step,
                epoch=self.current_epoch,
                event_type="override",
                detail=key,
                value=str(value),
            )

        save_training_config(self.cfg, self.run_dir / "config_resolved.yaml")

    def _masked_mean(self, values: torch.Tensor, frame_weight: torch.Tensor) -> torch.Tensor:
        denom = frame_weight.sum().clamp(min=1e-6)
        return (values * frame_weight).sum() / denom

    def _compute_losses(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        images = batch["images"].to(self.device, non_blocking=True)
        nav = batch["nav_command"].to(self.device, non_blocking=True)
        target_controls = batch["controls"].to(self.device, non_blocking=True)

        pred_controls, _, aux = self.model(images, nav)

        _bsz, _seq_len, _ = pred_controls.shape
        frame_weight = batch["loss_weight"].to(self.device, non_blocking=True).float()

        # Burn-in frames warm recurrent state but do not contribute to loss.
        burn_in = self.cfg.loss.burn_in
        if burn_in > 0:
            frame_weight = frame_weight.clone()
            frame_weight[:, :burn_in] = 0.0

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

        wp_target = batch["future_waypoints"].to(self.device, non_blocking=True).float()
        wp_pred = aux["future_waypoints"]
        wp_l1 = (wp_pred - wp_target).abs().mean(dim=(-1, -2))
        waypoint_loss = self._masked_mean(wp_l1, frame_weight)

        distill_loss = torch.zeros((), device=self.device)
        if self.teacher is not None:
            with torch.no_grad():
                teacher_controls, _, _ = self.teacher(images, nav)
            distill_mse = (pred_controls - teacher_controls).pow(2).mean(dim=-1)
            distill_loss = self._masked_mean(distill_mse, frame_weight)

        lcfg = self.cfg.loss
        total_loss = (
            lcfg.control_weight * control_loss
            + lcfg.speed_weight * speed_loss
            + lcfg.mode_weight * mode_loss
            + lcfg.waypoint_weight * waypoint_loss
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
            "distill_loss": float(distill_loss.detach().item()),
            "steer_mae": float(steer_mae.detach().item()),
            "accel_mae": float(accel_mae.detach().item()),
        }
        return total_loss, metrics

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _save_checkpoint(self, *, is_best: bool = False, tag: str | None = None) -> Path:
        payload: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
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
            "distill_loss": 0.0,
            "steer_mae": 0.0,
            "accel_mae": 0.0,
        }

        n_batches = 0
        max_batches = int(self.cfg.logging.val_max_batches)
        with torch.no_grad():
            for batch in self.val_loader:
                _, metrics = self._compute_losses(batch)
                for key, value in metrics.items():
                    agg[key] += value
                n_batches += 1
                if max_batches > 0 and n_batches >= max_batches:
                    break

        if n_batches == 0:
            out = {k: float("nan") for k in agg}
        else:
            out = {k: v / n_batches for k, v in agg.items()}

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

            for epoch in range(self.current_epoch, target_epochs):
                self.current_epoch = epoch

                pbar = tqdm(
                    enumerate(self.train_loader),
                    total=steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{target_epochs}",
                    dynamic_ncols=True,
                    leave=True,
                )
                for batch_idx, batch in pbar:
                    if epoch == resume_epoch and batch_idx < self.resume_batch_in_epoch:
                        pbar.update()
                        continue

                    self.global_step += 1
                    self.resume_batch_in_epoch = batch_idx + 1
                    self._poll_runtime_overrides()

                    self.optimizer.zero_grad(set_to_none=True)

                    t_step_start = time.time()
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=self.scaler.is_enabled(),
                    ):
                        total_loss, metrics = self._compute_losses(batch)

                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = float(
                        clip_grad_norm_(
                            self.model.parameters(), self.cfg.optimization.gradient_clip
                        ).item()
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    step_secs = time.time() - t_step_start

                    if self.ema is not None:
                        self.ema.update(self.model)

                    metrics["lr"] = self._current_lr()
                    metrics["grad_norm"] = grad_norm

                    steps_done = self.global_step - (resume_epoch * steps_per_epoch)
                    eta_secs = step_secs * (total_steps - steps_done)
                    postfix: dict[str, str] = {
                        "tr_loss": f"{metrics['total_loss']:.3f}",
                        "lr": f"{metrics['lr']:.1e}",
                        "gnorm": f"{grad_norm:.2f}",
                        "sps": f"{1.0 / step_secs:.1f}",
                        "eta": f"{eta_secs / 60:.0f}m",
                    }
                    if self._last_val_loss is not None:
                        postfix["val_loss"] = f"{self._last_val_loss:.3f}"
                    pbar.set_postfix(postfix)

                    if self.global_step % max(1, self.cfg.logging.log_every_steps) == 0:
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
                            f"accel_mae={val_metrics['accel_mae']:.4f}"
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
