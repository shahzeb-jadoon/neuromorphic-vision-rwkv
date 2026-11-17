"""Training configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from src.models.config import ModelConfig


@dataclass
class RunConfig:
    """Experiment identity and runtime controls."""

    output_root: str = "runs"
    name_prefix: str = "vrwkv"
    run_name: str | None = None
    project: str = "neuromorphic-vision-rwkv"
    entity: str | None = None
    tags: list[str] = field(default_factory=lambda: ["phase2", "rwkv", "imitation-learning"])
    seed: int = 42
    deterministic: bool = True
    resume: str = "auto"  # auto | never | /path/to/checkpoint_or_run_dir
    device: str = "auto"  # auto | cuda | cpu


@dataclass
class DatasetConfig:
    """Dataset and dataloader controls."""

    data_dir: str = ""  # empty: DATA_DIR from .env
    seq_len: int = 32
    stride: int = 16
    val_fraction: float = 0.2
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    weighted_sampling: bool = True
    augment_train: bool = True
    resize_height: int | None = None
    resize_width: int | None = None


@dataclass
class OptimizationConfig:
    """Optimizer-level settings."""

    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.0  # 0 disables EMA
    compile_encoder: bool = False  # torch.compile the RWKV encoder (mode='default')


@dataclass
class SchedulerConfig:
    """Learning rate schedule controls."""

    name: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1


@dataclass
class LossConfig:
    """Primary and auxiliary loss weights."""

    burn_in: int = 8
    control_weight: float = 1.0
    speed_weight: float = 0.1
    mode_weight: float = 0.1
    waypoint_weight: float = 0.5


@dataclass
class DistillationConfig:
    """Teacher distillation options."""

    enabled: bool = False
    teacher_checkpoint: str = ""
    weight: float = 0.2


@dataclass
class CheckpointConfig:
    """Checkpoint cadence and retention."""

    save_every_steps: int = 500
    save_every_epochs: int = 1
    keep_last_n: int = 5
    monitor_metric: str = "val/total_loss"
    monitor_mode: str = "min"  # min | max


@dataclass
class LoggingConfig:
    """Metrics and dashboard logging options."""

    log_every_steps: int = 20
    val_every_steps: int = 500
    use_tensorboard: bool = True
    use_wandb: bool = False
    plot_on_finish: bool = True
    val_max_batches: int = 0  # 0 = full validation set


@dataclass
class RuntimeOverrideConfig:
    """Hot-reloadable runtime parameter overrides."""

    enabled: bool = True
    file_name: str = "runtime_overrides.yaml"
    poll_every_steps: int = 100


@dataclass
class QuantizationConfig:
    """Optional post-training quantization export."""

    export_dynamic_int8: bool = False
    export_path: str = "quantized/model_dynamic_int8.pt"


@dataclass
class TrainingConfig:
    """Top-level training config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime_overrides: RuntimeOverrideConfig = field(default_factory=RuntimeOverrideConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    """Recursively update dataclass fields from a dict."""
    allowed = {f.name for f in fields(instance)}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f"Unknown config key '{key}' for {type(instance).__name__}")
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at the top level")
    return data


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a TrainingConfig from YAML."""
    raw = _load_yaml(path)
    cfg = TrainingConfig()
    _update_dataclass(cfg, raw)
    synchronize_training_config(cfg)

    if cfg.loss.burn_in >= cfg.dataset.seq_len:
        raise ValueError(
            f"loss.burn_in ({cfg.loss.burn_in}) must be < dataset.seq_len ({cfg.dataset.seq_len})"
        )
    return cfg


def synchronize_training_config(cfg: TrainingConfig) -> None:
    """Synchronize mirrored/derived fields across sections."""

    # Keep sequence settings aligned across model and dataset definitions.
    cfg.model.seq_len = cfg.dataset.seq_len
    cfg.model.burn_in = cfg.loss.burn_in
    cfg.model.batch_size = cfg.dataset.batch_size
    cfg.model.learning_rate = cfg.optimization.learning_rate
    cfg.model.weight_decay = cfg.optimization.weight_decay
    cfg.model.max_epochs = cfg.optimization.epochs
    cfg.model.gradient_clip = cfg.optimization.gradient_clip

    if cfg.dataset.resize_height is not None and cfg.dataset.resize_width is not None:
        cfg.model.tokenizer.img_height = cfg.dataset.resize_height
        cfg.model.tokenizer.img_width = cfg.dataset.resize_width

    if cfg.loss.burn_in >= cfg.dataset.seq_len:
        raise ValueError(
            f"loss.burn_in ({cfg.loss.burn_in}) must be < dataset.seq_len ({cfg.dataset.seq_len})"
        )


def apply_dotlist_overrides(cfg: TrainingConfig, overrides: dict[str, Any]) -> None:
    """Apply dotted-key overrides (e.g., optimization.learning_rate=5e-5)."""
    for path, value in overrides.items():
        target: Any = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"Unknown override path '{path}'")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"Unknown override path '{path}'")
        setattr(target, leaf, value)
    synchronize_training_config(cfg)


def training_config_to_dict(cfg: TrainingConfig) -> dict[str, Any]:
    """Serialize config dataclasses to a plain dict."""
    return asdict(cfg)


def save_training_config(cfg: TrainingConfig, path: str | Path) -> None:
    """Write resolved config to YAML."""
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(training_config_to_dict(cfg), f, sort_keys=False)
