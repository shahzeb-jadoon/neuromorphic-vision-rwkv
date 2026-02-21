"""Data collection, storage, and loading utilities."""

from src.data.driving_dataset import DrivingSequenceDataset
from src.data import hdf5_schema

# CARLA-dependent modules are imported lazily so the package is
# importable in environments without the carla wheel (e.g. CI).
try:
    from src.data.expert_collector import (
        ExpertDataCollector,
        EpisodeConfig,
        EpisodeResult,
    )
    from src.data.route_generator import BalancedRouteGenerator
except ImportError:
    pass

__all__ = [
    "DrivingSequenceDataset",
    "ExpertDataCollector",
    "EpisodeConfig",
    "EpisodeResult",
    "BalancedRouteGenerator",
    "hdf5_schema",
]
