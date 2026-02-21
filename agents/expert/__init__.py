"""Custom privileged expert agent for CARLA data collection."""

# Guard against missing carla package (e.g. CI / test environments).
try:
    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig, FrameInfo
except ImportError:
    pass

__all__ = ["PrivilegedExpert", "ExpertConfig", "FrameInfo"]
