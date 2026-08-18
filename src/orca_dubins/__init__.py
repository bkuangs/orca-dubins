"""orca_dubins — Kinodynamically-constrained ORCA + Dubins for fixed-wing swarms.

Research prototype with runnable ORCA planners, fixed-wing dynamics, classical
Dubins routing, mission guidance, scenarios, metrics, and visualization.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .types import (
    Agent,
    AircraftParams,
    AircraftState,
    Snapshot,
    vec2,
)

__all__ = [
    "__version__",
    "Agent",
    "AircraftParams",
    "AircraftState",
    "Snapshot",
    "vec2",
]
