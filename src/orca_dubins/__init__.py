"""orca_dubins — Kinodynamically-constrained ORCA + Dubins for fixed-wing swarms.

Research prototype. The simulation, scenarios and visualisation harness are
runnable today via the no-avoidance :class:`~orca_dubins.planners.PreferredVelocityPlanner`
baseline. The core avoidance algorithms (ORCA half-planes, velocity obstacles,
Dubins reachable sets and control primitives) are intentionally left as stubs.
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
