"""Fixed-wing kinematics helpers.

This module provides the *simulation substrate* — the constant-speed unicycle /
coordinated-turn integration used to move aircraft forward in time — plus small
helpers to convert bank-angle limits into turn-rate limits.

The collision-avoidance *algorithms* live in :mod:`orca_dubins.orca`,
:mod:`orca_dubins.dubins` and :mod:`orca_dubins.planners`; this module is
deliberately algorithm-free so the harness can run end-to-end.
"""

from __future__ import annotations

import numpy as np

from ..types import AircraftParams, AircraftState

G = 9.81  # gravitational acceleration (m/s^2)


def max_turn_rate(params: AircraftParams, g: float = G) -> float:
    """Maximum yaw rate for a coordinated turn: ``psi_dot = g*tan(phi_max)/v``.

    Follows the constant-speed fixed-wing model
    ``psi_dot = g * tan(phi) / v`` with ``|phi| <= phi_max``.
    """
    return g * np.tan(params.max_bank_angle) / params.speed


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def integrate(
    state: AircraftState,
    turn_rate: float,
    speed: float,
    dt: float,
) -> AircraftState:
    """Advance a constant-speed aircraft by ``dt`` under a commanded turn rate.

    Uses an exact arc integration for a constant turn rate (falls back to a
    straight segment as ``turn_rate -> 0``). This is plain kinematics, not an
    avoidance algorithm.
    """
    psi0 = state.heading
    if abs(turn_rate) < 1e-9:
        # Straight segment.
        dp = speed * dt * np.array([np.cos(psi0), np.sin(psi0)])
        new_heading = psi0
    else:
        psi1 = psi0 + turn_rate * dt
        r = speed / turn_rate  # signed turn radius
        dp = r * np.array([np.sin(psi1) - np.sin(psi0), -(np.cos(psi1) - np.cos(psi0))])
        new_heading = psi1
    return AircraftState(position=state.position + dp, heading=wrap_angle(new_heading))


def steer_toward_velocity(
    heading: float,
    desired_velocity: np.ndarray,
    max_rate: float,
    dt: float,
) -> float:
    """Return a turn-rate command that rotates ``heading`` toward a desired velocity.

    The command is the (bang-bang) rate that closes the heading error within
    ``dt``, clamped to ``[-max_rate, max_rate]``. This is a basic tracking
    controller used by the simulation to realise a planner's chosen velocity as
    a feasible fixed-wing turn; it is not part of ORCA or Dubins.
    """
    if float(np.linalg.norm(desired_velocity)) < 1e-9:
        return 0.0
    desired_heading = float(np.arctan2(desired_velocity[1], desired_velocity[0]))
    err = wrap_angle(desired_heading - heading)
    rate = err / dt
    return float(np.clip(rate, -max_rate, max_rate))
