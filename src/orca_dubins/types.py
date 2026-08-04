"""Core data types for the ORCA + Dubins fixed-wing prototype.

The prototype is 2D (constant altitude) but the types are intentionally kept
small and explicit so they can be extended to 3D later (e.g. add a vertical
component to :class:`AircraftState` and a climb-rate limit to
:class:`AircraftParams`).

Conventions
-----------
* Positions and velocities are ``numpy`` arrays of shape ``(2,)`` in metres and
  metres/second respectively, in a fixed world frame (x east, y north).
* ``heading`` (psi) is measured in radians, counter-clockwise from the +x axis.
* Fixed-wing aircraft here are modelled as *constant speed*: the control input
  is turn rate (equivalently bank angle), not acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Vector2 = np.ndarray  # semantic alias for an (2,) float array


def vec2(x: float, y: float) -> Vector2:
    """Construct a 2D float vector."""
    return np.array([x, y], dtype=float)


@dataclass
class AircraftParams:
    """Static kinodynamic parameters of a fixed-wing aircraft.

    Attributes
    ----------
    speed:
        Constant cruise speed ``v`` (m/s).
    max_bank_angle:
        Maximum bank angle ``phi_max`` (rad). Together with ``speed`` this
        bounds the achievable turn rate via ``psi_dot = g * tan(phi) / v``.
    radius:
        Collision radius (m) used to build velocity obstacles between agents.
    max_bank_rate:
        Optional bound on how fast bank angle can change (rad/s). ``None`` means
        bank angle can be commanded instantaneously. Reserved for future use.
    """

    speed: float
    max_bank_angle: float
    radius: float
    max_bank_rate: float | None = None


@dataclass
class AircraftState:
    """Instantaneous kinematic state of an aircraft (2D constant altitude)."""

    position: Vector2
    heading: float  # psi, radians

    def velocity(self, speed: float) -> Vector2:
        """Velocity vector implied by heading and a (constant) speed."""
        return speed * np.array([np.cos(self.heading), np.sin(self.heading)])


@dataclass
class Agent:
    """A single aircraft in the world: identity, parameters, state and goal."""

    id: str
    params: AircraftParams
    state: AircraftState
    goal: Vector2
    # Optional cruise-speed preference toward the goal; if None, the mission
    # layer will typically steer at ``params.speed`` toward ``goal``.
    preferred_speed: float | None = None

    def preferred_velocity(self) -> Vector2:
        """Preferred (mission) velocity ``v_pref`` toward the goal.

        Direction points from the current position to the goal; magnitude is
        the preferred cruise speed. This is the ``v_pref`` referenced by the
        planners' objective ``argmin ||v - v_pref||``.
        """
        speed = self.preferred_speed if self.preferred_speed is not None else self.params.speed
        to_goal = self.goal - self.state.position
        dist = float(np.linalg.norm(to_goal))
        if dist < 1e-9:
            return vec2(0.0, 0.0)
        return speed * to_goal / dist


@dataclass
class Snapshot:
    """A recorded frame of the whole world, for visualisation/analysis."""

    time: float
    positions: dict[str, Vector2]
    headings: dict[str, float]
    # Per-agent commanded velocity chosen by the planner at this frame (optional).
    commanded: dict[str, Vector2] = field(default_factory=dict)
