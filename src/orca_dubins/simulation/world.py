"""Multi-agent simulation loop.

The :class:`World` owns the agents and a planner, advances everyone
synchronously each timestep, and records :class:`Snapshot` frames for
visualisation. The avoidance decision is delegated to the planner; movement is
realised with the fixed-wing kinematics substrate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..dynamics import integrate, max_turn_rate, steer_toward_velocity
from ..planners.base import AvoidancePlanner
from ..types import Agent, Snapshot
from .guidance import MissionGuidance, PointGoalGuidance


@dataclass
class World:
    """A collection of aircraft advanced by a shared planner.

    Parameters
    ----------
    agents:
        The aircraft in the scene.
    planner:
        Avoidance planner applied to every agent (decentralised: each agent is
        planned independently using the others as neighbours).
    guidance:
        Mission layer that supplies preferred velocities and completion state.
    record_diagnostics:
        Record mission and no-neighbor planner commands for visualization.
    dt:
        Control/integration timestep (s).
    horizon:
        Planning horizon ``T_h`` handed to the planner (s).
    goal_tolerance:
        Distance (m) within which an agent is considered to have arrived.
    """

    agents: list[Agent]
    planner: AvoidancePlanner
    guidance: MissionGuidance = field(default_factory=PointGoalGuidance, kw_only=True)
    record_diagnostics: bool = field(default=False, kw_only=True)
    dt: float = 0.1
    horizon: float = 3.0
    goal_tolerance: float = 5.0
    time: float = 0.0
    history: list[Snapshot] = field(default_factory=list)

    def snapshot(
        self,
        commanded: dict[str, np.ndarray] | None = None,
        preferred: dict[str, np.ndarray] | None = None,
        nominal: dict[str, np.ndarray] | None = None,
    ) -> Snapshot:
        return Snapshot(
            time=self.time,
            positions={a.id: a.state.position.copy() for a in self.agents},
            headings={a.id: a.state.heading for a in self.agents},
            commanded=commanded or {},
            preferred=preferred or {},
            nominal=nominal or {},
        )

    def neighbours_of(self, ego: Agent) -> list[Agent]:
        """Return all other agents. Override/extend for sensing-range filtering."""
        return [a for a in self.agents if a.id != ego.id]

    def arrived(self, agent: Agent) -> bool:
        return self.guidance.is_complete(
            agent,
            self.agents,
            self.time,
            self.goal_tolerance,
        )

    def step(self) -> Snapshot:
        """Advance the whole world by one timestep and record a snapshot."""
        commanded: dict[str, np.ndarray] = {}
        preferred: dict[str, np.ndarray] = {}
        nominal: dict[str, np.ndarray] = {}
        # Plan for all agents against the *current* states, then apply together.
        planned = []
        for ego in self.agents:
            if self.arrived(ego):
                planned.append((ego, None))
                continue
            neighbours = self.neighbours_of(ego)
            v_pref = self.guidance.preferred_velocity(
                ego,
                self.agents,
                self.time,
                self.goal_tolerance,
            )
            if self.record_diagnostics:
                preferred[ego.id] = np.asarray(v_pref, dtype=float)
                v_nominal = self.planner.compute_velocity(
                    ego,
                    [],
                    v_pref,
                    self.horizon,
                    self.dt,
                )
                nominal[ego.id] = np.asarray(v_nominal, dtype=float)
                v_star = (
                    self.planner.compute_velocity(
                        ego,
                        neighbours,
                        v_pref,
                        self.horizon,
                        self.dt,
                    )
                    if neighbours
                    else v_nominal
                )
            else:
                v_star = self.planner.compute_velocity(
                    ego,
                    neighbours,
                    v_pref,
                    self.horizon,
                    self.dt,
                )
            commanded[ego.id] = np.asarray(v_star, dtype=float)
            planned.append((ego, np.asarray(v_star, dtype=float)))

        for ego, v_star in planned:
            if v_star is None:
                continue
            rate = steer_toward_velocity(
                ego.state.heading, v_star, max_turn_rate(ego.params), self.dt
            )
            ego.state = integrate(ego.state, rate, ego.params.speed, self.dt)

        self.time += self.dt
        snap = self.snapshot(commanded, preferred, nominal)
        self.history.append(snap)
        return snap

    def run(self, steps: int, record_initial: bool = True) -> list[Snapshot]:
        """Run for a fixed number of steps and return the recorded history."""
        if record_initial and not self.history:
            self.history.append(self.snapshot())
        for _ in range(steps):
            self.step()
        return self.history

    def run_until_complete(
        self,
        max_steps: int,
        record_initial: bool = True,
    ) -> list[Snapshot]:
        """Run until every mission is complete or ``max_steps`` is reached."""
        if record_initial and not self.history:
            self.history.append(self.snapshot())
        for _ in range(max_steps):
            if self.all_arrived():
                break
            self.step()
        return self.history

    def all_arrived(self) -> bool:
        return all(self.arrived(a) for a in self.agents)

    def copy(self) -> "World":
        """Deep copy (handy for running the same scenario with several planners)."""
        return copy.deepcopy(self)
