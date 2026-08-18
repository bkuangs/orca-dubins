"""Mission-level guidance for the simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..dubins import DubinsPath, dubins_shortest_path, sample_dubins_path
from ..dynamics import max_turn_rate, wrap_angle
from ..types import Agent, AircraftState


class MissionGuidance(Protocol):
    """Provide mission intent independently of local collision avoidance."""

    def preferred_velocity(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> np.ndarray:
        """Return ego's preferred velocity before collision avoidance."""
        ...

    def is_complete(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> bool:
        """Return whether ego has completed its mission."""
        ...


class PointGoalGuidance:
    """Preserve the existing behavior of steering each agent to its point goal."""

    def preferred_velocity(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> np.ndarray:
        return ego.preferred_velocity()

    def is_complete(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> bool:
        return float(np.linalg.norm(ego.goal - ego.state.position)) <= tolerance


@dataclass(frozen=True)
class SlotSwap:
    """Exchange two followers' assigned slots at a mission time."""

    time: float
    first_id: str
    second_id: str


@dataclass
class LeaderPathGuidance:
    """Guide one leader along a Dubins path to a terminal pose."""

    leader_id: str
    goal_state: AircraftState
    lookahead_distance: float | None = None
    replan_distance: float | None = None
    cross_track_gain: float = 0.5
    heading_tolerance: float = float(np.deg2rad(10.0))
    _route_start: AircraftState | None = field(default=None, init=False)
    _path: DubinsPath | None = field(default=None, init=False)
    _sample_distances: np.ndarray | None = field(default=None, init=False)
    _sample_positions: np.ndarray | None = field(default=None, init=False)
    _progress: float = field(default=0.0, init=False)
    _turn_radius: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.leader_id:
            raise ValueError("leader_id must not be empty")
        if self.lookahead_distance is not None and (
            not np.isfinite(self.lookahead_distance)
            or self.lookahead_distance <= 0.0
        ):
            raise ValueError("lookahead_distance must be positive and finite")
        if self.replan_distance is not None and (
            not np.isfinite(self.replan_distance)
            or self.replan_distance <= 0.0
        ):
            raise ValueError("replan_distance must be positive and finite")
        if not np.isfinite(self.cross_track_gain) or self.cross_track_gain <= 0.0:
            raise ValueError("cross_track_gain must be positive and finite")
        if not np.isfinite(self.heading_tolerance) or self.heading_tolerance < 0.0:
            raise ValueError("heading_tolerance must be nonnegative and finite")

    @property
    def path(self) -> DubinsPath | None:
        return self._path

    @property
    def route_start(self) -> AircraftState | None:
        return self._route_start

    @property
    def progress(self) -> float:
        return self._progress

    def route_points(self, spacing: float = 5.0) -> np.ndarray:
        """Sample the active route for visualization and metrics."""
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("spacing must be positive and finite")
        if self._path is None or self._route_start is None:
            return np.empty((0, 2), dtype=float)

        sample_count = max(2, int(np.ceil(self._path.length / spacing)) + 1)
        distances = np.linspace(0.0, self._path.length, sample_count)
        return np.array([
            sample_dubins_path(
                self._route_start,
                self._path,
                distance,
            ).position
            for distance in distances
        ])

    def _require_leader(self, ego: Agent) -> None:
        if ego.id != self.leader_id:
            raise ValueError(
                f"LeaderPathGuidance expected {self.leader_id!r}, got {ego.id!r}"
            )

    def _plan_from(self, leader: Agent) -> None:
        turn_rate = float(max_turn_rate(leader.params))
        if not np.isfinite(turn_rate) or turn_rate <= 0.0:
            raise ValueError("leader must have a positive finite turn rate")

        turn_radius = leader.params.speed / turn_rate
        route_start = AircraftState(
            position=leader.state.position.copy(),
            heading=leader.state.heading,
        )
        path = dubins_shortest_path(
            route_start,
            self.goal_state,
            turn_radius,
        )
        lookahead = self.lookahead_distance or 0.1 * turn_radius
        spacing = min(max(turn_radius / 20.0, 0.1), lookahead / 2.0)
        sample_count = max(2, int(np.ceil(path.length / spacing)) + 1)
        sample_distances = np.linspace(0.0, path.length, sample_count)

        self._route_start = route_start
        self._path = path
        self._sample_distances = sample_distances
        self._sample_positions = np.array([
            sample_dubins_path(route_start, path, distance).position
            for distance in sample_distances
        ])
        self._progress = 0.0
        self._turn_radius = turn_radius

    def _update_progress(self, leader: Agent) -> None:
        if self._path is None:
            self._plan_from(leader)

        assert self._sample_distances is not None
        assert self._sample_positions is not None
        assert self._turn_radius is not None

        lookahead = self.lookahead_distance or 0.1 * self._turn_radius
        lower = max(0.0, self._progress - lookahead)
        upper = min(self._path.length, self._progress + 2.0 * lookahead)
        candidates = np.flatnonzero(
            (self._sample_distances >= lower)
            & (self._sample_distances <= upper)
        )
        candidate_positions = self._sample_positions[candidates]
        errors = np.linalg.norm(
            candidate_positions - leader.state.position,
            axis=1,
        )
        nearest_index = int(candidates[int(np.argmin(errors))])
        cross_track_error = float(
            np.linalg.norm(
                self._sample_positions[nearest_index] - leader.state.position
            )
        )
        replan_distance = self.replan_distance or 2.0 * self._turn_radius
        if cross_track_error > replan_distance:
            self._plan_from(leader)
            return

        self._progress = max(
            self._progress,
            float(self._sample_distances[nearest_index]),
        )

    def preferred_velocity(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> np.ndarray:
        self._require_leader(ego)
        self._update_progress(ego)

        assert self._path is not None
        assert self._route_start is not None
        assert self._turn_radius is not None

        lookahead = self.lookahead_distance or 0.1 * self._turn_radius
        target_distance = min(self._progress + lookahead, self._path.length)
        target = sample_dubins_path(
            self._route_start,
            self._path,
            target_distance,
        )
        to_target = target.position - ego.state.position
        preferred = (
            target.velocity(ego.params.speed)
            + self.cross_track_gain * to_target
        )
        preferred_norm = float(np.linalg.norm(preferred))
        if preferred_norm < 1e-9:
            return target.velocity(ego.params.speed)
        return ego.params.speed * preferred / preferred_norm

    def is_complete(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> bool:
        self._require_leader(ego)
        position_error = float(
            np.linalg.norm(self.goal_state.position - ego.state.position)
        )
        heading_error = abs(wrap_angle(self.goal_state.heading - ego.state.heading))
        return (
            position_error <= tolerance
            and heading_error <= self.heading_tolerance
        )


@dataclass
class FormationGuidance:
    """Guide followers toward body-frame slots around a Dubins leader."""

    leader_guidance: LeaderPathGuidance
    slots: dict[str, np.ndarray]
    position_gain: float = 0.5
    slot_swaps: tuple[SlotSwap, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_gain) or self.position_gain <= 0.0:
            raise ValueError("position_gain must be positive and finite")
        if self.leader_guidance.leader_id in self.slots:
            raise ValueError("slots must not include the leader")

        self.slots = {
            agent_id: np.asarray(offset, dtype=float).copy()
            for agent_id, offset in self.slots.items()
        }
        for offset in self.slots.values():
            if offset.shape != (2,) or not np.all(np.isfinite(offset)):
                raise ValueError("formation offsets must be finite 2D vectors")
        self.slot_swaps = tuple(sorted(self.slot_swaps, key=lambda swap: swap.time))
        for swap in self.slot_swaps:
            if not np.isfinite(swap.time) or swap.time < 0.0:
                raise ValueError("slot swap times must be nonnegative and finite")
            if swap.first_id == swap.second_id:
                raise ValueError("slot swaps require two different followers")
            if swap.first_id not in self.slots or swap.second_id not in self.slots:
                raise ValueError("slot swaps must reference configured followers")

    def _leader(self, agents: list[Agent]) -> Agent:
        for agent in agents:
            if agent.id == self.leader_guidance.leader_id:
                return agent
        raise ValueError(
            f"leader {self.leader_guidance.leader_id!r} is not in the world"
        )

    def slot_offset(self, follower_id: str, time: float) -> np.ndarray:
        """Return the body-frame slot assigned to a follower at ``time``."""
        if follower_id not in self.slots:
            raise ValueError(
                f"no formation slot configured for {follower_id!r}"
            )

        assigned_slot = follower_id
        for swap in self.slot_swaps:
            if swap.time > time:
                break
            if assigned_slot == swap.first_id:
                assigned_slot = swap.second_id
            elif assigned_slot == swap.second_id:
                assigned_slot = swap.first_id
        return self.slots[assigned_slot]

    def slot_position(
        self,
        leader: Agent,
        follower_id: str,
        time: float,
    ) -> np.ndarray:
        """Return a follower's assigned slot in world coordinates."""
        return self.slot_position_from_pose(
            leader.state.position,
            leader.state.heading,
            follower_id,
            time,
        )

    def slot_position_from_pose(
        self,
        leader_position: np.ndarray,
        leader_heading: float,
        follower_id: str,
        time: float,
    ) -> np.ndarray:
        """Transform a follower's assigned slot from body to world coordinates."""
        offset = self.slot_offset(follower_id, time)
        c = np.cos(leader_heading)
        s = np.sin(leader_heading)
        rotation = np.array([[c, -s], [s, c]])
        return leader_position + rotation @ offset

    def preferred_velocity(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> np.ndarray:
        leader = self._leader(agents)
        if ego.id == leader.id:
            return self.leader_guidance.preferred_velocity(
                ego,
                agents,
                time,
                tolerance,
            )

        slot_position = self.slot_position(leader, ego.id, time)
        correction = self.position_gain * (slot_position - ego.state.position)
        if self.leader_guidance.is_complete(
            leader,
            agents,
            time,
            tolerance,
        ):
            preferred = correction
        else:
            preferred = leader.state.velocity(leader.params.speed) + correction
        preferred_norm = float(np.linalg.norm(preferred))
        if preferred_norm < 1e-9:
            return ego.state.velocity(ego.params.speed)
        return ego.params.speed * preferred / preferred_norm

    def is_complete(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> bool:
        leader = self._leader(agents)
        leader_complete = self.leader_guidance.is_complete(
            leader,
            agents,
            time,
            tolerance,
        )
        return leader_complete
