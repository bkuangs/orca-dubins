"""Matplotlib visualisation for the fixed-wing avoidance prototype.

Two entry points:

* :func:`plot_trajectories` — static plot of full recorded paths.
* :func:`animate` — :class:`matplotlib.animation.FuncAnimation` playback of a
  recorded :class:`~orca_dubins.types.Snapshot` history.

These are visualisation utilities (harness), not algorithms, so they work today
with the runnable baseline planner.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ..simulation.guidance import (
    FormationGuidance,
    LeaderPathGuidance,
    MissionGuidance,
)
from ..simulation.metrics import avoidance_is_active
from ..types import Agent, Snapshot


def _agent_ids(history: list[Snapshot]) -> list[str]:
    return list(history[0].positions.keys())


def _bounds(history: list[Snapshot], margin: float = 50.0) -> tuple[float, float, float, float]:
    xs = [p[0] for snap in history for p in snap.positions.values()]
    ys = [p[1] for snap in history for p in snap.positions.values()]
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def _leader_guidance(
    guidance: MissionGuidance | None,
) -> LeaderPathGuidance | None:
    if isinstance(guidance, FormationGuidance):
        return guidance.leader_guidance
    if isinstance(guidance, LeaderPathGuidance):
        return guidance
    return None


def _formation_slot(
    snapshot: Snapshot,
    guidance: FormationGuidance,
    follower_id: str,
) -> np.ndarray:
    leader_id = guidance.leader_guidance.leader_id
    return guidance.slot_position_from_pose(
        snapshot.positions[leader_id],
        snapshot.headings[leader_id],
        follower_id,
        snapshot.time,
    )


def _running_minimum_separations(history: list[Snapshot]) -> list[float]:
    agent_ids = _agent_ids(history)
    minimum = float("inf")
    running = []
    for snapshot in history:
        for index, first in enumerate(agent_ids):
            for second in agent_ids[index + 1:]:
                minimum = min(
                    minimum,
                    float(
                        np.linalg.norm(
                            snapshot.positions[first]
                            - snapshot.positions[second]
                        )
                    ),
                )
        running.append(minimum)
    return running


def plot_trajectories(
    history: list[Snapshot],
    agents: list[Agent] | None = None,
    ax: plt.Axes | None = None,
    title: str = "trajectories",
    guidance: MissionGuidance | None = None,
):
    """Plot full recorded trajectories with start/goal markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ids = _agent_ids(history)
    cmap = plt.get_cmap("tab20" if len(ids) > 10 else "tab10")
    for i, aid in enumerate(ids):
        color = cmap(i % cmap.N)
        xy = np.array([snap.positions[aid] for snap in history])
        label = aid if len(ids) <= 10 or aid == "L" else "_nolegend_"
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color, label=label, lw=1.5)
        ax.plot(xy[0, 0], xy[0, 1], "o", color=color, ms=6)  # start
        ax.plot(xy[-1, 0], xy[-1, 1], "s", color=color, ms=6)  # end
    leader_guidance = _leader_guidance(guidance)
    if leader_guidance is not None:
        route = leader_guidance.route_points()
        if len(route) > 0:
            ax.plot(
                route[:, 0],
                route[:, 1],
                "k--",
                lw=1.2,
                alpha=0.7,
                label="leader reference",
            )
        ax.plot(
            leader_guidance.goal_state.position[0],
            leader_guidance.goal_state.position[1],
            "k*",
            ms=14,
        )

    if isinstance(guidance, FormationGuidance):
        for i, follower_id in enumerate(guidance.slots, start=1):
            slot_path = np.array([
                _formation_slot(snapshot, guidance, follower_id)
                for snapshot in history
            ])
            ax.plot(
                slot_path[:, 0],
                slot_path[:, 1],
                ":",
                color=cmap(i % cmap.N),
                lw=0.8,
                alpha=0.45,
            )

    if agents is not None and not isinstance(guidance, FormationGuidance):
        for i, agent in enumerate(agents):
            color = cmap(i % cmap.N)
            start = history[0].positions[agent.id]
            ax.plot(
                [start[0], agent.goal[0]],
                [start[1], agent.goal[1]],
                "--",
                color=color,
                lw=0.8,
                alpha=0.25,
                zorder=0,
            )
            ax.plot(agent.goal[0], agent.goal[1], "*", color=color, ms=14, mec="k")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def animate(
    history: list[Snapshot],
    agents: list[Agent] | None = None,
    radius: float | dict[str, float] | None = None,
    interval_ms: int = 40,
    trail: int = 40,
    title: str = "ORCA + Dubins fixed-wing",
    guidance: MissionGuidance | None = None,
) -> FuncAnimation:
    """Animate a recorded history. Returns the ``FuncAnimation`` (keep a ref!).

    ``radius`` may be a scalar, a per-id mapping, or ``None`` to skip collision
    discs. Save with ``anim.save("out.mp4")`` or view with ``plt.show()``.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ids = _agent_ids(history)
    cmap = plt.get_cmap("tab20" if len(ids) > 10 else "tab10")
    xmin, xmax, ymin, ymax = _bounds(history)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    def radius_for(aid: str) -> float | None:
        if isinstance(radius, dict):
            return radius.get(aid)
        return radius

    leader_guidance = _leader_guidance(guidance)
    if leader_guidance is not None:
        route = leader_guidance.route_points()
        if len(route) > 0:
            ax.plot(
                route[:, 0],
                route[:, 1],
                "k--",
                lw=1.2,
                alpha=0.7,
                label="leader reference",
            )
        ax.plot(
            leader_guidance.goal_state.position[0],
            leader_guidance.goal_state.position[1],
            "k*",
            ms=14,
        )

    if agents is not None and not isinstance(guidance, FormationGuidance):
        for i, agent in enumerate(agents):
            color = cmap(i % cmap.N)
            start = history[0].positions[agent.id]
            ax.plot(
                [start[0], agent.goal[0]],
                [start[1], agent.goal[1]],
                "--",
                color=color,
                lw=0.8,
                alpha=0.25,
                zorder=0,
            )
            ax.plot(agent.goal[0], agent.goal[1], "*", color=color, ms=14, mec="k")

    markers, trails, discs, colors = {}, {}, {}, {}
    for i, aid in enumerate(ids):
        color = cmap(i % cmap.N)
        colors[aid] = color
        label = aid if len(ids) <= 10 or aid == "L" else "_nolegend_"
        agent_radius = radius_for(aid)
        marker = "." if agent_radius is not None else "o"
        marker_size = 2 if agent_radius is not None else 7
        (markers[aid],) = ax.plot(
            [],
            [],
            marker,
            color=color,
            ms=marker_size,
            label=label,
            zorder=3,
        )
        (trails[aid],) = ax.plot([], [], "-", color=color, lw=1.0, alpha=0.6)
        if agent_radius is not None:
            disc = plt.Circle(
                (0, 0),
                agent_radius,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                alpha=0.3,
                zorder=2,
            )
            ax.add_patch(disc)
            discs[aid] = disc

    slot_markers = {}
    if isinstance(guidance, FormationGuidance):
        for i, follower_id in enumerate(guidance.slots, start=1):
            (slot_markers[follower_id],) = ax.plot(
                [],
                [],
                "x",
                color=cmap(i % cmap.N),
                ms=7,
                alpha=0.7,
            )

    if any(snapshot.nominal for snapshot in history):
        ax.plot(
            [],
            [],
            "o",
            markerfacecolor="none",
            markeredgecolor="red",
            label="ORCA active",
        )

    ax.legend(loc="best", fontsize=8)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)
    running_minimum = _running_minimum_separations(history)

    def update(frame: int):
        snap = history[frame]
        lo = max(0, frame - trail)
        active_by_id = {
            aid: avoidance_is_active(snap, aid)
            for aid in ids
        }
        for aid in ids:
            p = snap.positions[aid]
            markers[aid].set_data([p[0]], [p[1]])
            active = active_by_id[aid]
            markers[aid].set_markeredgecolor("red" if active else colors[aid])
            xy = np.array([history[f].positions[aid] for f in range(lo, frame + 1)])
            trails[aid].set_data(xy[:, 0], xy[:, 1])
            if aid in discs:
                discs[aid].center = (p[0], p[1])
                discs[aid].set_edgecolor("red" if active else colors[aid])
                discs[aid].set_linewidth(2.5 if active else 1.0)
        if isinstance(guidance, FormationGuidance):
            for follower_id, marker in slot_markers.items():
                slot = _formation_slot(snap, guidance, follower_id)
                marker.set_data([slot[0]], [slot[1]])
        separation = running_minimum[frame]
        separation_text = (
            ""
            if not np.isfinite(separation)
            else f"\nmin separation = {separation:5.1f} m"
        )
        active_count = sum(active_by_id.values())
        time_text.set_text(
            f"t = {snap.time:5.1f} s"
            f"{separation_text}"
            f"\nactive avoidances = {active_count}"
        )
        return (
            list(markers.values())
            + list(trails.values())
            + list(discs.values())
            + list(slot_markers.values())
            + [time_text]
        )

    return FuncAnimation(fig, update, frames=len(history), interval=interval_ms, blit=False)
