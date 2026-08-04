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

from ..types import Agent, Snapshot


def _agent_ids(history: list[Snapshot]) -> list[str]:
    return list(history[0].positions.keys())


def _bounds(history: list[Snapshot], margin: float = 50.0) -> tuple[float, float, float, float]:
    xs = [p[0] for snap in history for p in snap.positions.values()]
    ys = [p[1] for snap in history for p in snap.positions.values()]
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def plot_trajectories(
    history: list[Snapshot],
    agents: list[Agent] | None = None,
    ax: plt.Axes | None = None,
    title: str = "trajectories",
):
    """Plot full recorded trajectories with start/goal markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ids = _agent_ids(history)
    cmap = plt.get_cmap("tab10")
    for i, aid in enumerate(ids):
        color = cmap(i % 10)
        xy = np.array([snap.positions[aid] for snap in history])
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color, label=aid, lw=1.5)
        ax.plot(xy[0, 0], xy[0, 1], "o", color=color, ms=6)  # start
        ax.plot(xy[-1, 0], xy[-1, 1], "s", color=color, ms=6)  # end
    if agents is not None:
        for i, agent in enumerate(agents):
            ax.plot(agent.goal[0], agent.goal[1], "*", color=cmap(i % 10), ms=14, mec="k")
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
) -> FuncAnimation:
    """Animate a recorded history. Returns the ``FuncAnimation`` (keep a ref!).

    ``radius`` may be a scalar, a per-id mapping, or ``None`` to skip collision
    discs. Save with ``anim.save("out.mp4")`` or view with ``plt.show()``.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ids = _agent_ids(history)
    cmap = plt.get_cmap("tab10")
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

    if agents is not None:
        for i, agent in enumerate(agents):
            ax.plot(agent.goal[0], agent.goal[1], "*", color=cmap(i % 10), ms=14, mec="k")

    markers, trails, discs = {}, {}, {}
    for i, aid in enumerate(ids):
        color = cmap(i % 10)
        (markers[aid],) = ax.plot([], [], "o", color=color, ms=7, label=aid)
        (trails[aid],) = ax.plot([], [], "-", color=color, lw=1.0, alpha=0.6)
        r = radius_for(aid)
        if r is not None:
            disc = plt.Circle((0, 0), r, color=color, alpha=0.15)
            ax.add_patch(disc)
            discs[aid] = disc
    ax.legend(loc="best", fontsize=8)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)

    def update(frame: int):
        snap = history[frame]
        lo = max(0, frame - trail)
        for aid in ids:
            p = snap.positions[aid]
            markers[aid].set_data([p[0]], [p[1]])
            xy = np.array([history[f].positions[aid] for f in range(lo, frame + 1)])
            trails[aid].set_data(xy[:, 0], xy[:, 1])
            if aid in discs:
                discs[aid].center = (p[0], p[1])
        time_text.set_text(f"t = {snap.time:5.1f} s")
        return list(markers.values()) + list(trails.values()) + list(discs.values()) + [time_text]

    return FuncAnimation(fig, update, frames=len(history), interval=interval_ms, blit=False)
