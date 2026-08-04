# orca-dubins

Prototyping **kinodynamically-constrained ORCA + Dubins** for local collision
avoidance in **fixed-wing aircraft swarms**.

> ⚠️ **Research scaffold.** The simulation, scenarios and visualisation harness
> run today, but the core avoidance algorithms (ORCA half-planes, velocity
> obstacles, Dubins reachable sets and control primitives) are intentionally
> left as **stubs** (`raise NotImplementedError`) for you to fill in.

## Idea

ORCA decides which local velocity directions are safe; a Dubins-style planner
turns that into a fixed-wing-feasible maneuver. A naive `ORCA → Dubins`
*projection* can fail: ORCA may pick a theoretically safe velocity, but by the
time the aircraft turns toward it the collision is already unavoidable, and
projecting onto a Dubins-feasible maneuver can invalidate ORCA's safety
guarantee.

The defensible version constrains ORCA to the aircraft's **reachable** velocity
set over a short horizon `T_h`:

```
v* = argmin_v || v - v_pref ||
subject to
    v ∈ V_ORCA-safe          (collision-avoidance half-planes)
    v ∈ V_Dubins-reachable   (reachable heading arc under fixed-wing dynamics)
```

with the constant-speed fixed-wing model `psi_dot = g·tan(phi)/v`, `|phi| ≤ phi_max`.

A practical variant (also scaffolded) skips continuous optimisation: generate a
small fan of **control primitives** (max-left, moderate-left, straight,
moderate-right, max-right), propagate each over `T_h`, test against ORCA's
constraints, and pick the feasible one closest to `v_pref`.

See [`docs/design.md`](docs/design.md) for the full write-up.

## Layout

```
src/orca_dubins/
├── types.py               # Agent / AircraftState / AircraftParams / Snapshot
├── dynamics/              # constant-speed kinematics + turn-rate limits (runnable)
├── orca/                  # velocity obstacles + ORCA half-planes            (STUBS)
├── dubins/                # Dubins paths, control primitives, reachable set  (STUBS)
├── planners/
│   ├── base.py            # AvoidancePlanner interface
│   ├── baseline.py        # PreferredVelocityPlanner — no avoidance (runnable)
│   ├── reachable_orca.py  # kinodynamic ORCA over reachable set             (STUB)
│   └── primitive_orca.py  # discrete control-primitive planner              (STUB)
├── simulation/            # multi-agent World + scenarios (head-on, crossing, swarm)
└── viz/                   # Matplotlib static plots + FuncAnimation playback
examples/run_demo.py       # end-to-end demo (runs with the baseline planner)
tests/test_smoke.py        # harness smoke tests + stub assertions
```

## Getting started

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create venv + install deps
uv run pytest                # smoke tests
uv run python examples/run_demo.py --scenario crossing --show
```

The demo uses the **no-avoidance baseline**, so aircraft will fly straight
through each other — that's expected until a real planner is implemented. Swap
in `ReachableOrcaPlanner` / `PrimitiveOrcaPlanner` once their `compute_velocity`
is written.

## Where to implement

| You want to implement… | File |
|---|---|
| Velocity obstacles | `src/orca_dubins/orca/velocity_obstacle.py` |
| ORCA half-planes / safe set | `src/orca_dubins/orca/orca.py` |
| Dubins reachable set + control primitives | `src/orca_dubins/dubins/primitives.py` |
| Kinodynamic ORCA planner | `src/orca_dubins/planners/reachable_orca.py` |
| Control-primitive planner | `src/orca_dubins/planners/primitive_orca.py` |

## Roadmap

- [ ] Velocity obstacles + ORCA half-planes
- [ ] Dubins reachable arc over `T_h`
- [ ] Control-primitive generation + rollout
- [ ] `ReachableOrcaPlanner.compute_velocity`
- [ ] `PrimitiveOrcaPlanner.compute_velocity`
- [ ] Metrics (min separation, collisions, path efficiency)
- [ ] 3D extension (altitude / climb-rate)

## License

MIT — see [LICENSE](LICENSE).
