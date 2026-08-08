# orca-dubins

Research prototype for ORCA + Dubins local collision avoidance, geared towards fixed-wing aircraft swarms.

## Idea

ORCA decides which local velocities are safe; Dubins
turns that into a fixed-wing-feasible maneuver. A naive ORCA → Dubins
projection can fail because while ORCA may pick a theoretically safe velocity,
by the time the aircraft turns toward it the collision is already unavoidable. 
Projecting onto a Dubins-feasible maneuver can invalidate ORCA's safety guarantee.

We will instead constrain ORCA to the aircraft's reachable velocity
set over a short horizon `T_h`:

```
v* = argmin_v || v - v_pref ||
subject to
    v ∈ V_ORCA-safe          (collision-avoidance half-planes)
    v ∈ V_Dubins-reachable   (reachable heading arc under fixed-wing dynamics)
```

with the constant-speed fixed-wing model `psi_dot = g·tan(phi)/v`, `|phi| ≤ phi_max`.

Another practical variant skips continuous optimization, instead generating a
small fan of control primitives (max-left, moderate-left, straight,
moderate-right, max-right), propagating each over `T_h`, testing against ORCA's
constraints, and pick the feasible one closest to `v_pref`.

## Bringup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev          # create venv + install deps
uv run pytest                # smoke tests
uv run python examples/run_demo.py --scenario crossing --show
```

The demo uses the **no-avoidance baseline**, so aircraft will fly straight
through each other — that's expected until a real planner is implemented. Swap
in `ReachableOrcaPlanner` / `PrimitiveOrcaPlanner` once their `compute_velocity`
is written.

## Layout

| Theory | File |
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
