# trajkit

*Vehicle GPS track conditioning: EKF/RTS sensor fusion, G2-continuous clothoid approximation, and curvature-adaptive waypoint sampling.*

## Features

- **Sample-and-hold removal** — recovers true GPS update epochs from high-rate DAQ streams (500–2000 Hz)
- **Extended Kalman Filter** — CTRV motion model fusing GPS + wheel speed + steering angle (bicycle model yaw rate)
- **RTS smoother** — offline backward pass for optimal state estimates
- **G2 clothoid approximation** — Bertolazzi-Frego SolveG2 chain with guaranteed max error < 0.5 m
- **Curvature-adaptive sampling** — variable-density waypoints driven by curvature or cumulative heading change
- **Numba JIT** — 14× speedup over pure Python; optional JAX backend for GPU/TPU

## Installation

```bash
pip install trajkit

# With JAX GPU support
pip install "trajkit[jax]"
```

## Quick Start

```python
from trajkit import GPSProcessor

# Kalman pipeline (sensor fusion)
proc = GPSProcessor(v_max_kmh=250, smoothing="kalman")
track = proc.process(
    lon_raw, lat_raw, v_wheel_ms,
    steering_angle_deg=steering_wheel_angle,
)

# G2 clothoid fit
proc = GPSProcessor(v_max_kmh=250, smoothing="g2")
track = proc.process(lon_raw, lat_raw, v_wheel_ms,
                     steering_angle_deg=steering_wheel_angle)
```

## Smoothing Modes

| Mode | Backend | Use Case |
|------|---------|----------|
| `butterworth` | SciPy | Quick visualization, position-only |
| `kalman` | Numba | Production: best accuracy (4 s / 850k samples) |
| `kalman_jax` | JAX/XLA | GPU clusters, batch vmap |
| `g2` | Numba + pyclothoids | G2 clothoid chain (max < 0.5 m lateral error) |

## Documentation

Full API reference, tuning guide, and mathematical background are included in the
package under `trajkit.docs` (accessible via `importlib.resources`).

## Requirements

- Python ≥ 3.10
- numpy, scipy, numba, matplotlib, contextily
- Optional: jax, jaxlib

## License

BSD 2 — the project contributors
