# trajkit

*Vehicle GPS track conditioning: EKF/RTS sensor fusion, G2-continuous clothoid approximation, and curvature-adaptive waypoint sampling.*

## Features

- **Sample-and-hold removal** — recovers true GPS update epochs from high-rate DAQ streams (500–2000 Hz)
- **Extended Kalman Filter** — CTRV motion model fusing GPS + wheel speed + steering angle (bicycle model yaw rate)
- **RTS smoother** — offline backward pass for optimal state estimates
- **ABS braking robustness** — speed interpolation + Butterworth position blend during ABS events (19× better position accuracy: 90 cm → 5 cm)
- **GPS freeze repair** — detects and interpolates single-axis "stuck" GPS coordinates (optional, `freeze_repair=True`)
- **G2 clothoid approximation** — Bertolazzi-Frego SolveG2 chain with guaranteed max error < 0.5 m
- **Curvature-adaptive sampling** — variable-density waypoints driven by curvature (`sample_adaptive`) or cumulative heading change (`sample_angular_step`)
- **Numba JIT** — 14× speedup over pure Python; optional JAX backend for GPU/TPU

## Installation

Not yet published to PyPI — install directly from the repo:

```bash
pip install -e /path/to/trajkit

# With JAX GPU support
pip install -e "/path/to/trajkit[jax]"
```

## Quick Start

```python
from trajkit import GPSProcessor

# Kalman pipeline (sensor fusion, with ABS-robust blending)
proc = GPSProcessor(v_max_kmh=250, smoothing="kalman")
track = proc.process(
    lon_raw, lat_raw, v_wheel_ms,
    steering_angle_deg=steering_wheel_angle,
    abs_flag=abs_active_flag,
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
| `kalman` | Numba | Production: best accuracy (4.1 s / 850k samples) |
| `kalman_jax` | JAX/XLA | GPU clusters, batch vmap (7.7 s CPU / ~0.5 s GPU) |
| `g2` | Numba + SciPy | G2 clothoid chain (max < 0.5 m lateral error) |

## Documentation

Full API reference, tuning guide, and mathematical background live in
[`README.md`](README.md) and the Typst technical write-ups
(`kalman_description.typ`, `clothoid_description.typ`).

## Requirements

- Python ≥ 3.10
- numpy, scipy, numba, matplotlib, contextily
- Optional: jax, jaxlib

## License

BSD 2 — the project contributors
