# trajkit

*Vehicle GPS track conditioning: EKF/RTS sensor fusion, G2-continuous clothoid approximation, and curvature-adaptive waypoint sampling.*

Handles the complete signal chain from raw sample-and-hold GPS signals to
filtered, speed-annotated trajectories with G2-continuous clothoid curvature
analysis and curvature-adaptive waypoint sampling.

## Installation

**Development install** (editable, no copy):

```bash
pip install -e /path/to/trajkit
```

**With optional JAX support:**

```bash
pip install -e "/path/to/trajkit[jax]"
```

Then import as:

```python
from trajkit import GPSProcessor, GPSTrack
```

Dependencies: `numpy`, `scipy`, `numba`, `contextily`, `matplotlib`  
Optional: `jax`, `jaxlib` (GPU/TPU acceleration)

## Module Structure

```
trajkit/                      (installed via: pip install -e /path/to/trajkit)
├── src/trajkit/
│   ├── __init__.py      # Public API: GPSProcessor, GPSTrack, GPSKalmanFilter,
│   │                    #   KalmanConfig, KalmanResult,
│   │                    #   ClothoidSegment, G2ClothoidApproximator,
│   │                    #   G2ClothoidConfig, G2ClothoidFitResult
│   ├── processor.py     # GPSTrack (dataclass), GPSProcessor (main class)
│   ├── kalman.py        # KalmanConfig, KalmanResult, GPSKalmanFilter (Numba/CPU)
│   ├── jax_kalman.py    # JAXKalmanFilter (XLA/GPU-ready, lax.scan)
│   └── clothoid.py      # ClothoidSegment, G2ClothoidApproximator,
│                        #   G2ClothoidConfig, G2ClothoidFitResult (Bertolazzi-Frego SolveG2)
├── docs/                # Typst technical documentation
├── notebooks/           # Usage examples
└── pyproject.toml
```

## Quick Start

```python
from trajkit import GPSProcessor

# Butterworth pipeline (fast, position-only)
proc = GPSProcessor(v_max_kmh=250)
track = proc.process(lon_raw, lat_raw, v_reference_ms)

# Kalman pipeline (sensor fusion: GPS + wheel speed + steering angle)
proc = GPSProcessor(v_max_kmh=250, smoothing="kalman")
track = proc.process(lon_raw, lat_raw, v_reference_ms,
                     steering_angle_deg=steering_wheel_angle,
                     abs_flag=abs_active_flag)  # ABS_FLAG channel

# Plot with basemap
fig = proc.plot(track, basemap=True, color_by="speed")
```

## Processing Pipeline

```
Raw GPS (S&H @ f_s Hz, typically ≥ 500 Hz)
     │
     ▼
0a. Compose Coordinates               (if lon_int/lat_int provided)
     │                                 → round(int_part) + fraction
     ▼
0b. Mark Missing Fixes                 → lon==0 | lat==0 → NaN
     │
     ▼
1. Detect GPS update epochs          _detect_updates()
     │                                 → 5–20 Hz actual GPS within high-rate stream
     ▼
2. Remove Sample-and-Hold             _interpolate()
     │                                 → np.interp between update epochs
     ▼
3. Calibrate Sampling Rate            _calibrate_dt()
     │                                 → median(dist / (WHEEL_SPEED_KMH × gaps))
     │                                 → dt auto-calibrated from data
     ▼
4. Position Smoothing                 (one of:)
     ├─ butterworth                    butter(N=2) + filtfilt, f_c = 7 Hz
     ├─ kalman                         EKF + RTS (Numba, CPU)
     ├─ kalman_jax                     EKF + RTS (JAX/XLA, GPU-ready)
     └─ g2                             Kalman + G2ClothoidApproximator (Bertolazzi-Frego SolveG2)
     │
     ▼
5. Speed Derivation
     ├─ butterworth: WHEEL_SPEED_KMH ÷ 3.6 pass-through (position-diff too noisy)
     └─ kalman:      EKF state estimate (fuses GPS + WHEEL_SPEED_KMH/3.6 + ω)
     │
     ▼
6. Outlier Masking                    speed > v_max_kmh → flagged
     │
     ▼
GPSTrack (dataclass)
```

## Smoothing Modes

| Mode | Backend | Speed Source | Latency | Use Case |
|------|---------|--------------|---------|----------|
| `butterworth` | SciPy | WHEEL_SPEED_KMH (pass-through) | ≈ 0s | Quick visualization, position-only |
| `kalman` | Numba | EKF state | 4.1s (850k) | Production: best accuracy |
| `kalman_jax` | JAX/XLA | EKF state | 7.7s CPU / ~0.5s GPU | GPU clusters, batch vmap |
| `g2` | Numba + SciPy | EKF state | ~3 s / 4 km | G2 clothoid chain (Bertolazzi-Frego SolveG2, max < 0.5 m) |

## API Reference

### `GPSProcessor`

```python
GPSProcessor(
    v_max_kmh=250.0,             # Outlier speed threshold [km/h]
    filter_order=2,              # Butterworth filter order
    cutoff_factor=0.5,           # LP cutoff as fraction of GPS Nyquist
    min_speed_calibration=1.0,   # Min wheel speed for dt calibration [m/s]
    smoothing='butterworth',     # 'butterworth' | 'kalman' | 'kalman_jax' | 'g2'
    speed_smoothing_window=501,  # Savgol window (Butterworth mode, 0=disabled)
    kalman_config=None,          # KalmanConfig — EKF tuning (dt auto-calibrated if None)
    g2_config=None,              # G2ClothoidConfig — SolveG2 fit params (smoothing='g2')
)
```

Config objects and their defaults:

| Parameter | Config class | Key fields |
|---|---|---|
| `kalman_config` | `KalmanConfig` | `sigma_pos_gps`, `sigma_v_wheel`, `steering_ratio=15.5`, `wheelbase_m=2.68` |
| `g2_config` | `G2ClothoidConfig` | `position_tolerance_m=0.5`, `resample_spacing_m=2.0`, `kappa_clip` |

All two default to `None`, in which case sensible defaults are used automatically.

#### `.process(lon_raw, lat_raw, v_reference_ms, aux_channels=None, yaw_rate_rad=None, steering_angle_deg=None, abs_flag=None, lon_int=None, lat_int=None)`

Run the full pipeline. Returns `GPSTrack`.

**GPS coordinate composition** (for split-channel daq data):
- `lon_int` / `lat_int` — integer parts of GPS coordinates (`GPS_LON_INT`, `GPS_LAT_INT`).
  When provided, the integer parts are rounded to the nearest integer (correcting
  float rounding artifacts like 8.9999 → 9) and summed with `lon_raw` / `lat_raw`
  (the fractional parts `GPS_LON`, `GPS_LAT`) to form the full coordinate. A one-time
  `logger.warning()` is emitted if rounding corrections were applied.
- **Zero-coordinate handling**: Samples where `lon == 0` or `lat == 0` are treated
  as missing GPS fixes and replaced with NaN. These epochs are excluded from
  `update_idx` so the interpolation bridges over the gaps seamlessly.

**Yaw rate input** (required for `kalman` mode):
- `steering_angle_deg` — signed steering WHEEL angle [°] (e.g. `STEER_ANGLE_DEG`).  
  **Takes priority** over `yaw_rate_rad` when both are provided (a warning is logged).  
  Internally converted via bicycle model: `ω = v · tan(δ / steering_ratio) / wheelbase_m`  
  using `steering_ratio` and `wheelbase_m` from `kalman_config` (defaults: 15.5, 2.68 m).
  To use custom vehicle parameters, pass a `KalmanConfig` with adjusted values.
- `yaw_rate_rad` — pre-computed signed yaw rate [rad/s]. Only used if
  `steering_angle_deg` is not provided.

**ABS speed clamping** (optional but recommended):
- `abs_flag` — binary ABS active flag (1 = ABS intervening, 0 = normal).  
  Channel: `ABS_FLAG` in the daq data.  
  When provided, wheel speed is clamped to the last pre-ABS value during ABS
  events before computing yaw rate via the bicycle model. This prevents
  ABS-induced wheel-speed oscillations (~15 Hz) from corrupting the heading
  estimate and degrading the Kalman-filtered trajectory.

### `G2ClothoidApproximator` (G2-continuous clothoid chain)

Fits a G2-continuous chain of 3-arc clothoid intervals (Bertolazzi-Frego `SolveG2`)  
directly to the Kalman track. A greedy forward-search + bisection finds the minimum  
number of support points subject to a lateral tolerance.

```python
from trajkit import G2ClothoidApproximator, G2ClothoidConfig

g2_cfg = G2ClothoidConfig(
    gps_sigma_m=2.0,
    resample_spacing_m=2.0,      # 2 m uniform arc-length grid
    kappa_smooth_window=11,      # moving-average window (samples)
    kappa_clip=0.20,             # max |κ| [1/m], R_min=5 m; None for clean tracks
    position_tolerance_m=0.5,    # max lateral error per interval [m]
)
g2_fit = G2ClothoidApproximator(g2_cfg).fit(x_m, y_m)
print(g2_fit.n_intervals, g2_fit.n_segments)       # e.g. 80 × 3 = 240
print(g2_fit.max_error_m, g2_fit.median_error_m)   # e.g. 0.499 m, 0.06 m
print(g2_fit.runtime_s)                            # ~3 s for 3.86 km / 200k samples
```

Key `G2ClothoidConfig` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gps_sigma_m` | 2.0 m | B-spline smoothing budget: `s = n × σ²` |
| `dedup_threshold_m` | 1e-3 m | Remove S&H duplicate GPS points before spline fitting |
| `resample_spacing_m` | 2.0 m | Uniform arc-length grid spacing |
| `kappa_smooth_window` | 11 | Moving-average window for analytic κ |
| `kappa_clip` | None | Max \|κ\| [1/m] — set ~0.20 for GPS-artifact tracks |
| `position_tolerance_m` | 0.5 m | Greedy lateral tolerance per interval |

**Preprocessing pipeline** (key implementation points):
- Spline parametrized by **chord length** (not sample index) — distributes the smoothing budget geometrically, matching `splprep`-based reference
- Curvature computed **analytically**: κ = (x′y″ − y′x″) / (x′² + y′²)^(3/2) from spline 2nd derivatives — more accurate than `np.gradient(θ)`
- Heading θ = θ₀ + ∫κ ds — ensures θ and κ are mutually consistent for `SolveG2`
- S&H duplicates removed before fitting (`dedup_threshold_m=1e-3 m`)

#### `.plot(track, basemap=True, zoom=16, color_by=None, cmap='coolwarm', ...)`

Plot track with optional basemap and color-coding.

#### `.sample_adaptive(track, kappa_scale=50.0, base_spacing_m=10.0, min_spacing_m=1.0, v_min_ms=5.0, smooth_kappa_m=10.0)` → `dict`

Sample the Kalman trace at curvature-dependent resolution.  Denser in curves,
sparser on straights.  The local step size is:

```
ds(s) = max(base_spacing / (1 + kappa_scale * |κ(s)|), min_spacing)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track` | — | `GPSTrack` from Kalman processing |
| `kappa_scale` | 50.0 | Curvature sensitivity (higher = more curve points). Typical: 20–100. |
| `base_spacing_m` | 10.0 | Spacing on straight segments [m] |
| `min_spacing_m` | 1.0 | Minimum spacing cap (hairpin limit) [m] |
| `v_min_ms` | 5.0 | Minimum speed for valid heading [m/s] |
| `smooth_kappa_m` | 10.0 | Savgol smoothing window for curvature [m] |

**Returns** a dict with keys:
- `longitude`, `latitude` — sampled coordinates [°]
- `time_s` — time from track start [s]
- `arc_m` — arc-length position [m]
- `kappa` — curvature at each sample [1/m]
- `speed_ms` — speed at each sample [m/s]
- `spacing_m` — local spacing actually used [m]
- `n_points` — total number of sampled points

```python
proc = GPSProcessor(v_max_kmh=250, smoothing="kalman")
track = proc.process(lon, lat, v, steering_angle_deg=x["STEER_ANGLE_DEG"],
                     abs_flag=x["ABS_FLAG"])
sampled = proc.sample_adaptive(track, kappa_scale=50)
print(sampled['n_points'], sampled['time_s'][-1])  # e.g. 716 pts, 116.0 s
```

#### `.sample_angular_step(track, delta_theta_deg=2.0, base_spacing_m=50.0, min_spacing_m=0.5, v_min_ms=5.0, kappa=None)` → `dict`

Sample the Kalman trace with a **cumulative angular step** constraint.
Between any two consecutive samples, the accumulated heading change
(∫|κ| ds) never exceeds `delta_theta_deg`.  This guarantees geometric
fidelity in curves while keeping straights sparse — and is robust against
instantaneous curvature noise because the integration averages it out.

```
ds(s): walk until ∫|κ| ds ≥ Δθ_max  OR  distance ≥ base_spacing
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track` | — | `GPSTrack` from Kalman processing |
| `delta_theta_deg` | 2.0 | Max cumulative heading change between samples [°]. Typical: 1–5. |
| `base_spacing_m` | 50.0 | Max spacing on straights [m] |
| `min_spacing_m` | 0.5 | Integration step size / minimum spacing [m] |
| `v_min_ms` | 5.0 | Minimum speed for valid arc [m/s] |
| `kappa` | None | Pre-computed curvature (same length as track). Recommended: pass G2 knot κ interpolated to the Kalman arc grid (`np.interp(arc, g2_fit.knot_s_m, g2_fit.knot_kappa)`). |

**Returns** a dict with the same keys as `sample_adaptive`:
`longitude`, `latitude`, `time_s`, `arc_m`, `kappa`, `speed_ms`,
`spacing_m`, `n_points`.

```python
# Source κ from G2 chain (interpolated to Kalman arc grid)
kappa_g2 = np.interp(arc_kalman, g2_fit.knot_s_m, g2_fit.knot_kappa)

# Sample with 10° cumulative heading budget
proc = GPSProcessor(v_max_kmh=250, smoothing="kalman")
track = proc.process(lon, lat, v, steering_angle_deg=x["STEER_ANGLE_DEG"],
                     abs_flag=x["ABS_FLAG"])
sampled = proc.sample_angular_step(track, delta_theta_deg=10.0,
                                   kappa=kappa_g2)
# 381 pts at Δθ≤10°, spacing 1.5–50 m
```

#### `.summary(track)` → `dict`

Returns: `fs_hz`, `dt_ms`, `n_samples`, `n_valid`, `pct_removed`, `v_*_kmh`.

### `GPSTrack` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `longitude` | NDArray | Filtered longitude [°] |
| `latitude` | NDArray | Filtered latitude [°] |
| `speed_ms` | NDArray | Speed [m/s] |
| `mask` | NDArray[bool] | Valid (non-outlier) samples |
| `fs` | float | Sampling frequency [Hz] |
| `dt` | float | Sampling period [s] |
| `aux` | dict | Auxiliary channels |

Properties: `speed_kmh`, `n_samples`, `n_valid`  
Method: `get_masked(channel)` → masked auxiliary channel

### `KalmanConfig` (dataclass)

```python
KalmanConfig(
    dt,                        # Sampling period [s] (auto-calibrated)
    sigma_pos_gps=2.0,         # GPS position noise [m]
    sigma_v_wheel=0.1,         # Wheel speed noise [m/s]
    sigma_yaw_rate=0.02,       # Yaw rate noise [rad/s]
    sigma_a=2.0,               # Process noise: acceleration [m/s²]
    sigma_yaw_acc=0.5,         # Process noise: yaw acceleration [rad/s²]
    sigma_pos_drift=5.0,       # Position drift noise [m/√s] — critical for high fs!
    initial_speed=0.0,         # Initial speed estimate [m/s]
    initial_heading=0.0,       # Initial heading [rad]
    use_rts_smoother=True,     # Enable RTS backward pass
    steering_ratio=15.5,       # Steering wheel → road wheel gear ratio
    wheelbase_m=2.68,          # Vehicle wheelbase [m] (bicycle model)
    gps_low_speed_gain=10.0,   # GPS noise multiplier at standstill
    gps_low_speed_v_scale=2.0, # Speed scale for GPS noise decay [m/s]
)
```

## Kalman Filter Architecture

**State vector**: `[x, y, v, θ, ω]` in local ENU frame  
**Motion model**: CTRV (Constant Turn Rate and Velocity)  
**Measurements**: GPS position (sparse, 5–20 Hz), wheel speed (dense, f_s), yaw rate (dense, f_s)  
**Smoother**: Rauch-Tung-Striebel (RTS) backward pass

### Yaw Rate Source

The yaw rate measurement is derived from **steering angle** via the bicycle model
(not from YAW_RATE_DPS, which is rarely available in measurement data):

```
ω = v · tan(δ_steering / steering_ratio) / wheelbase_m
```

Channel: `STEER_ANGLE_DEG` (signed steering wheel angle [°]).  
Defaults: `steering_ratio=15.5`, `wheelbase_m=2.68` (typical compact car).  
Custom values can be set via `kalman_config=KalmanConfig(dt=1.0, steering_ratio=..., wheelbase_m=...)`.  
This is handled automatically when passing `steering_angle_deg` to `process()`.
If both `steering_angle_deg` and `yaw_rate_rad` are provided, `steering_angle_deg`
takes priority and a warning is logged.

**ABS speed clamping**: During ABS braking, wheel speed (WHEEL_SPEED_KMH) oscillates at
~15 Hz from lock/unlock cycles. These oscillations corrupt the bicycle-model
yaw rate and cause heading drift in the Kalman filter. When `abs_flag=ABS_FLAG`
is provided, the speed used for yaw rate computation is clamped to the last
pre-ABS value during each ABS event. The raw WHEEL_SPEED_KMH still feeds the Kalman
filter as a wheel speed measurement (correctly reflects deceleration).

### Tuning Guide

| Parameter | Effect of ↑ |
|-----------|-------------|
| `sigma_pos_gps` | Less trust in GPS → smoother path, more model-driven |
| `sigma_v_wheel` | Less trust in wheel speed → speed follows GPS-derived estimate |
| `sigma_yaw_rate` | Less trust in gyro → heading follows GPS-derived heading |
| `sigma_a` | Allows faster speed changes → better tracking of acceleration |
| `sigma_yaw_acc` | Allows faster yaw changes → better tracking in tight curves |
| `sigma_pos_drift` | Stronger GPS corrections → prevents dead-reckoning drift |
| `gps_low_speed_gain` | More GPS suppression at standstill → less multipath zig-zag |
| `gps_low_speed_v_scale` | Wider transition → GPS suppressed at higher speeds |

### Dynamic GPS Noise (Multipath Suppression at Standstill)

At standstill, GPS receivers exhibit 1–3 m multipath jitter from building
reflections. The CTRV model predicts "stay put" but each GPS update shifts the
position — creating a visible zig-zag in the Kalman trace.

**Fix**: The GPS measurement noise R is speed-dependent:

```
σ_GPS_eff(v) = σ_GPS · (1 + gain · exp(-|v| / v_scale))
```

| Speed | Effective σ_GPS | K_GPS (approx) |
|-------|---------------|----------------|
| 0 m/s (standstill) | 22.0 m | 0.01 |
| 2 m/s (crawling) | 10.1 m | 0.05 |
| 5 m/s (parking) | 3.6 m | 0.28 |
| 10 m/s (urban) | 2.1 m | 0.54 |
| >15 m/s | ≈2.0 m | 0.56 |

The transition is smooth: at parking-lot speeds (2–5 m/s) the filter already
regains ~30% GPS trust. At highway speed the behavior is unchanged from a
static σ_GPS = 2.0.

**Result**: Kalman position steps at standstill reduced from 2.1 m to 1.0 m
(50% reduction in zig-zag amplitude).

### Position Drift Noise (Critical Fix for High Sample Rates)

The standard Q matrix has position noise ∝ dt⁴. At dt=2ms (500 Hz) this
yields ~4×10⁻¹² m² per step — effectively zero. The position covariance P[:2,:2]
never grows between GPS updates (N = f_s/f_GPS steps), causing:

- Kalman gain K_GPS → 0 (filter ignores GPS corrections)
- Unbounded dead-reckoning drift from heading/yaw-rate bias

**Fix**: `sigma_pos_drift` adds σ²·dt to Q[0,0] and Q[1,1] (scales with dt, not dt⁴):

```
Example: 500 Hz sample rate, 5 Hz GPS → N = 100 steps between fixes
P_position after N steps ≈ N × σ²_drift × dt = 100 × 25 × 0.002 = 5.0 m²
K_GPS = 5.0 / (5.0 + σ²_gps) = 5.0 / 9.0 ≈ 0.56  (was ~0)

Example: 500 Hz sample rate, 20 Hz GPS → N = 25 steps
P_position ≈ 25 × 25 × 0.002 = 1.25 m²
K_GPS = 1.25 / (1.25 + 4.0) ≈ 0.24
```

| σ_pos_drift | Median error vs GPS | P95   | Max   |
|-------------|--------------------:|------:|------:|
| 0 (broken)  | >100 m (unbounded)  | —     | —     |
| 5.0 (fixed) | 1.4 m               | 3.4 m | 6.6 m |

**Rule of thumb**: For any EKF where f_s >> f_GPS, position process noise must
scale with dt (not dt⁴) to maintain GPS observability. The critical parameter
is N = f_s / f_GPS (prediction steps between corrections).

### GPS Single-Axis Freeze Repair

GPS receivers occasionally "freeze" one coordinate axis (e.g. latitude constant
while longitude continues updating). The `_repair_gps_freezes()` method detects
and repairs these artifacts before Kalman filtering:

**Detection** (all three must hold for a run of ≥3 consecutive epochs):
1. One axis has zero change (frozen)
2. The other axis has non-zero change (receiver is still updating)
3. Expected movement (from WHEEL_SPEED_KMH) exceeds 5.0 m over the run

**Repair**: The frozen axis is linearly interpolated between the last valid
position before and first valid position after the freeze. The non-frozen axis
is preserved unchanged.

**Impact** (4 km test track):

| Region | Without fix | With fix | Δ |
|--------|-------------|----------|---|
| All | 7.71 m | 8.13 m | +0.43 |
| Straights (R>100m) | 6.52 m | 6.54 m | +0.02 |
| Moderate (33<R<100m) | 7.69 m | 7.01 m | **−0.67** |
| Tight curves (R<33m) | 7.87 m | 8.40 m | +0.53 |

The fix primarily benefits moderate curves where GPS freeze events corrupt the
Kalman heading estimate. In tight curves the effect is masked by higher GPS noise.

### Speed Bias Analysis

The RTS smoother shortens the path in curves ("corner cutting"), causing a
systematic speed underestimate proportional to speed × curvature:

| σ_v_wheel | Mean Bias | Std | Bias @ 200+ km/h |
|-----------|-----------|-----|-------------------|
| 0.5 (old) | -4.7 km/h | 5.4 | -9.3 km/h |
| 0.1 (new) | -0.2 km/h | 1.6 | -0.4 km/h |

**Recommendation**: Use `sigma_v_wheel=0.1` (default) for accurate speed.

## Performance

| Backend | 853k samples | Speedup vs Python |
|---------|-------------|-------------------|
| Pure Python | 58.0s | 1× |
| Numba (CPU) | 4.1s | **14×** |
| JAX (CPU, f64) | 7.7s | 7.5× |
| JAX (GPU, projected) | ~0.5s | ~100× |

First run includes JIT compilation overhead (~6–11s).

## Data Format

Input: channel-based DAQ recordings (e.g. automotive measurement systems
sampling CAN at high rate with GPS sample-and-hold). Minimum required channels:

| Channel | Description |
|---------|-------------|
| `GPS_LON` + `GPS_LON_INT` | Longitude (fractional + integer parts) |
| `GPS_LAT` + `GPS_LAT_INT` | Latitude (fractional + integer parts) |
| `WHEEL_SPEED_KMH` | Wheel speed [km/h] |
| `STEER_ANGLE_DEG` | Signed steering WHEEL angle [°] (primary yaw rate source) |
| `YAW_RATE_DPS` | Unsigned yaw rate [°/s] (RARE — not available in most recordings) |
| `YAW_RATE_SIGN` | Yaw rate sign (0=pos, 1=neg) (RARE) |

**Coordinate composition**: Pass `GPS_LON` / `GPS_LAT` as `lon_raw` / `lat_raw` and
`GPS_LON_INT` / `GPS_LAT_INT` as `lon_int` / `lat_int`. The processor rounds the
integer parts and composes them automatically. Zero coordinates (no GPS fix) are
replaced with NaN and bridged by interpolation.

**Preferred**: Pass `STEER_ANGLE_DEG` as `steering_angle_deg` when available.
The module derives yaw rate via bicycle model internally.

### Public Datasets

The S&H artifact removal targets professional DAQ equipment (500–2000 Hz base rate
with 5–20 Hz GPS). For experimentation without proprietary data:

* **[comma.ai comma2k19](https://github.com/commaai/comma2k19)** — real car CAN+GPS
  (steering angle, wheel speed, ABS equivalent); most relevant for the Kalman pipeline.
* **[KITTI Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php)** — GPS+INS
  from a test vehicle; well-documented; no CAN channels.
* **[Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/)** — GPS+INS+wheel
  odometry from repeated urban runs.

These datasets have native GPS at 10–20 Hz with no S&H padding; the S&H detection
step is a no-op, and the rest of the pipeline applies unchanged.

```python
# Full daq usage with split-channel GPS (recommended)
track = proc.process(
    x["GPS_LON"], x["GPS_LAT"], x["WHEEL_SPEED_KMH"] / 3.6,  # km/h → m/s
    lon_int=x["GPS_LON_INT"], lat_int=x["GPS_LAT_INT"],
    steering_angle_deg=x["STEER_ANGLE_DEG"], abs_flag=x["ABS_FLAG"],
)

# Steering angle approach (always available, recommended)
track = proc.process(lon, lat, v, steering_angle_deg=x["STEER_ANGLE_DEG"])

# Direct yaw rate (only when YAW_RATE_DPS is present)
yaw_rate_rad = np.radians(np.where(
    x["YAW_RATE_SIGN"] == 1, -x["YAW_RATE_DPS"], x["YAW_RATE_DPS"]
))
track = proc.process(lon, lat, v, yaw_rate_rad=yaw_rate_rad)
```

## Future Extensions

- [x] `pip install -e .` packaging (`pyproject.toml` with src-layout)
- [ ] Unit tests (pytest)
- [ ] Batch processing of multiple channel-based DAQ recordings (JAX vmap)
- [ ] RTK-GPS support (sigma_pos_gps=0.02)
- [x] G2-continuous clothoid approximation (Bertolazzi-Frego SolveG2, max error < 0.5 m)
- [ ] JAX differentiable Kalman for parameter optimization
- [ ] Production speed gate: skip v < 20 km/h segments (stationary κ pathology)
- [ ] Optimal spacing selection: auto-tune based on target P95 error

## License

BSD 2 — the project contributors

## Acknowledgements

Initial implementation and documentation drafted with AI assistance
(Claude Opus); reviewed and validated by the project maintainer.
