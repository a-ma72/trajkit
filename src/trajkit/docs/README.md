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
│   │                    #   KalmanConfig, KalmanResult, gps_enu,
│   │                    #   G2ClothoidApproximator, G2ClothoidConfig,
│   │                    #   G2ClothoidFitResult
│   ├── processor.py     # GPSTrack (dataclass), GPSProcessor (main class)
│   ├── kalman.py        # KalmanConfig, KalmanResult, GPSKalmanFilter (Numba/CPU)
│   ├── jax_kalman.py    # JAXKalmanFilter (XLA/GPU-ready, lax.scan)
│   └── clothoid.py      # gps_enu, ClothoidSegment, G2ClothoidApproximator,
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
| `kalman_jax` | JAX/XLA | EKF state | 7.7s CPU / ~0.5s GPU | GPU clusters, batch vmap ⚠️ see note below |
| `g2` | Numba + SciPy | EKF state | ~3 s / 4 km | G2 clothoid chain (Bertolazzi-Frego SolveG2, max < 0.5 m) |

- `smoothing='g2'` implies Kalman pre-filtering: `GPSProcessor` automatically
  resolves it to `'kalman+g2'` (logged via `logger.info`) unless a
  position-smoothing backend is already given explicitly, e.g.
  `smoothing='kalman_jax+g2'` or `smoothing='butterworth+g2'`.
- ⚠️ **`kalman_jax` is not yet numerically equivalent to `kalman`**: it
  omits the position-drift fix (`sigma_pos_drift`) and the speed-dependent
  GPS noise scaling (`gps_low_speed_gain`/`gps_low_speed_v_scale`), and it
  does not support the ABS noise-inflation hooks. See
  `docs/kalman_description.typ` § "JAX Backend Parity" for details. Use
  `kalman` for production tracks with high sample rates, standstill
  segments, or ABS events.

## API Reference

### `GPSProcessor`

```python
GPSProcessor(
    v_max_kmh=250.0,             # Outlier speed threshold [km/h]
    filter_order=2,              # Butterworth filter order
    cutoff_factor=0.5,           # LP cutoff as fraction of GPS Nyquist
    min_speed_calibration=1.0,   # Min wheel speed for dt calibration [m/s]
    smoothing='butterworth',     # 'butterworth' | 'kalman' | 'kalman_jax' | 'g2' (combinable with '+')
    speed_smoothing_s=0.3,       # Savgol duration [s]; window = int(speed_smoothing_s * fs) | 1, 0=disabled
    kalman_config=None,          # KalmanConfig — EKF tuning (dt auto-calibrated if None)
    g2_config=None,              # G2ClothoidConfig — SolveG2 fit params (smoothing='g2')
    freeze_repair=False,         # keyword-only; repair single-axis GPS freezes (kalman modes only)
)
```

Config objects and their defaults:

| Parameter | Config class | Key fields |
|---|---|---|
| `kalman_config` | `KalmanConfig` | `sigma_pos_gps`, `sigma_v_wheel`, `steering_ratio=15.5`, `wheelbase_m=2.68` |
| `g2_config` | `G2ClothoidConfig` | `position_tolerance_m=0.5`, `resample_spacing_m=2.0`, `kappa_clip` |

All two default to `None`, in which case sensible defaults are used automatically.

#### `.process(lon_raw, lat_raw, v_reference_ms, aux_channels=None, yaw_rate_rad=None, steering_angle_deg=None, lateral_acceleration_ms2=None, abs_flag=None, lon_int=None, lat_int=None, gps_samplerate_hz=None)`

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

**Yaw rate input** (required for `kalman` mode — provide one of the three):
- `steering_angle_deg` — signed steering WHEEL angle [°] (e.g. `STEER_ANGLE_DEG`).  
  **Takes priority** over `yaw_rate_rad` and `lateral_acceleration_ms2` when
  more than one is provided (a warning is logged).  
  Internally converted via bicycle model: `ω = v · tan(δ / steering_ratio) / wheelbase_m`  
  using `steering_ratio` and `wheelbase_m` from `kalman_config` (defaults: 15.5, 2.68 m).
  To use custom vehicle parameters, pass a `KalmanConfig` with adjusted values.
- `lateral_acceleration_ms2` — signed lateral acceleration [m/s²]. Used to
  derive yaw rate as `ω = a_lat / v` when `steering_angle_deg` is not
  provided. Assumed pre-conditioned; no filtering is applied to this
  channel internally.
- `yaw_rate_rad` — pre-computed signed yaw rate [rad/s]. Only used if
  neither `steering_angle_deg` nor `lateral_acceleration_ms2` is provided.

**ABS braking robustness** (optional but recommended):
- `abs_flag` — binary ABS active flag (1 = ABS intervening, 0 = normal).  
  Channel: `ABS_FLAG` in the daq data.  
  When provided, two mechanisms activate:
  1. **Speed interpolation**: Wheel speed through ABS epochs is linearly
     interpolated between pre/post-ABS anchors (removes 15 Hz oscillation).
  2. **Wheel-speed noise inflation**: The wheel-speed measurement variance
     fed to the EKF is inflated during (and briefly around) each ABS
     event, so GPS and the motion model dominate the speed estimate
     instead of the unreliable wheel speed.
  See §"ABS Braking Robustness" below for details.

**Sampling rate override** (optional):
- `gps_samplerate_hz` — nominal sample rate of the input arrays [Hz]. If
  provided, `dt = 1/gps_samplerate_hz` is used directly instead of being
  auto-calibrated from the reference speed and detected GPS update
  epochs (see `_calibrate_dt()`).

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

#### `.sample_adaptive(track, kappa_scale=50.0, base_spacing_m=10.0, min_spacing_m=1.0, v_min_ms=5.0, smooth_kappa_m=30.0, kappa_clip_percentile=98.0, kappa=None)` → `dict`

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
| `smooth_kappa_m` | 30.0 | Savgol smoothing window for curvature [m] |
| `kappa_clip_percentile` | 98.0 | Clip \|κ\| above this percentile of moving samples (noise-spike suppression). 100.0 disables clipping. |
| `kappa` | None | Pre-computed curvature array (same length as track); skips the internal heading/curvature derivation when provided. |

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

#### `.sample_angular_step(track, delta_theta_deg=2.0, base_spacing_m=50.0, min_spacing_m=0.5, v_min_ms=5.0, smooth_kappa_m=30.0, kappa=None)` → `dict`

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
| `smooth_kappa_m` | 30.0 | Savgol smoothing window for internally-derived curvature [m]. Ignored when `kappa` is provided. |
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
    sigma_yaw_rate=0.2,        # Yaw rate noise [rad/s]
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

### ABS Braking Robustness

During ABS braking, wheel speed (WHEEL_SPEED_KMH) oscillates at ~15 Hz from
lock/unlock cycles. Without treatment, these oscillations cause three problems:

1. **Bicycle-model yaw rate is invalid** — ω = v·tan(δ)/L uses oscillating v,
   producing error/σ ratios of ~3.5× that drift heading up to 8° over a 2 s event.
2. **Wheel speed is unreliable** — the raw signal swings ±10 km/h around the true
   deceleration curve.
3. **CTRV prediction is wrong** — with invalid ω and v, position error accumulates
   ~0.28 m per prediction epoch between GPS corrections.

**Two mechanisms** (activated when `abs_flag` is provided):

#### Stage 1: Speed interpolation (`_clamp_speed_during_abs`)

Wheel speed during ABS epochs is linearly interpolated between the last non-ABS
sample before and the first non-ABS sample after each event (`np.interp`).
This removes the 15 Hz oscillation while capturing the deceleration trend
(offline processing knows both anchor points).

The interpolated `v_smooth` is used for both the bicycle-model yaw rate
computation (ω = v·tan(δ)/L) and as the Kalman speed measurement input.

#### Stage 2: Wheel-speed measurement-noise inflation (`_build_abs_noise_scale`)

Rather than crossfading the *position* to a model-free Butterworth track
(an earlier design, since removed), the full GPS + wheel-speed + yaw-rate
fusion stays active throughout the ABS event. Instead, the wheel-speed
measurement variance is inflated so the EKF trusts GPS and the CTRV model
more than the (interpolated, still approximate) wheel speed while ABS is
active:

```
R_v[k] = sigma_v_wheel**2 * s_v[k],   s_v[k] in [1, inflation]
```

The yaw rate channel is left untouched — a direct sensor stays valid under
tire saturation, and a bicycle-model-derived yaw rate already benefits from
`v_smooth` in Stage 1.

**Temporal margin**: because the ABS flag typically lags the actual wheel
oscillation, the inflated-noise window is extended before/after each
contiguous ABS run:
- Pre-ABS margin: `_ABS_PRE_MARGIN_S` (default 0.3 s)
- Post-ABS margin: `_ABS_POST_MARGIN_S` (default 0.5 s)

**Raised-cosine ramp** (`_ABS_RAMP_S`, default 0.15 s): each window edge is
smoothed with a half-wave cosine, avoiding a step change in measurement
trust that would otherwise inject a kink into the estimate.

**Peak inflation**: `_ABS_SPEED_NOISE_INFLATION` defaults to 400 (variance
scale, i.e. ~20× in standard deviation).

| Constant | Default |
|----------|---------|
| `_ABS_FLAG_THRESHOLD` | 0.5 |
| `_ABS_SPEED_NOISE_INFLATION` | 400.0 (≈20× in σ) |
| `_ABS_PRE_MARGIN_S` | 0.3 s |
| `_ABS_POST_MARGIN_S` | 0.5 s |
| `_ABS_RAMP_S` | 0.15 s |

**Current scope**: only the wheel-speed channel is inflated by
`GPSProcessor.process()` today. `GPSKalmanFilter.run()` also accepts
`yaw_rate_noise_scale` and `process_noise_scale` for finer-grained control,
but the processor pipeline does not currently populate them.

No end-to-end accuracy benchmark is available yet for this mechanism (the
figures from the earlier position-blend design no longer apply, since that
code path was removed). See `docs/kalman_description.typ` § "ABS Braking
Robustness" for the full derivation.

### GPS Freeze Repair

Some GPS receivers intermittently freeze a single coordinate axis (e.g.
longitude stays constant for several fixes while latitude keeps updating),
producing a false straight-line segment and a heading kink once the freeze
ends. Enable detection and repair with `freeze_repair=True` (default
`False`; only applies to `kalman`/`kalman_jax` modes):

```python
proc = GPSProcessor(v_max_kmh=250, smoothing="kalman", freeze_repair=True)
```

A coordinate is classified as frozen only if it is unchanged for at least
`min_run_length` GPS epochs (default 3), the reference speed exceeds
`min_speed` (default 5.0 m/s), and the heading-projected expected
displacement in that axis exceeds `min_expected_change_m` (default 5.0 m)
— the last check avoids false positives on tight curves where one axis
legitimately stays near-constant. Detected freezes are repaired by linear
interpolation between the last valid fix before and the first valid fix
after the run; the non-frozen axis is left unchanged. A `UserWarning` is
raised summarizing how many epochs were repaired. See
`docs/kalman_description.typ` § "GPS Freeze Repair" for the full
detection criterion and a comparison against alternative repair
strategies (interpolation outperforms noise-inflation-based partial
updates on tight curves).

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
| `ABS_FLAG` | Binary ABS active flag (1 = ABS intervening, 0 = normal) |

```python
# Full daq usage with split-channel GPS (recommended)
track = proc.process(
    x["GPS_LON"], x["GPS_LAT"], x["WHEEL_SPEED_KMH"] / 3.6,
    lon_int=x["GPS_LON_INT"], lat_int=x["GPS_LAT_INT"],
    steering_angle_deg=x["STEER_ANGLE_DEG"], abs_flag=x["ABS_FLAG"],
)
```

## Future Extensions

- [x] `pip install -e .` packaging (`pyproject.toml` with src-layout)
- [ ] Unit tests (pytest)
- [x] G2-continuous clothoid approximation (Bertolazzi-Frego SolveG2, max error < 0.5 m)
- [ ] JAX differentiable Kalman for parameter optimization

## Known Limitations

- **`kalman_jax` is not a numerical drop-in for `kalman`.** It omits the
  position-drift fix (`sigma_pos_drift`) and the speed-dependent GPS noise
  scaling (`gps_low_speed_gain`/`gps_low_speed_v_scale`), and it does not
  accept the ABS noise-inflation arguments
  (`speed_noise_scale`/`yaw_rate_noise_scale`/`process_noise_scale`).
  Prefer `kalman` for high sample rates, standstill segments, or ABS
  events until these are ported.
- **ABS noise inflation currently covers only the wheel-speed channel.**
  `GPSKalmanFilter.run()` supports per-sample yaw-rate and process-noise
  scaling as well, but `GPSProcessor.process()` does not populate them
  yet — no benchmark exists for those paths.
- **No automated test suite yet** (tracked above under Future Extensions);
  changes to the pipeline are currently verified manually.
- **`EARTH_METERS_PER_DEGREE` is a single fixed constant** (111,139.0
  m/°), not a latitude-dependent WGS84 value. This introduces a small
  (~0.1–0.2%) systematic scale error in all ENU conversions; acceptable
  for the flat-Earth approximation used here, but worth knowing if you
  compare distances against a geodesic library.

## License

BSD 2 — the project contributors
