// kalman_description.typ — College-level description of kalman.py
// GPS Track Conditioning Pipeline — Extended Kalman Filter Module

#set document(
  title: [Extended Kalman Filter for GPS/INS Fusion],
  author: "GPS Track Conditioning Project",
  date: datetime(year: 2026, month: 6, day: 27),
)

#set page(
  margin: 2.5cm,
  numbering: "1",
  footer: [
    #text(size: 8pt, style: "italic")[
      Drafted with AI assistance; reviewed by the project maintainer.
    ]
    #h(1fr)
    #context counter(page).display()
  ],
)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1.1.1")
#set par(justify: true)
#set math.equation(numbering: "(1)")

#show heading.where(level: 1): set text(size: 14pt)
#show heading.where(level: 2): set text(size: 12pt)
#show raw.where(block: true): set text(size: 9pt)
#show figure.where(kind: table): set block(breakable: true)

// Title block
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Extended Kalman Filter for GPS/INS Fusion
  ]
  #v(0.5em)
  #text(size: 12pt, style: "italic")[
    Module Documentation: `kalman.py`
  ]
  #v(0.3em)
  #text(size: 10pt)[GPS Track Conditioning Pipeline]
  #v(2em)
]

#outline(title: "Contents", indent: auto)

#pagebreak()

= Introduction

The `kalman.py` module implements a sensor-fusion algorithm that combines
three independent measurement sources—GPS position, wheel speed, and
yaw rate—into a single, optimal state estimate of a vehicle's trajectory.
The core algorithm is an *Extended Kalman Filter* (EKF) with a
nonlinear motion model, augmented by a *Rauch–Tung–Striebel* (RTS) smoother @rauch1965
for offline batch processing.

The module is designed for high-rate vehicle measurement data
(sampling rate $f_s approx 500$ Hz) where GPS updates occur sparsely
at 5–20 Hz. Between GPS fixes, the filter propagates position
using wheel speed and yaw rate (dead reckoning), then corrects when a
GPS measurement arrives.

== Design Goals

- *Accuracy*: Sub-meter position accuracy by fusing complementary sensors
- *Robustness*: Graceful handling of GPS dropouts via dead reckoning
- *Performance*: Real-time capable via Numba JIT compilation (14× speedup over Python)
- *Modularity*: Pure-function math core enabling future JAX port

= State-Space Formulation

== State Vector

The filter maintains a five-dimensional state vector:

$ bold(x) = vec(x, y, v, theta, omega) $ <state-vector>

where:
- $x, y$ — position in a local East-North-Up (ENU) frame #h(1fr) [m]
- $v$ — scalar longitudinal speed #h(1fr) [m/s]
- $theta$ — heading angle (0 = East, $pi slash 2$ = North) #h(1fr) [rad]
- $omega$ — yaw rate (angular velocity about vertical axis) #h(1fr) [rad/s]

The local ENU frame is centered at the first GPS fix, with the $x$-axis
pointing East and the $y$-axis pointing North. This avoids numerical
issues from operating in WGS84 coordinates directly.

== Motion Model: CTRV

The *Constant Turn Rate and Velocity* (CTRV) model assumes that between
time steps, the vehicle maintains constant speed $v$ and constant yaw
rate $omega$. The resulting trajectory is a circular arc.

For $omega != 0$:

$ x_(k+1) = x_k + v_k / omega_k (sin(theta_k + omega_k Delta t) - sin theta_k) $
$ y_(k+1) = y_k + v_k / omega_k (cos theta_k - cos(theta_k + omega_k Delta t)) $
$ theta_(k+1) = theta_k + omega_k Delta t $

For $omega approx 0$ (straight-line motion):

$ x_(k+1) = x_k + v_k cos(theta_k) dot Delta t $
$ y_(k+1) = y_k + v_k sin(theta_k) dot Delta t $

Speed and yaw rate persist: $v_(k+1) = v_k$, $omega_(k+1) = omega_k$.

This model is *nonlinear* in the state—the position depends on trigonometric
functions of $theta$ and $omega$—which is why a standard (linear) Kalman
filter cannot be used. The EKF linearizes this model at each step.

== Jacobian Matrix $bold(F)$

The EKF requires the Jacobian of the state transition function
$bold(f)(bold(x))$ with respect to the state. For $omega != 0$:

$ bold(F) = (partial bold(f)) / (partial bold(x)) = mat(
  1, 0, (sin(theta + omega Delta t) - sin theta) / omega, dots.h, dots.h;
  0, 1, (cos theta - cos(theta + omega Delta t)) / omega, dots.h, dots.h;
  0, 0, 1, 0, 0;
  0, 0, 0, 1, Delta t;
  0, 0, 0, 0, 1;
) $

The full expressions for $F_(0,3), F_(0,4), F_(1,3), F_(1,4)$ involve
partial derivatives of the circular-arc equations with respect to $theta$
and $omega$, computed analytically in the source code.

= Process Noise

== Standard Process Noise Matrix $bold(Q)$

The process noise models unmodeled accelerations (longitudinal $sigma_a$)
and yaw acceleration ($sigma_(dot(omega))$). The resulting covariance
matrix has the block structure:

$ bold(Q) = mat(
  Delta t^4 / 4 dot sigma_a^2 + sigma_"drift"^2 dot Delta t, 0, Delta t^3 / 2 dot sigma_a^2, 0, 0;
  0, Delta t^4 / 4 dot sigma_a^2 + sigma_"drift"^2 dot Delta t, Delta t^3 / 2 dot sigma_a^2, 0, 0;
  Delta t^3 / 2 dot sigma_a^2, Delta t^3 / 2 dot sigma_a^2, Delta t^2 dot sigma_a^2, 0, 0;
  0, 0, 0, Delta t^2 dot sigma_(dot(omega))^2, Delta t dot sigma_(dot(omega))^2;
  0, 0, 0, Delta t dot sigma_(dot(omega))^2, sigma_(dot(omega))^2;
) $

== Position Drift Fix (Critical for High Sample Rates) <drift-fix>

At $Delta t = 2$ ms (500 Hz), the standard position noise term
$Delta t^4 slash 4 dot sigma_a^2 approx 4 times 10^(-12)$ m² per step.
Between GPS updates—typically 25 to 100 steps depending on whether GPS
runs at 20 Hz or 5 Hz—the position covariance $bold(P)_(0:1, 0:1)$
barely grows. The Kalman gain for GPS corrections approaches zero,
causing unbounded dead-reckoning drift.

The severity depends on the ratio $f_s slash f_"GPS"$. At 500 Hz with
5 Hz GPS, 100 prediction steps occur between corrections. At 500 Hz
with 20 Hz GPS, only 25 steps elapse. In both cases, the $Delta t^4$
scaling is insufficient.

*Fix*: An additional term $sigma_"drift"^2 dot Delta t$ is added to
$Q_(0,0)$ and $Q_(1,1)$. This scales *linearly* with $Delta t$ (not
$Delta t^4$), ensuring adequate position uncertainty growth:

$ P_"pos" "after" N "steps" approx N times sigma_"drift"^2 times Delta t $

For example, with $N = 100$ steps (5 Hz GPS at 500 Hz sample rate):

$ P_"pos" approx 100 times 25 times 0.002 = 5.0 "m"^2 $

$ K_"GPS" = P_"pos" / (P_"pos" + sigma_"GPS"^2) = 5.0 / (5.0 + 4.0) approx 0.56 $

With $N = 25$ steps (20 Hz GPS):

$ P_"pos" approx 25 times 25 times 0.002 = 1.25 "m"^2 $

$ K_"GPS" = 1.25 / (1.25 + 4.0) approx 0.24 $

Without this fix, $K_"GPS" approx 0$ in both cases and GPS measurements
are effectively ignored.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header[$sigma_"drift"$][Median error][P95][Max],
    [0 (broken)], [> 100 m], [—], [—],
    [5.0 (fixed)], [1.4 m], [3.4 m], [6.6 m],
  ),
  caption: [Effect of position drift noise on filter accuracy (500 Hz, 14 Hz GPS).],
) <tab-drift>

*Rule of thumb*: For any EKF where $f_s >> f_"GPS"$, position process
noise must scale with $Delta t$ (not $Delta t^4$) to maintain GPS
observability. The critical parameter is the number of prediction steps
between GPS fixes: $N = f_s slash f_"GPS"$.

= Measurement Updates

The filter processes three measurement types at each time step, applied
sequentially (scalar updates for efficiency):

== GPS Position (Sparse, 5–20 Hz)

When a GPS fix is available (indicated by a boolean mask):

$ bold(z)_"GPS" = vec(x_"GPS", y_"GPS"), quad bold(H)_"GPS" = mat(1, 0, 0, 0, 0; 0, 1, 0, 0, 0) $

The GPS measurement noise is *speed-dependent* to suppress multipath
jitter at standstill:

$ sigma_"GPS,eff"(v) = sigma_"GPS" dot (1 + G dot e^(-|v| slash v_s)) $

where $G$ = `gps_low_speed_gain` (default 10.0) and $v_s$ =
`gps_low_speed_v_scale` (default 2.0 m/s). The measurement covariance
becomes:

$ bold(R)_"GPS"(v) = mat(sigma_"GPS,eff"^2, 0; 0, sigma_"GPS,eff"^2) $

At standstill ($v approx 0$): $sigma_"eff" = 2.0 dot 11 = 22$ m $arrow.r$
$K_"GPS" approx 0.01$ (nearly ignores GPS). At highway speed ($v > 10$ m/s):
$sigma_"eff" approx sigma_"GPS"$ (unchanged).

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    table.header[Speed][Effective $sigma_"GPS"$][$K_"GPS"$ (approx.)],
    [0 m/s (standstill)], [22.0 m], [0.01],
    [2 m/s (crawling)], [10.1 m], [0.05],
    [5 m/s (parking)], [3.6 m], [0.28],
    [10 m/s (urban)], [2.1 m], [0.54],
    [$>$15 m/s], [$approx$ 2.0 m], [0.56],
  ),
  caption: [Speed-dependent GPS noise scaling. Gain $G=10$, $v_s = 2$ m/s, $sigma_"GPS" = 2.0$ m.],
) <tab-dynamic-gps>

*Physical rationale*: At standstill, GPS receivers exhibit 1–3 m
multipath jitter from building reflections. The CTRV model predicts
"stay put" but each GPS update shifts the position by the jitter
amount. Without dynamic scaling, the filter follows this jitter
(producing a visible zig-zag). With $G=10$, the effective measurement
noise at $v=0$ is 22 m, causing the filter to almost entirely ignore
GPS updates — relying on the (correct) zero-speed prediction.

The transition is smooth: at typical parking-lot speeds (2–5 m/s)
the filter already regains ~30% GPS trust, which is appropriate for
the lower multipath environment at non-zero velocity.

The 2×2 innovation covariance $bold(S) = bold(H) bold(P) bold(H)^top + bold(R)(v)$
is inverted analytically (closed-form 2×2 inverse) for performance.

== Wheel Speed (Dense, Full Rate)

$ z_v = v_"wheel", quad H_v = mat(0, 0, 1, 0, 0), quad R_v = sigma_v^2 dot s_v [k] $

This is a scalar update: only the speed state is directly observed.
The Kalman gain simplifies to $bold(K)_v = bold(P)_(: , 2) slash (P_(2,2) + R_v [k])$.

The per-sample scale factor $s_v [k]$ defaults to 1.0 under normal
conditions and is inflated during ABS braking (see @abs-robustness).

== Yaw Rate (Dense, Full Rate)

$ z_omega = omega_"gyro", quad H_omega = mat(0, 0, 0, 0, 1), quad R_omega = sigma_omega^2 dot s_omega [k] $

Similarly scalar: $bold(K)_omega = bold(P)_(: , 4) slash (P_(4,4) + R_omega [k])$.

Like $s_v$, the scale $s_omega [k]$ is 1.0 normally and inflated
during ABS events when the yaw rate source is unreliable (see @abs-robustness).

= Rauch–Tung–Striebel Smoother

For offline (batch) processing, the RTS smoother @rauch1965 runs a backward pass
after the forward filter, incorporating future measurements to refine
past estimates:

$ bold(G)_k = bold(P)_k^f bold(F)_(k+1)^top (bold(P)_(k+1)^p)^(-1) $
$ hat(bold(x))_k^s = hat(bold(x))_k^f + bold(G)_k (hat(bold(x))_(k+1)^s - hat(bold(x))_(k+1)^p) $
$ bold(P)_k^s = bold(P)_k^f + bold(G)_k (bold(P)_(k+1)^s - bold(P)_(k+1)^p) bold(G)_k^top $

where superscripts $f$, $p$, $s$ denote filtered, predicted, and smoothed
estimates respectively.

The smoother eliminates filter lag and produces the minimum-variance
estimate given *all* data (past and future). For vehicle trajectories,
this results in smooth paths through curves without the overshoot
typical of causal filters.

= Coordinate System

== WGS84 ↔ Local ENU Conversion

GPS coordinates (longitude, latitude in degrees) are converted to a
local tangent-plane approximation:

$ x_"ENU" = ("lon" - "lon"_0) dot M_"earth" dot cos("lat"_0) $
$ y_"ENU" = ("lat" - "lat"_0) dot M_"earth" $

where $M_"earth" = 111 space 319.49$ m/° is the meters-per-degree
constant on the WGS84 ellipsoid. The inverse conversion is applied
to produce the final output in geographic coordinates.

This flat-Earth approximation introduces negligible error for
trajectories spanning < 50 km from the origin.

= Implementation Architecture

== Numba JIT Compilation

The computational core `_ekf_forward_rts()` is a single `@njit`-decorated
function @numba2015 containing the forward loop (predict + 3 updates per step) and
the backward RTS loop. This design:

- Avoids Python interpreter overhead for the $O(n)$ tight loop
- Keeps all intermediate arrays (states, covariances, Jacobians) in
  Numba's native memory layout
- Achieves *14× speedup* over equivalent pure-Python code

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    table.header[Backend][$n = 853 "k"$][Speedup],
    [Pure Python], [58.0 s], [1×],
    [Numba (CPU)], [4.1 s], [*14×*],
    [JAX (CPU)], [7.7 s], [7.5×],
  ),
  caption: [Runtime comparison for the full EKF + RTS pipeline.],
) <tab-perf>

== Class Structure

```python
class GPSKalmanFilter:
    def __init__(self, config: KalmanConfig) -> None
    def run(self, longitude, latitude, v_reference_ms,
            yaw_rate_rad, gps_update_mask, heading_init=None
    ) -> KalmanResult
```

The `run()` method performs:
1. Coordinate conversion (WGS84 → ENU)
2. Initial heading estimation from first two GPS fixes
3. Forward EKF + optional RTS (via `_ekf_forward_rts`)
4. Inverse coordinate conversion (ENU → WGS84)

= Tuning Parameters <tuning>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header[Parameter][Default][Effect of increasing],
    [$sigma_"pos,GPS"$], [2.0 m], [Less trust in GPS → smoother path],
    [$sigma_"v,wheel"$], [0.1 m/s], [Less trust in wheel speed → follows GPS-derived speed],
    [$sigma_omega$], [0.02 rad/s], [Less trust in gyro → heading follows GPS],
    [$sigma_a$], [2.0 m/s²], [Allows faster speed changes → better acceleration tracking],
    [$sigma_(dot(omega))$], [0.5 rad/s²], [Allows faster yaw changes → better cornering],
    [$sigma_"drift"$], [5.0 m/$sqrt(s)$], [Stronger GPS pull → prevents dead-reckoning drift],
    [`gps_low_speed_gain`], [10.0], [More GPS suppression at standstill → less multipath jitter],
    [`gps_low_speed_v_scale`], [2.0 m/s], [Wider transition region → GPS suppressed at higher speeds],
    [`steering_ratio`], [15.5], [Steering wheel → road wheel gear ratio (bicycle model)],
    [`wheelbase_m`], [2.68 m], [Vehicle wheelbase for yaw rate derivation],
  ),
  caption: [Tuning parameters and their effects on filter behavior.],
) <tab-tuning>

== Speed Bias from RTS Smoothing

The RTS smoother tends to "cut corners" in curves—the smoothed trajectory
is shorter than the actual path, causing a systematic speed underestimate:

$ Delta v approx -v dot kappa dot delta_"lateral" $

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header[$sigma_"v,wheel"$][Mean bias][Std][Bias at 200+ km/h],
    [0.5 (old)], [−4.7 km/h], [5.4 km/h], [−9.3 km/h],
    [0.1 (new)], [−0.2 km/h], [1.6 km/h], [−0.4 km/h],
  ),
  caption: [Speed bias reduction by tightening wheel-speed noise.],
) <tab-bias>

Reducing $sigma_"v,wheel"$ from 0.5 to 0.1 forces the filter to trust
the wheel speed sensor more closely, suppressing the corner-cutting
artifact.

= Data Flow

The module fits into the larger GPS Track Conditioning pipeline:

#align(center)[
  #box(stroke: 0.5pt, inset: 10pt, radius: 4pt)[
    #set text(size: 9pt)
    #grid(
      columns: 1,
      row-gutter: 6pt,
      [*Raw GPS* (S&H at $f_s$ Hz)],
      [#h(2em) ↓ `processor.py`: detect updates, interpolate],
      [*Interpolated GPS* + wheel speed + yaw rate],
      [#h(2em) ↓ `kalman.py`: EKF + RTS],
      [*Smoothed trajectory* (x, y, v, θ, ω)],
      [#h(2em) ↓ `clothoid.py`: curvature analysis],
      [*Clothoid segments* (κ₀, σ, L per segment)],
    )
  ]
]

== Input Requirements

- `longitude`, `latitude`: Interpolated GPS (S&H removed, freeze-repaired)
- `v_reference_ms`: Wheel speed at full sample rate ($f_s$)
- `yaw_rate_rad`: Signed yaw rate at full sample rate. Derived from steering
  angle via bicycle model: $omega = v dot tan(delta_"steering" slash i_s) slash L$
  where $i_s = 15.5$ (steering ratio) and $L = 2.68$ m (wheelbase).
  Alternatively from ESP gyroscope when available (rare in practice)
- `gps_update_mask`: Boolean mask marking actual GPS measurement epochs

== Output: `KalmanResult`

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    table.header[Field][Type][Description],
    [`x_m`, `y_m`], [`NDArray`], [Position in local ENU frame \[m\]],
    [`speed_ms`], [`NDArray`], [Filtered speed \[m/s\]],
    [`heading_rad`], [`NDArray`], [Filtered heading \[rad\]],
    [`yaw_rate_rad`], [`NDArray`], [Filtered yaw rate \[rad/s\]],
    [`longitude`, `latitude`], [`NDArray`], [Converted back to WGS84 \[°\]],
    [`P_diag`], [`NDArray (n,5)`], [State covariance diagonal],
  ),
  caption: [Fields of the `KalmanResult` output structure.],
) <tab-output>

= ABS Braking Robustness <abs-robustness>

During ABS (Anti-lock Braking System) intervention, the wheel speed
signal oscillates at $approx 15$ Hz due to repeated lock/unlock cycles.
Without treatment, three mechanisms corrupt the EKF:

+ *Bicycle-model yaw rate is invalid*: $omega = v dot tan(delta) slash L$
  uses oscillating $v$, producing error/$sigma$ ratios of $approx 3.5 times$.
  With $sigma_omega = 0.02$ rad/s, the filter trusts the corrupted $omega$
  and heading drifts up to 8° over a 2 s ABS event.

+ *Wheel speed is unreliable*: The raw signal swings $plus.minus 10$ km/h
  around the true deceleration curve.

+ *CTRV prediction is wrong*: With invalid $omega$ and $v$, position error
  accumulates $approx 0.28$ m per prediction epoch between GPS corrections.

== Stage 1: Speed Interpolation

The method `_clamp_speed_during_abs()` replaces wheel speed during ABS
epochs with a linear interpolation between the last non-ABS sample
before and the first non-ABS sample after each event:

$ v_"smooth" [k] = cases(
  v_"raw" [k] &"if ABS inactive",
  "interp"(k, {j: "ABS"[j] = 0}, v_"raw") quad &"if ABS active"
) $

This removes the 15 Hz oscillation while preserving the deceleration
trend (offline processing knows both anchor points). The interpolated
$v_"smooth"$ is used for both the bicycle-model yaw rate computation
and as the Kalman speed measurement input.

== Stage 2: Per-Sample Noise Inflation

Even with interpolated speed, additional protection is needed because
(a) the bicycle model is invalid during tire saturation, and (b) the
linear speed approximation is imperfect. Three per-sample scale arrays
are passed to the EKF kernel:

$ R_omega [k] = sigma_omega^2 dot s_omega [k], quad
  R_v [k] = sigma_v^2 dot s_v [k], quad
  bold(Q)[k] = bold(Q) dot s_Q [k] $

The inflation level depends on the *yaw rate source*:

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header[Source][$s_omega$ (peak)][$s_v$ (peak)][$s_Q$ (peak)],
    [Bicycle model (`steering_angle_deg`)], [49 (7× $sigma$)], [4 (2× $sigma$)], [10],
    [Direct sensor (`yaw_rate_rad`, e.g.\ ESP)], [1 (unchanged)], [4 (2× $sigma$)], [10],
  ),
  caption: [Per-sample noise inflation during ABS. Scale factors multiply the base variance.],
) <tab-abs-inflation>

*Rationale for source-dependent scaling*: A direct ESP yaw rate sensor
measures angular velocity independent of tire slip—it remains valid
during ABS. Only speed uncertainty and CTRV prediction need mild
inflation. The bicycle model, however, produces invalid $omega$ because
$v$ oscillates and tires saturate ($tan delta$ no longer maps to true
slip angle).

== Temporal Margin and Cosine Ramp

The ABS flag typically activates 100–200 ms *after* the first wheel
oscillation begins. Two techniques ensure smooth transitions:

*Margin (1000 ms)*: The ABS mask is dilated by $approx 1000$ ms on
both sides.  The ABS flag lags behind actual tire saturation by
200--700 ms (pedal → pressure build-up → saturation → flag);
during curve-braking the steering angle corrupts the model earlier
still.  The 1 s margin fully covers the pre-onset transient.

*Cosine ramp (100 ms)*: Instead of a step function, scale factors
transition smoothly:

$ s(t) = 1 + (s_"peak" - 1) dot 1/2 (1 - cos(pi t slash t_"ramp")) $

This avoids discontinuities in the Kalman gain that would cause
position transients at the ABS boundaries.

== Results

#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header[Metric][Value],
    [Mean |Kalman − BW| during ABS (straight)], [\~15 cm],
    [Max |Kalman − BW| during ABS (straight)], [< 90 cm],
    [Mean |Kalman − BW| during ABS (curve, worst case)], [21 cm],
    [Max |Kalman − BW| during ABS (curve, worst case)], [90 cm],
    [Speed tracking during ABS], [smooth deceleration, no oscillation],
    [Normal driving accuracy], [unchanged],
  ),
  caption: [ABS handling performance (steering-angle model, full stops from 46–56 km/h).],
) <tab-abs-results>

With inflation, the Kalman during ABS achieves position accuracy within
6% of Butterworth while remaining 1.6× better during normal driving.

= GPS Freeze Repair (Preprocessing) <freeze-repair>

Before coordinates reach the EKF, `processor.py` applies
`_repair_gps_freezes()` to handle a specific GPS receiver pathology:
*single-axis coordinate freezes*.

== Phenomenon

Some GPS receivers intermittently freeze one coordinate axis—for example,
reporting the same longitude `10.78352077°` for 5–8 consecutive epochs
while latitude updates normally. At 120 km/h, the vehicle travels
$approx 2.5$ m per GPS fix in the frozen direction. The receiver reports
no movement in that axis, creating a false purely-northward trajectory
segment.

*Consequences* without repair:
- The EKF partially tracks the frozen longitude, introducing a heading
  kink when the freeze ends
- The heading distortion corrupts downstream curvature derivation
  ($kappa = d theta slash d s$) and clothoid segmentation
- The RTS smoother propagates the kink globally (it is not local)

== Detection: Innovation-Based Criterion

A coordinate is classified as *frozen* when ALL of the following hold:

+ The axis value is unchanged for $>=$ `min_run_length` consecutive GPS
  epochs (default: 3)
+ The vehicle speed exceeds `min_speed` (default: 5.0 m/s)
+ The *expected* displacement in the frozen axis—computed from
  heading and arc distance—exceeds `min_expected_change_m` (default: 5.0 m)

The third criterion is critical: in tight curves, one coordinate
legitimately stays constant for several epochs when the vehicle moves
perpendicular to that axis (e.g., driving North → longitude barely changes).
The heading-projection test eliminates these *false positives*.

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    table.header[Condition][Tight curve (valid)][True freeze (invalid)],
    [Axis unchanged], [✓ (3+ epochs)], [✓ (5+ epochs)],
    [Speed > 5 m/s], [✓], [✓],
    [Heading predicts movement], [✗ (heading ⊥ axis)], [✓ (heading ∥ axis)],
    [*Classified as freeze?*], [*No*], [*Yes*],
  ),
  caption: [Innovation check distinguishes true freezes from normal quantization.],
) <tab-freeze-detection>

== Repair Strategy: Linear Interpolation

For each confirmed freeze run, the frozen coordinate values are replaced
with a linear interpolation between the last valid epoch before the freeze
and the first valid epoch after:

$ "lon"_"repaired"(t) = "lon"_"before" + ("lon"_"after" - "lon"_"before") dot
  (t - t_"before") / (t_"after" - t_"before") $

The *valid* axis is preserved unchanged. The EKF then receives the
repaired coordinates with standard measurement noise $sigma_"GPS"^2$ for
both axes—maintaining full Kalman gain and position anchoring.

== Why Not Partial (1D) Measurement Updates?

An alternative approach was tested: inflate the GPS noise for the frozen
axis to $approx oo$ (or $100 times sigma^2$), giving the Kalman zero
(or reduced) gain for that axis while preserving the valid axis at full
gain. This is the textbook "partial observability" solution.

*Result*: it performs significantly worse (tight curves: +3.4 m vs +0.5 m
median error) because:

+ Reduced gain → more position drift during the freeze forward pass
+ The RTS backward smoother is *global*: drift in one region propagates
  to the entire trajectory
+ Tight curves far from the freeze region are affected by the rebalancing

The interpolation approach works better because it *maintains position
anchoring* (the linear bridge limits drift) while providing a smooth
signal that the RTS smoother treats as unremarkable.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header[Approach][Tight Δ][Moderate Δ][Straight Δ],
    [No fix (baseline)], [0.00 m], [0.00 m], [0.00 m],
    [Interpolation (current)], [+0.53 m], [−0.67 m], [+0.02 m],
    [Epoch removal], [+0.63 m], [−0.56 m], [+0.01 m],
    [Noise inflation (100×)], [+3.42 m], [−0.69 m], [+0.03 m],
    [Noise inflation (∞)], [+4.15 m], [−0.67 m], [+0.02 m],
  ),
  caption: [Median error Δ vs baseline for GPS freeze repair approaches (G2 clothoid chain, `kappa_clip=None`).],
) <tab-repair-comparison>

== Code Reference

```python
@staticmethod
def _repair_gps_freezes(
    lon, lat, update_idx, speed,
    min_speed=5.0, min_run_length=3,
    min_expected_change_m=5.0,
) -> tuple[NDArray, NDArray]:
    """Detect and interpolate single-axis GPS freezes."""
```

Called inside `GPSProcessor._apply_kalman()` after S&H interpolation
and update detection, before building the GPS mask and calling
`GPSKalmanFilter.run()`.

= Summary

The `kalman.py` module provides a production-ready EKF/RTS implementation
tailored to the specific challenges of high-rate vehicle GPS processing:

+ *Multi-rate fusion*: Dense wheel speed and gyro ($f_s approx 500$ Hz)
  with sparse GPS (5–20 Hz) in a unified framework
+ *Position drift fix*: Custom process noise scaling for high sample rates
  (see @drift-fix)
+ *ABS braking robustness*: Source-dependent per-sample noise inflation
  with temporal margin and cosine ramp (see @abs-robustness)
+ *GPS freeze repair*: Innovation-based detection and linear interpolation
  of single-axis coordinate freezes (see @freeze-repair)
+ *Offline optimality*: RTS smoother gives minimum-variance estimate
  from all data
+ *Numba acceleration*: 4.1 s for 853k samples (see @tab-perf)
+ *Clean API*: Single `run()` call with geographic I/O, internal ENU math

#pagebreak()

#bibliography("refs.bib", title: "References", style: "ieee")
