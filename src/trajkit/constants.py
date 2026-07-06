"""Shared numeric and sampling constants for the trajkit pipeline.

All module-level guard values, sentinel values, and sampling-size
constants live here so that they can be imported by any submodule
without creating circular dependencies.

Sections
--------
- Clothoid numerical guard constants  (used by ``clothoid.py``)
- Clothoid sampling-size constants    (used by ``clothoid.py``)
- Processor constants                 (used by ``processor.py``)

"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Clothoid numerical guard constants
# ---------------------------------------------------------------------------

_KAPPA_EPS: float = 1e-12
"""Curvature [1/m] below which a clothoid is treated as a straight line
(equivalent radius > 10^12 m)."""

_STEP_EPS_M: float = 1e-12
"""Minimum arc-length integration step [m] in segment walking loops.
Prevents infinite loops when a remaining segment length rounds to zero."""

_SPEED_SQ_MIN: float = 1e-12
"""Minimum squared parametric speed [m^2/param^2] used as a floor in the
analytic kappa denominator (dx_u^2 + dy_u^2) in _smooth_and_parametrize.
Corresponds to a parametric speed of ~1e-6 m/param."""

_SPEED_CUBED_MIN: float = 1e-12
"""Minimum (dx^2+dy^2)^1.5 [m^3/param^3] used as a floor in the
Frenet-Serret curvature denominator guard.
Corresponds to a parametric speed of ~1e-4 m/param."""

_SEG_LEN_EPS_M: float = 1e-12
"""Minimum segment length [m].
Prevents division by zero when two consecutive support points are co-located."""

_SIGMA_EPS: float = 1e-10
"""Curvature rate [1/m^2] below which a clothoid segment is treated as a
circular arc (sigma=0) or straight line (sigma=0, kappa=0).
Distinct from _KAPPA_EPS: guards sigma, not kappa."""

_DS_CURVATURE_EPS_M: float = 0.01
"""Minimum arc-length step [m] used as a floor when computing the
curvature denominator guard (dtheta/ds computation).
Prevents division by zero for co-located downsampled points."""

_CHORD_EPS_M: float = 1e-9
"""Chord length [m] below which two consecutive points are treated as
co-located (used for deduplication and endpoint proximity guards)."""

_FLOAT_EPS: float = 1e-6
"""Generic float comparison epsilon used as a guard in inequality tests."""

_LARGE_ERROR_SENTINEL: float = 1e9
"""Sentinel value [m] returned by _interval_error() when no clothoid
segments are available.  Guarantees the greedy search rejects the
degenerate interval."""

# ---------------------------------------------------------------------------
# Clothoid sampling-size constants
# ---------------------------------------------------------------------------

_DENSE_SAMPLES_MIN: int = 200
"""Minimum number of samples for the dense spline evaluation grid.
Ensures adequate resolution even for very short tracks."""

_DENSE_OVERSAMPLE_FACTOR: float = 0.25
"""Fraction of resample_spacing_m used as the dense-grid spacing floor.
Dense grid spacing <= resample_spacing_m * _DENSE_OVERSAMPLE_FACTOR,
guaranteeing at least 4x oversampling relative to the output grid."""

_UNI_GRID_MIN_PTS: int = 10
"""Minimum number of points in the uniform arc-length grid produced by
_smooth_and_parametrize.  Prevents degenerate grids on very short tracks."""

_SEG_EVAL_MIN_PTS: int = 4
"""Minimum number of sample points per clothoid arc in _interval_error.
Ensures the KD-tree has enough points for a meaningful nearest-neighbour
query even on very short arcs."""

_SEG_BUILD_MIN_PTS: int = 3
"""Minimum number of sample points per clothoid arc in _build_result.
Slightly lower than _SEG_EVAL_MIN_PTS: one interior point is sufficient
for error statistics on very short terminal segments."""

# ---------------------------------------------------------------------------
# Processor constants
# ---------------------------------------------------------------------------

EARTH_METERS_PER_DEGREE: float = 111_319.49
"""Meters per degree of longitude at the equator (WGS84 ellipsoidal
approximation, see :func:`meters_per_degree_lon` at ``lat_deg=0``).

Kept as a single constant for backward compatibility (some call sites
historically used it for both axes without latitude correction). New
code should prefer :func:`meters_per_degree_lat` /
:func:`meters_per_degree_lon`, which are accurate across the full
latitude range instead of only at the equator; the true meters-per-degree
of *latitude* ranges from about 110,574 m at the equator to 111,694 m at
the poles, a ~1% variation this single constant does not capture.
"""


def meters_per_degree_lat(lat_deg: float) -> float:
    """Meters per degree of latitude at ``lat_deg`` [deg] (WGS84).

    Ellipsoidal approximation (Snyder, *Map Projections: A Working
    Manual*, USGS Professional Paper 1395, 1987), accurate to within a
    few mm across the full latitude range. Vectorized: accepts scalars
    or NumPy arrays.
    """
    phi = np.radians(lat_deg)
    return (
        111_132.92
        - 559.82 * np.cos(2 * phi)
        + 1.175 * np.cos(4 * phi)
        - 0.0023 * np.cos(6 * phi)
    )


def meters_per_degree_lon(lat_deg: float) -> float:
    """Meters per degree of longitude at ``lat_deg`` [deg] (WGS84).

    Ellipsoidal approximation (same source as
    :func:`meters_per_degree_lat`); replaces the common spherical
    shortcut ``EARTH_METERS_PER_DEGREE * cos(lat)`` with the slightly
    more accurate ellipsoidal formula. Vectorized: accepts scalars or
    NumPy arrays.
    """
    phi = np.radians(lat_deg)
    return (
        111_412.84 * np.cos(phi)
        - 93.5 * np.cos(3 * phi)
        + 0.118 * np.cos(5 * phi)
    )

_SAVGOL_MIN_WINDOW: int = 51
"""Minimum Savgol filter window [samples].

Ensures stable polynomial fitting even when the arc-length grid is coarse
(high speed, sparse GPS) so that the window computed from a distance budget
would otherwise fall below the polyorder+2 threshold.  Must be odd; 51
gives ~50-sample support for the degree-3 polynomial used throughout.
"""

_SAVGOL_POLYORDER: int = 3
"""Polynomial order for all Savitzky-Golay filter calls.

Degree 3 balances smoothness (removes GPS-scale noise) with fidelity
to the signal shape (preserves curvature peaks in the heading signal).
"""

_MS_TO_KMH: float = 3.6
"""Conversion factor from m/s to km/h (1 m/s = 3.6 km/h)."""

_S_TO_MS: float = 1000.0
"""Conversion factor from seconds to milliseconds."""

_PCT_FACTOR: float = 100.0
"""Multiplication factor to convert a fraction [0, 1] to percent [0, 100]."""

_DS_EPS_M: float = 1e-6
"""Minimum arc-length step [m] used as a guard against division by zero
when computing curvature as dθ/ds.  Below this threshold, consecutive
GPS samples are considered co-located."""

_ABS_FLAG_THRESHOLD: float = 0.5
"""Decision threshold for the binary ABS-active flag (stored as 0.0/1.0).
Values above this are treated as ABS active."""

_ABS_SPEED_NOISE_INFLATION: float = 400.0
"""Peak multiplier applied to the wheel-speed measurement variance during
ABS intervention.  The interpolated wheel speed is an unreliable proxy for
the true non-linear deceleration, so its variance is inflated (≈20x in std)
and the EKF lets GPS drive the speed estimate through the ABS window.
Applied as a variance scale; std inflation is sqrt of this value."""

_ABS_PRE_MARGIN_S: float = 0.3
"""Temporal margin [s] added before each ABS run when building the
speed-noise scale.  Covers the lag between actual wheel oscillation onset
and the ABS flag rising edge."""

_ABS_POST_MARGIN_S: float = 0.5
"""Temporal margin [s] added after each ABS run.  Covers residual wheel
oscillation after the flag falling edge before wheel speed is trustworthy
again."""

_ABS_RAMP_S: float = 0.15
"""Raised-cosine ramp length [s] on each edge of the ABS noise-scale
window.  Avoids a step change in measurement trust that would inject a
kink into the Kalman estimate."""

_FREEZE_MIN_EPOCHS: int = 3
"""Minimum number of GPS update epochs required by _repair_gps_freezes.

Three epochs are needed to anchor the repair: one epoch before the freeze,
at least one frozen epoch, and one epoch after for interpolation.
"""
