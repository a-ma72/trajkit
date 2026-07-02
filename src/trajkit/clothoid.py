"""Piecewise clothoid (Euler spiral) smoothing for GPS trajectories.

Clothoids are curves with linearly changing curvature along arc length,
making them the natural model for vehicle paths (constant steering rate
→ linearly changing curvature). This module fits a G2-continuous piecewise
clothoid chain directly to a Kalman-filtered GPS trajectory.

Pipeline position::

    S&H removal → Interpolation → Kalman EKF → **Clothoid Smoothing**

Each interval between support points is solved by ``pyclothoids.SolveG2``
(Bertolazzi-Frego), which enforces position, tangent, and curvature
continuity at both endpoints in closed form.

Typical usage::

    from .clothoid import G2ClothoidApproximator, G2ClothoidConfig

    approx = G2ClothoidApproximator(G2ClothoidConfig())
    result = approx.fit(x_m, y_m)
"""

# ruff: noqa: C901, D107, N806, PLR0912, PLR0913

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .constants import (
    EARTH_METERS_PER_DEGREE,
    _LARGE_ERROR_SENTINEL,
    _SEG_BUILD_MIN_PTS,
    _SEG_EVAL_MIN_PTS,
    _SPEED_SQ_MIN,
    _UNI_GRID_MIN_PTS,
)

# ---------------------------------------------------------------------------
# pyclothoids backend — required for G2ClothoidApproximator.
# _PYCLOTHOIDS / _PYC_CC are module-level flags checked at instantiation time.
# ---------------------------------------------------------------------------
try:
    import pyclothoids as _pyclothoids
    _PYC_CC = _pyclothoids._clothoids_cpp.ClothoidCurve
    _PYCLOTHOIDS = True
except (ImportError, AttributeError):
    _PYC_CC = None
    _PYCLOTHOIDS = False

# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def gps_enu(
    a: "NDArray[np.float64]",
    b: "NDArray[np.float64]",
    *,
    origin: tuple[float, float] | None = None,
    inverse: bool = False,
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]", tuple[float, float]]:
    """Convert between WGS84 geographic and local ENU coordinates.

    A single function for both directions of the flat-Earth tangent-plane
    approximation used throughout trajkit.

    Parameters
    ----------
    a, b :
        Forward (``inverse=False``): longitude [°], latitude [°].
        Inverse (``inverse=True``): x_east [m], y_north [m].
    origin :
        Reference point ``(lon0, lat0)`` in degrees.
        Forward: if *None*, the first sample ``(a[0], b[0])`` is used.
        Inverse: **required** — must match the origin used in the forward pass.
    inverse :
        If *False* (default): GPS → ENU.  If *True*: ENU → GPS.

    Returns
    -------
    (c, d, origin_used) :
        Forward: ``(x_east_m, y_north_m, (lon0, lat0))``
        Inverse: ``(longitude, latitude, (lon0, lat0))``

    Examples
    --------
    >>> x, y, origin = gps_enu(lon_arr, lat_arr)
    >>> lon_back, lat_back, _ = gps_enu(x, y, origin=origin, inverse=True)
    """
    if not inverse:
        # GPS → ENU
        lon, lat = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if origin is None:
            origin = (float(lon[0]), float(lat[0]))
        lon0, lat0 = origin
        cos_lat0 = np.cos(np.radians(lat0))
        x_m = (lon - lon0) * EARTH_METERS_PER_DEGREE * cos_lat0
        y_m = (lat - lat0) * EARTH_METERS_PER_DEGREE
        return x_m, y_m, origin

    # ENU → GPS
    if origin is None:
        msg = "origin=(lon0, lat0) is required for inverse conversion"
        raise ValueError(msg)
    x_m, y_m = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    lon0, lat0 = origin
    cos_lat0 = np.cos(np.radians(lat0))
    lon = x_m / (EARTH_METERS_PER_DEGREE * cos_lat0) + lon0
    lat = y_m / EARTH_METERS_PER_DEGREE + lat0
    return lon, lat, origin


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ClothoidSegment:
    """Single clothoid arc — output unit of :class:`G2ClothoidApproximator`.

    Parameters
    ----------
    x0, y0 : float
        Start position [m] in local ENU frame.
    theta0 : float
        Start heading [rad].
    kappa0 : float
        Start curvature [1/m].
    sigma : float
        Curvature rate [1/m²] (= d kappa / d s).
    length : float
        Arc length [m].

    """

    x0: float
    y0: float
    theta0: float
    kappa0: float
    sigma: float
    length: float


@dataclass
class G2ClothoidConfig:
    """Configuration for :class:`G2ClothoidApproximator`.

    Parameters
    ----------
    gps_sigma_m :
        Expected GPS noise (m).  Used as the spline smoothing budget
        ``s = n * gps_sigma_m**2``.
    spline_degree :
        B-Spline degree for reference smoothing.  Degree 5 (default) gives
        a smooth 2nd derivative which is required for stable analytic
        curvature computation.
    spline_smoothing_factor :
        Override spline smoothing factor ``s``.  ``None`` → ``n * sigma**2``.
    dense_samples_factor :
        Oversampling multiplier used for arc-length parametrisation.
    resample_spacing_m :
        Target spacing of the uniform arc-length grid [m].
    kappa_smooth_window :
        Uniform moving-average window applied to the analytic curvature
        before the greedy search.  Must be odd; 51 at 2 m spacing ≈ 100 m.
    position_tolerance_m :
        Maximum allowed nearest-neighbour lateral error between the clothoid
        chain and the smoothed reference [m].
    eval_spacing_m :
        Clothoid sampling density for interval error evaluation [m].
        Should match the sampling used in ``_build_result`` (0.3 m)
        so that the greedy acceptance criterion is consistent with
        the final error statistics.
    initial_step :
        Initial forward step (in grid indices) for the exponential search.
    min_knot_spacing :
        Minimum advance when SolveG2 fails even for a tiny interval.

    """

    gps_sigma_m: float = 2.0
    spline_degree: int = 5
    spline_smoothing_factor: float | None = None
    dedup_threshold_m: float = 1e-3
    """Minimum step between consecutive raw points [m].  Points closer than
    this are treated as sample-and-hold duplicates and removed before
    spline fitting.  Set to 0 to disable deduplication."""
    dense_samples_factor: int = 4
    resample_spacing_m: float = 2.0
    kappa_smooth_window: int = 11
    kappa_clip: float | None = None
    """Maximum absolute curvature [1/m].  GPS artefacts can produce
    impossibly tight radii (R < 2 m) that break SolveG2.  Set e.g.
    ``kappa_clip=0.25`` (R_min = 4 m) to clamp the reference curvature
    to physically reachable values.  ``None`` = no clipping."""
    position_tolerance_m: float = 0.5
    eval_spacing_m: float = 0.3
    initial_step: int = 16
    min_knot_spacing: int = 8
    """If True, the ``"g2"`` processor mode returns the reconstructed
    clothoid path (resampled onto the input samples by arc length) instead
    of the unchanged Kalman positions.  Default False keeps it a drop-in
    curvature-only regulariser, consistent with the ``"clothoid"`` mode."""


@dataclass
class G2ClothoidFitResult:
    """Result of a :class:`G2ClothoidApproximator` fit.

    Attributes
    ----------
    segments :
        All clothoid arcs as :class:`ClothoidSegment` objects
        (3 per interval, compatible with :func:`_evaluate_segments_at`).
    knot_indices :
        Indices into the uniform arc-length grid at support points.
    knot_s_m :
        Arc-length positions [m] of support points.
    knot_kappa :
        Curvature values at support points [1/m].
    n_intervals :
        Number of G2 intervals (= ``len(knot_indices) - 1``).
    n_segments :
        Total number of clothoid arcs (= ``3 * n_intervals``).
    median_error_m, rms_error_m, max_error_m :
        Nearest-neighbour lateral error statistics [m].
    runtime_s :
        Wall-clock time for :meth:`G2ClothoidApproximator.fit` [s].
    s_ref, x_ref, y_ref :
        Uniform arc-length grid and smoothed reference positions.

    """

    segments: list[ClothoidSegment]
    knot_indices: NDArray[np.int64]
    knot_s_m: NDArray[np.float64]
    knot_kappa: NDArray[np.float64]
    n_intervals: int
    n_segments: int
    median_error_m: float
    rms_error_m: float
    max_error_m: float
    runtime_s: float
    s_ref: NDArray[np.float64]
    x_ref: NDArray[np.float64]
    y_ref: NDArray[np.float64]

    def resample(self, spacing_m: float = 0.5) -> dict:
        """Sample the G2 clothoid chain at uniform arc-length spacing.

        Parameters
        ----------
        spacing_m :
            Approximate distance between consecutive output points [m].

        Returns
        -------
        dict with keys:
            x, y : NDArray[float64]
                Cartesian positions (same frame as fit input).
            kappa : NDArray[float64]
                Curvature at each sample [1/m].
            s : NDArray[float64]
                Cumulative arc length [m].
        """
        xs, ys, ks, ss = [], [], [], []
        s_cum = 0.0
        for seg in self.segments:
            n_pts = max(3, int(seg.length / spacing_m))
            s_arr = np.linspace(0.0, seg.length, n_pts)
            cc = _PYC_CC()
            cc.build(seg.x0, seg.y0, seg.theta0, seg.kappa0, seg.sigma,
                     seg.length)
            s_list = s_arr.tolist()
            xs.extend([cc.X(s) for s in s_list])
            ys.extend([cc.Y(s) for s in s_list])
            ks.extend([seg.kappa0 + seg.sigma * s for s in s_arr])
            ss.extend((s_cum + s_arr).tolist())
            s_cum += seg.length
        return {
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "kappa": np.asarray(ks),
            "s": np.asarray(ss),
        }


class G2ClothoidApproximator:
    """Fit a G2-continuous (curvature-smooth) piecewise clothoid chain.

    Uses ``pyclothoids.SolveG2`` (Bertolazzi-Frego) to connect support
    points with exactly 3 clothoid arcs per interval, satisfying position,
    tangent, and curvature constraints at both ends.  A greedy search
    minimises the number of support points while keeping the lateral error
    below :attr:`G2ClothoidConfig.position_tolerance_m`.

    Parameters
    ----------
    config :
        Algorithm parameters.  Defaults to :class:`G2ClothoidConfig`.

    Raises
    ------
    ImportError
        If ``pyclothoids`` is not installed.

    Example
    -------
    >>> cfg = G2ClothoidConfig(position_tolerance_m=0.5)
    >>> result = G2ClothoidApproximator(cfg).fit(x_m, y_m)
    >>> print(result.n_intervals, result.max_error_m)

    """

    def __init__(self, config: G2ClothoidConfig | None = None) -> None:
        if not _PYCLOTHOIDS:
            msg = (
                "pyclothoids is required for G2ClothoidApproximator.  "
                "Install with: pip install pyclothoids"
            )
            raise ImportError(msg)
        if config is None:
            config = G2ClothoidConfig()
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        x_m: NDArray[np.float64],
        y_m: NDArray[np.float64],
    ) -> G2ClothoidFitResult:
        """Fit a G2 clothoid chain to GPS positions in local ENU [m].

        Parameters
        ----------
        x_m, y_m :
            Raw GPS positions in a local ENU frame [m].  Noise level
            is governed by :attr:`G2ClothoidConfig.gps_sigma_m`.

        Returns
        -------
        G2ClothoidFitResult

        """
        t0 = time.perf_counter()

        xs, ys, s_uni, theta, kappa = self._smooth_and_parametrize(x_m, y_m)
        knot_idx = self._greedy_knots(xs, ys, theta, kappa)
        result = self._build_result(knot_idx, xs, ys, theta, kappa, s_uni)
        result.runtime_s = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _smooth_and_parametrize(
        self,
        x_m: NDArray[np.float64],
        y_m: NDArray[np.float64],
    ) -> tuple:
        """Smooth, resample uniformly in arc length, compute θ and κ.

        Returns
        -------
        xs, ys : NDArray
            Smoothed positions on a uniform arc-length grid.
        s_uni : NDArray
            Uniform arc-length values [m].
        theta : NDArray
            Heading [rad], consistent with κ via θ(s) = θ₀ + ∫κ ds.
        kappa : NDArray
            Curvature [1/m], analytic from spline + moving-average.

        """
        cfg = self.cfg

        # Remove sample-and-hold duplicates: high-rate GPS logged at 100-500 Hz
        # holds the last fix between 5-20 Hz updates, producing runs of
        # identical points (~7% on typical vehicle logs).  Left in, they
        # distort the spline parametrisation (many knots map to zero arc
        # length) and inject curvature noise.  We deduplicate on a small
        # spatial threshold before fitting.
        x_m = np.asarray(x_m, dtype=np.float64)
        y_m = np.asarray(y_m, dtype=np.float64)
        step_ds = np.hypot(np.diff(x_m), np.diff(y_m))
        keep = np.concatenate([[True], step_ds > cfg.dedup_threshold_m])
        x_m = x_m[keep]
        y_m = y_m[keep]
        n = len(x_m)

        sf = cfg.spline_smoothing_factor
        if sf is None:
            sf = float(n) * cfg.gps_sigma_m ** 2

        # Parametrise by chord length, NOT sample index.  Index parametrisation
        # (t = arange(n)) assumes uniform spacing; after dedup the spacing is
        # still uneven (variable speed), and an index parameter makes the
        # spline's smoothing budget act unevenly along the track — over-
        # smoothing dense (slow) regions and under-smoothing sparse (fast)
        # ones, which is what roughened the curvature profile.  Chord length is
        # a near-arc-length parameter and distributes the smoothing budget
        # geometrically, matching the splprep-based reference.
        t = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_m), np.diff(y_m)))])

        # Degree-5 B-Spline: smooth 2nd derivative for stable analytic κ
        sp_x = UnivariateSpline(t, x_m, k=cfg.spline_degree, s=sf)
        sp_y = UnivariateSpline(t, y_m, k=cfg.spline_degree, s=sf)
        sp_dx = sp_x.derivative()
        sp_dy = sp_y.derivative()
        sp_ddx = sp_dx.derivative()
        sp_ddy = sp_dy.derivative()

        # Dense sampling for arc-length parametrisation
        t_dense = np.linspace(t[0], t[-1], n * cfg.dense_samples_factor)
        dx_d = sp_dx(t_dense)
        dy_d = sp_dy(t_dense)
        # Arc length: ∫|r'(t)| dt
        speed = np.sqrt(dx_d ** 2 + dy_d ** 2)
        s_dense = np.concatenate([[0.0], cumulative_trapezoid(speed, t_dense)])

        total_len = float(s_dense[-1])
        n_uni = max(_UNI_GRID_MIN_PTS, int(total_len / cfg.resample_spacing_m) + 1)
        s_uni = np.linspace(0.0, total_len, n_uni)

        # Map uniform arc length back to spline parameter t
        t_uni = interp1d(
            s_dense,
            t_dense,
            kind="linear",
            bounds_error=False,
            fill_value=(t_dense[0], t_dense[-1]),
        )(s_uni)

        xs = sp_x(t_uni)
        ys = sp_y(t_uni)
        dx_u = sp_dx(t_uni)
        dy_u = sp_dy(t_uni)
        ddx_u = sp_ddx(t_uni)
        ddy_u = sp_ddy(t_uni)

        # Initial heading from spline tangent at the first point.
        theta0_ = float(np.arctan2(dy_u[0], dx_u[0]))

        # Curvature: analytic formula κ = (x'·y'' - y'·x'') / (x'² + y'²)^(3/2).
        # Uses the spline second derivatives computed above — more accurate than
        # finite-differencing theta (np.gradient adds numerical noise on top of
        # discretisation error and is inconsistent after the moving average).
        denom = dx_u ** 2 + dy_u ** 2
        denom = np.where(denom < _SPEED_SQ_MIN, _SPEED_SQ_MIN, denom)
        kappa_raw = (dx_u * ddy_u - dy_u * ddx_u) / denom ** 1.5

        # Moving-average smoothing (uniform window, reflect padding).
        # Reference uses 51 samples ≈ 102 m at 2 m spacing to suppress residual
        # spline noise without distorting curvature peaks.
        w = cfg.kappa_smooth_window | 1  # ensure odd
        if w > 1 and len(kappa_raw) > w:
            pad = w // 2
            kappa_padded = np.pad(kappa_raw, pad, mode="reflect")
            kappa = np.convolve(
                kappa_padded, np.ones(w) / w, mode="valid",
            )[:len(kappa_raw)]
        else:
            kappa = kappa_raw

        # Optional curvature clipping: GPS artefacts can produce impossibly
        # tight radii that break SolveG2.
        if cfg.kappa_clip is not None:
            kappa = np.clip(kappa, -abs(cfg.kappa_clip), abs(cfg.kappa_clip))

        # Heading from integrated curvature — guarantees θ and κ are mutually
        # consistent by construction: θ = θ0 + ∫κ ds.  SolveG2 imposes position,
        # tangent, and curvature constraints simultaneously; if θ and κ come from
        # different sources (atan2 vs smoothed gradient) they disagree after
        # moving-average smoothing, causing the solver to produce excess knots.
        theta = theta0_ + np.concatenate([[0.0], cumulative_trapezoid(kappa, s_uni)])

        return xs, ys, s_uni, theta, kappa

    # ------------------------------------------------------------------
    # Interval solver and error metric
    # ------------------------------------------------------------------

    @staticmethod
    def _interval_segs(
        xs: NDArray,
        ys: NDArray,
        theta: NDArray,
        kappa: NDArray,
        ia: int,
        ib: int,
    ) -> list | None:
        """Call SolveG2 for interval [ia, ib].

        Returns a list of :class:`ClothoidSegment` (3 arcs) or ``None``
        when the solver fails or returns an empty chain.
        """
        try:
            clothoids = _pyclothoids.SolveG2(
                float(xs[ia]), float(ys[ia]), float(theta[ia]), float(kappa[ia]),
                float(xs[ib]), float(ys[ib]), float(theta[ib]), float(kappa[ib]),
            )
        except Exception:
            return None
        if not clothoids:
            return None
        segs = []
        for c in clothoids:
            p = c.Parameters  # (x0, y0, theta0, kappa0, dk, length)
            segs.append(ClothoidSegment(
                x0=p[0], y0=p[1], theta0=p[2], kappa0=p[3],
                sigma=p[4], length=p[5],
            ))
        return segs

    def _interval_error(
        self,
        ia: int,
        ib: int,
        segs: list,
        xs: NDArray,
        ys: NDArray,
    ) -> float:
        """Max nearest-neighbour lateral error: clothoid chain → reference.

        The clothoid chain is sampled at :attr:`G2ClothoidConfig.eval_spacing_m`
        and a KD-tree is built; the reference points xs[ia:ib+1], ys[ia:ib+1]
        are then queried.
        """
        pts = []
        for seg in segs:
            n_pts = max(_SEG_EVAL_MIN_PTS, int(seg.length / self.cfg.eval_spacing_m))
            s_arr = np.linspace(0.0, seg.length, n_pts)
            cc = _PYC_CC()
            cc.build(seg.x0, seg.y0, seg.theta0, seg.kappa0, seg.sigma, seg.length)
            sf_list = s_arr.tolist()
            pts.append(np.column_stack([[cc.X(s) for s in sf_list],
                                         [cc.Y(s) for s in sf_list]]))
        if not pts:
            return _LARGE_ERROR_SENTINEL
        P = np.vstack(pts)
        tree = cKDTree(P)
        ref = np.column_stack([xs[ia:ib + 1], ys[ia:ib + 1]])
        dd, _ = tree.query(ref)
        return float(dd.max())

    # ------------------------------------------------------------------
    # Greedy knot search
    # ------------------------------------------------------------------

    def _greedy_knots(
        self,
        xs: NDArray,
        ys: NDArray,
        theta: NDArray,
        kappa: NDArray,
    ) -> NDArray[np.int64]:
        """Find the minimum support-point set via greedy forward search.

        Algorithm (faithful to Bertolazzi-Frego g2fit.py):

        1. From current knot *i*: exponentially expand the candidate
           endpoint *j* (step 16, 32, 64, …) while the interval error
           stays below tolerance.
        2. Once tolerance is exceeded, bisect between the last valid *j*
           and the first failing *j* to find the exact boundary.
        3. Append that boundary as the next knot and advance *i*.
        4. If even the minimal step fails, advance by *min_knot_spacing*
           without bisection.
        """
        cfg = self.cfg
        N = len(xs)
        knot: list[int] = [0]
        i = 0

        def ok(j: int) -> bool:
            """Return True if interval [i, j] solves and stays within tolerance."""
            segs = self._interval_segs(xs, ys, theta, kappa, i, j)
            if not segs:
                return False
            return self._interval_error(i, j, segs, xs, ys) <= cfg.position_tolerance_m

        while i < N - 1:
            # --- Exponential forward search for the upper bracket ---
            # lo = largest j known to be acceptable, hi = smallest known to fail.
            lo = i   # [i, i] is trivially fine
            hi = -1  # not yet found
            step = cfg.initial_step
            j = min(i + step, N - 1)
            while True:
                if ok(j):
                    lo = j
                    if j == N - 1:
                        break          # reached the end, all good
                    step *= 2
                    j = min(i + step, N - 1)
                else:
                    hi = j             # first failing endpoint
                    break

            if lo == N - 1:
                knot.append(N - 1)
                break

            # --- Bisection between last valid (lo) and first failing (hi) ---
            # This runs regardless of whether the FIRST step succeeded: when the
            # initial step already fails (tight curve), lo == i and we bisect
            # down toward i+1, finding the largest admissible interval.  This is
            # what keeps the knot count resolution-independent — the previous
            # version fell back to fixed min_knot_spacing steps here, which
            # produced one knot every few grid points on fine grids.
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if ok(mid):
                    lo = mid
                else:
                    hi = mid

            if lo > i:
                # Found the largest admissible interval [i, lo].
                knot.append(lo)
                i = lo
                continue

            # --- lo == i: even [i, i+1] violates tolerance ---
            # Either SolveG2 fails or the reference is locally inconsistent
            # (GPS artefact / impossibly tight radius).  Advance by the
            # smallest step that at least produces a valid SolveG2 chain so the
            # output stays gap-free; consider raising kappa_clip if this fires.
            jf = min(i + 1, N - 1)
            for cand in range(i + 1, min(i + cfg.min_knot_spacing, N - 1) + 1):
                if self._interval_segs(xs, ys, theta, kappa, i, cand) is not None:
                    jf = cand
                    break
            knot.append(jf)
            i = jf

        knot = sorted(set(knot))
        if knot[-1] != N - 1:
            knot.append(N - 1)
        return np.array(knot, dtype=np.int64)

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        knot_idx: NDArray[np.int64],
        xs: NDArray,
        ys: NDArray,
        theta: NDArray,
        kappa: NDArray,
        s_uni: NDArray,
    ) -> G2ClothoidFitResult:
        """Build the full clothoid chain and compute error statistics.

        When SolveG2 fails for an interval (unusual for well-conditioned
        input), a G1-Hermite fallback is used to prevent chain gaps.
        """
        all_segs: list[ClothoidSegment] = []
        for a, b in itertools.pairwise(knot_idx):
            segs = self._interval_segs(xs, ys, theta, kappa, int(a), int(b))
            if segs is None:
                # G1-Hermite fallback: 1 clothoid arc matching only
                # position + heading (weaker than G2, but gap-free).
                try:
                    c_fb = _pyclothoids.Clothoid.G1Hermite(
                        float(xs[a]), float(ys[a]), float(theta[a]),
                        float(xs[b]), float(ys[b]), float(theta[b]),
                    )
                    p = c_fb._ClothoidCurve  # noqa: SLF001
                    segs = [ClothoidSegment(
                        x0=float(xs[a]), y0=float(ys[a]),
                        theta0=float(theta[a]),
                        kappa0=p.KappaStart(), sigma=p.dk(),
                        length=p.length(),
                    )]
                except Exception:
                    segs = []  # degenerate: empty (chain gap)
            if segs:
                all_segs.extend(segs)

        # Sample the full chain at eval_spacing_m for accurate error statistics
        pts = []
        for seg in all_segs:
            n_pts = max(_SEG_BUILD_MIN_PTS, int(seg.length / self.cfg.eval_spacing_m))
            s_arr = np.linspace(0.0, seg.length, n_pts)
            cc = _PYC_CC()
            cc.build(seg.x0, seg.y0, seg.theta0, seg.kappa0, seg.sigma, seg.length)
            sf_list = s_arr.tolist()
            pts.append(np.column_stack([[cc.X(s) for s in sf_list],
                                         [cc.Y(s) for s in sf_list]]))

        P = np.vstack(pts) if pts else np.zeros((1, 2))
        tree = cKDTree(P)
        ref = np.column_stack([xs, ys])
        dd, _ = tree.query(ref)

        return G2ClothoidFitResult(
            segments=all_segs,
            knot_indices=knot_idx,
            knot_s_m=s_uni[knot_idx],
            knot_kappa=kappa[knot_idx],
            n_intervals=int(len(knot_idx) - 1),
            n_segments=len(all_segs),
            median_error_m=float(np.median(dd)),
            rms_error_m=float(np.sqrt(np.mean(dd ** 2))),
            max_error_m=float(dd.max()),
            runtime_s=0.0,  # overwritten by fit()
            s_ref=s_uni,
            x_ref=xs,
            y_ref=ys,
        )
