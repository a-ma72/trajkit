"""GPS track conditioning pipeline for sample-and-hold vehicle data.

This module provides the core processing classes for conditioning GPS
position data from high-rate vehicle measurement systems (typically
≥ 500 Hz sample rate with 5-20 Hz GPS updates). It handles the typical
signal chain: sample-and-hold removal, interpolation, low-pass filtering,
speed derivation, and outlier detection.

The Kalman smoothing mode fuses GPS position with wheel
speed and yaw rate for sub-meter accuracy.
"""

# ruff: noqa: C901, D107, PLC0415, PLR0912, PLR0913, PLR0915

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import butter, filtfilt, savgol_filter

from .clothoid import G2ClothoidApproximator, G2ClothoidConfig
from .kalman import KalmanConfig

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Defined centrally in constants.py so that clothoid.py can import
# EARTH_METERS_PER_DEGREE without importing processor.py - importing it here
# at module level would close an import cycle (processor -> clothoid ->
# processor) during package initialisation.  Re-exported below for backward
# compatibility with code that does `from trajkit.processor import
# EARTH_METERS_PER_DEGREE`.

from .constants import (  # noqa: E402
    _ABS_FLAG_THRESHOLD,
    _ABS_POST_MARGIN_S,
    _ABS_PRE_MARGIN_S,
    _ABS_RAMP_S,
    _ABS_SPEED_NOISE_INFLATION,
    _DS_EPS_M,
    _FREEZE_MIN_EPOCHS,
    _MS_TO_KMH,
    _PCT_FACTOR,
    _S_TO_MS,
    _SAVGOL_MIN_WINDOW,
    _SAVGOL_POLYORDER,
    EARTH_METERS_PER_DEGREE,
)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GPSTrack:
    """Processed GPS track with metadata and auxiliary channels.

    Attributes
    ----------
    longitude : NDArray[np.float64]
        Longitude in degrees (WGS84).
    latitude : NDArray[np.float64]
        Latitude in degrees (WGS84).
    speed_ms : NDArray[np.float64]
        Speed in m/s (same length as longitude / latitude).
    mask : NDArray[np.bool_]
        Boolean mask indicating valid (non-outlier) samples.
    fs : float
        Sampling frequency in Hz.
    dt : float
        Sampling period in seconds.
    aux : dict[str, NDArray[np.float64]]
        Auxiliary channels aligned with the position arrays.
        Populated via the ``aux_channels`` argument of
        :meth:`GPSProcessor.process`.

    """

    longitude: NDArray[np.float64]
    latitude: NDArray[np.float64]
    speed_ms: NDArray[np.float64]
    mask: NDArray[np.bool_]
    fs: float
    dt: float
    aux: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def speed_kmh(self) -> NDArray[np.float64]:
        """Speed in km/h (same length as longitude / latitude)."""
        return self.speed_ms * _MS_TO_KMH

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return len(self.longitude)

    @property
    def n_valid(self) -> int:
        """Number of valid (non-outlier) samples."""
        return int(np.sum(self.mask))

    def get_masked(self, channel: str) -> NDArray[np.float64]:
        """Return an auxiliary channel with the validity mask applied.

        Parameters
        ----------
        channel : str
            Name of the auxiliary channel.

        Returns
        -------
        NDArray[np.float64]
            Channel values at valid samples only.

        Raises
        ------
        KeyError
            If the channel name is not found in :attr:`aux`.

        """
        return self.aux[channel][self.mask]


# ---------------------------------------------------------------------------
# GPS Processor
# ---------------------------------------------------------------------------


class GPSProcessor:
    """Pipeline for GPS signal conditioning from sample-and-hold raw data.

    Handles the typical signal chain for vehicle-logged GPS:

    1. Detect actual GPS update epochs within high-rate sampled data.
    2. Remove sample-and-hold via linear interpolation.
    3. Calibrate sampling rate from a reference speed channel.
    4. Apply position smoothing (Butterworth LP or Kalman EKF).
    5. Derive speed from the conditioned position signal.
    6. Flag outliers exceeding a physical speed threshold.

    Parameters
    ----------
    v_max_kmh : float, default 250.0
        Maximum physically plausible speed in km/h.
        Points exceeding this are flagged as outliers.
    filter_order : int, default 2
        Order of the Butterworth low-pass filter.
    cutoff_factor : float, default 0.5
        Fraction of the effective GPS Nyquist frequency
        used as filter cutoff (0 < cutoff_factor <= 1).
    min_speed_calibration : float, default 1.0
        Minimum reference speed (m/s) for dt calibration.
        Samples below this are excluded from calibration.
    smoothing : {'butterworth', 'kalman', 'kalman_jax', 'g2'}, default 'butterworth'
        Position smoothing method.
        - 'kalman': Numba-accelerated EKF (CPU, ~4s for 850k samples).
        - 'kalman_jax': JAX/XLA EKF (GPU-ready, uses lax.scan).
          (θ-continuous chain, physically consistent κ profile, no drift).
        Kalman variants require `yaw_rate_rad` or
        `steering_angle_deg` in :meth:`process`.
    speed_smoothing_s : float, default 0.3
        Savitzky-Golay smoothing duration [s] for post-smoothing the
        speed signal derived from position differences. Only applies
        to the Butterworth pipeline. The actual window in samples is
        computed at runtime as ``int(speed_smoothing_s * fs) | 1``,
        ensuring rate-independent behaviour. Set 0.0 to disable.
    kalman_config : KalmanConfig or None, optional
        Tuning parameters for the Kalman filter. If None and
        smoothing='kalman', default KalmanConfig is used (dt is
        auto-calibrated from data).

    Examples
    --------
    >>> from trajkit import GPSProcessor
    >>> proc = GPSProcessor(v_max_kmh=250, smoothing='kalman')
    >>> track = proc.process(lon_raw, lat_raw, v_ref, yaw_rate_rad=yr)
    >>> proc.plot(track, basemap=True)

    """

    def __init__(
        self,
        v_max_kmh: float = 250.0,
        filter_order: int = 2,
        cutoff_factor: float = 0.5,
        min_speed_calibration: float = 1.0,
        smoothing: str | None = "butterworth",
        speed_smoothing_s: float = 0.3,
        kalman_config: Mapping[str, object] | None = None,
        g2_config: Mapping[str, object] | None = None,
        *,
        freeze_repair: bool = False,
    ) -> None:
        # Check that smoothing mode is valid
        valid_smoothings = ("butterworth", "kalman", "kalman_jax", "g2")
        smoothings = [] if not smoothing else smoothing.split("+")
        for s in smoothings:
            if s not in valid_smoothings:
                msg = (
                    f"Invalid smoothing mode '{s}'. "
                    f"Valid options: {valid_smoothings}."
                )
                raise ValueError(msg)
        if "butterworth" in smoothings and "kalman" in smoothings:
            msg = (
                "Cannot combine 'butterworth' and 'kalman' smoothing. "
                "Choose one or the other."
            )
            raise ValueError(msg)
        self.v_max_kmh = v_max_kmh
        self.filter_order = filter_order
        self.cutoff_factor = cutoff_factor
        self.min_speed_calibration = min_speed_calibration
        self.smoothing = smoothing
        self.speed_smoothing_s = speed_smoothing_s
        self.kalman_config = kalman_config
        self.g2_config = g2_config
        self.freeze_repair = freeze_repair
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        lon_raw: NDArray[np.float64],
        lat_raw: NDArray[np.float64],
        v_reference_ms: NDArray[np.float64],
        aux_channels: Mapping[str, NDArray[np.float64]] | None = None,
        yaw_rate_rad: NDArray[np.float64] | None = None,
        steering_angle_deg: NDArray[np.float64] | None = None,
        lateral_acceleration_ms2: NDArray[np.float64] | None = None,
        abs_flag: NDArray[np.float64] | None = None,
        lon_int: NDArray[np.float64] | None = None,
        lat_int: NDArray[np.float64] | None = None,
        gps_samplerate_hz: float | None = None,
    ) -> GPSTrack:
        """Run the full GPS conditioning pipeline.

        Parameters
        ----------
        lon_raw : NDArray[np.float64]
            Raw longitude samples (with sample-and-hold).
        lat_raw : NDArray[np.float64]
            Raw latitude samples (with sample-and-hold).
        v_reference_ms : NDArray[np.float64]
            Reference speed channel in m/s (e.g. wheel-based).
            Must have the same length as lon_raw / lat_raw.
        aux_channels : Mapping[str, NDArray[np.float64]] | None, optional
            Additional time-aligned channels to carry through the
            pipeline (e.g. steering angle, yaw rate). Each array
            must have the same length as the position arrays.
            They are stored unmodified in :attr:`GPSTrack.aux`
            and can be accessed with :meth:`GPSTrack.get_masked`.
        yaw_rate_rad : NDArray[np.float64] | None, optional
            Signed yaw rate in rad/s (required when smoothing='kalman').
            Must have the same length as lon_raw.
        steering_angle_deg : NDArray[np.float64] | None, optional
            Steering wheel angle in degrees (signed). Used to derive
            yaw rate via bicycle model when ``yaw_rate_rad`` is not
            provided. Must have the same length as lon_raw.
        lateral_acceleration_ms2 : NDArray[np.float64] | None, optional
            Lateral acceleration in m/s² (signed). Used to derive
            yaw rate when ``yaw_rate_rad`` is not provided. Must have
            the same length as lon_raw.
            Note: There is no filter applied to this channel; it is
            assumed to be pre-conditioned.
        abs_flag : NDArray[np.float64] | None, optional
            Binary ABS-active flag (1 = active). When provided alongside
            ``steering_angle_deg``, wheel speed is clamped during ABS
            events before computing bicycle-model yaw rate.
        lon_int : NDArray[np.float64] | None, optional
            Integer part of longitude from split-channel GPS (e.g.
            rpc3 GPS_X_Int). When provided together with ``lat_int``,
            the integer parts are rounded to the nearest integer
            (correcting float rounding artifacts like 8.9999 → 9)
            and summed with ``lon_raw`` / ``lat_raw`` (the fractional
            parts) to form the full coordinate. A one-time warning is
            logged if rounding corrections were applied.
        lat_int : NDArray[np.float64] | None, optional
            Integer part of latitude (e.g. rpc3 GPS_Y_Int). Same
            semantics as ``lon_int``; both must be provided together.
        gps_samplerate_hz : float | None, optional
            Optional nominal sample rate of the input arrays.
            If provided, it is used to compute the effective GPS Nyquist
            frequency for the Butterworth filter cutoff.
            If None, the effective GPS Nyquist frequency is estimated
            from the detected GPS update epochs.

        Returns
        -------
        GPSTrack
            Processed track with filtered positions, derived speed,
            validity mask, and auxiliary channels.

        """
        n = len(lon_raw)
        if len(lat_raw) != n or len(v_reference_ms) != n:
            msg = "All input arrays must have equal length."
            raise ValueError(msg)

        # Validate auxiliary channels
        aux = aux_channels or {}
        for name, arr in aux.items():
            if len(arr) != n:
                msg_0 = (
                    f"Auxiliary channel '{name}' has length {len(arr)}, "
                    f"expected {n}."
                )
                raise ValueError(
                    msg_0,
                )

        # Step 0a: Compose coordinates from integer + fraction parts
        if lon_int is not None and lat_int is not None:
            lon_int_rounded = np.rint(lon_int, dtype=np.float64)
            lat_int_rounded = np.rint(lat_int, dtype=np.float64)
            n_lon_corrected = int(np.sum(~np.isclose(lon_int_rounded, lon_int)))
            n_lat_corrected = int(np.sum(~np.isclose(lat_int_rounded, lat_int)))
            if n_lon_corrected > 0 or n_lat_corrected > 0:
                logger.warning(
                    "GPS integer parts contained rounding errors: "
                    "%d lon + %d lat samples "
                    "corrected to nearest integer before composition.",
                    n_lon_corrected, n_lat_corrected,
                )
            lon_raw = lon_int_rounded + lon_raw
            lat_raw = lat_int_rounded + lat_raw

        # Step 0b: Mark zero coordinates as NaN (no GPS fix)
        no_fix = (lon_raw == 0.0) | (lat_raw == 0.0)
        if no_fix.any():
            lon_raw = lon_raw.copy()
            lat_raw = lat_raw.copy()
            lon_raw[no_fix] = np.nan
            lat_raw[no_fix] = np.nan

        # Interpolate wheel speed through ABS epochs (removes 15 Hz
        # lock/unlock oscillation, preserves deceleration trend).
        # Used for both bicycle-model yaw rate and Kalman speed input.
        v_smooth = v_reference_ms
        if abs_flag is not None:
            v_smooth = self._clamp_speed_during_abs(v_reference_ms, abs_flag)

        # Derive yaw rate from steering angle, optional
        if steering_angle_deg is not None:
            if yaw_rate_rad is not None:
                logger.warning(
                    "Both yaw_rate_rad and steering_angle_deg provided; "
                    "yaw_rate_rad will be overwritten by bicycle-model "
                    "derivation from steering_angle_deg.",
                )
            _cfg = self.kalman_config or KalmanConfig(dt=1.0)
            delta_road_rad = np.radians(steering_angle_deg) / _cfg.steering_ratio
            yaw_rate_rad = v_smooth * np.tan(delta_road_rad) / _cfg.wheelbase_m

        # Derive yaw rate from lateral acceleration, optional
        if lateral_acceleration_ms2 is not None:
            if yaw_rate_rad is not None:
                logger.warning(
                    "Both yaw_rate_rad and lateral_acceleration_ms2 provided; "
                    "yaw_rate_rad will be overwritten by derivation from "
                    "lateral_acceleration_ms2.",
                )
            yaw_rate_rad = lateral_acceleration_ms2 / v_smooth

        # Track whether yaw rate comes from a model (invalid during ABS)
        # or from a direct sensor (valid during ABS).
        _yr_from_model = (
            steering_angle_deg is not None
            or lateral_acceleration_ms2 is not None
        )

        # Step 1: Detect GPS update epochs
        update_idx = self._detect_updates(lon_raw, lat_raw)

        # Remove epochs where coordinates are NaN (no GPS fix)
        valid_coords = ~(np.isnan(lon_raw) | np.isnan(lat_raw))
        update_idx = update_idx[valid_coords[update_idx]]

        # Step 2: Remove sample-and-hold
        lon_interp, lat_interp = self._interpolate(
            lon_raw, lat_raw, update_idx,
        )

        # Step 3: Calibrate sampling rate
        if gps_samplerate_hz is not None:
            dt = 1.0 / gps_samplerate_hz
            fs = gps_samplerate_hz
        else:
            dt = self._calibrate_dt(lon_raw, lat_raw, update_idx, v_reference_ms)
            fs = 1.0 / dt

        # During ABS, the interpolated wheel speed (v_smooth) is only a
        # rough approximation of the true, non-linear deceleration.  Rather
        # than blending the *position* toward a model-free Butterworth
        # filter (which discards the good GPS+yaw-rate fusion), we inflate
        # the wheel-speed *measurement* noise so the EKF lets GPS drive the
        # speed estimate through the ABS window.  The yaw rate comes from a
        # direct sensor here (valid under tire saturation), so its noise is
        # left untouched.  A cosine-ramped temporal margin avoids transient
        # kinks at the ABS flag edges (the flag often lags the actual wheel
        # oscillation).
        speed_noise_scale = None
        smoothings = [] if not self.smoothing else self.smoothing.split("+")
        if abs_flag is not None and smoothings:
            speed_noise_scale = self._build_abs_noise_scale(
                abs_flag, dt, n,
                inflation=_ABS_SPEED_NOISE_INFLATION,
                pre_margin_s=_ABS_PRE_MARGIN_S,
                post_margin_s=_ABS_POST_MARGIN_S,
                ramp_s=_ABS_RAMP_S,
            )

        # Step 4: Position smoothing
        if any(s in ("kalman", "kalman_jax") for s in smoothings):
            # Butterworth low-pass (default)
            f_gps = len(update_idx) / (n * dt)
            f_cutoff = f_gps * 0.4  # Use a lower cutoff for Kalman to avoid overshoot
            lon_filt, lat_filt = self._lowpass(
                lon_interp, lat_interp, f_cutoff, fs,
            )
            lon_filt, lat_filt, speed_ms = self._apply_kalman(
                lon_filt, lat_filt, v_smooth,
                yaw_rate_rad, update_idx, dt,
                speed_noise_scale=speed_noise_scale,
            )
        elif "butterworth" in smoothings:
            # Butterworth low-pass (default)
            f_gps = len(update_idx) / (n * dt)
            f_cutoff = f_gps * self.cutoff_factor
            lon_filt, lat_filt = self._lowpass(
                lon_interp, lat_interp, f_cutoff, fs,
            )
            # Step 5: Speed - use reference (wheel speed) directly.
            # Position-derived speed via np.diff is inherently noisy
            # because GPS position error (~2m) / dt_update ≈ high noise.
            # Only Kalman can estimate speed properly from GPS position.
            speed_ms = np.copy(v_reference_ms)
        else:
            # No smoothing, no speed derivation
            lon_filt, lat_filt = lon_interp, lat_interp
            speed_ms = np.copy(v_reference_ms)

        # Optional Savitzky-Golay post-smoothing (rate-adaptive window)
        _sg_win = self._speed_smoothing_window(fs)
        if _sg_win > 0 and len(speed_ms) >= _sg_win:
            speed_ms = savgol_filter(
                speed_ms, _sg_win, polyorder=_SAVGOL_POLYORDER,
            )

        # Step 4b: G2 clothoid spline (SolveG2) on pre-filtered output
        if "g2" in smoothings:
            lon_filt, lat_filt, speed_ms = self._apply_g2(
                lon_filt, lat_filt, speed_ms, dt,
            )

        # Step 6: Outlier mask
        mask = self._build_mask(speed_ms, n)

        return GPSTrack(
            longitude=lon_filt,
            latitude=lat_filt,
            speed_ms=speed_ms,
            mask=mask,
            fs=fs,
            dt=dt,
            aux=aux,
        )


    def _apply_g2(
        self,
        lon_kalman: NDArray[np.float64],
        lat_kalman: NDArray[np.float64],
        speed_ms: NDArray[np.float64],
        _dt: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Fit a G2-continuous clothoid spline (SolveG2) to the Kalman track.

        Uses
        :class:`~trajkit.clothoid.G2ClothoidApproximator` to fit
        a curvature-continuous (C1, i.e. G2) chain of clothoid arcs anchored to
        support points on the smoothed track.  Each support interval is solved
        by ``pyclothoids.SolveG2`` (Bertolazzi-Frego, 3 arcs), so position,
        tangent, and curvature are all continuous across joints and there is no
        forward-integration drift.

        The full fit (segments, support points, error statistics, reconstructed
        path) is stored in ``self._g2_result``.  By default the Kalman
        positions are returned unchanged so this mode is a drop-in replacement
        Positions are always those of the Kalman filter.

        Parameters
        ----------
        lon_kalman, lat_kalman : NDArray
            Kalman-filtered position.
        speed_ms : NDArray
            Kalman speed estimates [m/s].
        dt : float
            Sampling period [s] (unused; kept for signature symmetry).

        Returns
        -------
        lon_out, lat_out, speed_ms : NDArray
            Position (Kalman or reconstructed, per config) and speed.

        """
        cfg = self.g2_config if self.g2_config is not None else G2ClothoidConfig()

        # Local equirectangular frame.
        lon0, lat0 = lon_kalman[0], lat_kalman[0]
        cos_lat0 = np.cos(np.radians(lat0))
        x_m = (lon_kalman - lon0) * EARTH_METERS_PER_DEGREE * cos_lat0
        y_m = (lat_kalman - lat0) * EARTH_METERS_PER_DEGREE

        result = G2ClothoidApproximator(cfg).fit(x_m, y_m)
        self._g2_result = result

        logger.info(
            "G2 clothoid fit: %d intervals (%d arcs), max %.3f m, rms %.3f m, %.1f s",
            result.n_intervals, result.n_segments,
            result.max_error_m, result.rms_error_m, result.runtime_s,
        )

        return lon_kalman, lat_kalman, speed_ms

    def plot(
        self,
        track: GPSTrack,
        zoom: int = 16,
        figsize: tuple[float, float] = (12, 10),
        color: str = "red",
        linewidth: float = 0.8,
        label: str | None = None,
        title: str | None = None,
        color_by: str | None = None,
        cmap: str = "coolwarm",
        clabel: str | None = None,
        percentile_clip: float = 99.0,
        fig: plt.Figure | None = None,
        *,
        basemap: bool = True,
        symmetric_cmap: bool = True,
    ) -> None:
        """Plot the GPS track with optional basemap and color coding.

        Parameters
        ----------
        track : GPSTrack
            Processed track from :meth:`process`.
        basemap : bool, default True
            Whether to add an OpenStreetMap basemap.
        zoom : int, default 16
            Tile zoom level for the basemap.
        figsize : tuple, default (12, 10)
            Figure size in inches.
        color : str, default 'red'
            Line color (ignored when ``color_by`` is set).
        linewidth : float, default 0.8
            Line width.
        label : str or None
            Legend label. If None, no legend is shown.
        title : str or None
            Custom title. If None, auto-generated.
        color_by : str or None
            Name of an auxiliary channel or ``'speed'`` to color-code
            the trajectory. If None, uses solid ``color``.
        cmap : str, default 'coolwarm'
            Matplotlib colormap name (used with ``color_by``).
        clabel : str or None
            Colorbar label. Auto-generated if None.
        fig : matplotlib.figure.Figure or None
            If provided, plot into this figure instead of creating a new one.
        symmetric_cmap : bool, default True
            If True, center the colormap at zero.
        percentile_clip : float, default 99.0
            Percentile for clipping extreme values in colormap.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.

        """
        if fig is None:
            fig_provided = False
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig_provided = True
            ax = fig.gca()
        mask = track.mask
        lon = track.longitude[mask]
        lat = track.latitude[mask]

        if color_by is not None:
            # Resolve color channel
            if color_by == "speed":
                c_data = track.speed_kmh[
                    np.where(mask)[0][:-1]  # one color per segment: drop last valid index
                ]
                _clabel = clabel or "Speed [km/h]"
            else:
                c_data = track.get_masked(color_by)
                _clabel = clabel or f"{color_by}"

            # Build line segments
            points = np.column_stack([lon, lat]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color normalization
            if symmetric_cmap:
                vabs = np.percentile(np.abs(c_data), percentile_clip)
                norm = Normalize(vmin=-vabs, vmax=vabs)
            else:
                vmin = np.percentile(c_data, _PCT_FACTOR - percentile_clip)
                vmax = np.percentile(c_data, percentile_clip)
                norm = Normalize(vmin=vmin, vmax=vmax)

            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, label=label)
            lc.set_array(c_data)
            ax.add_collection(lc)
            ax.autoscale()
            fig.colorbar(lc, ax=ax, label=_clabel)
        else:
            ax.plot(lon, lat, color=color, linewidth=linewidth, label=label)

        if basemap and not fig_provided:
            try:
                cx.add_basemap(
                    ax,
                    crs="EPSG:4326",
                    source=cx.providers.OpenStreetMap.Mapnik,
                    zoom=zoom,
                )
            except Exception as exc:
                logger.warning("Basemap could not be loaded: %s", exc)

        if not fig_provided:
            _title = title or (
                f"GPS Track - {track.n_valid:,} valid / {track.n_samples:,} samples "
                f"({track.fs:.0f} Hz)"
            )
            ax.set_title(_title)
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")

        mean_lat = np.mean(lat)
        ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))

        plt.tight_layout()
        return fig

    def summary(self, track: GPSTrack) -> dict[str, float]:
        """Return summary statistics of the processed track.

        Parameters
        ----------
        track : GPSTrack
            Processed track.

        Returns
        -------
        dict
            Keys: fs_hz, dt_ms, n_samples, n_valid, pct_removed,
            v_min_kmh, v_median_kmh, v_max_kmh.

        """
        v = track.speed_kmh
        return {
            "fs_hz": track.fs,
            "dt_ms": track.dt * _S_TO_MS,
            "n_samples": track.n_samples,
            "n_valid": track.n_valid,
            "pct_removed": _PCT_FACTOR * (1 - track.n_valid / track.n_samples),
            "v_min_kmh": float(np.min(v)),
            "v_median_kmh": float(np.median(v)),
            "v_max_kmh": float(np.max(v)),
        }

    def sample_adaptive(
        self,
        track: GPSTrack,
        kappa_scale: float = 50.0,
        base_spacing_m: float = 10.0,
        min_spacing_m: float = 1.0,
        v_min_ms: float = 5.0,
        smooth_kappa_m: float = 30.0,
        kappa_clip_percentile: float = 98.0,
        kappa: NDArray[np.float64] | None = None,
    ) -> dict[str, np.ndarray | int]:
        """Sample Kalman trace at curvature-dependent resolution.

        Denser sampling in curves, sparser on straights.  The local step
        size is:  ds(s) = max(base_spacing / (1 + kappa_scale * |κ(s)|),
        min_spacing)

        Parameters
        ----------
        track : GPSTrack
            Kalman-filtered track (from GPSProcessor with
            smoothing='kalman').
        kappa_scale : float
            Scaling factor controlling curvature sensitivity.  Higher
            values produce more points in curves.  Typical: 20-100.
        base_spacing_m : float
            Spacing on perfectly straight segments [m].
        min_spacing_m : float
            Minimum spacing cap (prevents over-sampling in
            hairpins) [m].
        v_min_ms : float
            Minimum speed for valid heading [m/s].
        smooth_kappa_m : float
            Savgol smoothing window for curvature [m].  Larger values
            produce smoother spacing variation (less noise-sensitive).
        kappa_clip_percentile : float
            Clip |κ| above this percentile (of moving samples) to
            suppress noise spikes from heading jitter.  Set to 100.0
            to disable clipping.
        kappa : NDArray or None
            Pre-computed curvature array aligned with the track
            samples (same length as track.longitude).  When provided,
            the internal heading/curvature computation is skipped and
            this array is used directly.  Useful for passing the
            smoothed κ from a clothoid fit or the trasse_cut pipeline.

        Returns
        -------
        dict
            Keys:
            - 'longitude'  : NDArray - sampled longitudes [°]
            - 'latitude'   : NDArray - sampled latitudes [°]
            - 'time_s'     : NDArray - time at each sample [s]
            - 'arc_m'      : NDArray - arc-length position [m]
            - 'kappa'      : NDArray - curvature at sample [1/m]
            - 'speed_ms'   : NDArray - speed at each sample [m/s]
            - 'spacing_m'  : NDArray - local spacing used [m]
            - 'n_points'   : int    - total sampled points

        """
        n = track.n_samples
        fs = track.fs

        # ENU coordinates
        lon0, lat0 = track.longitude[0], track.latitude[0]
        cos_lat0 = np.cos(np.radians(lat0))
        x_m = (track.longitude - lon0) * EARTH_METERS_PER_DEGREE * cos_lat0
        y_m = (track.latitude - lat0) * EARTH_METERS_PER_DEGREE

        # Arc length
        ds_step = np.sqrt(np.diff(x_m) ** 2 + np.diff(y_m) ** 2)
        arc = np.concatenate([[0.0], np.cumsum(ds_step)])

        # Time vector
        time_s = np.arange(n) / fs

        # Curvature source
        moving = track.speed_ms > v_min_ms

        if kappa is not None:
            # Use externally provided curvature (e.g. from clothoid fit)
            if len(kappa) != n:
                msg = f"kappa length {len(kappa)} != track length {n}"
                raise ValueError(msg)
            kappa_arr = np.asarray(kappa, dtype=np.float64)
        else:
            # Derive curvature from positions
            dx = np.gradient(x_m)
            dy = np.gradient(y_m)
            theta = np.unwrap(np.arctan2(dy, dx))

            med_ds = float(np.nanmedian(ds_step[moving[:-1]]))
            win = int(smooth_kappa_m / med_ds) | 1
            win = max(win, _SAVGOL_MIN_WINDOW)

            # Interpolate heading through standstill, then smooth
            theta_interp = theta.copy()
            if moving.sum() > win:
                theta_interp[~moving] = np.interp(
                    np.where(~moving)[0],
                    np.where(moving)[0],
                    theta[moving],
                )
            theta_sm = savgol_filter(theta_interp, win, polyorder=_SAVGOL_POLYORDER)

            ds_safe = np.where(ds_step > _DS_EPS_M, ds_step, _DS_EPS_M)
            kappa_raw = np.concatenate(
                [[0.0], np.diff(theta_sm) / ds_safe],
            )
            kappa_raw[~moving] = 0.0

            # Second Savgol pass on κ to remove residual noise
            kappa_arr = savgol_filter(kappa_raw, win, polyorder=_SAVGOL_POLYORDER)
            kappa_arr[~moving] = 0.0

            # Clip extreme curvature values (noise spikes)
            if kappa_clip_percentile < _PCT_FACTOR:
                kappa_moving = np.abs(kappa_arr[moving])
                clip_val = float(np.percentile(
                    kappa_moving, kappa_clip_percentile,
                ))
                kappa_arr = np.clip(kappa_arr, -clip_val, clip_val)

        # Walk along arc with variable step
        sample_arcs: list[float] = [0.0]
        s = 0.0
        arc_max = float(arc[moving][-1]) if moving.any() else float(arc[-1])

        while s < arc_max:
            idx = int(np.searchsorted(arc, s).clip(0, n - 1))
            kappa_local = abs(kappa_arr[idx])
            step = base_spacing_m / (1.0 + kappa_scale * kappa_local)
            step = max(step, min_spacing_m)
            s += step
            if s <= arc_max:
                sample_arcs.append(s)

        sample_arcs_arr = np.asarray(sample_arcs)

        # Interpolate track quantities at sampled arc positions
        lon_sampled = np.interp(sample_arcs_arr, arc, track.longitude)
        lat_sampled = np.interp(sample_arcs_arr, arc, track.latitude)
        time_sampled = np.interp(sample_arcs_arr, arc, time_s)
        speed_sampled = np.interp(sample_arcs_arr, arc, track.speed_ms)
        kappa_sampled = np.interp(sample_arcs_arr, arc, kappa_arr)

        spacing_used = np.diff(sample_arcs_arr, prepend=0.0)
        spacing_used[0] = (
            spacing_used[1] if len(spacing_used) > 1 else base_spacing_m
        )

        return {
            "longitude": lon_sampled,
            "latitude": lat_sampled,
            "time_s": time_sampled,
            "arc_m": sample_arcs_arr,
            "kappa": kappa_sampled,
            "speed_ms": speed_sampled,
            "spacing_m": spacing_used,
            "n_points": len(sample_arcs_arr),
        }

    def sample_angular_step(
        self,
        track: GPSTrack,
        delta_theta_deg: float = 2.0,
        base_spacing_m: float = 50.0,
        min_spacing_m: float = 0.5,
        v_min_ms: float = 5.0,
        smooth_kappa_m: float = 30.0,
        kappa: NDArray[np.float64] | None = None,
    ) -> dict[str, np.ndarray | int]:
        """Sample Kalman trace with maximum angular step constraint.

        Between any two consecutive samples, the heading changes by at
        most ``delta_theta_deg`` degrees.  This guarantees geometric
        fidelity in curves: tighter curves automatically receive more
        points, while straights stay sparse.

        The local step size is:
            ds(s) = min(base_spacing, max(Δθ / |κ(s)|, min_spacing))

        Parameters
        ----------
        track : GPSTrack
            Kalman-filtered track.
        delta_theta_deg : float, default 2.0
            Maximum heading change per interval [deg].
        base_spacing_m : float, default 50.0
            Maximum spacing cap for near-zero curvature [m].
        min_spacing_m : float, default 0.5
            Minimum spacing floor [m].
        v_min_ms : float, default 5.0
            Minimum speed for valid heading [m/s].
        smooth_kappa_m : float, default 30.0
            Savgol smoothing window for curvature [m].
        kappa : NDArray or None
            Pre-computed curvature aligned with the track.

        Returns
        -------
        dict
            Keys:
            - 'longitude'  : NDArray - sampled longitudes [°]
            - 'latitude'   : NDArray - sampled latitudes [°]
            - 'time_s'     : NDArray - time at each sample [s]
            - 'arc_m'      : NDArray - arc-length position [m]
            - 'kappa'      : NDArray - curvature at sample [1/m]
            - 'speed_ms'   : NDArray - speed at each sample [m/s]
            - 'spacing_m'  : NDArray - local spacing used [m]
            - 'n_points'   : int    - total sampled points

        """
        n = track.n_samples
        fs = track.fs
        delta_theta = np.radians(delta_theta_deg)

        # ENU coordinates
        lon0, lat0 = track.longitude[0], track.latitude[0]
        cos_lat0 = np.cos(np.radians(lat0))
        x_m = (track.longitude - lon0) * EARTH_METERS_PER_DEGREE * cos_lat0
        y_m = (track.latitude - lat0) * EARTH_METERS_PER_DEGREE

        # Arc length
        ds_step = np.sqrt(np.diff(x_m) ** 2 + np.diff(y_m) ** 2)
        arc = np.concatenate([[0.0], np.cumsum(ds_step)])

        # Time vector
        time_s = np.arange(n) / fs

        # Curvature source
        moving = track.speed_ms > v_min_ms

        if kappa is not None:
            if len(kappa) != n:
                msg = f"kappa length {len(kappa)} != track length {n}"
                raise ValueError(msg)
            kappa_arr = np.asarray(kappa, dtype=np.float64)
        else:
            # Derive curvature from positions
            dx = np.gradient(x_m)
            dy = np.gradient(y_m)
            theta = np.unwrap(np.arctan2(dy, dx))

            med_ds = float(np.nanmedian(ds_step[moving[:-1]]))
            win = int(smooth_kappa_m / med_ds) | 1
            win = max(win, _SAVGOL_MIN_WINDOW)

            theta_interp = theta.copy()
            if moving.sum() > win:
                theta_interp[~moving] = np.interp(
                    np.where(~moving)[0],
                    np.where(moving)[0],
                    theta[moving],
                )
            theta_sm = savgol_filter(theta_interp, win, polyorder=_SAVGOL_POLYORDER)

            ds_safe = np.where(ds_step > _DS_EPS_M, ds_step, _DS_EPS_M)
            kappa_raw = np.concatenate(
                [[0.0], np.diff(theta_sm) / ds_safe],
            )
            kappa_raw[~moving] = 0.0
            kappa_arr = savgol_filter(kappa_raw, win, polyorder=_SAVGOL_POLYORDER)
            kappa_arr[~moving] = 0.0

        # Walk along arc with CUMULATIVE angular step constraint.
        # Instead of predicting step from instantaneous κ (noise-sensitive),
        # integrate |κ| and place a sample when the accumulated heading
        # change reaches delta_theta.  This averages out curvature noise
        # and gives a true geometric guarantee.
        arc_max = float(arc[moving][-1]) if moving.any() else float(arc[-1])
        sample_arcs: list[float] = [0.0]

        # Fine integration step for accumulating Δθ
        ds_walk = min_spacing_m
        s = 0.0
        theta_accum = 0.0
        dist_accum = 0.0

        while s < arc_max:
            idx = int(np.searchsorted(arc, s).clip(0, n - 1))
            kappa_local = abs(kappa_arr[idx])

            # Advance by a fine step
            s += ds_walk
            theta_accum += kappa_local * ds_walk
            dist_accum += ds_walk

            # Place sample when heading budget exhausted OR max distance
            if theta_accum >= delta_theta or dist_accum >= base_spacing_m:
                if s <= arc_max:
                    sample_arcs.append(s)
                theta_accum = 0.0
                dist_accum = 0.0

        sample_arcs_arr = np.asarray(sample_arcs)

        # Interpolate track quantities at sampled positions
        lon_sampled = np.interp(sample_arcs_arr, arc, track.longitude)
        lat_sampled = np.interp(sample_arcs_arr, arc, track.latitude)
        time_sampled = np.interp(sample_arcs_arr, arc, time_s)
        speed_sampled = np.interp(sample_arcs_arr, arc, track.speed_ms)
        kappa_sampled = np.interp(sample_arcs_arr, arc, kappa_arr)

        spacing_used = np.diff(sample_arcs_arr, prepend=0.0)
        spacing_used[0] = (
            spacing_used[1] if len(spacing_used) > 1 else base_spacing_m
        )

        return {
            "longitude": lon_sampled,
            "latitude": lat_sampled,
            "time_s": time_sampled,
            "arc_m": sample_arcs_arr,
            "kappa": kappa_sampled,
            "speed_ms": speed_sampled,
            "spacing_m": spacing_used,
            "n_points": len(sample_arcs_arr),
        }

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _speed_smoothing_window(self, fs: float) -> int:
        """Compute Savitzky-Golay window size from duration and sample rate.

        Returns an odd integer >= _SAVGOL_MIN_WINDOW, or 0 if smoothing
        is disabled (speed_smoothing_s == 0).
        """
        if self.speed_smoothing_s <= 0.0:
            return 0
        win = int(self.speed_smoothing_s * fs)
        win = win | 1  # ensure odd
        return max(win, _SAVGOL_MIN_WINDOW)

    @staticmethod
    def _detect_updates(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
    ) -> NDArray[np.intp]:
        """Find indices where GPS position actually changes.

        In sample-and-hold GPS streams, the same position is repeated at
        the base sample rate (f_s) until the next GPS fix arrives (5-20 Hz).
        This method detects the transition points.
        """
        changed = (np.diff(lon) != 0) | (np.diff(lat) != 0)
        return np.concatenate(
            [[0], np.where(changed)[0] + 1, [len(lon) - 1]],
        )

    @staticmethod
    def _interpolate(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        update_idx: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Linear interpolation to remove sample-and-hold."""
        sample_idx = np.arange(len(lon))
        lon_interp = np.interp(sample_idx, update_idx, lon[update_idx])
        lat_interp = np.interp(sample_idx, update_idx, lat[update_idx])
        return lon_interp, lat_interp

    def _calibrate_dt(
        self,
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        update_idx: NDArray[np.intp],
        v_ref: NDArray[np.float64],
    ) -> float:
        """Calibrate per-sample dt from reference speed and GPS distances.

        Uses the relationship: dt = distance / (speed × n_samples_between).
        Only GPS segments where the reference speed exceeds
        ``min_speed_calibration`` are used (low-speed segments have
        unreliable GPS distances).
        """
        gaps = np.diff(update_idx[:-1])
        v_avg = (v_ref[update_idx[:-2]] + v_ref[update_idx[1:-1]]) / 2.0

        dlat = np.diff(lat[update_idx[:-1]]) * EARTH_METERS_PER_DEGREE
        dlon = (
            np.diff(lon[update_idx[:-1]])
            * EARTH_METERS_PER_DEGREE
            * np.cos(np.radians(lat[update_idx[:-2]]))
        )
        dist = np.sqrt(dlat**2 + dlon**2)

        valid = v_avg > self.min_speed_calibration
        dt = float(np.nanmedian(dist[valid] / (v_avg[valid] * gaps[valid])))
        return dt  # noqa: RET504

    def _lowpass(
        self,
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        f_cutoff: float,
        fs: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply zero-phase Butterworth low-pass to position signals."""
        wn = f_cutoff / (fs / 2.0)
        b, a = butter(N=self.filter_order, Wn=wn, btype="low")
        lon_filt = filtfilt(b, a, lon)
        lat_filt = filtfilt(b, a, lat)
        return lon_filt, lat_filt

    @staticmethod
    def _compute_speed(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        dt: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Derive speed from consecutive position differences."""
        dlat = np.diff(lat) * EARTH_METERS_PER_DEGREE
        dlon = (
            np.diff(lon)
            * EARTH_METERS_PER_DEGREE
            * np.cos(np.radians(lat[:-1]))
        )
        return np.sqrt(dlat**2 + dlon**2) / dt

    def _build_mask(
        self,
        speed_ms: NDArray[np.float64],
        n_samples: int,
    ) -> NDArray[np.bool_]:
        """Create boolean mask flagging outliers above v_max."""
        v_threshold = self.v_max_kmh / _MS_TO_KMH
        if len(speed_ms) == n_samples - 1:
            speed_ok = speed_ms < v_threshold
            return np.concatenate([[True], speed_ok])
        # Kalman returns speed with same length as position
        return speed_ms < v_threshold

    @staticmethod
    def _build_abs_noise_scale(
        abs_flag: NDArray[np.float64],
        dt: float,
        n: int,
        inflation: float,
        pre_margin_s: float,
        post_margin_s: float,
        ramp_s: float,
    ) -> NDArray[np.float64]:
        """Build a per-sample measurement-noise scale array for ABS events.

        Returns an array of multipliers (1.0 outside ABS, up to
        ``inflation`` inside) that inflate a Kalman measurement-noise
        variance during ABS intervention.  A fixed temporal margin
        before and after each ABS run covers the flag-vs-signal lag,
        and a raised-cosine ramp on both edges avoids a step change in
        trust that would otherwise inject a kink into the estimate.

        Parameters
        ----------
        abs_flag : NDArray[np.float64]
            Binary ABS-active flag (values above the decision threshold
            are treated as active), shape (n,).
        dt : float
            Calibrated sampling period [s].
        n : int
            Number of samples (length of the position arrays).
        inflation : float
            Peak noise multiplier applied in the fully-active region.
        pre_margin_s, post_margin_s : float
            Temporal margins [s] added before / after each ABS run.
        ramp_s : float
            Raised-cosine ramp length [s] on each edge of the window.

        Returns
        -------
        scale : NDArray[np.float64]
            Per-sample noise scale in [1.0, inflation], shape (n,).

        """
        scale = np.ones(n, dtype=np.float64)
        abs_on = abs_flag > _ABS_FLAG_THRESHOLD
        if not abs_on.any() or dt <= 0.0:
            return scale

        pre = int(pre_margin_s / dt)
        post = int(post_margin_s / dt)
        ramp = max(int(ramp_s / dt), 1)

        # Dilate the ABS mask by the temporal margins.
        window = abs_on.copy()
        starts = np.where(np.diff(abs_on.astype(np.int8)) == 1)[0] + 1
        ends = np.where(np.diff(abs_on.astype(np.int8)) == -1)[0] + 1
        if abs_on[0]:
            starts = np.concatenate([[0], starts])
        if abs_on[-1]:
            ends = np.concatenate([ends, [n]])
        for s0, e0 in zip(starts, ends, strict=False):
            window[max(0, s0 - pre):min(n, e0 + post)] = True

        # Raised-cosine ramp on each contiguous window edge.
        target = 1.0 + (inflation - 1.0) * window.astype(np.float64)
        w_idx = np.where(window)[0]
        if len(w_idx) == 0:
            return scale
        breaks = np.where(np.diff(w_idx) > 1)[0]
        run_starts = np.concatenate([[w_idx[0]], w_idx[breaks + 1]])
        run_ends = np.concatenate([w_idx[breaks], [w_idx[-1]]])
        for rs, re in zip(run_starts, run_ends, strict=False):
            ramp_len = min(ramp, (re - rs) // 2)
            if ramp_len <= 0:
                continue
            up = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
            target[rs:rs + ramp_len] = 1.0 + (inflation - 1.0) * up
            dn = 0.5 * (1.0 + np.cos(np.pi * np.arange(ramp_len) / ramp_len))
            target[re - ramp_len + 1:re + 1] = 1.0 + (inflation - 1.0) * dn

        return target

    @staticmethod
    def _clamp_speed_during_abs(
        v_ms: NDArray[np.float64],
        abs_active: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Interpolate wheel speed across ABS events.

        During ABS intervention, wheel speed oscillates at ~15 Hz
        (lock/unlock cycles) around a value below the true vehicle
        speed. Using the oscillating signal in the bicycle model
        produces yaw rate spikes. Simple clamping to the last pre-ABS
        value overestimates speed (vehicle is decelerating).

        This method linearly interpolates between the last pre-ABS
        and first post-ABS speed for each contiguous ABS epoch,
        capturing the deceleration trend without oscillation.

        Parameters
        ----------
        v_ms : NDArray[np.float64]
            Wheel speed array [m/s].
        abs_active : NDArray[np.float64]
            ABS flag array (values near 1.0 = active).

        Returns
        -------
        NDArray[np.float64]
            Speed array with ABS epochs linearly bridged.

        """
        abs_on = abs_active > _ABS_FLAG_THRESHOLD
        if not abs_on.any():
            return v_ms

        v_out = v_ms.copy()
        # Indices where ABS is NOT active (valid anchor points)
        valid_idx = np.where(~abs_on)[0]

        if len(valid_idx) == 0:
            # Entire signal is ABS - nothing to interpolate from
            return v_out

        # Linear interpolation: use non-ABS samples as anchors,
        # interpolate through ABS epochs.
        abs_idx = np.where(abs_on)[0]
        v_out[abs_idx] = np.interp(abs_idx, valid_idx, v_ms[valid_idx])
        return v_out

    @staticmethod
    def _repair_gps_freezes(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        update_idx: NDArray[np.intp],
        speed: NDArray[np.float64],
        min_speed: float = 5.0,
        min_run_length: int = 3,
        min_expected_change_m: float = 5.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Detect single-axis GPS freezes and build per-axis noise array.

        Instead of removing freeze epochs or interpolating coordinates,
        this method returns a per-sample noise array that inflates the
        measurement variance for the frozen axis to effectively infinity.
        The Kalman filter then performs a true partial 1D measurement
        update: correcting from the valid axis while ignoring the
        frozen axis (Kalman gain → 0 for that axis).

        Detection uses an innovation-based criterion: a coordinate is
        frozen only if (a) it didn't change for ``min_run_length``
        consecutive epochs, AND (b) the heading predicts ≥
        ``min_expected_change_m`` of movement in that axis.

        Parameters
        ----------
        lon, lat : NDArray
            Interpolated coordinates (after S&H removal).
        update_idx : NDArray
            Indices of detected GPS update epochs.
        speed : NDArray
            Reference speed [m/s] (wheel-based).
        min_speed : float, default 5.0
            Minimum speed to consider a freeze.
        min_run_length : int, default 3
            Minimum consecutive frozen epochs.
        min_expected_change_m : float, default 5.0
            Minimum expected displacement in frozen axis.

        Returns
        -------
        lon, lat : NDArray
            Coordinate arrays with frozen-axis values replaced by
            linear interpolation. Valid-axis values unchanged.

        """
        lon_out = lon.copy()
        lat_out = lat.copy()

        if len(update_idx) < _FREEZE_MIN_EPOCHS:
            return lon_out, lat_out

        n_epochs = len(update_idx)

        # Heading between epochs (from position diffs)
        cos_lat = np.cos(np.radians(np.mean(lat[update_idx])))
        dx_m = np.diff(lon[update_idx]) * EARTH_METERS_PER_DEGREE * cos_lat
        dy_m = np.diff(lat[update_idx]) * EARTH_METERS_PER_DEGREE
        heading = np.arctan2(dy_m, dx_m)

        # Classify transitions: 0=normal, 1=lon frozen, 2=lat frozen
        epoch_type = np.zeros(n_epochs, dtype=np.int8)
        for i in range(1, n_epochs):
            lon_changed = lon[update_idx[i]] != lon[update_idx[i - 1]]
            lat_changed = lat[update_idx[i]] != lat[update_idx[i - 1]]

            if lon_changed and lat_changed:
                continue
            if lat_changed and not lon_changed:
                avg_speed = np.mean(speed[update_idx[i-1]:update_idx[i] + 1])
                if avg_speed > min_speed:
                    epoch_type[i] = 1
            elif lon_changed and not lat_changed:
                avg_speed = np.mean(speed[update_idx[i-1]:update_idx[i] + 1])
                if avg_speed > min_speed:
                    epoch_type[i] = 2

        # Find runs and apply innovation check + interpolation repair
        n_repaired = 0
        i = 0
        while i < n_epochs:
            if epoch_type[i] == 0:
                i += 1
                continue

            freeze_type = epoch_type[i]
            run_start = i
            while i < n_epochs and epoch_type[i] == freeze_type:
                i += 1
            run_end = i
            run_len = run_end - run_start

            if run_len < min_run_length:
                continue

            # Innovation check: expected displacement in frozen axis
            h_start = max(run_start - 2, 0)
            h_end = min(run_end, n_epochs - 2)
            if h_end > h_start and run_start > 0:
                total_dist = np.sum(np.sqrt(
                    dx_m[run_start-1:run_end-1]**2 +
                    dy_m[run_start-1:run_end-1]**2,
                ))
                avg_heading = np.mean(heading[h_start:h_end])
                if freeze_type == 1:
                    expected_change = abs(total_dist * np.cos(avg_heading))
                else:
                    expected_change = abs(total_dist * np.sin(avg_heading))
            else:
                expected_change = 0.0

            if expected_change < min_expected_change_m:
                continue

            # === Repair: interpolate frozen coordinate ===
            # Replace frozen values with linear bridge between anchors.
            # Keep normal noise (full gain) - the interpolated value is
            # smooth and provides continuous position anchoring.
            idx_before = update_idx[run_start - 1] if run_start > 0 else update_idx[0]
            idx_after = update_idx[min(run_end, n_epochs - 1)]
            span = np.arange(idx_before, idx_after + 1)

            if freeze_type == 1:  # lon (x) frozen
                lon_out[span] = np.interp(
                    span, [idx_before, idx_after],
                    [lon[idx_before], lon[idx_after]])
            else:  # lat (y) frozen
                lat_out[span] = np.interp(
                    span, [idx_before, idx_after],
                    [lat[idx_before], lat[idx_after]])

            n_repaired += run_len

        if n_repaired > 0:
            warnings.warn(
                f"GPS freeze repair: {n_repaired} epochs interpolated "
                f"(runs ≥{min_run_length}, expected ≥{min_expected_change_m}m). "
                f"Frozen axis linearly bridged, valid axis preserved.",
                stacklevel=2,
            )

        return lon_out, lat_out


    def _apply_kalman(
        self,
        lon_interp: NDArray[np.float64],
        lat_interp: NDArray[np.float64],
        v_reference_ms: NDArray[np.float64],
        yaw_rate_rad: NDArray[np.float64] | None,
        update_idx: NDArray[np.intp],
        dt: float,
        speed_noise_scale: NDArray[np.float64] | None = None,
        yaw_rate_noise_scale: NDArray[np.float64] | None = None,
        process_noise_scale: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Apply Kalman EKF + RTS smoother for position conditioning.

        Fuses sparse GPS (5-20 Hz) with dense wheel speed and yaw rate
        (both at f_s) using a CTRV motion model. The RTS backward pass
        produces minimum-variance smoothed estimates.

        Parameters
        ----------
        lon_interp : NDArray
            Interpolated longitude (S&H removed).
        lat_interp : NDArray
            Interpolated latitude (S&H removed).
        v_reference_ms : NDArray
            Wheel speed in m/s (dense, at f_s).
        yaw_rate_rad : NDArray or None
            Signed yaw rate in rad/s (dense, at f_s).
        update_idx : NDArray
            Indices of actual GPS updates (sparse, 5-20 Hz).
        dt : float
            Calibrated sampling period [s].
        yaw_rate_noise_scale : NDArray or None
            Optional per-sample yaw rate noise scaling factor.  When
            provided, the yaw rate measurement noise is multiplied by this
            factor (useful for inflating noise during ABS events).
        speed_noise_scale : NDArray or None
            Optional per-sample speed noise scaling factor.  When provided,
            the speed measurement noise is multiplied by this factor
            (useful for inflating noise during ABS events).
        process_noise_scale : NDArray or None
            Optional per-sample process noise scaling factor.  When provided,
            the process noise covariance is multiplied by this factor
            (useful for inflating noise during ABS events).

        Returns
        -------
        lon_filt : NDArray
            Kalman-filtered longitude.
        lat_filt : NDArray
            Kalman-filtered latitude.
        speed_ms : NDArray
            Kalman-estimated speed (same length as position).

        """
        from .kalman import KalmanConfig

        if self.smoothing == "kalman_jax":
            from .jax_kalman import JAXKalmanFilter as _FilterCls
        else:
            from .kalman import GPSKalmanFilter as _FilterCls

        if yaw_rate_rad is None:
            msg = "smoothing='kalman' requires the `yaw_rate_rad` argument."
            raise ValueError(msg)

        # Repair single-axis GPS freezes (interpolate frozen coordinate)
        if self.freeze_repair:
            lon_interp, lat_interp = self._repair_gps_freezes(
                lon_interp, lat_interp, update_idx, v_reference_ms,
            )

        # Build GPS update mask from ALL update indices (no epochs removed)
        n = len(lon_interp)
        gps_update_mask = np.zeros(n, dtype=bool)
        gps_update_mask[update_idx] = True

        # Configure Kalman filter
        if self.kalman_config is not None:
            cfg = self.kalman_config
            # Override dt with calibrated value; carry over ALL fields
            cfg = KalmanConfig(
                dt=dt,
                sigma_pos_gps=cfg.sigma_pos_gps,
                sigma_v_wheel=cfg.sigma_v_wheel,
                sigma_yaw_rate=cfg.sigma_yaw_rate,
                sigma_a=cfg.sigma_a,
                sigma_yaw_acc=cfg.sigma_yaw_acc,
                sigma_pos_drift=cfg.sigma_pos_drift,
                initial_speed=cfg.initial_speed,
                initial_heading=cfg.initial_heading,
                use_rts_smoother=cfg.use_rts_smoother,
                steering_ratio=cfg.steering_ratio,
                wheelbase_m=cfg.wheelbase_m,
                gps_low_speed_gain=cfg.gps_low_speed_gain,
                gps_low_speed_v_scale=cfg.gps_low_speed_v_scale,
            )
        else:
            cfg = KalmanConfig(dt=dt)

        kf = _FilterCls(cfg)
        result = kf.run(
            longitude=lon_interp,
            latitude=lat_interp,
            v_reference_ms=v_reference_ms,
            yaw_rate_rad=yaw_rate_rad,
            gps_update_mask=gps_update_mask,
            speed_noise_scale=speed_noise_scale,
            yaw_rate_noise_scale=yaw_rate_noise_scale,
            process_noise_scale=process_noise_scale,
        )

        return result.longitude, result.latitude, result.speed_ms
