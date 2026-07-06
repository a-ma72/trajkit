"""Extended Kalman Filter for GPS/IMU fusion.

Provides an EKF with a constant-turn-rate-and-velocity (CTRV) motion model
fused with GPS position and wheel-speed measurements. Designed for offline
batch processing with an optional Rauch-Tung-Striebel (RTS) backward smoother.

The implementation uses pure functions for the core math to facilitate
a future port to JAX (jax.numpy drop-in replacement).

Typical usage::

    from .kalman import GPSKalmanFilter, KalmanConfig

    kf = GPSKalmanFilter(KalmanConfig(dt=0.002))  # 500 Hz
    result = kf.run(lon, lat, v_ref, yaw_rate, gps_update_mask)
"""

# ruff: noqa: D107, N803, N806, PLR0913, PLR0915

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from .constants import meters_per_degree_lat, meters_per_degree_lon

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KalmanConfig:
    """Tuning parameters for the GPS Kalman filter.

    Parameters
    ----------
    dt : float
        Nominal sampling period in seconds.
    sigma_pos_gps : float, default 2.0
        GPS position measurement noise std [m].
    sigma_v_wheel : float, default 0.1
        Wheel speed measurement noise std [m/s].
    sigma_yaw_rate : float, default 0.2
        Yaw rate measurement noise std [rad/s].
    sigma_a : float, default 2.0
        Process noise for longitudinal acceleration [m/s^2].
    sigma_yaw_acc : float, default 0.5
        Process noise for yaw rate change [rad/s^2].
    sigma_pos_drift : float, default 5.0
        Position drift noise std [m/√s]. Prevents P[:2,:2] from
        collapsing at high sample rates where dt⁴σ_a² ≈ 0.
        Accumulates as σ²·dt per step; over N = f_s/f_GPS steps
        provides sufficient covariance for GPS correction.
    initial_speed : float, default 0.0
        Initial speed assumption [m/s].
    initial_heading : float, default 0.0
        Initial heading assumption [rad]. If None, estimated from
        first two GPS updates.
    use_rts_smoother : bool, default True
        Whether to apply backward RTS smoother after forward pass.
    steering_ratio : float, default 15.5
        Steering wheel to road wheel ratio (i_s). Used to derive
        yaw rate from steering angle via bicycle model:
        omega = v * tan(delta_steering / i_s) / wheelbase_m.
    wheelbase_m : float, default 2.68
        Vehicle wheelbase [m]. Used with steering_ratio to derive
        yaw rate from steering angle. Default is typical compact car.
    gps_low_speed_gain : float, default 10.0
        Multiplicative gain for GPS noise at standstill. At v=0,
        effective sigma_pos_gps = sigma_pos_gps * (1 + gain).
        Prevents GPS multipath jitter from creating zig-zag at
        low speed. Set to 0.0 to disable dynamic scaling.
    gps_low_speed_v_scale : float, default 2.0
        Speed scale [m/s] for the exponential decay. The gain
        halves every v_scale * ln(2) ≈ 1.4 m/s.

    """

    dt: float
    sigma_pos_gps: float = 2.0
    sigma_v_wheel: float = 0.1
    sigma_yaw_rate: float = 0.2
    sigma_a: float = 2.0
    sigma_yaw_acc: float = 0.5
    sigma_pos_drift: float = 5.0
    initial_speed: float = 0.0
    initial_heading: float = 0.0
    use_rts_smoother: bool = True
    steering_ratio: float = 15.5
    wheelbase_m: float = 2.68
    gps_low_speed_gain: float = 10.0
    gps_low_speed_v_scale: float = 2.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class KalmanResult:
    """Output of the Kalman filter.

    Attributes
    ----------
    x_m : NDArray[np.float64]
        Estimated x position in local frame [m].
    y_m : NDArray[np.float64]
        Estimated y position in local frame [m].
    speed_ms : NDArray[np.float64]
        Estimated speed [m/s].
    heading_rad : NDArray[np.float64]
        Estimated heading [rad].
    yaw_rate_rad : NDArray[np.float64]
        Estimated yaw rate [rad/s].
    longitude : NDArray[np.float64]
        Estimated longitude [deg] (converted back from local frame).
    latitude : NDArray[np.float64]
        Estimated latitude [deg] (converted back from local frame).
    P_diag : NDArray[np.float64]
        Diagonal of the state covariance at each step (n, 5).

    """

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    speed_ms: NDArray[np.float64]
    heading_rad: NDArray[np.float64]
    yaw_rate_rad: NDArray[np.float64]
    longitude: NDArray[np.float64]
    latitude: NDArray[np.float64]
    P_diag: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Pure math functions (Numba-accelerated)
# ---------------------------------------------------------------------------

# State vector: [x, y, v, theta, omega]
#   x, y    : position in local ENU frame [m]
#   v       : scalar speed [m/s]
#   theta   : heading angle [rad], 0 = East, pi/2 = North
#   omega   : yaw rate [rad/s]

_STATE_DIM: int = 5
_IDX_X, _IDX_Y, _IDX_V, _IDX_THETA, _IDX_OMEGA = range(_STATE_DIM)

_OMEGA_EPS: float = 1e-6
# Minimum absolute yaw rate [rad/s] for the curved CTRV branch.
# Below this threshold the turn radius v/omega overflows; the
# straight-line approximation is used instead.

_HEADING_INIT_SIGMA_RAD: float = 0.5
# Initial heading uncertainty [rad] (≈ 28.6°, roughly ±30°).
# Used as the std for the P0 heading diagonal entry.


@njit(cache=True)
def _ekf_forward_rts(
    gps_x: np.ndarray,
    gps_y: np.ndarray,
    v_ref: np.ndarray,
    yaw_rate: np.ndarray,
    gps_mask: np.ndarray,
    r_omega_arr: np.ndarray,
    r_v_arr: np.ndarray,
    q_scale_arr: np.ndarray,
    dt: float,
    sigma_pos_gps: float,
    sigma_v_wheel: float,
    sigma_yaw_rate: float,
    sigma_a: float,
    sigma_yaw_acc: float,
    sigma_pos_drift: float,
    gps_low_speed_gain: float,
    gps_low_speed_v_scale: float,
    state0: np.ndarray,
    P0: np.ndarray,
    *,
    use_rts: bool,
) -> tuple:
    """Numba-accelerated EKF forward pass + RTS smoother.

    Returns (states_out, P_diag_out) with shape (n, 5) each.
    """
    n = len(gps_x)
    dim = _STATE_DIM

    # Pre-compute process noise Q
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    sa2 = sigma_a ** 2
    sya2 = sigma_yaw_acc ** 2

    Q = np.zeros((dim, dim))
    Q[0, 0] = dt4 / 4.0 * sa2 + sigma_pos_drift**2 * dt
    Q[0, 2] = dt3 / 2.0 * sa2
    Q[1, 1] = dt4 / 4.0 * sa2 + sigma_pos_drift**2 * dt
    Q[1, 2] = dt3 / 2.0 * sa2
    Q[2, 0] = dt3 / 2.0 * sa2
    Q[2, 1] = dt3 / 2.0 * sa2
    Q[2, 2] = dt2 * sa2
    Q[3, 3] = dt2 * sya2
    Q[3, 4] = dt * sya2
    Q[4, 3] = dt * sya2
    Q[4, 4] = sya2

    # Measurement noise: r_v and r_omega are per-sample arrays
    # (inflated during ABS to reduce trust in interpolated values)

    # Storage
    states_fwd = np.zeros((n, dim))
    P_fwd = np.zeros((n, dim, dim))
    states_pred = np.zeros((n, dim))
    P_pred = np.zeros((n, dim, dim))
    F_store = np.zeros((n, dim, dim))

    state = state0.copy()
    P = P0.copy()
    states_fwd[0] = state
    P_fwd[0] = P

    I5 = np.eye(dim)

    for k in range(1, n):
        # --- Predict (CTRV) ---
        x_s, y_s, v, theta, omega = state[0], state[1], state[2], state[3], state[4]

        # State prediction
        if np.abs(omega) > _OMEGA_EPS:
            s_t = np.sin(theta)
            c_t = np.cos(theta)
            s_tw = np.sin(theta + omega * dt)
            c_tw = np.cos(theta + omega * dt)
            x_new = x_s + (v / omega) * (s_tw - s_t)
            y_new = y_s + (v / omega) * (c_t - c_tw)
            # Jacobian
            F = I5.copy()
            F[0, 2] = (s_tw - s_t) / omega
            F[0, 3] = (v / omega) * (c_tw - c_t)
            F[0, 4] = (v / omega) * c_tw * dt - (v / omega**2) * (s_tw - s_t)
            F[1, 2] = (c_t - c_tw) / omega
            F[1, 3] = (v / omega) * (s_tw - s_t)
            F[1, 4] = (v / omega) * s_tw * dt - (v / omega**2) * (c_t - c_tw)
        else:
            c_t = np.cos(theta)
            s_t = np.sin(theta)
            x_new = x_s + v * c_t * dt
            y_new = y_s + v * s_t * dt
            F = I5.copy()
            F[0, 2] = c_t * dt
            F[0, 3] = -v * s_t * dt
            F[1, 2] = s_t * dt
            F[1, 3] = v * c_t * dt

        F[3, 4] = dt
        theta_new = theta + omega * dt

        state_pred = np.array([x_new, y_new, v, theta_new, omega])
        P_p = F @ P @ F.T + Q * q_scale_arr[k]

        states_pred[k] = state_pred
        P_pred[k] = P_p
        F_store[k] = F

        state = state_pred
        P = P_p

        # --- Update: wheel speed (per-sample noise, inflated during ABS) ---
        innov_v = v_ref[k] - state[2]
        S_v = P[2, 2] + r_v_arr[k]
        K_v = P[:, 2].copy() / S_v  # .copy() for contiguous column
        state = state + K_v * innov_v
        P = P - np.outer(K_v, P[2, :].copy())

        # --- Update: yaw rate (per-sample noise, inflated during ABS) ---
        innov_w = yaw_rate[k] - state[4]
        S_w = P[4, 4] + r_omega_arr[k]
        K_w = P[:, 4].copy() / S_w
        state = state + K_w * innov_w
        P = P - np.outer(K_w, P[4, :].copy())

        # --- Update: GPS (sparse, speed-dependent noise) ---
        if gps_mask[k]:
            innov_gps = np.array([gps_x[k] - state[0], gps_y[k] - state[1]])
            # Dynamic sigma: increase at low speed to suppress multipath jitter
            v_cur = abs(state[2])
            sigma_gps_eff = sigma_pos_gps * (
                1.0 +
                gps_low_speed_gain * np.exp(-v_cur / gps_low_speed_v_scale))
            R_gps = np.array([[sigma_gps_eff**2, 0.0], [0.0, sigma_gps_eff**2]])
            # H_gps @ P @ H_gps.T + R_gps = P[:2,:2] + R_gps
            S_gps = P[:2, :2] + R_gps
            # Solve K = P @ H.T @ S^-1 -> P[:,:2] @ inv(S)
            det = S_gps[0, 0] * S_gps[1, 1] - S_gps[0, 1] * S_gps[1, 0]
            S_inv = np.array([[S_gps[1, 1], -S_gps[0, 1]],
                             [-S_gps[1, 0], S_gps[0, 0]]]) / det
            K_gps = np.ascontiguousarray(P[:, :2]) @ S_inv
            state = state + K_gps @ innov_gps
            P = P - K_gps @ np.ascontiguousarray(P[:2, :])

        states_fwd[k] = state
        P_fwd[k] = P

    # --- RTS Backward Smoother ---
    if use_rts:
        states_out = states_fwd.copy()
        P_out = P_fwd.copy()

        for k in range(n - 2, -1, -1):
            # G = P_fwd[k] @ F[k+1].T @ inv(P_pred[k+1])
            Pp = P_pred[k + 1].copy()
            Pp_inv = np.linalg.inv(Pp)
            Pk = P_fwd[k].copy()
            Ft = np.ascontiguousarray(F_store[k + 1].T)
            G = Pk @ Ft @ Pp_inv

            states_out[k] = (
                states_fwd[k]
                + G @ (states_out[k + 1] - states_pred[k + 1])
            )
            Gt = np.ascontiguousarray(G.T)
            P_out[k] = (
                Pk
                + G @ (P_out[k + 1] - P_pred[k + 1]) @ Gt
            )
    else:
        states_out = states_fwd
        P_out = P_fwd

    # Extract diagonal of P for output
    P_diag = np.zeros((n, dim))
    for k in range(n):
        for i in range(dim):
            P_diag[k, i] = P_out[k, i, i]

    return states_out, P_diag


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


def _geo_to_local(
    lon: NDArray, lat: NDArray, lon0: float, lat0: float,
) -> tuple[NDArray, NDArray]:
    """Convert WGS84 to local ENU meters.

    Uses the WGS84 ellipsoidal meters-per-degree at the local origin
    latitude ``lat0`` (see :func:`trajkit.constants.meters_per_degree_lat`)
    rather than a single global constant, which is accurate to within
    ~1% only near the equator.
    """
    x_m = (lon - lon0) * meters_per_degree_lon(lat0)
    y_m = (lat - lat0) * meters_per_degree_lat(lat0)
    return x_m, y_m


def _local_to_geo(
    x_m: NDArray, y_m: NDArray, lon0: float, lat0: float,
) -> tuple[NDArray, NDArray]:
    """Convert local ENU meters back to WGS84."""
    lon = x_m / meters_per_degree_lon(lat0) + lon0
    lat = y_m / meters_per_degree_lat(lat0) + lat0
    return lon, lat


# ---------------------------------------------------------------------------
# Kalman Filter class
# ---------------------------------------------------------------------------


class GPSKalmanFilter:
    """Extended Kalman Filter for GPS track smoothing.

    Implements a CTRV (Constant Turn Rate and Velocity) motion model
    with measurement updates from:
    - GPS position (sparse, 5-20 Hz in typical vehicle data)
    - Wheel speed (dense, full sample rate)
    - Yaw rate from ESP (dense, full sample rate)

    Parameters
    ----------
    config : KalmanConfig
        Filter tuning parameters.

    Examples
    --------
    >>> cfg = KalmanConfig(dt=0.002)  # 500 Hz sample rate
    >>> kf = GPSKalmanFilter(cfg)
    >>> result = kf.run(lon, lat, v_ref_ms, yaw_rate_rad, gps_mask)

    """

    def __init__(self, config: KalmanConfig) -> None:
        self.config = config

    def run(
        self,
        longitude: NDArray[np.float64],
        latitude: NDArray[np.float64],
        v_reference_ms: NDArray[np.float64],
        yaw_rate_rad: NDArray[np.float64],
        gps_update_mask: NDArray[np.bool_],
        heading_init: float | None = None,
        yaw_rate_noise_scale: NDArray[np.float64] | None = None,
        speed_noise_scale: NDArray[np.float64] | None = None,
        process_noise_scale: NDArray[np.float64] | None = None,
    ) -> KalmanResult:
        """Run the full EKF forward pass + optional RTS smoother.

        Parameters
        ----------
        longitude : NDArray[np.float64]
            Interpolated longitude (from S&H removal), shape (n,).
        latitude : NDArray[np.float64]
            Interpolated latitude (from S&H removal), shape (n,).
        v_reference_ms : NDArray[np.float64]
            Wheel-based speed in m/s, shape (n,).
        yaw_rate_rad : NDArray[np.float64]
            Signed yaw rate in rad/s, shape (n,).
        gps_update_mask : NDArray[np.bool_]
            Boolean mask where True = actual GPS measurement available.
        heading_init : float or None
            Initial heading in radians. If None, estimated from first
            GPS displacement.

        Returns
        -------
        KalmanResult
            Filtered (and optionally smoothed) state estimates.

        """
        cfg = self.config

        # Reference point for local frame
        lon0 = longitude[0]
        lat0 = latitude[0]

        # Convert to local frame
        gps_x, gps_y = _geo_to_local(longitude, latitude, lon0, lat0)

        # Estimate initial heading from first GPS displacement
        if heading_init is None:
            gps_updates = np.where(gps_update_mask)[0]
            if len(gps_updates) >= 2:  # noqa: PLR2004
                i0, i1 = gps_updates[0], gps_updates[1]
                dx = gps_x[i1] - gps_x[i0]
                dy = gps_y[i1] - gps_y[i0]
                heading_init = np.arctan2(dy, dx)
            else:
                heading_init = cfg.initial_heading

        # Initialize state and covariance
        state0 = np.array([
            gps_x[0], gps_y[0], v_reference_ms[0],
            heading_init, yaw_rate_rad[0],
        ])
        P0 = np.diag(np.array([
            cfg.sigma_pos_gps**2,
            cfg.sigma_pos_gps**2,
            cfg.sigma_v_wheel**2,
            _HEADING_INIT_SIGMA_RAD**2,
            cfg.sigma_yaw_rate**2,
        ]))

        # Per-sample yaw rate measurement noise (inflated during ABS)
        r_omega_base = cfg.sigma_yaw_rate ** 2
        if yaw_rate_noise_scale is not None:
            r_omega_arr = r_omega_base * yaw_rate_noise_scale
        else:
            r_omega_arr = np.full(len(longitude), r_omega_base)

        # Per-sample wheel speed measurement noise (inflated during ABS)
        r_v_base = cfg.sigma_v_wheel ** 2
        if speed_noise_scale is not None:
            r_v_arr = r_v_base * speed_noise_scale
        else:
            r_v_arr = np.full(len(longitude), r_v_base)

        # Per-sample process noise scale (inflated during ABS so
        # prediction covariance grows fast → GPS dominates)
        if process_noise_scale is not None:
            q_scale_arr = process_noise_scale
        else:
            q_scale_arr = np.ones(len(longitude))

        # Run Numba-accelerated EKF + RTS
        states_out, P_diag = _ekf_forward_rts(
            gps_x=np.ascontiguousarray(gps_x),
            gps_y=np.ascontiguousarray(gps_y),
            v_ref=np.ascontiguousarray(v_reference_ms),
            yaw_rate=np.ascontiguousarray(yaw_rate_rad),
            gps_mask=np.ascontiguousarray(gps_update_mask),
            r_omega_arr=np.ascontiguousarray(r_omega_arr),
            r_v_arr=np.ascontiguousarray(r_v_arr),
            q_scale_arr=np.ascontiguousarray(q_scale_arr),
            dt=cfg.dt,
            sigma_pos_gps=cfg.sigma_pos_gps,
            sigma_v_wheel=cfg.sigma_v_wheel,
            sigma_yaw_rate=cfg.sigma_yaw_rate,
            sigma_a=cfg.sigma_a,
            sigma_yaw_acc=cfg.sigma_yaw_acc,
            sigma_pos_drift=cfg.sigma_pos_drift,
            gps_low_speed_gain=cfg.gps_low_speed_gain,
            gps_low_speed_v_scale=cfg.gps_low_speed_v_scale,
            use_rts=cfg.use_rts_smoother,
            state0=np.ascontiguousarray(state0),
            P0=np.ascontiguousarray(P0),
        )

        # Convert back to geographic coordinates
        lon_out, lat_out = _local_to_geo(
            states_out[:, _IDX_X], states_out[:, _IDX_Y], lon0, lat0,
        )

        return KalmanResult(
            x_m=states_out[:, _IDX_X],
            y_m=states_out[:, _IDX_Y],
            speed_ms=states_out[:, _IDX_V],
            heading_rad=states_out[:, _IDX_THETA],
            yaw_rate_rad=states_out[:, _IDX_OMEGA],
            longitude=lon_out,
            latitude=lat_out,
            P_diag=P_diag,
        )
