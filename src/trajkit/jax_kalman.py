"""JAX-accelerated Extended Kalman Filter for GPS/IMU fusion.

Numerically-aligned alternative to the Numba-based EKF in ``kalman.py``:
same CTRV motion model, position-drift fix, speed-dependent GPS noise
scaling, and per-sample measurement/process-noise scaling for ABS
robustness (``speed_noise_scale`` / ``yaw_rate_noise_scale`` /
``process_noise_scale``, matching :meth:`GPSKalmanFilter.run`). Uses
``jax.lax.scan`` for the forward pass and RTS smoother, allowing XLA
compilation and GPU execution.

Additionally supports ``yaw_rate_mask`` (JAX-only, no Numba equivalent):
a boolean array that skips the yaw-rate measurement update entirely for
samples where it is ``False``, rather than merely de-weighting it via
``yaw_rate_noise_scale``.

Usage::

    from .jax_kalman import JAXKalmanFilter
    from .kalman import KalmanConfig

    kf = JAXKalmanFilter(KalmanConfig(dt=0.002))
    result = kf.run(lon, lat, v_ref, yaw_rate, gps_mask)

On GPU::

    # pip install jax[cuda12]
    import jax
    print(jax.devices())  # [GpuDevice(id=0)]
    # Same API — JAX dispatches to GPU automatically.
"""

# ruff: noqa: D107, N803, N806, PLR0913, PLR0915

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)  # Use float64 for numerical accuracy
    import jax.numpy as jnp
    from jax import lax
except ImportError as _jax_err:
    msg = (
        "JAX is required for jax_kalman. "
        "Install with: %pip install jax[cpu]  "
        "or for GPU: %pip install jax[cuda12]"
    )
    raise ImportError(msg) from _jax_err

from .constants import EARTH_METERS_PER_DEGREE
from .kalman import KalmanConfig, KalmanResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# JAX EKF core (pure functions, XLA-compilable)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("use_rts",))
def _ekf_forward_rts_jax(
    gps_x: jax.Array,
    gps_y: jax.Array,
    v_ref: jax.Array,
    yaw_rate: jax.Array,
    gps_mask: jax.Array,
    yaw_rate_mask: jax.Array,
    r_v_arr: jax.Array,
    r_omega_arr: jax.Array,
    q_scale_arr: jax.Array,
    dt: float,
    Q: jax.Array,
    sigma_pos_gps: float,
    gps_low_speed_gain: float,
    gps_low_speed_v_scale: float,
    state0: jax.Array,
    P0: jax.Array,
    *,
    use_rts: bool,
) -> tuple[jax.Array, jax.Array]:
    """XLA-compiled EKF + RTS using jax.lax.scan."""
    dim = 5
    I5 = jnp.eye(dim)

    # --- CTRV predict step ---
    def _predict(
        state: jax.Array,
        P: jax.Array,
        q_scale_k: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        x_s, y_s, v, theta, omega = state

        # Branchless CTRV: blend straight/curved based on |omega|
        eps = 1e-6
        omega_safe = jnp.where(jnp.abs(omega) > eps, omega, eps)

        s_t = jnp.sin(theta)
        c_t = jnp.cos(theta)
        s_tw = jnp.sin(theta + omega_safe * dt)
        c_tw = jnp.cos(theta + omega_safe * dt)

        # Curved
        x_curved = x_s + (v / omega_safe) * (s_tw - s_t)
        y_curved = y_s + (v / omega_safe) * (c_t - c_tw)

        # Straight
        x_straight = x_s + v * c_t * dt
        y_straight = y_s + v * s_t * dt

        is_curved = jnp.abs(omega) > eps
        x_new = jnp.where(is_curved, x_curved, x_straight)
        y_new = jnp.where(is_curved, y_curved, y_straight)
        theta_new = theta + omega * dt

        state_pred = jnp.array([x_new, y_new, v, theta_new, omega])

        # Jacobian F
        F = I5.copy()
        # Curved Jacobian entries
        F = F.at[0, 2].set(jnp.where(is_curved, (s_tw - s_t) / omega_safe, c_t * dt))
        F = F.at[0, 3].set(jnp.where(is_curved, (v / omega_safe) * (c_tw - c_t), -v * s_t * dt))
        F = F.at[0, 4].set(jnp.where(
            is_curved,
            (v / omega_safe) * c_tw * dt - (v / omega_safe**2) * (s_tw - s_t),
            0.0,
        ))
        F = F.at[1, 2].set(jnp.where(is_curved, (c_t - c_tw) / omega_safe, s_t * dt))
        F = F.at[1, 3].set(jnp.where(is_curved, (v / omega_safe) * (s_tw - s_t), v * c_t * dt))
        F = F.at[1, 4].set(jnp.where(
            is_curved,
            (v / omega_safe) * s_tw * dt - (v / omega_safe**2) * (c_t - c_tw),
            0.0,
        ))
        F = F.at[3, 4].set(dt)

        P_pred = F @ P @ F.T + Q * q_scale_k
        return state_pred, P_pred, F

    # --- Forward scan body ---
    def _forward_step(
        carry: tuple[jax.Array, jax.Array],
        inputs: tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
            jax.Array, jax.Array, jax.Array, jax.Array,
        ],
    ) -> tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        state, P = carry
        gx, gy, v_r, yr, g_mask, yr_mask_k, r_v_k, r_omega_k, q_scale_k = inputs

        # Predict
        state_pred, P_pred, F = _predict(state, P, q_scale_k)
        state = state_pred
        P = P_pred

        # Update: wheel speed (per-sample noise, inflated during ABS via
        # r_v_k = sigma_v_wheel**2 * speed_noise_scale[k])
        innov_v = v_r - state[2]
        S_v = P[2, 2] + r_v_k
        K_v = P[:, 2] / S_v
        state = state + K_v * innov_v
        P = P - jnp.outer(K_v, P[2, :])

        # Update: yaw rate (per-sample noise; update itself is skipped
        # where yaw_rate_mask is False, mirroring the GPS mask below)
        innov_w = yr - state[4]
        S_w = P[4, 4] + r_omega_k
        K_w = P[:, 4] / S_w
        state_yr = state + K_w * innov_w
        P_yr = P - jnp.outer(K_w, P[4, :])
        state = jnp.where(yr_mask_k, state_yr, state)
        P = jnp.where(yr_mask_k, P_yr, P)

        # Update: GPS (conditional via mask). Measurement noise is
        # speed-dependent to suppress standstill multipath jitter, mirroring
        # the Numba implementation (kalman.py): sigma_eff grows at low speed
        # via an exponential decay, using the speed *after* the wheel-speed
        # and yaw-rate updates above.
        v_cur = jnp.abs(state[2])
        sigma_gps_eff = sigma_pos_gps * (
            1.0 + gps_low_speed_gain * jnp.exp(-v_cur / gps_low_speed_v_scale)
        )
        R_gps = jnp.eye(2) * sigma_gps_eff**2
        innov_gps = jnp.array([gx - state[0], gy - state[1]])
        S_gps = P[:2, :2] + R_gps
        S_gps_inv = jnp.linalg.inv(S_gps)
        K_gps = P[:, :2] @ S_gps_inv
        state_gps = state + K_gps @ innov_gps
        P_gps = P - K_gps @ P[:2, :]

        # Apply GPS update only where mask is True
        state = jnp.where(g_mask, state_gps, state)
        P = jnp.where(g_mask, P_gps, P)

        carry = (state, P)
        outputs = (state, P, state_pred, P_pred, F)
        return carry, outputs

    # Run forward pass
    inputs = (
        gps_x[1:], gps_y[1:], v_ref[1:], yaw_rate[1:], gps_mask[1:],
        yaw_rate_mask[1:], r_v_arr[1:], r_omega_arr[1:], q_scale_arr[1:],
    )
    init_carry = (state0, P0)
    _, (states_fwd_tail, P_fwd_tail, states_pred_tail, P_pred_tail, F_tail) = lax.scan(
        _forward_step, init_carry, inputs,
    )

    # Prepend initial state
    states_fwd = jnp.concatenate([state0[None, :], states_fwd_tail], axis=0)
    P_fwd = jnp.concatenate([P0[None, :, :], P_fwd_tail], axis=0)
    states_pred = jnp.concatenate([state0[None, :], states_pred_tail], axis=0)
    P_pred = jnp.concatenate([P0[None, :, :], P_pred_tail], axis=0)
    F_all = jnp.concatenate([I5[None, :, :], F_tail], axis=0)

    # --- RTS Backward smoother ---
    if use_rts:
        def _rts_step(
            carry: tuple[jax.Array, jax.Array],
            inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            state_s_next, P_s_next = carry
            state_fwd_k, P_fwd_k, state_pred_kp1, P_pred_kp1, F_kp1 = inputs

            Pp_inv = jnp.linalg.inv(P_pred_kp1)
            G = P_fwd_k @ F_kp1.T @ Pp_inv

            state_s = state_fwd_k + G @ (state_s_next - state_pred_kp1)
            P_s = P_fwd_k + G @ (P_s_next - P_pred_kp1) @ G.T

            return (state_s, P_s), (state_s, P_s)

        # Reverse inputs (exclude last element which is the init for backward)
        rts_inputs = (
            states_fwd[:-1],      # state_fwd[k] for k=n-2..0
            P_fwd[:-1],
            states_pred[1:],      # state_pred[k+1]
            P_pred[1:],
            F_all[1:],            # F[k+1]
        )
        # Reverse for backward scan
        rts_inputs_rev = jax.tree.map(lambda x: x[::-1], rts_inputs)

        init_rts = (states_fwd[-1], P_fwd[-1])
        _, (states_smooth_rev, P_smooth_rev) = lax.scan(
            _rts_step, init_rts, rts_inputs_rev,
        )

        # Reverse back and append the last state
        states_out = jnp.concatenate([states_smooth_rev[::-1], states_fwd[-1:]], axis=0)
        P_out = jnp.concatenate([P_smooth_rev[::-1], P_fwd[-1:, :, :]], axis=0)
    else:
        states_out = states_fwd
        P_out = P_fwd

    P_diag = jax.vmap(jnp.diag)(P_out)
    return states_out, P_diag


# ---------------------------------------------------------------------------
# JAX Kalman Filter class
# ---------------------------------------------------------------------------


class JAXKalmanFilter:
    """JAX-accelerated EKF for GPS track smoothing.

    Numerically-aligned alternative to :class:`GPSKalmanFilter` (Numba)
    using JAX/XLA: same CTRV model, position-drift fix
    (``sigma_pos_drift``), speed-dependent GPS noise scaling
    (``gps_low_speed_gain``/``gps_low_speed_v_scale``), and per-sample
    measurement/process-noise scaling for ABS robustness
    (``speed_noise_scale``/``yaw_rate_noise_scale``/``process_noise_scale``,
    matching :meth:`GPSKalmanFilter.run`). Automatically uses GPU if
    available.

    Parameters
    ----------
    config : KalmanConfig
        Filter tuning parameters (same as Numba version).

    Notes
    -----
    - First call triggers XLA compilation (~10-30s).
    - Subsequent calls with same-shaped data are instant.
    - For GPU: install ``jax[cuda12]`` and ensure CUDA is available.
    - For batch processing: use ``jax.vmap(kf.run)`` over multiple tracks.

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
        yaw_rate_mask: NDArray[np.bool_] | None = None,
        yaw_rate_noise_scale: NDArray[np.float64] | None = None,
        speed_noise_scale: NDArray[np.float64] | None = None,
        process_noise_scale: NDArray[np.float64] | None = None,
    ) -> KalmanResult:
        """Run EKF + RTS via JAX/XLA.

        Parameters match :meth:`GPSKalmanFilter.run`, plus ``yaw_rate_mask``
        (JAX-only): a boolean array where ``False`` skips the yaw-rate
        measurement update entirely for that sample (e.g. known sensor
        dropout), as opposed to ``yaw_rate_noise_scale`` which only
        de-weights it.
        """
        cfg = self.config

        # Local coordinate frame
        lon0, lat0 = float(longitude[0]), float(latitude[0])
        cos_lat0 = np.cos(np.radians(lat0))
        gps_x = (longitude - lon0) * EARTH_METERS_PER_DEGREE * cos_lat0
        gps_y = (latitude - lat0) * EARTH_METERS_PER_DEGREE

        # Initial heading
        if heading_init is None:
            updates = np.where(gps_update_mask)[0]
            if len(updates) >= 2:  # noqa: PLR2004
                i0, i1 = updates[0], updates[1]
                heading_init = float(np.arctan2(
                    gps_y[i1] - gps_y[i0], gps_x[i1] - gps_x[i0],
                ))
            else:
                heading_init = cfg.initial_heading

        # Build matrices
        dt = cfg.dt
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        sa2 = cfg.sigma_a**2
        sya2 = cfg.sigma_yaw_acc**2

        Q = jnp.zeros((5, 5))
        sigma_drift2 = cfg.sigma_pos_drift**2
        Q = Q.at[0, 0].set(dt4 / 4 * sa2 + sigma_drift2 * dt)
        Q = Q.at[0, 2].set(dt3 / 2 * sa2)
        Q = Q.at[1, 1].set(dt4 / 4 * sa2 + sigma_drift2 * dt)
        Q = Q.at[1, 2].set(dt3 / 2 * sa2)
        Q = Q.at[2, 0].set(dt3 / 2 * sa2)
        Q = Q.at[2, 1].set(dt3 / 2 * sa2)
        Q = Q.at[2, 2].set(dt2 * sa2)
        Q = Q.at[3, 3].set(dt2 * sya2)
        Q = Q.at[3, 4].set(dt * sya2)
        Q = Q.at[4, 3].set(dt * sya2)
        Q = Q.at[4, 4].set(sya2)

        state0 = jnp.array([
            gps_x[0], gps_y[0], v_reference_ms[0],
            heading_init, yaw_rate_rad[0],
        ])
        P0 = jnp.diag(jnp.array([
            cfg.sigma_pos_gps**2,
            cfg.sigma_pos_gps**2,
            cfg.sigma_v_wheel**2,
            0.25,  # ~30 deg
            cfg.sigma_yaw_rate**2,
        ]))

        # Default: trust yaw rate for all samples
        if yaw_rate_mask is None:
            yaw_rate_mask = np.ones(len(longitude), dtype=np.bool_)

        # Per-sample measurement/process noise (inflated during ABS, same
        # semantics as GPSKalmanFilter.run(): scale=None -> no inflation)
        n = len(longitude)
        r_v_base = cfg.sigma_v_wheel ** 2
        r_v_arr = (
            r_v_base * np.asarray(speed_noise_scale, dtype=np.float64)
            if speed_noise_scale is not None
            else np.full(n, r_v_base)
        )

        r_omega_base = cfg.sigma_yaw_rate ** 2
        r_omega_arr = (
            r_omega_base * np.asarray(yaw_rate_noise_scale, dtype=np.float64)
            if yaw_rate_noise_scale is not None
            else np.full(n, r_omega_base)
        )

        q_scale_arr = (
            np.asarray(process_noise_scale, dtype=np.float64)
            if process_noise_scale is not None
            else np.ones(n)
        )

        # Transfer to JAX arrays
        gps_x_j = jnp.asarray(gps_x)
        gps_y_j = jnp.asarray(gps_y)
        v_ref_j = jnp.asarray(v_reference_ms)
        yr_j = jnp.asarray(yaw_rate_rad)
        mask_j = jnp.asarray(gps_update_mask)
        yr_mask_j = jnp.asarray(yaw_rate_mask)
        r_v_arr_j = jnp.asarray(r_v_arr)
        r_omega_arr_j = jnp.asarray(r_omega_arr)
        q_scale_arr_j = jnp.asarray(q_scale_arr)

        # Run JIT-compiled EKF
        states_out, P_diag = _ekf_forward_rts_jax(
            gps_x_j, gps_y_j, v_ref_j, yr_j, mask_j, yr_mask_j,
            r_v_arr_j, r_omega_arr_j, q_scale_arr_j,
            dt, Q,
            cfg.sigma_pos_gps,
            cfg.gps_low_speed_gain,
            cfg.gps_low_speed_v_scale,
            state0, P0,
            use_rts=cfg.use_rts_smoother,
        )

        # Back to NumPy
        states_np = np.asarray(states_out)
        P_diag_np = np.asarray(P_diag)

        # Convert to geo
        lon_out = states_np[:, 0] / (EARTH_METERS_PER_DEGREE * cos_lat0) + lon0
        lat_out = states_np[:, 1] / EARTH_METERS_PER_DEGREE + lat0

        return KalmanResult(
            x_m=states_np[:, 0],
            y_m=states_np[:, 1],
            speed_ms=states_np[:, 2],
            heading_rad=states_np[:, 3],
            yaw_rate_rad=states_np[:, 4],
            longitude=lon_out,
            latitude=lat_out,
            P_diag=P_diag_np,
        )
