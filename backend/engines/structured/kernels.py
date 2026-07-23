"""
Structured Product Numba Kernels
=================================

Numba-optimized evaluation kernels for structured product cashflows.
These provide significant speedups for the inner evaluation loops.

Kernels:
- evaluate_autocallable_paths: Autocall + conditional coupons + knock-in put
- evaluate_cpn_paths: Bond floor + upside participation
- evaluate_rc_paths: Fixed coupon + knock-in put
- check_barrier_breach_continuous: Continuous barrier monitoring

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

import numba as nb
from numba import prange
import numpy as np


# =============================================================================
# Autocallable Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_autocallable_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    discount_factors: np.ndarray,
    obs_dt: np.ndarray,
    notional: float,
    autocall_trigger: float,
    coupon_rate: float,
    coupon_barrier: float,
    ki_barrier: float,
    memory_coupon: bool,
    continuous_monitoring: bool,
    df_terminal: float = -1.0,
    s0_reference: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate autocallable product for all paths.

    Returns
    -------
    pv : np.ndarray
        Total present value per path, shape (n_paths,).
    called : np.ndarray
        Boolean array indicating autocalled paths, shape (n_paths,).
    coupon_pv : np.ndarray
        PV of coupons per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)

    pv = np.zeros(n_paths)
    coupon_pv = np.zeros(n_paths)
    called = np.zeros(n_paths, dtype=nb.boolean)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        unpaid = 0.0
        path_pv = 0.0
        path_coupon_pv = 0.0
        is_called = False

        for j in range(n_obs):
            if is_called:
                break

            spot_j = paths[i, obs_indices[j]]
            perf = spot_j / s0
            df = discount_factors[j]
            dt_j = obs_dt[j]

            # Conditional coupon
            coupon_j = coupon_rate * dt_j * notional
            if memory_coupon:
                unpaid += coupon_j
                if perf >= coupon_barrier:
                    path_coupon_pv += unpaid * df
                    path_pv += unpaid * df
                    unpaid = 0.0
            else:
                if perf >= coupon_barrier:
                    path_coupon_pv += coupon_j * df
                    path_pv += coupon_j * df

            # Autocall check
            if perf >= autocall_trigger:
                path_pv += notional * df
                is_called = True

        # If not autocalled: terminal payoff
        if not is_called:
            s_t = paths[i, -1]
            df_t = df_terminal if df_terminal > 0.0 else discount_factors[n_obs - 1]

            # Check knock-in barrier
            breached = False
            if continuous_monitoring:
                for step in range(paths.shape[1]):
                    if paths[i, step] <= ki_barrier * s0:
                        breached = True
                        break
            else:
                for j in range(n_obs):
                    if paths[i, obs_indices[j]] <= ki_barrier * s0:
                        breached = True
                        break

            if breached:
                # Deliver min(S_T/S_0, 1) * notional
                delivery = min(s_t / s0, 1.0)
                path_pv += notional * delivery * df_t
            else:
                # Return full notional
                path_pv += notional * df_t

        pv[i] = path_pv
        coupon_pv[i] = path_coupon_pv
        called[i] = is_called

    return pv, called, coupon_pv


# =============================================================================
# Snowball Autocallable Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_snowball_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    discount_factors: np.ndarray,
    obs_times: np.ndarray,
    notional: float,
    autocall_trigger: float,
    coupon_rate: float,
    ki_barrier: float,
    continuous_monitoring: bool,
    df_terminal: float = -1.0,
    s0_reference: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate snowball autocallable product for all paths.

    Unlike the standard autocallable, the snowball coupon is unconditional
    and grows with time: coupon_j = rate * t_j * notional.

    Returns
    -------
    pv : np.ndarray
        Total present value per path, shape (n_paths,).
    called : np.ndarray
        Boolean array indicating autocalled paths, shape (n_paths,).
    coupon_pv : np.ndarray
        PV of coupons per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)
    pv = np.zeros(n_paths)
    coupon_pv = np.zeros(n_paths)
    called = np.zeros(n_paths, dtype=nb.boolean)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        path_pv = 0.0
        path_coupon = 0.0
        is_called = False

        for j in range(n_obs):
            if is_called:
                break
            spot_j = paths[i, obs_indices[j]]
            perf = spot_j / s0
            df = discount_factors[j]

            # Snowball coupon: unconditional, grows with time
            coupon_j = coupon_rate * obs_times[j] * notional
            path_coupon += coupon_j * df
            path_pv += coupon_j * df

            # Autocall check
            if perf >= autocall_trigger:
                path_pv += notional * df
                is_called = True

        if not is_called:
            # Terminal: KI put
            s_t = paths[i, -1]
            df_t = df_terminal if df_terminal > 0.0 else discount_factors[n_obs - 1]

            breached = False
            if continuous_monitoring:
                for step in range(paths.shape[1]):
                    if paths[i, step] <= ki_barrier * s0:
                        breached = True
                        break
            else:
                for j2 in range(n_obs):
                    if paths[i, obs_indices[j2]] <= ki_barrier * s0:
                        breached = True
                        break

            if breached:
                path_pv += notional * min(s_t / s0, 1.0) * df_t
            else:
                path_pv += notional * df_t

        pv[i] = path_pv
        coupon_pv[i] = path_coupon
        called[i] = is_called

    return pv, called, coupon_pv


# =============================================================================
# Capital Protected Note Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_cpn_paths(
    paths: np.ndarray,
    discount_factor_terminal: float,
    notional: float,
    protection_level: float,
    participation: float,
    cap: float,
    has_cap: bool,
    s0_reference: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Capital Protected Note for all paths.

    Returns
    -------
    pv : np.ndarray
        Total PV per path, shape (n_paths,).
    bond_pv : np.ndarray
        Bond floor PV per path, shape (n_paths,).
    option_pv : np.ndarray
        Option component PV per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    pv = np.zeros(n_paths)
    bond_pv = np.zeros(n_paths)
    option_pv = np.zeros(n_paths)

    bond_floor = protection_level * notional * discount_factor_terminal

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        s_t = paths[i, -1]
        perf = s_t / s0

        upside = max(perf - 1.0, 0.0)
        if has_cap:
            upside = min(upside, cap - 1.0)

        opt = notional * participation * upside * discount_factor_terminal

        bond_pv[i] = bond_floor
        option_pv[i] = opt
        pv[i] = bond_floor + opt

    return pv, bond_pv, option_pv


# =============================================================================
# Reverse Convertible Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_rc_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    discount_factors: np.ndarray,
    obs_dt: np.ndarray,
    notional: float,
    coupon_rate: float,
    ki_barrier: float,
    continuous_monitoring: bool,
    df_terminal: float = -1.0,
    s0_reference: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Reverse Convertible for all paths.

    Returns
    -------
    pv : np.ndarray
        Total PV per path, shape (n_paths,).
    coupon_pv : np.ndarray
        Coupon PV per path, shape (n_paths,).
    terminal_pv : np.ndarray
        Terminal redemption PV per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)

    pv = np.zeros(n_paths)
    coupon_pv_arr = np.zeros(n_paths)
    terminal_pv = np.zeros(n_paths)

    # Total coupon PV (same for all paths since fixed)
    total_coupon_pv = 0.0
    for j in range(n_obs):
        total_coupon_pv += coupon_rate * obs_dt[j] * notional * discount_factors[j]

    df_t = df_terminal if df_terminal > 0.0 else discount_factors[n_obs - 1]

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        s_t = paths[i, -1]

        # Check barrier
        breached = False
        if continuous_monitoring:
            for step in range(paths.shape[1]):
                if paths[i, step] <= ki_barrier * s0:
                    breached = True
                    break
        else:
            for j in range(n_obs):
                if paths[i, obs_indices[j]] <= ki_barrier * s0:
                    breached = True
                    break

        if breached:
            term = notional * min(s_t / s0, 1.0) * df_t
        else:
            term = notional * df_t

        coupon_pv_arr[i] = total_coupon_pv
        terminal_pv[i] = term
        pv[i] = total_coupon_pv + term

    return pv, coupon_pv_arr, terminal_pv


# =============================================================================
# Barrier Breach Check
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def check_barrier_breach_continuous(
    paths: np.ndarray,
    barrier_level: float,
) -> np.ndarray:
    """
    Check if paths breach a barrier (continuous monitoring).

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    barrier_level : float
        Absolute barrier level.

    Returns
    -------
    np.ndarray
        Boolean array, True if barrier breached, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_steps = paths.shape[1]
    breached = np.zeros(n_paths, dtype=nb.boolean)

    for i in prange(n_paths):
        for j in range(n_steps):
            if paths[i, j] <= barrier_level:
                breached[i] = True
                break

    return breached


# =============================================================================
# Single-pass Result Aggregation
# =============================================================================


@nb.njit(cache=True, fastmath=True)
def aggregate_result_stats(
    pv: np.ndarray,
    bond_floor_pv: np.ndarray,
    option_pv: np.ndarray,
    coupon_pv: np.ndarray,
    notional: float,
    df_terminal: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Compute all result statistics in minimal passes.

    Returns
    -------
    mean_pv, std_err, bond_floor_pct, option_pct, coupon_pct,
    capital_loss_prob, worst_case_return, best_case_return
    """
    n = len(pv)
    inv_n = 1.0 / n

    # Pass 1: sums for mean + decomposition + loss count
    sum_pv = 0.0
    sum_sq = 0.0
    sum_bf = 0.0
    sum_opt = 0.0
    sum_cpn = 0.0
    loss_count = 0

    capital_threshold = notional

    for i in range(n):
        v = pv[i]
        sum_pv += v
        sum_sq += v * v
        sum_bf += bond_floor_pv[i]
        sum_opt += option_pv[i]
        sum_cpn += coupon_pv[i]
        if v < capital_threshold:
            loss_count += 1

    mean_pv = sum_pv * inv_n
    variance = sum_sq * inv_n - mean_pv * mean_pv
    std_err = math.sqrt(max(variance, 0.0)) / math.sqrt(n)

    bond_floor_pct = (sum_bf * inv_n) / notional * 100.0
    option_pct = (sum_opt * inv_n) / notional * 100.0
    coupon_pct = (sum_cpn * inv_n) / notional * 100.0
    capital_loss_prob = loss_count * inv_n

    # Pass 2: percentiles via O(n) partial selection instead of an O(n log n)
    # full sort — only the 5th and 95th order statistics are needed.
    returns = (pv / notional) - 1.0
    k5 = int(0.05 * n)
    k95 = min(int(0.95 * n), n - 1)
    worst_case = np.partition(returns, k5)[k5]
    best_case = np.partition(returns, k95)[k95]

    return (
        mean_pv,
        std_err,
        bond_floor_pct,
        option_pct,
        coupon_pct,
        capital_loss_prob,
        worst_case,
        best_case,
    )


# =============================================================================
# Generalized Barrier Breach Check
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def check_barrier_breach(
    paths: np.ndarray,
    barrier_fraction: float,
    obs_indices: np.ndarray,
    continuous: bool,
    breach_below: bool,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Check if paths breach a barrier level (generalized).

    Supports both knock-in (breach below) and knock-out (breach above),
    with continuous or discrete monitoring and early exit per path.

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    barrier_fraction : float
        Barrier as fraction of initial spot (e.g., 0.6 for 60%).
    obs_indices : np.ndarray
        Observation date indices (used only if continuous=False).
    continuous : bool
        True = check all time steps, False = check only observation dates.
    breach_below : bool
        True = breach if price <= barrier (knock-in put).
        False = breach if price >= barrier (knock-out call).
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Boolean array, True if barrier breached, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    breached = np.zeros(n_paths, dtype=nb.boolean)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        barrier = barrier_fraction * s0

        if continuous:
            for step in range(paths.shape[1]):
                if breach_below:
                    if paths[i, step] <= barrier:
                        breached[i] = True
                        break
                else:
                    if paths[i, step] >= barrier:
                        breached[i] = True
                        break
        else:
            n_obs = len(obs_indices)
            for j in range(n_obs):
                if breach_below:
                    if paths[i, obs_indices[j]] <= barrier:
                        breached[i] = True
                        break
                else:
                    if paths[i, obs_indices[j]] >= barrier:
                        breached[i] = True
                        break

    return breached


# =============================================================================
# Conditional Coupon Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_conditional_coupon_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    discount_factors: np.ndarray,
    obs_dt: np.ndarray,
    notional: float,
    coupon_rate: float,
    barrier: float,
    memory: bool,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate conditional coupon for all paths.

    Also covers VariableIncomeCoupon (memory=False, same logic).

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    obs_indices : np.ndarray
        Observation date indices, shape (n_obs,).
    discount_factors : np.ndarray
        Discount factors at observation dates, shape (n_obs,).
    obs_dt : np.ndarray
        Period lengths between observations, shape (n_obs,).
    notional : float
        Product notional.
    coupon_rate : float
        Annualized coupon rate.
    barrier : float
        Performance barrier (e.g., 0.7 = 70% of initial).
    memory : bool
        If True, unpaid coupons accumulate and are paid when barrier is met.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value of coupons per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        unpaid = 0.0
        path_pv = 0.0

        for j in range(n_obs):
            spot_j = paths[i, obs_indices[j]]
            perf = spot_j / s0
            coupon_j = coupon_rate * obs_dt[j] * notional

            if memory:
                unpaid += coupon_j
                if perf >= barrier:
                    path_pv += unpaid * discount_factors[j]
                    unpaid = 0.0
            else:
                if perf >= barrier:
                    path_pv += coupon_j * discount_factors[j]

        pv[i] = path_pv

    return pv


# =============================================================================
# CMI Conditional Coupon Kernel (CPPF/DPPF)
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_cmi_coupon_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    discount_factors: np.ndarray,
    obs_dt: np.ndarray,
    notional: float,
    coupon_rate: float,
    barrier: float,
    cppf: float,
    dppf: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate CMI conditional coupon with variable return (CPPF/DPPF).

    At each observation:
    - If above barrier AND no prior misses: fixed + cppf * max(perf-1, 0)
    - If above barrier AND prior misses: fixed + accumulated + dppf * max(perf-barrier, 0)
    - If below barrier: accumulate (memory)

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    obs_indices : np.ndarray
        Observation date indices, shape (n_obs,).
    discount_factors : np.ndarray
        Discount factors at observation dates, shape (n_obs,).
    obs_dt : np.ndarray
        Period lengths between observations, shape (n_obs,).
    notional : float
        Product notional.
    coupon_rate : float
        Annualized fixed coupon rate.
    barrier : float
        Performance barrier.
    cppf : float
        Current Participation Performance Factor.
    dppf : float
        Deferred Participation Performance Factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value of coupons per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        unpaid = 0.0
        unpaid_count = 0
        path_pv = 0.0

        for j in range(n_obs):
            spot_j = paths[i, obs_indices[j]]
            perf = spot_j / s0
            coupon_j = coupon_rate * obs_dt[j] * notional

            if perf >= barrier:
                # Variable return depends on miss history
                if unpaid_count > 0:
                    var_return = dppf * max(perf - barrier, 0.0) * notional
                else:
                    var_return = cppf * max(perf - 1.0, 0.0) * notional
                path_pv += (coupon_j + unpaid + var_return) * discount_factors[j]
                unpaid = 0.0
                unpaid_count = 0
            else:
                unpaid += coupon_j
                unpaid_count += 1

        pv[i] = path_pv

    return pv


# =============================================================================
# Knock-Out Participation Kernel (SharkNote)
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_knock_out_participation_paths(
    paths: np.ndarray,
    notional: float,
    participation: float,
    barrier: float,
    rebate: float,
    cap: float,
    has_cap: bool,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate knock-out participation for all paths.

    If upper barrier is touched: pays rebate.
    Otherwise: pays capped upside participation.

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    notional : float
        Product notional.
    participation : float
        Participation rate.
    barrier : float
        Upper knock-out barrier as fraction of initial (e.g., 1.40).
    rebate : float
        Fixed rebate if knocked out (fraction of notional).
    cap : float
        Effective cap on performance (e.g., barrier if no explicit cap).
    has_cap : bool
        Whether a cap applies.
    df_terminal : float
        Terminal discount factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_steps = paths.shape[1]
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        ko_barrier = barrier * s0

        # Check knock-out with early exit
        knocked_out = False
        for step in range(n_steps):
            if paths[i, step] >= ko_barrier:
                knocked_out = True
                break

        s_t = paths[i, -1]
        perf = s_t / s0

        if knocked_out:
            pv[i] = rebate * notional * df_terminal
        else:
            upside = max(perf - 1.0, 0.0)
            if has_cap:
                upside = min(upside, cap - 1.0)
            pv[i] = participation * upside * notional * df_terminal

    return pv


# =============================================================================
# Twin-Win Participation Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_twin_win_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    notional: float,
    participation: float,
    ki_barrier: float,
    continuous: bool,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate twin-win participation for all paths.

    If NOT breached: profit from absolute move (up or down).
    If breached: capital loss (deliver min(perf, 1) * notional).

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    obs_indices : np.ndarray
        Observation date indices (used if continuous=False).
    notional : float
        Product notional.
    participation : float
        Participation rate.
    ki_barrier : float
        Knock-in barrier as fraction of initial.
    continuous : bool
        True = continuous monitoring, False = discrete.
    df_terminal : float
        Terminal discount factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        barrier = ki_barrier * s0

        # Check barrier breach with early exit
        breached = False
        if continuous:
            for step in range(paths.shape[1]):
                if paths[i, step] <= barrier:
                    breached = True
                    break
        else:
            n_obs = len(obs_indices)
            for j in range(n_obs):
                if paths[i, obs_indices[j]] <= barrier:
                    breached = True
                    break

        s_t = paths[i, -1]
        perf = s_t / s0

        if breached:
            # Breached twin-win becomes a linear tracker (no upside cap)
            pv[i] = perf * notional * df_terminal
        else:
            abs_move = abs(perf - 1.0)
            pv[i] = (1.0 + participation * abs_move) * notional * df_terminal

    return pv


# =============================================================================
# Geared Knock-In Put Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_geared_ki_put_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    notional: float,
    barrier: float,
    gearing: float,
    continuous: bool,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate geared knock-in put for all paths.

    If breached: pays notional * max(1 - gearing * (1 - S_T/S_0), 0).
    Otherwise: pays full notional.

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    obs_indices : np.ndarray
        Observation date indices (used if continuous=False).
    notional : float
        Product notional.
    barrier : float
        Knock-in barrier as fraction of initial.
    gearing : float
        Loss amplification factor.
    continuous : bool
        True = continuous monitoring, False = discrete.
    df_terminal : float
        Terminal discount factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        ki_level = barrier * s0

        # Check barrier breach with early exit
        breached = False
        if continuous:
            for step in range(paths.shape[1]):
                if paths[i, step] <= ki_level:
                    breached = True
                    break
        else:
            n_obs = len(obs_indices)
            for j in range(n_obs):
                if paths[i, obs_indices[j]] <= ki_level:
                    breached = True
                    break

        if breached:
            s_t = paths[i, -1]
            loss_fraction = gearing * (1.0 - s_t / s0)
            pv[i] = notional * max(1.0 - loss_fraction, 0.0) * df_terminal
        else:
            pv[i] = notional * df_terminal

    return pv


# =============================================================================
# Cliquet Participation Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_cliquet_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    discount_factors: np.ndarray,
    notional: float,
    local_cap: float,
    local_floor: float,
    global_cap: float,
    global_floor: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate cliquet (ratchet) participation for all paths.

    Accumulates locally capped/floored periodic returns, then applies
    global cap/floor to the total accumulated return.

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    obs_indices : np.ndarray
        Observation date indices (reset dates), shape (n_obs,).
    discount_factors : np.ndarray
        Discount factors at observation dates, shape (n_obs,).
    notional : float
        Product notional.
    local_cap : float
        Maximum return per period (e.g., 0.05 = 5%).
    local_floor : float
        Minimum return per period (e.g., 0.0 = no negative contributions).
    global_cap : float
        Maximum total accumulated return (e.g., 0.50 = 50%).
    global_floor : float
        Minimum total accumulated return (e.g., 0.0 = principal protected).
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value of cliquet payoff per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)
    pv = np.zeros(n_paths)
    df_terminal = discount_factors[n_obs - 1]

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        accumulated = 0.0

        # First period: from s0 to first obs
        s_prev = s0
        for j in range(n_obs):
            s_curr = paths[i, obs_indices[j]]
            period_return = s_curr / s_prev - 1.0

            # Apply local cap and floor
            clamped = min(max(period_return, local_floor), local_cap)
            accumulated += clamped
            s_prev = s_curr

        # Apply global cap and floor
        total_return = min(max(accumulated, global_floor), global_cap)
        pv[i] = notional * total_return * df_terminal

    return pv


# =============================================================================
# Average (Asian) Participation Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_asian_participation_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    notional: float,
    participation: float,
    cap: float,
    has_cap: bool,
    is_average_strike: bool,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate Asian (average) participation for all paths.

    Supports two modes:
    - Average rate: payoff = participation * max(avg_perf - 1, 0)
    - Average strike: payoff = participation * max(S_T/S_0 - avg_perf, 0)

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    obs_indices : np.ndarray
        Averaging observation dates, shape (n_obs,).
    notional : float
        Product notional.
    participation : float
        Participation rate.
    cap : float
        Maximum return. Only used if has_cap=True.
    has_cap : bool
        Whether to apply a cap.
    is_average_strike : bool
        True = average strike mode, False = average rate mode.
    df_terminal : float
        Terminal discount factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]

        # Compute average performance
        avg_perf = 0.0
        for j in range(n_obs):
            avg_perf += paths[i, obs_indices[j]] / s0
        avg_perf /= n_obs

        if is_average_strike:
            # Average strike: payoff based on terminal vs average
            terminal_perf = paths[i, -1] / s0
            raw_return = max(terminal_perf - avg_perf, 0.0)
        else:
            # Average rate: payoff based on average vs initial
            raw_return = max(avg_perf - 1.0, 0.0)

        if has_cap:
            raw_return = min(raw_return, cap)

        pv[i] = notional * participation * raw_return * df_terminal

    return pv


# =============================================================================
# Lookback Participation Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_lookback_participation_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    notional: float,
    participation: float,
    cap: float,
    has_cap: bool,
    use_max: bool,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate lookback participation for all paths.

    Two modes:
    - use_max=True: payoff = participation * max(max_perf - 1, 0)
      (best-of timing — investor benefits from the highest price)
    - use_max=False: payoff = participation * max(1 - min_perf, 0)
      (worst-of timing — protection from the lowest price, i.e. floating strike put)

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    obs_indices : np.ndarray
        Observation dates for lookback, shape (n_obs,).
    notional : float
        Product notional.
    participation : float
        Participation rate.
    cap : float
        Maximum return. Only used if has_cap=True.
    has_cap : bool
        Whether to apply a cap.
    use_max : bool
        True = lookback on max (call-like), False = lookback on min (put-like).
    df_terminal : float
        Terminal discount factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]

        if use_max:
            extreme_perf = 0.0
            for j in range(n_obs):
                perf = paths[i, obs_indices[j]] / s0
                if perf > extreme_perf:
                    extreme_perf = perf
            raw_return = max(extreme_perf - 1.0, 0.0)
        else:
            extreme_perf = 1e30
            for j in range(n_obs):
                perf = paths[i, obs_indices[j]] / s0
                if perf < extreme_perf:
                    extreme_perf = perf
            raw_return = max(1.0 - extreme_perf, 0.0)

        if has_cap:
            raw_return = min(raw_return, cap)

        pv[i] = notional * participation * raw_return * df_terminal

    return pv


# =============================================================================
# Range Accrual Coupon Kernel
# =============================================================================


# =============================================================================
# Barrier Type Constants
# =============================================================================

BARRIER_MATURITY = 0
BARRIER_PRINCIPAL_PROTECTED = 1
BARRIER_GEARED_BUFFER = 2
BARRIER_BUFFER = 3
BARRIER_NONE = 4


# =============================================================================
# Generalized Autocallable Kernel (per-observation arrays)
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_generalized_ac_paths(
    paths: np.ndarray,
    coupon_obs_indices: np.ndarray,
    coupon_amounts: np.ndarray,
    coupon_barriers: np.ndarray,
    call_obs_indices: np.ndarray,
    call_triggers: np.ndarray,
    call_redemptions: np.ndarray,
    call_baselines: np.ndarray,
    participation_factor: float,
    barrier_fraction: float,
    barrier_type: int,
    gearing: float,
    notional: float,
    discount_factors_coupon: np.ndarray,
    discount_factors_call: np.ndarray,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generalized autocallable kernel with per-observation varying parameters.

    Coupon and call events can occur on different dates with different schedules.
    All thresholds are expressed as performance fractions (S / S_0).
    Terminal protection uses barrier_type (maturity-only check, no path-dependent KI).

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    coupon_obs_indices : np.ndarray
        Step indices for coupon observations, shape (n_coupons,).
    coupon_amounts : np.ndarray
        Dollar coupon amount per event, shape (n_coupons,).
    coupon_barriers : np.ndarray
        Performance barrier per coupon (e.g., 0.7), shape (n_coupons,).
    call_obs_indices : np.ndarray
        Step indices for call observations, shape (n_calls,).
    call_triggers : np.ndarray
        Performance trigger per call (e.g., 1.0), shape (n_calls,).
    call_redemptions : np.ndarray
        Redemption amount per call (dollar), shape (n_calls,).
    call_baselines : np.ndarray
        Baseline for variable return per call (dollar), shape (n_calls,).
    participation_factor : float
        Participation rate for variable return on call.
    barrier_fraction : float
        Terminal barrier as fraction of S_0 (e.g., 0.7).
    barrier_type : int
        Terminal protection: BARRIER_MATURITY, BARRIER_PRINCIPAL_PROTECTED, etc.
    gearing : float
        Gearing for BARRIER_GEARED_BUFFER.
    notional : float
        Product notional.
    discount_factors_coupon : np.ndarray
        Discount factors at coupon dates, shape (n_coupons,).
    discount_factors_call : np.ndarray
        Discount factors at call dates, shape (n_calls,).
    df_terminal : float
        Terminal discount factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    pv : np.ndarray
        Present value per path, shape (n_paths,).
    last_obs_step : np.ndarray
        Step index of last payment per path, shape (n_paths,).
    coupon_pv : np.ndarray
        PV of coupons per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    dtm = paths.shape[1] - 1
    n_coupons = len(coupon_obs_indices)
    n_calls = len(call_obs_indices)

    pv = np.zeros(n_paths)
    last_obs_step = np.full(n_paths, -1.0)
    coupon_pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        barrier_abs = barrier_fraction * s0
        alive = True
        path_pv = 0.0
        path_coupon = 0.0
        last_step = -1.0

        # Merge-iterate coupon and call events in chronological order
        coupon_idx = 0
        call_idx = 0

        while (coupon_idx < n_coupons or call_idx < n_calls) and alive:
            coupon_step = (
                coupon_obs_indices[coupon_idx] if coupon_idx < n_coupons else dtm + 1
            )
            call_step = call_obs_indices[call_idx] if call_idx < n_calls else dtm + 1

            if coupon_step <= call_step and coupon_idx < n_coupons:
                step = coupon_step
                # step==0 means the event falls on time_t; the legacy
                # _EventDatesIterator (data_management.py) uses a strict
                # ``time_t > event.event_date`` filter and keeps such
                # events. paths[i, 0] == S_t for all paths so the barrier
                # check triggers uniformly.
                if 0 <= step <= dtm:
                    perf = paths[i, step] / s0
                    if perf >= coupon_barriers[coupon_idx]:
                        amt = (
                            coupon_amounts[coupon_idx]
                            * discount_factors_coupon[coupon_idx]
                        )
                        path_pv += amt
                        path_coupon += amt
                        last_step = step
                coupon_idx += 1
            else:
                step = call_step
                if 0 <= step <= dtm:
                    spot = paths[i, step]
                    perf = spot / s0
                    if perf >= call_triggers[call_idx]:
                        upside = participation_factor * max(
                            spot - call_baselines[call_idx], 0.0
                        )
                        path_pv += (
                            call_redemptions[call_idx] + upside
                        ) * discount_factors_call[call_idx]
                        last_step = step
                        alive = False
                call_idx += 1

        # Terminal payoff for surviving paths
        if alive:
            s_t = paths[i, -1]
            if barrier_type == BARRIER_MATURITY:
                terminal = s_t if s_t < barrier_abs else notional
            elif barrier_type == BARRIER_PRINCIPAL_PROTECTED:
                terminal = notional
            elif barrier_type == BARRIER_GEARED_BUFFER:
                terminal = notional - gearing * max(barrier_abs - s_t, 0.0)
            elif barrier_type == BARRIER_BUFFER:
                terminal = notional - max(barrier_abs - s_t, 0.0)
            else:
                terminal = s_t
            path_pv += terminal * df_terminal
            last_step = dtm

        pv[i] = path_pv
        last_obs_step[i] = last_step
        coupon_pv[i] = path_coupon

    return pv, last_obs_step, coupon_pv


# =============================================================================
# Terminal Product Kernel (non-autocallable: BTS, BO, FC/FR, PART)
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_terminal_product_paths(
    paths: np.ndarray,
    coupon_obs_indices: np.ndarray,
    coupon_amounts: np.ndarray,
    coupon_barriers: np.ndarray,
    discount_factors_coupon: np.ndarray,
    notional: float,
    participation: float,
    barrier_fraction: float,
    barrier_type: int,
    gearing: float,
    bonus_amount: float,
    bonus_trigger_perf: float,
    variable_strike_perf: float,
    has_bonus: bool,
    has_cap: bool,
    cap_perf: float,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Terminal payoff kernel for non-autocallable products.

    Handles protection + optional bonus + optional capped participation + coupons.

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    coupon_obs_indices : np.ndarray
        Step indices for coupon observations, shape (n_coupons,).
    coupon_amounts : np.ndarray
        Dollar coupon per event, shape (n_coupons,).
    coupon_barriers : np.ndarray
        Performance barrier per coupon, shape (n_coupons,).
    discount_factors_coupon : np.ndarray
        Discount factors at coupon dates, shape (n_coupons,).
    notional : float
        Product notional.
    participation : float
        Upside participation rate.
    barrier_fraction : float
        Barrier as fraction of S_0.
    barrier_type : int
        Terminal protection type.
    gearing : float
        Gearing for BARRIER_GEARED_BUFFER.
    bonus_amount : float
        Bonus dollar amount.
    bonus_trigger_perf : float
        Performance level above which bonus is paid.
    variable_strike_perf : float
        Performance level above which variable return starts.
    has_bonus : bool
        Whether product has a bonus feature.
    has_cap : bool
        Whether participation is capped.
    cap_perf : float
        Cap as performance level (e.g., 1.5 for 150%).
    df_terminal : float
        Terminal discount factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    pv : np.ndarray
        Present value per path, shape (n_paths,).
    coupon_pv : np.ndarray
        PV of coupons per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    dtm = paths.shape[1] - 1
    n_coupons = len(coupon_obs_indices)

    pv = np.zeros(n_paths)
    coupon_pv_arr = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        barrier_abs = barrier_fraction * s0
        path_coupon = 0.0

        # Conditional coupons (step==0 means the event falls on time_t;
        # the legacy keeps such events, see sn_interface._extract_future_coupon_arrays).
        for idx in range(n_coupons):
            step = coupon_obs_indices[idx]
            if 0 <= step <= dtm:
                perf = paths[i, step] / s0
                if perf >= coupon_barriers[idx]:
                    path_coupon += coupon_amounts[idx] * discount_factors_coupon[idx]

        # Terminal payoff
        s_t = paths[i, -1]
        perf_t = s_t / s0

        # Protection
        if barrier_type == BARRIER_MATURITY:
            protection = s_t if s_t < barrier_abs else notional
        elif barrier_type == BARRIER_PRINCIPAL_PROTECTED:
            protection = notional
        elif barrier_type == BARRIER_GEARED_BUFFER:
            protection = notional - gearing * max(barrier_abs - s_t, 0.0)
        elif barrier_type == BARRIER_BUFFER:
            protection = notional - max(barrier_abs - s_t, 0.0)
        else:
            protection = notional

        # Bonus + variable return. When the bonus is *not* triggered, mimic the legacy Portfolio's additive cancellation (+bonus_amount * Bond - bonus_amount * DigitalPut) so that the 1-ULP floating-point residual below the protection level matches the expected scores stored in the database.
        if has_bonus:
            if perf_t >= bonus_trigger_perf:
                protection += bonus_amount
                upside = max(perf_t - variable_strike_perf, 0.0)
            else:
                protection = (protection + bonus_amount) - bonus_amount
                upside = 0.0
        else:
            upside = max(perf_t - 1.0, 0.0)

        if has_cap and upside > 0.0:
            upside = min(upside, cap_perf - 1.0)
        variable = participation * upside * notional

        pv[i] = (protection + variable) * df_terminal + path_coupon
        coupon_pv_arr[i] = path_coupon

    return pv, coupon_pv_arr


# =============================================================================
# Marathon Kernel
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_marathon_paths(
    paths: np.ndarray,
    notional: float,
    participation: float,
    barrier_fraction: float,
    barrier_type: int,
    partial_protection_fraction: float,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Marathon family: participation in both upside and downside.

    Variants (selected by barrier_type):
      BARRIER_NONE:   S_0 + pf * max(S_T-S_0, 0) - max(S_0-S_T, 0)
      BARRIER_MATURITY: same, but S_T if below barrier (cliff)
      BARRIER_BUFFER:  S_0 + pf * max(S_T-S_0, 0) - max(B-S_T, 0)
      5 (partial):     B + max(S_T-B, 0) + (pf-1) * max(S_T-S_0, 0)

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, n_steps+1).
    notional : float
        Product notional (= S_0).
    participation : float
        Upside participation rate.
    barrier_fraction : float
        Barrier as fraction of S_0.
    barrier_type : int
        Protection variant.
    partial_protection_fraction : float
        Protection level for partial variant (as fraction of S_0).
    df_terminal : float
        Terminal discount factor.
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    result = np.zeros(n_paths)
    PARTIAL = 5

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        s_t = paths[i, -1]
        barrier_abs = barrier_fraction * s0

        if barrier_type == BARRIER_NONE:
            val = s0 + participation * max(s_t - s0, 0.0) - max(s0 - s_t, 0.0)
        elif barrier_type == BARRIER_MATURITY:
            if s_t < barrier_abs:
                val = s_t
            else:
                val = s0 + participation * max(s_t - s0, 0.0)
        elif barrier_type == BARRIER_BUFFER:
            val = s0 + participation * max(s_t - s0, 0.0) - max(barrier_abs - s_t, 0.0)
        elif barrier_type == PARTIAL:
            prot = partial_protection_fraction * s0
            val = (
                prot + max(s_t - prot, 0.0) + (participation - 1.0) * max(s_t - s0, 0.0)
            )
        else:
            val = s_t

        result[i] = val * df_terminal

    return result


# =============================================================================
# CMI Autocallable Kernel (per-event cppf/dppf arrays)
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_cmi_ac_paths(
    paths: np.ndarray,
    coupon_obs_indices: np.ndarray,
    coupon_amounts: np.ndarray,
    coupon_barriers: np.ndarray,
    coupon_cppf: np.ndarray,
    coupon_dppf: np.ndarray,
    call_obs_indices: np.ndarray,
    call_triggers: np.ndarray,
    call_redemptions: np.ndarray,
    call_baselines: np.ndarray,
    participation_factor: float,
    barrier_fraction: float,
    barrier_type: int,
    gearing: float,
    notional: float,
    discount_factors_coupon: np.ndarray,
    discount_factors_call: np.ndarray,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Autocallable kernel with CMI (Contingent Memory Income) coupons.

    Extends evaluate_generalized_ac_paths with per-event cppf/dppf
    variable return logic. When threshold is met:
      - No prior misses: variable_return = cppf * max(perf - 1, 0) * notional
      - Prior misses:    variable_return = dppf * max(perf - barrier, 0) * notional

    Parameters
    ----------
    coupon_cppf : np.ndarray
        Current Payment Participation Factor per coupon, shape (n_coupons,).
    coupon_dppf : np.ndarray
        Deferred Payment Participation Factor per coupon, shape (n_coupons,).
    """
    n_paths = paths.shape[0]
    dtm = paths.shape[1] - 1
    n_coupons = len(coupon_obs_indices)
    n_calls = len(call_obs_indices)

    pv = np.zeros(n_paths)
    last_obs_step = np.full(n_paths, -1.0)
    coupon_pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        barrier_abs = barrier_fraction * s0
        alive = True
        path_pv = 0.0
        path_coupon = 0.0
        last_step = -1.0
        cmi_unpaid = 0.0
        has_missed = False

        coupon_idx = 0
        call_idx = 0

        while (coupon_idx < n_coupons or call_idx < n_calls) and alive:
            coupon_step = (
                coupon_obs_indices[coupon_idx] if coupon_idx < n_coupons else dtm + 1
            )
            call_step = call_obs_indices[call_idx] if call_idx < n_calls else dtm + 1

            if coupon_step <= call_step and coupon_idx < n_coupons:
                step = coupon_step
                if 0 <= step <= dtm:
                    spot_j = paths[i, step]
                    perf = spot_j / s0
                    base_coupon = coupon_amounts[coupon_idx]

                    if perf >= coupon_barriers[coupon_idx]:
                        # Variable return: cppf if no misses, dppf if misses
                        if has_missed:
                            var_ret = max(
                                coupon_dppf[coupon_idx]
                                * (spot_j - coupon_barriers[coupon_idx] * s0),
                                0.0,
                            )
                        else:
                            var_ret = max(coupon_cppf[coupon_idx] * (spot_j - s0), 0.0)

                        payment = (
                            base_coupon + cmi_unpaid + var_ret
                        ) * discount_factors_coupon[coupon_idx]
                        path_pv += payment
                        path_coupon += payment
                        last_step = step
                        cmi_unpaid = 0.0
                        has_missed = False
                    else:
                        cmi_unpaid += base_coupon
                        has_missed = True
                coupon_idx += 1
            else:
                step = call_step
                if 0 <= step <= dtm:
                    spot_j = paths[i, step]
                    perf = spot_j / s0
                    if perf >= call_triggers[call_idx]:
                        upside = participation_factor * max(
                            spot_j - call_baselines[call_idx], 0.0
                        )
                        path_pv += (
                            call_redemptions[call_idx] + upside
                        ) * discount_factors_call[call_idx]
                        last_step = step
                        alive = False
                call_idx += 1

        if alive:
            s_t = paths[i, -1]
            if barrier_type == BARRIER_MATURITY:
                terminal = s_t if s_t < barrier_abs else notional
            elif barrier_type == BARRIER_PRINCIPAL_PROTECTED:
                terminal = notional
            elif barrier_type == BARRIER_GEARED_BUFFER:
                terminal = notional - gearing * max(barrier_abs - s_t, 0.0)
            elif barrier_type == BARRIER_BUFFER:
                terminal = notional - max(barrier_abs - s_t, 0.0)
            else:
                terminal = s_t
            path_pv += terminal * df_terminal
            last_step = dtm

        pv[i] = path_pv
        last_obs_step[i] = last_step
        coupon_pv[i] = path_coupon

    return pv, last_obs_step, coupon_pv


# =============================================================================
# Low-Point Forward Kernel (MALP)
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_lowpoint_forward_paths(
    paths: np.ndarray,
    fixings_indices: np.ndarray,
    notional: float,
    participation: float,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Low-Point Forward payoff: S_0 + pf * S_0 * (S_T / min_obs - 1).

    The reference price is replaced by the minimum over fixings observations.
    The forward is geared by S_0 / min_obs.

    Parameters
    ----------
    fixings_indices : np.ndarray
        Step indices for fixings observations, shape (n_fixings,).
    """
    n_paths = paths.shape[0]
    n_fixings = len(fixings_indices)
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]

        # Find minimum over fixings observations
        min_obs = paths[i, fixings_indices[0]]
        for idx in range(1, n_fixings):
            val = paths[i, fixings_indices[idx]]
            if val < min_obs:
                min_obs = val

        s_t = paths[i, -1]
        # Geared forward: (S_T - min) * (S_0 / min) = S_0 * (S_T/min - 1)
        forward_return = participation * s0 * (s_t / min_obs - 1.0)
        pv[i] = (s0 + forward_return) * df_terminal

    return pv


# =============================================================================
# Daily Barrier Kernel (DailySP)
# =============================================================================


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_daily_barrier_paths(
    paths: np.ndarray,
    notional: float,
    barrier_fraction: float,
    has_cap: bool,
    cap_fraction: float,
    df_terminal: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Daily-monitored barrier product (DailySP).

    If barrier NOT touched on any day: payoff = min(S_T, cap_level) if capped, else S_T.
    Principal protected via down-and-out put.
    If barrier touched: payoff = S_T (full underlying exposure, no protection).

    Parameters
    ----------
    barrier_fraction : float
        Barrier as fraction of S_0 (e.g., 0.7).
    cap_fraction : float
        Cap as fraction of S_0 (e.g., 1.3). Ignored if has_cap=False.
    """
    n_paths = paths.shape[0]
    n_steps = paths.shape[1]
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        barrier_abs = barrier_fraction * s0
        s_t = paths[i, -1]

        # Check daily barrier with early exit
        knocked_out = False
        for step in range(n_steps):
            if paths[i, step] <= barrier_abs:
                knocked_out = True
                break

        if knocked_out:
            pv[i] = s_t * df_terminal
        else:
            # Protected: payoff = max(S_T, S_0) with optional cap
            protected = max(s_t, s0)
            if has_cap:
                cap_abs = cap_fraction * s0
                protected = min(protected, cap_abs)
            pv[i] = protected * df_terminal

    return pv


# =============================================================================
# Fixings-Based Call Kernel (GT, GTD, PART with average/lookback reference)
# =============================================================================

FIXINGS_LOOKBACK_MIN = 0
FIXINGS_AVERAGE = 1


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_fixings_call_paths(
    paths: np.ndarray,
    fixings_indices: np.ndarray,
    notional: float,
    participation: float,
    fixings_mode: int,
    geared: bool,
    has_cap: bool,
    cap_perf: float,
    df_terminal: float,
    s0_reference: float = -1.0,
    past_reference: float = 0.0,
    n_past_fixings: int = 0,
) -> np.ndarray:
    """
    Principal protection + participation in a call with floating reference price.

    The floating reference combines historically observed fixings (``past_reference`` + ``n_past_fixings``) with simulated future fixings (``fixings_indices`` into ``paths``). AVERAGE mode expects ``past_reference`` to be the historical ``floating_strike`` (mean of past fixings) and weights it by ``n_past_fixings``. LOOKBACK_MIN expects ``past_reference`` to be the running minimum across historical observations; ``n_past_fixings > 0`` then signals that the minimum is already seeded from history.

    If geared=True (GTD): payoff = S_0 + pf * (S_0/ref) * max(S_T - ref, 0)
    If geared=False (GT, PART): payoff = S_0 + pf * max(S_T - ref, 0)
    """
    n_paths = paths.shape[0]
    n_fixings = len(fixings_indices)
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]

        if fixings_mode == FIXINGS_LOOKBACK_MIN:
            if n_past_fixings > 0:
                ref = past_reference
                start_idx = 0
            else:
                ref = paths[i, fixings_indices[0]]
                start_idx = 1
            for idx in range(start_idx, n_fixings):
                val = paths[i, fixings_indices[idx]]
                if val < ref:
                    ref = val
        else:  # FIXINGS_AVERAGE
            total = past_reference * n_past_fixings
            for idx in range(n_fixings):
                total += paths[i, fixings_indices[idx]]
            ref = total / (n_past_fixings + n_fixings)

        s_t = paths[i, -1]
        call_value = max(s_t - ref, 0.0)

        if geared:
            call_value *= s0 / ref

        if has_cap:
            max_call = (cap_perf - 1.0) * s0
            call_value = min(call_value, max_call)

        pv[i] = (s0 + participation * call_value) * df_terminal

    return pv


@nb.njit(cache=True, parallel=True, fastmath=True)
def evaluate_range_accrual_paths(
    paths: np.ndarray,
    obs_indices: np.ndarray,
    discount_factors: np.ndarray,
    obs_dt: np.ndarray,
    notional: float,
    coupon_rate: float,
    lower_barrier: float,
    upper_barrier: float,
    s0_reference: float = -1.0,
) -> np.ndarray:
    """
    Evaluate range accrual coupon for all paths.

    Coupon at each observation period is proportional to the fraction of
    simulation days the underlying spent within the range [lower, upper].

    Parameters
    ----------
    paths : np.ndarray
        Price paths, shape (n_paths, n_steps+1).
    obs_indices : np.ndarray
        Observation (coupon payment) dates, shape (n_obs,).
    discount_factors : np.ndarray
        Discount factors at observation dates, shape (n_obs,).
    obs_dt : np.ndarray
        Period lengths between observations, shape (n_obs,).
    notional : float
        Product notional.
    coupon_rate : float
        Maximum annualized coupon rate (paid in full if always in range).
    lower_barrier : float
        Lower range barrier as fraction of initial (e.g., 0.80 = 80%).
    upper_barrier : float
        Upper range barrier as fraction of initial (e.g., 1.20 = 120%).
    s0_reference : float
        Fixed reference spot. If <= 0, uses paths[i, 0].

    Returns
    -------
    np.ndarray
        Present value of range accrual coupons per path, shape (n_paths,).
    """
    n_paths = paths.shape[0]
    n_obs = len(obs_indices)
    pv = np.zeros(n_paths)

    for i in prange(n_paths):
        s0 = s0_reference if s0_reference > 0.0 else paths[i, 0]
        lo = lower_barrier * s0
        hi = upper_barrier * s0
        path_pv = 0.0

        # For each observation period, count fraction of days in range
        start_step = 0
        for j in range(n_obs):
            end_step = obs_indices[j]
            total_steps = end_step - start_step
            if total_steps <= 0:
                start_step = end_step
                continue

            in_range_count = 0
            for step in range(start_step, end_step):
                if paths[i, step] >= lo and paths[i, step] <= hi:
                    in_range_count += 1

            fraction = in_range_count / total_steps
            coupon_j = coupon_rate * obs_dt[j] * notional * fraction
            path_pv += coupon_j * discount_factors[j]
            start_step = end_step

        pv[i] = path_pv

    return pv
