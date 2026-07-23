"""
Numba-compiled exotic Greeks / price kernels
============================================

Stateless ``@njit`` option-type dispatch + central finite-difference Greeks and
price kernels for exotic options, extracted verbatim from ``engine.py``. Kept in
one module so the inter-kernel njit calls stay intra-module.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from backend.engines.exotic._option_types import (
    ASIAN_GEO,
    ASSET_OR_NOTHING,
    BARRIER,
    CHOOSER,
    DIGITAL,
    GAP,
    LOOKBACK_FIXED,
    LOOKBACK_FLOATING,
    POWER,
)
from backend.engines.exotic.asian import asian_geometric_price
from backend.engines.exotic.asset_or_nothing import asset_or_nothing_price
from backend.engines.exotic.barrier import _bs_vanilla_price, barrier_option_price
from backend.engines.exotic.chooser import chooser_price
from backend.engines.exotic.digital import digital_price
from backend.engines.exotic.gap import gap_option_price
from backend.engines.exotic.lookback import (
    lookback_fixed_price,
    lookback_floating_price,
)
from backend.engines.exotic.power import power_option_price
from backend.utils.math import DAYS_PER_YEAR


@njit(fastmath=True, cache=True)
def _exotic_price(
    option_type: int,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    H: float,
    M_min: float,
    M_max: float,
    is_knock_in: bool,
    is_up: bool,
    rebate: float,
    payout: float,
    extra1: float = 0.0,
) -> float:
    """Dispatch to the correct exotic pricing kernel by option type.

    extra1 encodes type-specific parameters:
        CHOOSER: t_c (choice time)
        POWER: n (power exponent)
        GAP: K2 (trigger strike)
    """
    if option_type == BARRIER:
        return barrier_option_price(
            S, K, H, T, r, q, sigma, is_call, is_knock_in, is_up, rebate
        )
    if option_type == ASIAN_GEO:
        return asian_geometric_price(S, K, T, r, q, sigma, is_call)
    if option_type == DIGITAL:
        return digital_price(S, K, T, r, q, sigma, is_call, payout)
    if option_type == LOOKBACK_FIXED:
        return lookback_fixed_price(S, K, M_min, M_max, T, r, q, sigma, is_call)
    if option_type == LOOKBACK_FLOATING:
        return lookback_floating_price(S, M_min, M_max, T, r, q, sigma, is_call)
    if option_type == CHOOSER:
        return chooser_price(S, K, T, extra1, r, q, sigma)
    if option_type == ASSET_OR_NOTHING:
        return asset_or_nothing_price(S, K, T, r, q, sigma, is_call)
    if option_type == POWER:
        return power_option_price(S, K, T, r, q, sigma, is_call, extra1)
    if option_type == GAP:
        return gap_option_price(S, K, extra1, T, r, q, sigma, is_call)
    return 0.0


@njit(fastmath=True, cache=True)
def exotic_greeks_batch(
    option_type: int,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    H: float,
    M_min: float,
    M_max: float,
    is_knock_in: bool,
    is_up: bool,
    rebate: float,
    payout: float,
    extra1: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """Compute (price, delta, gamma, vega, theta, rho) with single-dispatch."""
    dS = S * 0.01
    dV = 0.01
    dT = 1.0 / DAYS_PER_YEAR
    dR = 0.0001
    sigma_dn = max(sigma - dV, 1e-8)
    has_theta = dT < T
    T_dn = T - dT if has_theta else T

    if option_type == BARRIER:
        price = barrier_option_price(
            S, K, H, T, r, q, sigma, is_call, is_knock_in, is_up, rebate
        )
        p_up = barrier_option_price(
            S + dS, K, H, T, r, q, sigma, is_call, is_knock_in, is_up, rebate
        )
        p_dn = barrier_option_price(
            S - dS, K, H, T, r, q, sigma, is_call, is_knock_in, is_up, rebate
        )
        p_vu = barrier_option_price(
            S, K, H, T, r, q, sigma + dV, is_call, is_knock_in, is_up, rebate
        )
        p_vd = barrier_option_price(
            S, K, H, T, r, q, sigma_dn, is_call, is_knock_in, is_up, rebate
        )
        p_td = (
            barrier_option_price(
                S, K, H, T_dn, r, q, sigma, is_call, is_knock_in, is_up, rebate
            )
            if has_theta
            else price
        )
        p_ru = barrier_option_price(
            S, K, H, T, r + dR, q, sigma, is_call, is_knock_in, is_up, rebate
        )
        p_rd = barrier_option_price(
            S, K, H, T, r - dR, q, sigma, is_call, is_knock_in, is_up, rebate
        )
    elif option_type == ASIAN_GEO:
        price = asian_geometric_price(S, K, T, r, q, sigma, is_call)
        p_up = asian_geometric_price(S + dS, K, T, r, q, sigma, is_call)
        p_dn = asian_geometric_price(S - dS, K, T, r, q, sigma, is_call)
        p_vu = asian_geometric_price(S, K, T, r, q, sigma + dV, is_call)
        p_vd = asian_geometric_price(S, K, T, r, q, sigma_dn, is_call)
        p_td = (
            asian_geometric_price(S, K, T_dn, r, q, sigma, is_call)
            if has_theta
            else price
        )
        p_ru = asian_geometric_price(S, K, T, r + dR, q, sigma, is_call)
        p_rd = asian_geometric_price(S, K, T, r - dR, q, sigma, is_call)
    elif option_type == DIGITAL:
        price = digital_price(S, K, T, r, q, sigma, is_call, payout)
        p_up = digital_price(S + dS, K, T, r, q, sigma, is_call, payout)
        p_dn = digital_price(S - dS, K, T, r, q, sigma, is_call, payout)
        p_vu = digital_price(S, K, T, r, q, sigma + dV, is_call, payout)
        p_vd = digital_price(S, K, T, r, q, sigma_dn, is_call, payout)
        p_td = (
            digital_price(S, K, T_dn, r, q, sigma, is_call, payout)
            if has_theta
            else price
        )
        p_ru = digital_price(S, K, T, r + dR, q, sigma, is_call, payout)
        p_rd = digital_price(S, K, T, r - dR, q, sigma, is_call, payout)
    elif option_type == LOOKBACK_FIXED:
        price = lookback_fixed_price(S, K, M_min, M_max, T, r, q, sigma, is_call)
        p_up = lookback_fixed_price(S + dS, K, M_min, M_max, T, r, q, sigma, is_call)
        p_dn = lookback_fixed_price(S - dS, K, M_min, M_max, T, r, q, sigma, is_call)
        p_vu = lookback_fixed_price(S, K, M_min, M_max, T, r, q, sigma + dV, is_call)
        p_vd = lookback_fixed_price(S, K, M_min, M_max, T, r, q, sigma_dn, is_call)
        p_td = (
            lookback_fixed_price(S, K, M_min, M_max, T_dn, r, q, sigma, is_call)
            if has_theta
            else price
        )
        p_ru = lookback_fixed_price(S, K, M_min, M_max, T, r + dR, q, sigma, is_call)
        p_rd = lookback_fixed_price(S, K, M_min, M_max, T, r - dR, q, sigma, is_call)
    elif option_type == LOOKBACK_FLOATING:
        # Adjust extremes for spot bumps: M_min must be <= S, M_max must be >= S
        if is_call:
            mi_up = M_min  # S+dS > M_min, minimum unchanged
            mi_dn = min(M_min, S - dS)  # new minimum if S drops below M_min
            ma_up = M_max
            ma_dn = M_max
        else:
            mi_up = M_min
            mi_dn = M_min
            ma_up = max(M_max, S + dS)  # new maximum if S rises above M_max
            ma_dn = M_max  # S-dS < M_max, maximum unchanged
        price = lookback_floating_price(S, M_min, M_max, T, r, q, sigma, is_call)
        p_up = lookback_floating_price(S + dS, mi_up, ma_up, T, r, q, sigma, is_call)
        p_dn = lookback_floating_price(S - dS, mi_dn, ma_dn, T, r, q, sigma, is_call)
        p_vu = lookback_floating_price(S, M_min, M_max, T, r, q, sigma + dV, is_call)
        p_vd = lookback_floating_price(S, M_min, M_max, T, r, q, sigma_dn, is_call)
        p_td = (
            lookback_floating_price(S, M_min, M_max, T_dn, r, q, sigma, is_call)
            if has_theta
            else price
        )
        p_ru = lookback_floating_price(S, M_min, M_max, T, r + dR, q, sigma, is_call)
        p_rd = lookback_floating_price(S, M_min, M_max, T, r - dR, q, sigma, is_call)
    elif option_type == CHOOSER:
        price = chooser_price(S, K, T, extra1, r, q, sigma)
        p_up = chooser_price(S + dS, K, T, extra1, r, q, sigma)
        p_dn = chooser_price(S - dS, K, T, extra1, r, q, sigma)
        p_vu = chooser_price(S, K, T, extra1, r, q, sigma + dV)
        p_vd = chooser_price(S, K, T, extra1, r, q, sigma_dn)
        p_td = chooser_price(S, K, T_dn, extra1, r, q, sigma) if has_theta else price
        p_ru = chooser_price(S, K, T, extra1, r + dR, q, sigma)
        p_rd = chooser_price(S, K, T, extra1, r - dR, q, sigma)
    elif option_type == ASSET_OR_NOTHING:
        price = asset_or_nothing_price(S, K, T, r, q, sigma, is_call)
        p_up = asset_or_nothing_price(S + dS, K, T, r, q, sigma, is_call)
        p_dn = asset_or_nothing_price(S - dS, K, T, r, q, sigma, is_call)
        p_vu = asset_or_nothing_price(S, K, T, r, q, sigma + dV, is_call)
        p_vd = asset_or_nothing_price(S, K, T, r, q, sigma_dn, is_call)
        p_td = (
            asset_or_nothing_price(S, K, T_dn, r, q, sigma, is_call)
            if has_theta
            else price
        )
        p_ru = asset_or_nothing_price(S, K, T, r + dR, q, sigma, is_call)
        p_rd = asset_or_nothing_price(S, K, T, r - dR, q, sigma, is_call)
    elif option_type == POWER:
        price = power_option_price(S, K, T, r, q, sigma, is_call, extra1)
        p_up = power_option_price(S + dS, K, T, r, q, sigma, is_call, extra1)
        p_dn = power_option_price(S - dS, K, T, r, q, sigma, is_call, extra1)
        p_vu = power_option_price(S, K, T, r, q, sigma + dV, is_call, extra1)
        p_vd = power_option_price(S, K, T, r, q, sigma_dn, is_call, extra1)
        p_td = (
            power_option_price(S, K, T_dn, r, q, sigma, is_call, extra1)
            if has_theta
            else price
        )
        p_ru = power_option_price(S, K, T, r + dR, q, sigma, is_call, extra1)
        p_rd = power_option_price(S, K, T, r - dR, q, sigma, is_call, extra1)
    elif option_type == GAP:
        price = gap_option_price(S, K, extra1, T, r, q, sigma, is_call)
        p_up = gap_option_price(S + dS, K, extra1, T, r, q, sigma, is_call)
        p_dn = gap_option_price(S - dS, K, extra1, T, r, q, sigma, is_call)
        p_vu = gap_option_price(S, K, extra1, T, r, q, sigma + dV, is_call)
        p_vd = gap_option_price(S, K, extra1, T, r, q, sigma_dn, is_call)
        p_td = (
            gap_option_price(S, K, extra1, T_dn, r, q, sigma, is_call)
            if has_theta
            else price
        )
        p_ru = gap_option_price(S, K, extra1, T, r + dR, q, sigma, is_call)
        p_rd = gap_option_price(S, K, extra1, T, r - dR, q, sigma, is_call)
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    delta = (p_up - p_dn) / (2.0 * dS)
    gamma = (p_up - 2.0 * price + p_dn) / (dS * dS)
    vega = (p_vu - p_vd) / (2.0 * dV) / 100.0
    # Per-(calendar-)day theta to match bs_theta and the portfolio path: dT is a
    # one-day bump (1/365), so the per-day decay is just bumped-minus-base.
    # Dividing by dT here returned a per-YEAR theta, inconsistent with vanilla.
    theta = (p_td - price) if has_theta else 0.0
    rho = (p_ru - p_rd) / (2.0 * dR) / 100.0

    return price, delta, gamma, vega, theta, rho


@njit(fastmath=True, cache=True)
def exotic_calculate_greeks(
    option_type: int,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    H: float,
    M_min: float,
    M_max: float,
    is_knock_in: bool,
    is_up: bool,
    rebate: float,
    payout: float,
    extra1: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate option Greeks using finite differences for all exotic types.

    Parameters
    ----------
    option_type : int
        Option type constant (BARRIER, ASIAN_GEO, DIGITAL, etc.)
    S, K, T, r, q, sigma : float
        Standard option parameters (q = continuous dividend yield)
    is_call : bool
        True for call
    H : float
        Barrier level (for barrier options)
    M_min, M_max : float
        Running min/max (for lookback options)
    is_knock_in, is_up : bool
        Barrier type flags
    rebate, payout : float
        Rebate (barrier) or payout (digital)
    extra1 : float
        Type-specific parameter (t_c for chooser, n for power, K2 for gap)

    Returns
    -------
    tuple of 6 floats
        (price, delta, gamma, vega, theta, rho)
    """
    return exotic_greeks_batch(
        option_type,
        S,
        K,
        T,
        r,
        q,
        sigma,
        is_call,
        H,
        M_min,
        M_max,
        is_knock_in,
        is_up,
        rebate,
        payout,
        extra1,
    )


@njit(parallel=True, fastmath=True, cache=True)
def exotic_greeks_surface(
    option_type: int,
    spot_range: np.ndarray,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    H: float,
    M_min: float,
    M_max: float,
    is_knock_in: bool,
    is_up: bool,
    rebate: float,
    payout: float,
    extra1: float = 0.0,
) -> np.ndarray:
    """Compute Greeks over spot range with parallel execution."""
    n = len(spot_range)
    result = np.empty((n, 6))
    for i in prange(n):
        S = spot_range[i]
        if option_type == LOOKBACK_FLOATING:
            # Floating lookback: extremes track spot relative to reference
            mi = min(S, M_min) if M_min > 0 else S
            ma = max(S, M_max) if M_max > 0 else S
        elif option_type == LOOKBACK_FIXED:
            mi = S
            ma = S
        else:
            mi = M_min
            ma = M_max
        (
            result[i, 0],
            result[i, 1],
            result[i, 2],
            result[i, 3],
            result[i, 4],
            result[i, 5],
        ) = exotic_greeks_batch(
            option_type,
            S,
            K,
            T,
            r,
            q,
            sigma,
            is_call,
            H,
            mi,
            ma,
            is_knock_in,
            is_up,
            rebate,
            payout,
            extra1,
        )
    return result


@njit(parallel=True, fastmath=True, cache=True)
def exotic_price_surface(
    option_type: int,
    spot_range: np.ndarray,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    H: float,
    M_min: float,
    M_max: float,
    is_knock_in: bool,
    is_up: bool,
    rebate: float,
    payout: float,
    extra1: float = 0.0,
) -> np.ndarray:
    """Compute prices over spot range with parallel execution."""
    n = len(spot_range)
    result = np.empty(n)
    for i in prange(n):
        S = spot_range[i]
        if option_type == LOOKBACK_FLOATING:
            mi = min(S, M_min) if M_min > 0 else S
            ma = max(S, M_max) if M_max > 0 else S
        elif option_type == LOOKBACK_FIXED:
            mi = S
            ma = S
        else:
            mi = M_min
            ma = M_max
        result[i] = _exotic_price(
            option_type,
            S,
            K,
            T,
            r,
            q,
            sigma,
            is_call,
            H,
            mi,
            ma,
            is_knock_in,
            is_up,
            rebate,
            payout,
            extra1,
        )
    return result


@njit(parallel=True, fastmath=True, cache=True)
def exotic_price_param_sweep(
    option_type: int,
    param_values: np.ndarray,
    param_is_vol: bool,
    S: float,
    K: float,
    T_base: float,
    r: float,
    q: float,
    sigma_base: float,
    is_call: bool,
    H: float,
    M_min: float,
    M_max: float,
    is_knock_in: bool,
    is_up: bool,
    rebate: float,
    payout: float,
    extra1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exotic and vanilla prices over a parameter sweep."""
    n = len(param_values)
    exotic_prices = np.empty(n)
    vanilla_prices = np.empty(n)
    for i in prange(n):
        T = T_base if param_is_vol else param_values[i]
        sig = param_values[i] if param_is_vol else sigma_base
        mi = (
            S
            if (option_type == LOOKBACK_FIXED or option_type == LOOKBACK_FLOATING)
            else M_min
        )
        ma = (
            S
            if (option_type == LOOKBACK_FIXED or option_type == LOOKBACK_FLOATING)
            else M_max
        )
        exotic_prices[i] = _exotic_price(
            option_type,
            S,
            K,
            T,
            r,
            q,
            sig,
            is_call,
            H,
            mi,
            ma,
            is_knock_in,
            is_up,
            rebate,
            payout,
            extra1,
        )
        vanilla_prices[i] = _bs_vanilla_price(S, K, T, r, q, sig, is_call)
    return exotic_prices, vanilla_prices
