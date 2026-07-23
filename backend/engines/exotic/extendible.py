"""
Extendible-maturity option kernel (Longstaff 1990, Haug 4.14).

The option expires at ``t1`` but its life may be extended to ``T2`` with the
strike reset from ``X1`` to ``X2``.

* holder-extendible: the holder may extend at ``t1`` by paying the fee ``A``.
  Time-t1 value: call ``max(S - X1, c_BSM(S, X2, T2-t1) - A, 0)``; put
  ``max(X1 - S, p_BSM(S, X2, T2-t1) - A, 0)``. Two critical asset prices bound
  the "extend" band: the extend-vs-lapse boundary (``BSM(I, X2, T2-t1) = A``) and
  the exercise-vs-extend boundary (``BSM(I, X2, T2-t1) = eta*(I - X1) + A``).
* writer-extendible: the writer extends for free if the option is out-of-the-
  money at ``t1`` (no fee).

The WRITER variant is ported VERBATIM from Haug's published VBA ``ExtendibleWriter``
(== R fExoticOptions ``WriterExtendibleOption``; eq 4.37/4.38). The HOLDER variant
has NO published reference implementation -- Haug's book gives only formulas
4.35/4.36 (and the printed worked example mis-uses ``X1`` instead of ``X2`` in the
extend boundary). It is implemented here from first principles via the tower
property: the time-t1 payoff is split into exercise / extend-band / lapse regions
whose discounted expectations are univariate (``norm_cdf``) and bivariate
(``cbnd``) probabilities. Validated against an independent Brownian Monte-Carlo of
the contract for both call and put, with and without dividends. (NB: the book's
printed holder call 9.4029 corresponds to the economically-wrong ``X1`` boundary;
the correct value -- this kernel, the book *formula* text, and Monte-Carlo -- is
~9.4233.)

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.engines.exotic._rootfind import (
    critical_value_bsm_combo,
    critical_value_extendible_exercise,
)
from backend.engines.exotic.barrier import _bs_vanilla_price
from backend.utils.math import cbnd, norm_cdf

# Clamp for (+/-)infinite standardized thresholds: norm_cdf(+/-40)=1/0 and
# cbnd(+/-40, x, rho) collapses to the marginal, to machine precision.
_CLAMP = 40.0
# Sentinel for an infinite critical value (S* = +inf); mapped directly to
# +_CLAMP by _a_threshold so the corresponding region vanishes for ANY inputs.
_INF_CRIT = 1e15


@njit(fastmath=True, cache=True)
def _a_threshold(crit: float, S: float, t1: float, drift: float, vt1: float) -> float:
    """Standardized threshold for ``S(t1) < crit`` (clamped to +/-_CLAMP).

    The ``crit <= 0`` (S* = 0) and ``crit >= _INF_CRIT`` (S* = +inf) sentinels map
    straight to -/+ _CLAMP so their region saturates regardless of S/sigma/maturity.
    """
    if crit <= 0.0:
        return -_CLAMP
    if crit >= _INF_CRIT:
        return _CLAMP
    a = (math.log(crit / S) - drift * t1) / vt1
    if a > _CLAMP:
        return _CLAMP
    if a < -_CLAMP:
        return -_CLAMP
    return a


@njit(fastmath=True, cache=True)
def extendible_price(
    S: float,
    X1: float,
    X2: float,
    t1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    A: float,
    is_call: bool,
    is_holder: bool,
) -> float:
    """
    Price an extendible-maturity option (Longstaff 1990).

    Parameters
    ----------
    S : float
        Spot price.
    X1 : float
        Initial strike.
    X2 : float
        Extended strike.
    t1 : float
        Initial expiry (``0 < t1 < T2``).
    T2 : float
        Extended expiry.
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.
    A : float
        Holder extension fee (>= 0); ignored for the writer variant.
    is_call : bool
        Call or put.
    is_holder : bool
        Holder-extendible (True) or writer-extendible (False).

    Returns
    -------
    float
        Option price.
    """
    if sigma <= 0.0 or t1 <= 0.0 or T2 <= t1 or S <= 0.0:
        return 0.0

    b = r - q
    half_vsq = 0.5 * sigma * sigma
    drift = b - half_vsq
    sqt1 = math.sqrt(t1)
    sqT2 = math.sqrt(T2)
    vt1 = sigma * sqt1
    vT2 = sigma * sqT2
    rho = math.sqrt(t1 / T2)
    er_t1 = math.exp((b - r) * t1)
    d_t1 = math.exp(-r * t1)
    er_T2 = math.exp((b - r) * T2)
    d_T2 = math.exp(-r * T2)

    if not is_holder:
        # Writer-extendible (Haug 4.37/4.38), no fee.
        z1 = (math.log(S / X2) + (b + half_vsq) * T2) / vT2
        z2 = (math.log(S / X1) + (b + half_vsq) * t1) / vt1
        if is_call:
            return (
                _bs_vanilla_price(S, X1, t1, r, q, sigma, True)
                + S * er_T2 * cbnd(z1, -z2, -rho)
                - X2 * d_T2 * cbnd(z1 - vT2, -z2 + vt1, -rho)
            )
        return (
            _bs_vanilla_price(S, X1, t1, r, q, sigma, False)
            + X2 * d_T2 * cbnd(-z1 + vT2, z2 - vt1, -rho)
            - S * er_T2 * cbnd(-z1, z2, -rho)
        )

    # Holder-extendible (Haug 4.35/4.36) via the tower property, with the
    # economically-correct X2 extend boundary.
    tau = T2 - t1
    hi = max(S, X1, X2) * 1000.0
    if is_call:
        # I1: extend-vs-lapse boundary c_BSM(I1, X2, tau) = A (increasing).
        if A <= 0.0:
            i1 = 0.0
        else:
            i1 = critical_value_bsm_combo(
                1.0, X2, tau, True, 0.0, X2, tau, False, A, r, q, sigma, 1e-8, hi
            )
        # I2: exercise-vs-extend boundary. As S -> inf, (extend - exercise) ~
        # S*(exp((b-r)tau) - 1) + X1 - X2*exp(-r*tau) - A, so exercise eventually
        # dominates (finite I2) iff b < r (q > 0); for b > r (q < 0) extend always
        # wins (I2 = +inf); for b == r (q == 0) it is Haug's level condition.
        if q < 0.0:
            i2 = _INF_CRIT
        elif q == 0.0 and A < X1 - X2 * math.exp(-r * tau):
            i2 = _INF_CRIT
        else:
            i2 = critical_value_extendible_exercise(
                True, X2, tau, X1, A, r, q, sigma, 1e-8, hi
            )
    else:
        # I1: exercise-vs-extend boundary p_BSM(I1, X2, tau) = X1 - I1 + A.
        i1 = critical_value_extendible_exercise(
            False, X2, tau, X1, A, r, q, sigma, 1e-8, hi
        )
        # I2: extend-vs-lapse boundary p_BSM(I2, X2, tau) = A (decreasing);
        # +inf when A == 0 (the put is always extended above I1).
        if A <= 0.0:
            i2 = _INF_CRIT
        else:
            i2 = critical_value_bsm_combo(
                1.0, X2, tau, False, 0.0, X2, tau, False, A, r, q, sigma, 1e-8, hi
            )

    # Empty extend band (fee so high the holder never extends): collapse to the
    # plain time-t1 option (the I1 == I2 limit).
    if i1 >= i2:
        return _bs_vanilla_price(S, X1, t1, r, q, sigma, is_call)

    a1 = _a_threshold(i1, S, t1, drift, vt1)
    a2 = _a_threshold(i2, S, t1, drift, vt1)
    c2 = (math.log(X2 / S) - drift * T2) / vT2  # S(T2) < X2  <=>  W2 < c2

    if is_call:
        # Exercise (S > I2): pays S - X1. Extend band pays c_BSM (S(T2) > X2).
        reg_ex = S * er_t1 * norm_cdf(vt1 - a2) - X1 * d_t1 * norm_cdf(-a2)
        reg_fee = -A * d_t1 * (norm_cdf(a2) - norm_cdf(a1))
        term_S = (
            S
            * er_T2
            * (cbnd(a2 - vt1, -(c2 - vT2), -rho) - cbnd(a1 - vt1, -(c2 - vT2), -rho))
        )
        term_X2 = -X2 * d_T2 * (cbnd(a2, -c2, -rho) - cbnd(a1, -c2, -rho))
        return reg_ex + reg_fee + term_S + term_X2

    # Exercise (S < I1): pays X1 - S. Extend band pays p_BSM (S(T2) < X2).
    reg_ex = X1 * d_t1 * norm_cdf(a1) - S * er_t1 * norm_cdf(a1 - vt1)
    reg_fee = -A * d_t1 * (norm_cdf(a2) - norm_cdf(a1))
    term_X2 = X2 * d_T2 * (cbnd(a2, c2, rho) - cbnd(a1, c2, rho))
    term_S = (
        -S * er_T2 * (cbnd(a2 - vt1, c2 - vT2, rho) - cbnd(a1 - vt1, c2 - vT2, rho))
    )
    return reg_ex + reg_fee + term_X2 + term_S
