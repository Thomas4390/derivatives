"""
Critical-value root finder for closed-form exotic pricers (Haug catalog).

Several Haug pricers need the critical asset price ``S*`` at a decision date
where two Black-Scholes-Merton legs balance:

* complex chooser (§4.12.2): ``c_BSM(S*, Xc, Tc - t) = p_BSM(S*, Xp, Tp - t)``
* compound / Geske (§4.13): ``c_or_p_BSM(S*, X1, T2 - t1) = X2``
* extendible (§4.14): an analogous BSM-vs-level balance.

Each condition has the shape

    coef_a * BSM(S*, Xa, tau_a, is_call_a)
    + coef_b * BSM(S*, Xb, tau_b, is_call_b) = target,

monotone in ``S*``. It is solved with a safeguarded **bisection**: the objective
is two cheap BSM evaluations, and for a bracketed monotone function bisection is
bulletproof -- no derivative, no regula-falsi stagnation, and (unlike passing a
Numba function as an argument) it keeps ``cache=True`` on every caller.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

from numba import njit

from backend.engines.exotic.barrier import _bs_vanilla_price
from backend.utils.constants.exotic import ROOTFIND_MAX_ITER, ROOTFIND_TOL


@njit(fastmath=True, cache=True)
def _bsm_combo(
    S: float,
    coef_a: float,
    Xa: float,
    tau_a: float,
    is_call_a: bool,
    coef_b: float,
    Xb: float,
    tau_b: float,
    is_call_b: bool,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """``coef_a*BSM(S,Xa,tau_a,..) + coef_b*BSM(S,Xb,tau_b,..)`` (one leg may have coef 0)."""
    return coef_a * _bs_vanilla_price(
        S, Xa, tau_a, r, q, sigma, is_call_a
    ) + coef_b * _bs_vanilla_price(S, Xb, tau_b, r, q, sigma, is_call_b)


@njit(fastmath=True, cache=True)
def critical_value_bsm_combo(
    coef_a: float,
    Xa: float,
    tau_a: float,
    is_call_a: bool,
    coef_b: float,
    Xb: float,
    tau_b: float,
    is_call_b: bool,
    target: float,
    r: float,
    q: float,
    sigma: float,
    lo: float,
    hi: float,
) -> float:
    """
    Solve ``combo(S*) = target`` for ``S*`` in ``[lo, hi]`` by safeguarded bisection.

    ``combo`` is the two-leg BSM combination of :func:`_bsm_combo`; the caller
    must supply a bracket over which it is monotone and (for a guaranteed
    straddle) where ``combo(lo) <= target <= combo(hi)``. If the bracket does not
    straddle the root, the endpoint with the smaller residual is returned.

    Parameters
    ----------
    coef_a, Xa, tau_a, is_call_a : leg A coefficient, strike, maturity, call flag.
    coef_b, Xb, tau_b, is_call_b : leg B coefficient, strike, maturity, call flag
        (pass ``coef_b = 0.0`` for a single-leg condition such as the compound option).
    target : float
        Right-hand-side level.
    r, q, sigma : float
        Rate, dividend yield, volatility.
    lo, hi : float
        Bracket bounds (``0 < lo < hi``).

    Returns
    -------
    float
        The critical asset price ``S*``.
    """
    fa = (
        _bsm_combo(
            lo, coef_a, Xa, tau_a, is_call_a, coef_b, Xb, tau_b, is_call_b, r, q, sigma
        )
        - target
    )
    fb = (
        _bsm_combo(
            hi, coef_a, Xa, tau_a, is_call_a, coef_b, Xb, tau_b, is_call_b, r, q, sigma
        )
        - target
    )
    if fa == 0.0:
        return lo
    if fb == 0.0:
        return hi
    if (fa > 0.0) == (fb > 0.0):
        # Bracket does not straddle the root: return the nearer endpoint.
        return lo if abs(fa) < abs(fb) else hi

    a = lo
    b = hi
    for _ in range(ROOTFIND_MAX_ITER):
        m = 0.5 * (a + b)
        fm = (
            _bsm_combo(
                m,
                coef_a,
                Xa,
                tau_a,
                is_call_a,
                coef_b,
                Xb,
                tau_b,
                is_call_b,
                r,
                q,
                sigma,
            )
            - target
        )
        if abs(fm) < ROOTFIND_TOL or 0.5 * (b - a) < ROOTFIND_TOL * (1.0 + abs(m)):
            return m
        if (fa > 0.0) != (fm > 0.0):
            b = m
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)


@njit(fastmath=True, cache=True)
def critical_value_extendible_exercise(
    is_call: bool,
    X2: float,
    tau: float,
    X1: float,
    A: float,
    r: float,
    q: float,
    sigma: float,
    lo: float,
    hi: float,
) -> float:
    """
    Exercise-vs-extend boundary for the holder-extendible option (Longstaff 1990).

    Solve ``BSM(S, X2, tau, is_call) = eta*(S - X1) + A`` for ``S`` in ``[lo, hi]``
    where ``eta = +1`` (call) / ``-1`` (put) -- the spot at which exercising now
    (``eta*(S - X1)``) is worth exactly as much as extending (``BSM(S, X2, tau) -
    A``). The residual is monotone, so a safeguarded bisection converges; if the
    bracket does not straddle a root the nearer endpoint is returned (the caller
    detects the ``S* = +inf`` no-exercise case separately).

    Parameters
    ----------
    is_call : bool
        Underlying-option call/put flag (sets ``eta``).
    X2 : float
        Extended strike.
    tau : float
        Remaining life of the extended option (``T2 - t1``).
    X1 : float
        Initial strike.
    A : float
        Extension fee.
    r, q, sigma : float
        Rate, dividend yield, volatility.
    lo, hi : float
        Bracket bounds (``0 < lo < hi``).

    Returns
    -------
    float
        The exercise-boundary asset price.
    """
    eta = 1.0 if is_call else -1.0
    fa = _bs_vanilla_price(lo, X2, tau, r, q, sigma, is_call) - eta * (lo - X1) - A
    fb = _bs_vanilla_price(hi, X2, tau, r, q, sigma, is_call) - eta * (hi - X1) - A
    if fa == 0.0:
        return lo
    if fb == 0.0:
        return hi
    if (fa > 0.0) == (fb > 0.0):
        return lo if abs(fa) < abs(fb) else hi
    a = lo
    b = hi
    for _ in range(ROOTFIND_MAX_ITER):
        m = 0.5 * (a + b)
        fm = _bs_vanilla_price(m, X2, tau, r, q, sigma, is_call) - eta * (m - X1) - A
        if abs(fm) < ROOTFIND_TOL or 0.5 * (b - a) < ROOTFIND_TOL * (1.0 + abs(m)):
            return m
        if (fa > 0.0) != (fm > 0.0):
            b = m
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)
