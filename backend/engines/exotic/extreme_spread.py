"""
Extreme-spread option kernel (Bermin 1996b, Haug 4.15.5).

The option's life is split into a first period ``[0, t1]`` and a second period
``[t1, T2]``. With ``Smax_i`` / ``Smin_i`` the extremes of period ``i``:

* extreme-spread call pays ``max(Smax_2 - Smax_1, 0)``;
* extreme-spread put  pays ``max(Smin_1 - Smin_2, 0)``;
* reverse extreme-spread call pays ``max(Smin_2 - Smin_1, 0)``;
* reverse extreme-spread put  pays ``max(Smax_1 - Smax_2, 0)``.

The first-period extreme is seeded by the observed running extreme carried into
the contract: ``S_max`` for the variants whose payoff references a maximum,
``S_min`` for those referencing a minimum (a freshly issued option carries the
spot as that extreme).

Transcribed from Haug's published formulas 4.49/4.50, cross-read against the R
``ExtremeSpreadOption`` (fExoticOptions, Diethelm Wuertz) for the two ambiguous
reflection-term arguments. ONE deliberate divergence from the R port: the
first-period discount factor uses the book's ``exp(-b*(T2-t1))``, not the R
port's ``exp(-r*(T2-t1))``. The two agree on all of Haug's Table 4-11 (which
fixes ``b == r``) but the R form is wrong for ``b != r`` -- it makes the price
*rise* with the dividend yield. The ``-b`` form matches a Brownian-bridge
Monte-Carlo for every dividend yield and is consistent with the sibling
partial-time fixed-strike lookback (4.47/4.48). Validated against Haug's Table
4-11 (all 24 cells, max abs error ~1.9e-4 -- the book's 4-decimal rounding) and
an independent Monte-Carlo cross-check at ``b != r``.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

from numba import njit

from backend.utils.math import norm_cdf


@njit(fastmath=True, cache=True)
def extreme_spread_price(
    S: float,
    S_min: float,
    S_max: float,
    t1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    is_reverse: bool,
) -> float:
    """
    Price an extreme-spread / reverse extreme-spread option (Bermin 1996b).

    Parameters
    ----------
    S : float
        Spot price.
    S_min, S_max : float
        Observed running minimum / maximum carried into the contract (the
        first-period extreme baseline). Pass ``S`` for both on a fresh option.
    t1 : float
        End of the first period / start of the second (``0 < t1 < T2``).
    T2 : float
        Time to expiration.
    r, q, sigma : float
        Rate, dividend yield, volatility. Cost of carry ``b = r - q``.
    is_call : bool
        Call (``eta = +1``) or put (``eta = -1``).
    is_reverse : bool
        Reverse extreme-spread (``kappa = -1``) or plain (``kappa = +1``).

    Returns
    -------
    float
        Option price.
    """
    if T2 <= 0.0 or sigma <= 0.0:
        return 0.0
    b = r - q
    # The closed form divides by 2b; nudge a (near-)zero carry off zero so the
    # b -> 0 limit is approximated rather than dividing by 0 (no book b=0 form).
    if -1e-8 < b < 1e-8:
        b = 1e-8

    # The formula divides by sqrt(t1) and sqrt(T2 - t1); keep t1 strictly inside
    # (0, T2). t1 -> 0 reproduces the degenerate single-point first period (the
    # standard-lookback-spread limit); 1e-10 is comfortably converged.
    if t1 <= 0.0:
        time1 = 1e-10
    elif t1 >= T2:
        time1 = T2 * (1.0 - 1e-10)
    else:
        time1 = t1

    v = sigma
    vsq = v * v
    Time = T2

    eta = 1.0 if is_call else -1.0
    kappa = -1.0 if is_reverse else 1.0
    # M0 = S_max when kappa*eta == +1 (extreme call / reverse put), else S_min.
    Mo = S_max if kappa * eta > 0.0 else S_min

    mu1 = b - 0.5 * vsq
    mu = mu1 + vsq  # = b + 0.5 * vsq
    m = math.log(Mo / S)

    carry_df = math.exp((b - r) * Time)
    disc = math.exp(-r * Time)
    # Discount on the first-period (length t1) leg: the book formulas 4.49/4.50
    # use exp(-b*(T2-t1)). The widely-mirrored fExoticOptions R port writes
    # exp(-r*(T2-t1)) here -- identical when b == r (all of Haug's Table 4-11),
    # but wrong for b != r: it makes the price rise with the dividend yield. The
    # -b form matches a Brownian-bridge Monte-Carlo for every q and is
    # consistent with the sibling partial-time fixed lookback (4.47/4.48).
    disc_t1 = math.exp(-b * (Time - time1))
    reflect = vsq / (2.0 * b) * math.exp(2.0 * mu1 * m / vsq)
    box = 1.0 + vsq / (2.0 * b)
    sqrt_Time = math.sqrt(Time)
    sqrt_t1 = math.sqrt(time1)

    if kappa > 0.0:
        # Extreme spread (Haug 4.49).
        return eta * (
            S * carry_df * box * norm_cdf(eta * (-m + mu * Time) / (v * sqrt_Time))
            - disc_t1
            * S
            * carry_df
            * box
            * norm_cdf(eta * (-m + mu * time1) / (v * sqrt_t1))
            + disc * Mo * norm_cdf(eta * (m - mu1 * Time) / (v * sqrt_Time))
            - disc * Mo * reflect * norm_cdf(eta * (-m - mu1 * Time) / (v * sqrt_Time))
            - disc * Mo * norm_cdf(eta * (m - mu1 * time1) / (v * sqrt_t1))
            + disc * Mo * reflect * norm_cdf(eta * (-m - mu1 * time1) / (v * sqrt_t1))
        )

    # Reverse extreme spread (Haug 4.50): note the leading -eta.
    sqrt_Tt = math.sqrt(Time - time1)
    return -eta * (
        S * carry_df * box * norm_cdf(eta * (m - mu * Time) / (v * sqrt_Time))
        + disc * Mo * norm_cdf(eta * (-m + mu1 * Time) / (v * sqrt_Time))
        - disc * Mo * reflect * norm_cdf(eta * (m + mu1 * Time) / (v * sqrt_Time))
        - S * carry_df * box * norm_cdf(eta * (-mu * (Time - time1)) / (v * sqrt_Tt))
        - disc_t1
        * S
        * carry_df
        * (1.0 - vsq / (2.0 * b))
        * norm_cdf(eta * (mu1 * (Time - time1)) / (v * sqrt_Tt))
    )
