"""
Binary-barrier option kernel (Reiner-Rubinstein 1991, Haug 4.19.5).

Twenty-eight one-touch / barrier-digital types, split into two families:

- **cash-or-nothing barrier** options pay a fixed cash amount ``K`` (or nothing)
  depending on whether the barrier ``H`` has been hit;
- **asset-or-nothing barrier** options pay one unit of the asset (or nothing).

Each family covers in/out x down/up x (at-hit / at-expiration / strike-gated
call / strike-gated put). A single closed form is assembled from nine building
blocks ``A1..A5`` / ``B1..B4`` selected per type by the integer flags
``eta``/``phi`` and the strike-vs-barrier branch (``X`` ><  ``H``).

Ported verbatim from Haug's published VBA ``BinaryBarrier`` (Digitals.bas). The
kernel reproduces Haug's Table 4-22 to all printed digits (validated against the
*corrected* table -- see the test module for two transcription errors in the
printed book table that the closed form exposes).

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from backend.utils.math import norm_cdf

# --- Binary-barrier type codes (Haug 4.19.5, items 1..28) ---
BB_DOWN_IN_CASH_ATHIT = 1  # down-and-in cash-(at-hit)-or-nothing      (S > H)
BB_UP_IN_CASH_ATHIT = 2  # up-and-in cash-(at-hit)-or-nothing          (S < H)
BB_DOWN_IN_ASSET_ATHIT = 3  # down-and-in asset-(at-hit)-or-nothing    (S > H)
BB_UP_IN_ASSET_ATHIT = 4  # up-and-in asset-(at-hit)-or-nothing        (S < H)
BB_DOWN_IN_CASH_ATEXP = 5  # down-and-in cash-(at-expiration)          (S > H)
BB_UP_IN_CASH_ATEXP = 6  # up-and-in cash-(at-expiration)              (S < H)
BB_DOWN_IN_ASSET_ATEXP = 7  # down-and-in asset-(at-expiration)        (S > H)
BB_UP_IN_ASSET_ATEXP = 8  # up-and-in asset-(at-expiration)            (S < H)
BB_DOWN_OUT_CASH = 9  # down-and-out cash-or-nothing                   (S > H)
BB_UP_OUT_CASH = 10  # up-and-out cash-or-nothing                      (S < H)
BB_DOWN_OUT_ASSET = 11  # down-and-out asset-or-nothing                (S > H)
BB_UP_OUT_ASSET = 12  # up-and-out asset-or-nothing                    (S < H)
BB_DOWN_IN_CASH_CALL = 13  # down-and-in cash-or-nothing call          (S > H)
BB_UP_IN_CASH_CALL = 14  # up-and-in cash-or-nothing call              (S < H)
BB_DOWN_IN_ASSET_CALL = 15  # down-and-in asset-or-nothing call        (S > H)
BB_UP_IN_ASSET_CALL = 16  # up-and-in asset-or-nothing call            (S < H)
BB_DOWN_IN_CASH_PUT = 17  # down-and-in cash-or-nothing put            (S > H)
BB_UP_IN_CASH_PUT = 18  # up-and-in cash-or-nothing put                (S < H)
BB_DOWN_IN_ASSET_PUT = 19  # down-and-in asset-or-nothing put          (S > H)
BB_UP_IN_ASSET_PUT = 20  # up-and-in asset-or-nothing put              (S < H)
BB_DOWN_OUT_CASH_CALL = 21  # down-and-out cash-or-nothing call        (S > H)
BB_UP_OUT_CASH_CALL = 22  # up-and-out cash-or-nothing call            (S < H)
BB_DOWN_OUT_ASSET_CALL = 23  # down-and-out asset-or-nothing call      (S > H)
BB_UP_OUT_ASSET_CALL = 24  # up-and-out asset-or-nothing call          (S < H)
BB_DOWN_OUT_CASH_PUT = 25  # down-and-out cash-or-nothing put          (S > H)
BB_UP_OUT_CASH_PUT = 26  # up-and-out cash-or-nothing put              (S < H)
BB_DOWN_OUT_ASSET_PUT = 27  # down-and-out asset-or-nothing put        (S > H)
BB_UP_OUT_ASSET_PUT = 28  # up-and-out asset-or-nothing put            (S < H)

# eta / phi per type code (1-indexed; arrays are 0-indexed), Haug pp. 177-182.
# eta selects the down (+1) / up (-1) reflection in the A3/A4/B3/B4/A5 blocks;
# phi selects call (+1) / put (-1) in the A1/A2/B1/B2 blocks. For the at-hit
# types (1..4) only eta matters (phi is unused by A5).
_BB_ETA = np.array(
    [
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
    ],
    dtype=np.float64,
)
_BB_PHI = np.array(
    [
        1,
        1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
    ],
    dtype=np.float64,
)


@njit(fastmath=True, cache=True)
def binary_barrier_price(
    S: float,
    X: float,
    H: float,
    cash: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    binary_type: int,
) -> float:
    """
    Price a binary-barrier option (Reiner-Rubinstein 1991, Haug 4.19.5).

    Parameters
    ----------
    S, X, H : float
        Spot, strike, barrier (all > 0). ``X`` is the strike used by the
        strike-gated call/put types (13..28); the at-hit / at-expiration types
        (1..12) do not depend on it but it still parameterises the formula.
    cash : float
        Cash payout ``K`` for the cash-or-nothing types. Ignored for the
        asset-or-nothing types (which pay the asset). For the asset-(at-hit)
        types 3 & 4 the payout is the asset value at the hit, i.e. ``H``, so the
        kernel internally sets ``cash = H`` for those (Haug's ``K = H``).
    T, r, q, sigma : float
        Maturity, rate, dividend yield, volatility. Cost of carry ``b = r - q``.
    binary_type : int
        One of the ``BB_*`` constants (1..28).

    Returns
    -------
    float
        Option price. Returns 0 for the degenerate ``T <= 0`` / ``sigma <= 0``.
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or H <= 0.0 or X <= 0.0:
        return 0.0

    b = r - q
    eta = float(_BB_ETA[binary_type - 1])
    phi = float(_BB_PHI[binary_type - 1])
    k = cash
    # Asset-(at-hit): the asset delivered at the hit is worth H -> K = H.
    if binary_type == BB_DOWN_IN_ASSET_ATHIT or binary_type == BB_UP_IN_ASSET_ATHIT:
        k = H

    vst = sigma * math.sqrt(T)
    vsq = sigma * sigma
    mu = (b - 0.5 * vsq) / vsq
    lam = math.sqrt(mu * mu + 2.0 * r / vsq)

    x1 = math.log(S / X) / vst + (mu + 1.0) * vst
    x2 = math.log(S / H) / vst + (mu + 1.0) * vst
    y1 = math.log(H * H / (S * X)) / vst + (mu + 1.0) * vst
    y2 = math.log(H / S) / vst + (mu + 1.0) * vst
    z = math.log(H / S) / vst + lam * vst

    df_b = math.exp((b - r) * T)
    df = math.exp(-r * T)
    hs_2mu1 = math.pow(H / S, 2.0 * (mu + 1.0))
    hs_2mu = math.pow(H / S, 2.0 * mu)

    a1 = S * df_b * norm_cdf(phi * x1)
    b1 = k * df * norm_cdf(phi * x1 - phi * vst)
    a2 = S * df_b * norm_cdf(phi * x2)
    b2 = k * df * norm_cdf(phi * x2 - phi * vst)
    a3 = S * df_b * hs_2mu1 * norm_cdf(eta * y1)
    b3 = k * df * hs_2mu * norm_cdf(eta * y1 - eta * vst)
    a4 = S * df_b * hs_2mu1 * norm_cdf(eta * y2)
    b4 = k * df * hs_2mu * norm_cdf(eta * y2 - eta * vst)
    a5 = k * (
        math.pow(H / S, mu + lam) * norm_cdf(eta * z)
        + math.pow(H / S, mu - lam) * norm_cdf(eta * z - 2.0 * eta * lam * vst)
    )

    # Strike-vs-barrier branch. The at-hit / at-expiration / one-touch types
    # (1..12) are strike-independent, so both branches agree; only the
    # strike-gated types (13..28) differ. X == H is folded into the X >= H
    # branch (the sensible boundary limit; Haug's VBA returns 0 there). A single
    # assignment + return mirrors the VBA ``BinaryBarrier = ...`` Select Case.
    value = 0.0
    if X >= H:
        if binary_type < 5:
            value = a5
        elif binary_type < 7:
            value = b2 + b4
        elif binary_type < 9:
            value = a2 + a4
        elif binary_type < 11:
            value = b2 - b4
        elif binary_type < 13:
            value = a2 - a4
        elif binary_type == 13:
            value = b3
        elif binary_type == 14:
            value = b3
        elif binary_type == 15:
            value = a3
        elif binary_type == 16:
            value = a1
        elif binary_type == 17:
            value = b2 - b3 + b4
        elif binary_type == 18:
            value = b1 - b2 + b4
        elif binary_type == 19:
            value = a2 - a3 + a4
        elif binary_type == 20:
            value = a1 - a2 + a3
        elif binary_type == 21:
            value = b1 - b3
        elif binary_type == 22:
            value = 0.0
        elif binary_type == 23:
            value = a1 - a3
        elif binary_type == 24:
            value = 0.0
        elif binary_type == 25:
            value = b1 - b2 + b3 - b4
        elif binary_type == 26:
            value = b2 - b4
        elif binary_type == 27:
            value = a1 - a2 + a3 - a4
        elif binary_type == 28:
            value = a2 - a4
    else:  # X < H
        if binary_type < 5:
            value = a5
        elif binary_type < 7:
            value = b2 + b4
        elif binary_type < 9:
            value = a2 + a4
        elif binary_type < 11:
            value = b2 - b4
        elif binary_type < 13:
            value = a2 - a4
        elif binary_type == 13:
            value = b1 - b2 + b4
        elif binary_type == 14:
            value = b2 - b3 + b4
        elif binary_type == 15:
            value = a1 - a2 + a4
        elif binary_type == 16:
            value = a2 - a3 + a4
        elif binary_type == 17:
            value = b1
        elif binary_type == 18:
            value = b3
        elif binary_type == 19:
            value = a1
        elif binary_type == 20:
            value = a3
        elif binary_type == 21:
            value = b2 - b4
        elif binary_type == 22:
            value = b1 - b2 + b3 - b4
        elif binary_type == 23:
            value = a2 - a4
        elif binary_type == 24:
            value = a1 - a2 + a3 - a4
        elif binary_type == 25:
            value = 0.0
        elif binary_type == 26:
            value = b1 - b3
        elif binary_type == 27:
            value = 0.0
        elif binary_type == 28:
            value = a1 - a3
    return value
