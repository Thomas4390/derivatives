"""
Simulation Runner Service for Monte Carlo Simulation Explorer.

Provides P&L calculation from simulation results.
Supports hybrid vanilla (Numba) + exotic (Python) computation.
"""

import time as time_module
from typing import Any

import numpy as np

from backend.portfolio.pnl import (
    calculate_portfolio_pnl_vectorized,
    calculate_portfolio_pnl_with_stock,
    compute_payoff_curve,
    compute_risk_metrics,
)
from backend.simulation.base import SimulationResult

from streamlit_app.simulation.utils.exotic_loader import (
    get_exotic_payoff_fn as _get_exotic_payoff_fn,
)


def has_exotic_legs(position_arrays: dict) -> bool:
    """Check if any legs are exotic."""
    exotic_metadata = position_arrays.get("exotic_metadata", [])
    return any(m["instrument_class"] != "vanilla" for m in exotic_metadata)


_PATH_DEPENDENT_CLASSES = {
    "barrier",
    "asian",
    "lookback_floating",
    "lookback_fixed",
    "chooser",
    # Haug advanced barriers — monitored on the realised path:
    "double_barrier",
    "discrete_barrier",
    "partial_barrier",
    "binary_barrier",
}


def _discrete_monitor(price_path: np.ndarray, m: int) -> np.ndarray:
    """Sub-sample ``price_path`` at ``m`` evenly-spaced discrete monitoring dates.

    Endpoints (inception + expiry) are always included; ``m >= len-1`` (or
    non-positive) returns the full path (continuous monitoring).
    """
    n = len(price_path)
    if m <= 0 or m >= n - 1:
        return price_path
    idx = np.linspace(0, n - 1, m + 1).round().astype(int)
    return price_path[idx]


def _path_dependent_payoff(price_path: np.ndarray, meta: dict) -> float:
    """Compute per-share payoff for path-dependent exotics using the full path."""
    inst = meta.get("instrument_class", "vanilla")
    is_call = meta.get("option_type") == "call"
    strike = float(meta.get("strike", 0.0))
    spot_t = float(price_path[-1])
    params = meta.get("params", {}) or {}

    def _vanilla() -> float:
        return max(spot_t - strike, 0.0) if is_call else max(strike - spot_t, 0.0)

    # ── Haug advanced barriers (continuous / discrete / partial-time / binary) ──
    if inst == "double_barrier":
        lower = float(params.get("lower", 0.8 * strike))
        upper = float(params.get("upper", 1.2 * strike))
        is_ki = bool(params.get("is_knock_in", meta.get("is_knock_in", False)))
        breached = bool(np.any((price_path <= lower) | (price_path >= upper)))
        if is_ki:
            return _vanilla() if breached else 0.0
        return 0.0 if breached else _vanilla()

    if inst == "discrete_barrier":
        bar = float(params.get("barrier", 1.1 * strike))
        is_up = bool(params.get("is_up", meta.get("is_up", True)))
        is_ki = bool(params.get("is_knock_in", meta.get("is_knock_in", False)))
        m = int(params.get("monitoring_points", 252))
        monitored = _discrete_monitor(price_path, m)
        breached = (
            bool(np.any(monitored >= bar)) if is_up else bool(np.any(monitored <= bar))
        )
        if is_ki:
            return _vanilla() if breached else 0.0
        return 0.0 if breached else _vanilla()

    if inst == "partial_barrier":
        # Heynen-Kat partial-time knock-OUT: the barrier is live only over a window
        # — type A / B1 monitor the early window [0, t1], type B2 the late [t1, T].
        bar = float(params.get("barrier", 1.1 * strike))
        btype = str(params.get("barrier_type", "out_B1"))
        t1 = float(params.get("t1", 0.5 * float(meta.get("maturity", 1.0))))
        maturity = float(meta.get("maturity", 1.0))
        n = len(price_path)
        frac = min(max(t1 / maturity, 0.0), 1.0) if maturity > 0 else 1.0
        cut = int(round(frac * (n - 1)))
        window = price_path[cut:] if btype.endswith("B2") else price_path[: cut + 1]
        is_up = "up" in btype
        breached = bool(np.any(window >= bar)) if is_up else bool(np.any(window <= bar))
        return 0.0 if breached else _vanilla()

    if inst == "binary_barrier":
        from streamlit_app.simulation.utils.exotic_loader import get_binary_parse_map

        bar = float(params.get("barrier", 1.1 * strike))
        cash = float(params.get("cash", 10.0))
        btype = int(params.get("binary_type", 13))
        is_down, is_in, is_asset, gate = get_binary_parse_map().get(
            btype, (False, True, False, "none")
        )
        hit = (
            bool(np.any(price_path <= bar))
            if is_down
            else bool(np.any(price_path >= bar))
        )
        amount = spot_t if is_asset else cash
        if gate == "call":
            amount = amount if spot_t > strike else 0.0
        elif gate == "put":
            amount = amount if spot_t < strike else 0.0
        if is_in:
            return amount if hit else 0.0
        return 0.0 if hit else amount

    if inst == "barrier":
        barrier = float(meta.get("barrier", 0.0))
        is_up = meta.get("is_up", True)
        is_knock_in = meta.get("is_knock_in", False)
        vanilla_payoff = (
            max(spot_t - strike, 0.0) if is_call else max(strike - spot_t, 0.0)
        )
        if is_up:
            barrier_hit = bool(np.any(price_path >= barrier))
        else:
            barrier_hit = bool(np.any(price_path <= barrier))
        if is_knock_in:
            return vanilla_payoff if barrier_hit else 0.0
        return 0.0 if barrier_hit else vanilla_payoff

    if inst == "asian":
        log_avg = float(np.mean(np.log(price_path)))
        geo_avg = float(np.exp(log_avg))
        if is_call:
            return max(geo_avg - strike, 0.0)
        return max(strike - geo_avg, 0.0)

    if inst == "lookback_floating":
        if is_call:
            return max(spot_t - float(np.min(price_path)), 0.0)
        return max(float(np.max(price_path)) - spot_t, 0.0)

    if inst == "lookback_fixed":
        if is_call:
            return max(float(np.max(price_path)) - strike, 0.0)
        return max(strike - float(np.min(price_path)), 0.0)

    if inst == "chooser":
        choice_pct = float(meta.get("choice_time_pct", 0.5))
        n_steps = len(price_path) - 1
        step_tc = max(1, min(int(choice_pct * n_steps), n_steps - 1))
        s_tc = float(price_path[step_tc])
        r = float(meta.get("r", 0.0))
        maturity = float(meta.get("maturity", 1.0))
        t_c = choice_pct * maturity
        threshold = strike * np.exp(-(r) * (maturity - t_c))
        if s_tc > threshold:
            return max(spot_t - strike, 0.0)
        return max(strike - spot_t, 0.0)

    return 0.0


def _ko_breached_so_far(
    inst: str, params: dict, path_history: np.ndarray, strike: float
) -> bool:
    """True if a knock-OUT Haug barrier has already died on the path observed so far.

    Knock-in types (and partial/binary, whose in-out semantics resolve at expiry)
    are treated as still alive mid-life — they return ``False``.
    """
    if inst == "double_barrier":
        if bool(params.get("is_knock_in", False)):
            return False
        lower = float(params.get("lower", 0.8 * strike))
        upper = float(params.get("upper", 1.2 * strike))
        return bool(np.any((path_history <= lower) | (path_history >= upper)))
    if inst == "discrete_barrier":
        if bool(params.get("is_knock_in", False)):
            return False
        bar = float(params.get("barrier", 1.1 * strike))
        is_up = bool(params.get("is_up", True))
        monitored = _discrete_monitor(
            path_history, int(params.get("monitoring_points", 252))
        )
        return (
            bool(np.any(monitored >= bar)) if is_up else bool(np.any(monitored <= bar))
        )
    return False


def _registry_price_alive(
    inst: str,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    is_call: bool,
    meta: dict,
) -> float:
    """Registry closed-form value of a Haug barrier at the remaining maturity ``tau``.

    Re-prices fresh (same approximation as the basic barrier's mid-life value);
    always returns a finite, non-negative float.
    """
    from streamlit_app.simulation.utils.exotic_loader import get_exotic_price_fn

    try:
        px = get_exotic_price_fn()(
            exotic_type=inst,
            spot=spot,
            strike=strike,
            maturity=tau,
            rate=rate,
            sigma=sigma,
            is_call=is_call,
            dividend_yield=0.0,
            cap=float(meta.get("cap", 0.0)),
            params=meta.get("params"),
        )
        return max(float(px), 0.0) if np.isfinite(px) else 0.0
    except Exception:
        return 0.0


def _calculate_hybrid_pnl(
    terminal_prices: np.ndarray,
    position_arrays: dict,
    multiplier: float = 100.0,
    price_paths: np.ndarray | None = None,
) -> np.ndarray:
    """Hybrid P&L: Numba for vanilla legs, Python loop for exotic legs."""
    exotic_metadata = position_arrays.get("exotic_metadata", [])
    vanilla_idx = [
        i for i, m in enumerate(exotic_metadata) if m["instrument_class"] == "vanilla"
    ]
    exotic_idx = [
        i for i, m in enumerate(exotic_metadata) if m["instrument_class"] != "vanilla"
    ]

    # Vanilla via Numba (existing vectorized path)
    if vanilla_idx:
        v_idx = np.array(vanilla_idx)
        pnl = calculate_portfolio_pnl_vectorized(
            terminal_prices,
            position_arrays["strikes"][v_idx],
            position_arrays["option_types"][v_idx],
            position_arrays["position_types"][v_idx],
            position_arrays["quantities"][v_idx],
            position_arrays["premiums"][v_idx],
            multiplier,
        )
    else:
        pnl = np.zeros(len(terminal_prices))

    # Exotic via Python
    exotic_payoff_fn = _get_exotic_payoff_fn()
    for j in exotic_idx:
        meta = exotic_metadata[j]
        premium = float(position_arrays["premiums"][j])
        direction = float(position_arrays["position_types"][j])
        qty = float(position_arrays["quantities"][j])
        is_path_dep = meta.get("instrument_class") in _PATH_DEPENDENT_CLASSES

        if is_path_dep and price_paths is not None:
            for i in range(len(terminal_prices)):
                payoff = _path_dependent_payoff(price_paths[i], meta)
                pnl[i] += direction * (payoff - premium) * qty * multiplier
        else:
            for i, spot in enumerate(terminal_prices):
                payoff = exotic_payoff_fn(float(spot), meta)
                pnl[i] += direction * (payoff - premium) * qty * multiplier

    # Stock P&L
    stock_qty = position_arrays.get("stock_quantity", 0.0)
    if stock_qty != 0.0:
        stock_entry = position_arrays.get("stock_entry_price", 0.0)
        pnl += stock_qty * (terminal_prices - stock_entry)

    return pnl


def compute_mtm_pnl_at_step(
    paths: np.ndarray,
    step_idx: int,
    position_arrays: dict,
    rate: float,
    sigma: float,
    time_to_expiry: float,
    multiplier: float = 100.0,
) -> np.ndarray:
    """Compute mark-to-market P&L at an intermediate time step.

    For vanilla legs, uses vectorised Black-Scholes repricing (intrinsic at
    expiry).  For path-dependent exotics, inspects the path history up to
    ``step_idx`` so that, e.g., a knocked-out barrier has value 0.

    Parameters
    ----------
    paths : np.ndarray
        Shape ``(n_paths, n_steps+1)`` simulated price paths.
    step_idx : int
        Time step index to evaluate at (0 = inception, ``n_steps`` = expiry).
    position_arrays : dict
        Portfolio leg arrays (strikes, option_types, position_types,
        quantities, premiums, exotic_metadata, stock_quantity,
        stock_entry_price).
    rate : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised).
    time_to_expiry : float
        Total time to expiry in years (T).
    multiplier : float
        Contract multiplier (default 100).

    Returns
    -------
    np.ndarray
        Shape ``(n_paths,)`` P&L per path at the given step.
    """
    # ── lazy imports (heavy / Numba-compiled) ──────────────────────────
    from backend.engines.exotic_engine import (
        asset_or_nothing_price,
        barrier_option_price,
        chooser_price,
        digital_price,
        gap_option_price,
        lookback_fixed_price,
        lookback_floating_price,
        power_option_price,
    )
    from backend.utils.math import bs_price

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1
    dt = time_to_expiry / n_steps if n_steps > 0 else 0.0
    tau = time_to_expiry - step_idx * dt  # remaining time
    spot_at_step = paths[:, step_idx]  # (n_paths,)

    exotic_metadata = position_arrays.get("exotic_metadata", [])
    strikes = position_arrays["strikes"]
    option_types = position_arrays["option_types"]
    position_types = position_arrays["position_types"]
    quantities = position_arrays["quantities"]
    premiums = position_arrays["premiums"]

    pnl = np.zeros(n_paths)

    for j, meta in enumerate(exotic_metadata):
        inst = meta.get("instrument_class", "vanilla")
        K = float(strikes[j])
        direction = float(position_types[j])
        qty = float(quantities[j])
        premium = float(premiums[j])
        is_call = option_types[j] == 1.0

        # ── Vanilla ────────────────────────────────────────────────────
        if inst == "vanilla":
            if tau < 1e-10:
                values = np.where(
                    is_call,
                    np.maximum(spot_at_step - K, 0.0),
                    np.maximum(K - spot_at_step, 0.0),
                )
            else:
                values = np.array(
                    [
                        bs_price(float(s), K, tau, rate, sigma, is_call)
                        for s in spot_at_step
                    ]
                )
            pnl += direction * (values - premium) * qty * multiplier
            continue

        # ── Path-dependent & other exotics (per-path loop) ─────────────
        for i in range(n_paths):
            s = float(spot_at_step[i])
            path_history = paths[i, : step_idx + 1]

            if inst == "barrier":
                barrier = float(meta.get("barrier", 0.0))
                is_up = meta.get("is_up", True)
                is_ki = meta.get("is_knock_in", False)

                if is_up:
                    hit = bool(np.any(path_history >= barrier))
                else:
                    hit = bool(np.any(path_history <= barrier))

                if not is_ki and hit:
                    # Knock-out and barrier breached → dead
                    value = 0.0
                elif is_ki and hit:
                    # Knock-in activated → now vanilla
                    value = (
                        bs_price(s, K, tau, rate, sigma, is_call)
                        if tau > 1e-10
                        else (max(s - K, 0.0) if is_call else max(K - s, 0.0))
                    )
                elif is_ki and not hit and tau < 1e-10:
                    # Knock-in never activated at expiry → 0
                    value = 0.0
                else:
                    # Not yet triggered, still alive
                    value = barrier_option_price(
                        s, K, barrier, tau, rate, 0.0, sigma, is_call, is_ki, is_up
                    )

            elif inst == "lookback_floating":
                running_min = float(np.min(path_history))
                running_max = float(np.max(path_history))
                if tau < 1e-10:
                    value = (
                        max(s - running_min, 0.0)
                        if is_call
                        else max(running_max - s, 0.0)
                    )
                else:
                    value = lookback_floating_price(
                        s, running_min, running_max, tau, rate, 0.0, sigma, is_call
                    )

            elif inst == "lookback_fixed":
                running_min = float(np.min(path_history))
                running_max = float(np.max(path_history))
                if tau < 1e-10:
                    value = (
                        max(running_max - K, 0.0)
                        if is_call
                        else max(K - running_min, 0.0)
                    )
                else:
                    value = lookback_fixed_price(
                        s, K, running_min, running_max, tau, rate, 0.0, sigma, is_call
                    )

            elif inst == "chooser":
                choice_pct = float(meta.get("choice_time_pct", 0.5))
                t_c_total = choice_pct * time_to_expiry
                elapsed = step_idx * dt
                if elapsed >= t_c_total:
                    # Choice already made: determine from path at choice step
                    choice_step = max(1, min(int(choice_pct * n_steps), n_steps - 1))
                    s_tc = float(paths[i, choice_step])
                    r_local = float(meta.get("r", rate))
                    threshold = K * np.exp(-r_local * (time_to_expiry - t_c_total))
                    chose_call = s_tc > threshold
                    if tau < 1e-10:
                        value = max(s - K, 0.0) if chose_call else max(K - s, 0.0)
                    else:
                        value = bs_price(s, K, tau, rate, sigma, chose_call)
                else:
                    # Choice not yet made → use chooser formula
                    t_c_remaining = t_c_total - elapsed
                    value = chooser_price(s, K, tau, t_c_remaining, rate, 0.0, sigma)

            elif inst == "digital":
                payout = float(meta.get("payout", 1.0))
                if tau < 1e-10:
                    if is_call:
                        value = payout if s > K else 0.0
                    else:
                        value = payout if s < K else 0.0
                else:
                    value = digital_price(s, K, tau, rate, 0.0, sigma, is_call, payout)

            elif inst == "asset_or_nothing":
                if tau < 1e-10:
                    if is_call:
                        value = s if s > K else 0.0
                    else:
                        value = s if s < K else 0.0
                else:
                    value = asset_or_nothing_price(s, K, tau, rate, 0.0, sigma, is_call)

            elif inst == "power":
                n_pow = float(meta.get("power_n", 2.0))
                if tau < 1e-10:
                    sn = s**n_pow
                    value = max(sn - K, 0.0) if is_call else max(K - sn, 0.0)
                else:
                    value = power_option_price(
                        s, K, tau, rate, 0.0, sigma, is_call, n_pow
                    )

            elif inst == "gap":
                trigger = float(meta.get("gap_trigger", K))
                if tau < 1e-10:
                    if is_call:
                        value = (s - K) if s > trigger else 0.0
                    else:
                        value = (K - s) if s < trigger else 0.0
                else:
                    value = gap_option_price(
                        s, K, trigger, tau, rate, 0.0, sigma, is_call
                    )

            elif inst == "asian":
                # Running geometric average from the path observed so far.
                # Note: asian_geometric_price assumes a *fresh* averaging
                # period and is therefore unsuitable for partial-path
                # evaluation where some of the averaging window has already
                # elapsed.  BS on the running average is the pragmatic
                # approximation here.
                geo_avg = float(np.exp(np.mean(np.log(path_history))))
                if tau < 1e-10:
                    value = max(geo_avg - K, 0.0) if is_call else max(K - geo_avg, 0.0)
                else:
                    value = bs_price(geo_avg, K, tau, rate, sigma, is_call)

            elif inst in (
                "double_barrier",
                "discrete_barrier",
                "partial_barrier",
                "binary_barrier",
            ):
                # Haug advanced barriers: at expiry the exact path-monitored
                # terminal payoff; mid-life the registry closed form, zeroed once a
                # knock-out barrier has already breached on the path so far.
                p = meta.get("params", {}) or {}
                if tau < 1e-10:
                    value = _path_dependent_payoff(path_history, meta)
                elif _ko_breached_so_far(inst, p, path_history, K):
                    value = 0.0
                else:
                    value = _registry_price_alive(
                        inst, s, K, tau, rate, sigma, is_call, meta
                    )

            else:
                # Unknown exotic – fall back to 0
                value = 0.0

            pnl[i] += direction * (value - premium) * qty * multiplier

    # ── Stock leg ──────────────────────────────────────────────────────
    stock_qty = position_arrays.get("stock_quantity", 0.0)
    if stock_qty != 0.0:
        stock_entry = position_arrays.get("stock_entry_price", 0.0)
        pnl += stock_qty * (spot_at_step - stock_entry)

    return pnl


def compute_hybrid_payoff_curve(
    spot_range: np.ndarray,
    position_arrays: dict,
    multiplier: float = 100.0,
) -> np.ndarray:
    """Hybrid payoff curve: Numba for vanilla, Python for exotic legs."""
    exotic_metadata = position_arrays.get("exotic_metadata", [])
    vanilla_idx = [
        i for i, m in enumerate(exotic_metadata) if m["instrument_class"] == "vanilla"
    ]
    exotic_idx = [
        i for i, m in enumerate(exotic_metadata) if m["instrument_class"] != "vanilla"
    ]

    # Vanilla via Numba compute_payoff_curve
    if vanilla_idx:
        v_idx = np.array(vanilla_idx)
        payoff = compute_payoff_curve(
            spot_range,
            position_arrays["strikes"][v_idx],
            position_arrays["option_types"][v_idx],
            position_arrays["position_types"][v_idx],
            position_arrays["quantities"][v_idx],
            position_arrays["premiums"][v_idx],
            0.0,
            0.0,  # no stock in payoff curve (handled separately)
        )
    else:
        payoff = np.zeros(len(spot_range))

    # Exotic legs
    exotic_payoff_fn = _get_exotic_payoff_fn()
    for j in exotic_idx:
        meta = exotic_metadata[j]
        premium = float(position_arrays["premiums"][j])
        direction = float(position_arrays["position_types"][j])
        qty = float(position_arrays["quantities"][j])
        for i, spot in enumerate(spot_range):
            exotic_payoff = exotic_payoff_fn(float(spot), meta)
            payoff[i] += direction * (exotic_payoff - premium) * qty * multiplier

    # Stock component
    stock_qty = position_arrays.get("stock_quantity", 0.0)
    if stock_qty != 0.0:
        stock_entry = position_arrays.get("stock_entry_price", 0.0)
        payoff += stock_qty * (spot_range - stock_entry)

    return payoff


def calculate_pnl_from_paths(
    price_result: SimulationResult, params: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Calculate portfolio P&L from existing price simulation result.

    Uses terminal prices from the simulation to compute P&L for all
    option positions and optional stock position. Automatically selects
    hybrid (Numba+Python) path when exotic legs are present.

    Parameters
    ----------
    price_result : SimulationResult
        Result from price simulation containing paths
    params : dict
        Parameters including position_arrays with:
        - strikes: np.ndarray
        - option_types: np.ndarray (1=call, -1=put)
        - position_types: np.ndarray (1=long, -1=short)
        - quantities: np.ndarray
        - premiums: np.ndarray
        - stock_quantity: float
        - stock_entry_price: float
        - exotic_metadata: list[dict] (optional)

    Returns
    -------
    dict or None
        Dictionary containing:
        - terminal_prices: np.ndarray
        - pnl_values: np.ndarray
        - risk_metrics: RiskMetrics
        - computation_time: float
        - num_paths: int
        Returns None if no positions are defined.
    """
    position_arrays = params.get("position_arrays", {})

    if len(position_arrays.get("strikes", [])) == 0:
        return None

    start_time = time_module.perf_counter()

    # Use terminal prices from the simulation result
    terminal_prices = price_result.terminal_prices

    # Check for exotic legs — use hybrid path (pass full paths for path-dependent)
    if has_exotic_legs(position_arrays):
        pnl_values = _calculate_hybrid_pnl(
            terminal_prices,
            position_arrays,
            price_paths=price_result.price_paths,
        )
    else:
        stock_qty = position_arrays.get("stock_quantity", 0.0)
        stock_entry = position_arrays.get("stock_entry_price", 0.0)

        if stock_qty != 0.0:
            pnl_values = calculate_portfolio_pnl_with_stock(
                terminal_prices,
                position_arrays["strikes"],
                position_arrays["option_types"],
                position_arrays["position_types"],
                position_arrays["quantities"],
                position_arrays["premiums"],
                stock_qty,
                stock_entry,
                multiplier=100.0,
            )
        else:
            pnl_values = calculate_portfolio_pnl_vectorized(
                terminal_prices,
                position_arrays["strikes"],
                position_arrays["option_types"],
                position_arrays["position_types"],
                position_arrays["quantities"],
                position_arrays["premiums"],
                multiplier=100.0,
            )

    risk_metrics = compute_risk_metrics(pnl_values)
    computation_time = time_module.perf_counter() - start_time

    return {
        "terminal_prices": terminal_prices,
        "pnl_values": pnl_values,
        "risk_metrics": risk_metrics,
        "computation_time": computation_time,
        "num_paths": len(terminal_prices),
    }
