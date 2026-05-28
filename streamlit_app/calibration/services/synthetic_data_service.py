"""
Synthetic-data generation
============================

Bridges the sidebar's "true parameters + data config" choices to the
backend's pricing / simulation engines so the user can generate ground-
truth surfaces and return series to calibrate against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.calibration.market_data import (
    HistoricalReturns,
    OptionMarketData,
    OptionQuote,
)
from backend.core.market import MarketEnvironment
from backend.engines.fft_engine import FFTEngine
from backend.instruments.options import VanillaOption
from backend.models.bates import BatesModel
from backend.models.heston import HestonModel
from backend.models.heston_nandi import HestonNandiGARCHModel
from backend.models.merton import MertonModel
from backend.simulation.models.garch import GARCHSimulator
from backend.simulation.models.gjr_garch import GJRGARCHSimulator
from backend.simulation.models.ngarch import NGARCHSimulator
from backend.utils.math import implied_volatility


@dataclass(frozen=True)
class _GBMStandIn:
    """Pickle-safe stand-in for the GBM "model" returned by the IV path."""
    sigma: float
    name: str = "GBM"


@dataclass(frozen=True)
class SurfaceData:
    market_data: OptionMarketData
    true_model: Any
    iv_grid: np.ndarray  # (n_T, n_K) implied vols
    strikes: np.ndarray
    maturities: np.ndarray


@dataclass(frozen=True)
class ReturnsData:
    market_data: HistoricalReturns
    true_params: dict[str, float]
    prices: np.ndarray
    log_returns: np.ndarray
    sample_volatility_ann: float


# --------------------------------------------------------------------- #
# Surface generation (Heston / Merton / Bates / GBM)
# --------------------------------------------------------------------- #


def _build_model(model_key: str, true_params: dict) -> Any:
    if model_key == "heston":
        return HestonModel(**true_params)
    if model_key == "merton":
        return MertonModel(**true_params)
    if model_key == "bates":
        return BatesModel(**true_params)
    if model_key == "heston_nandi":
        return HestonNandiGARCHModel(**true_params)
    if model_key == "ngarch_q":
        from backend.models.ngarch_q import NGARCHRiskNeutralModel

        return NGARCHRiskNeutralModel(**true_params)
    if model_key == "garch_q":
        from backend.models.ngarch_q import GARCHRiskNeutralModel

        return GARCHRiskNeutralModel(**true_params)
    if model_key == "gjr_q":
        from backend.models.ngarch_q import GJRGARCHRiskNeutralModel

        return GJRGARCHRiskNeutralModel(**true_params)
    raise NotImplementedError(f"Surface generation for {model_key} not supported.")


def generate_surface(
    model_key: str,
    true_params: dict,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_strikes: int,
    n_maturities: int,
    strike_min: float,
    strike_max: float,
    maturity_min: float,
    maturity_max: float,
    noise_std: float,
    seed: int,
) -> SurfaceData:
    """Generate an OptionMarketData surface under the chosen model."""
    if model_key == "iv_gbm":
        sigma = float(true_params.get("sigma", 0.2))
        return _generate_gbm_surface(
            sigma=sigma,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield,
            n_strikes=n_strikes,
            n_maturities=n_maturities,
            strike_min=strike_min,
            strike_max=strike_max,
            maturity_min=maturity_min,
            maturity_max=maturity_max,
            noise_std=noise_std,
            seed=seed,
        )

    model = _build_model(model_key, true_params)
    market = MarketEnvironment(spot=spot, rate=rate, dividend_yield=dividend_yield)
    # Nonaffine Duan NGARCH-Q has no closed form → price the ground-truth
    # surface by Monte-Carlo (high path count for a clean target); affine models
    # use the exact FFT engine.
    is_mc_model = model_key in ("ngarch_q", "garch_q", "gjr_q")
    mc_sim = model.create_simulator() if is_mc_model else None
    engine = None if is_mc_model else FFTEngine()
    rng = np.random.default_rng(seed)

    strikes = np.linspace(strike_min, strike_max, n_strikes)
    maturities = np.linspace(maturity_min, maturity_max, n_maturities)
    iv_grid = np.full((len(maturities), len(strikes)), np.nan)

    quotes: list[OptionQuote] = []
    for i_T, T in enumerate(maturities):
        forward = spot * np.exp((rate - dividend_yield) * float(T))
        # Modern surface convention: keep the *out-of-the-money* option at each
        # strike (put below the forward, call at/above it). OTM prices are pure
        # time value, so the BS inversion stays well conditioned — pricing only
        # calls leaves deep-ITM cells almost all intrinsic, which inverts to NaN
        # under Monte-Carlo noise (and is poorly scaled for price_mse even when
        # exact). Applied to every model for one consistent, realistic surface.
        quote_is_call = strikes >= forward
        if is_mc_model:
            # MC: price calls and puts off one shared seed (identical terminals
            # → put-call parity holds exactly), then keep the OTM side.
            mc_seed = seed + i_T
            calls = mc_sim.price_strikes(
                spot, strikes, rate, float(T),
                n_paths=80_000, rng=np.random.default_rng(mc_seed), is_call=True,
            )
            puts = mc_sim.price_strikes(
                spot, strikes, rate, float(T),
                n_paths=80_000, rng=np.random.default_rng(mc_seed), is_call=False,
            )
        else:
            # FFT (affine): exact call prices; the OTM put follows by put-call
            # parity P = C - S·e^{-qT} + K·e^{-rT} (exact for these models).
            template = VanillaOption(strike=spot, maturity=float(T), is_call=True)
            calls = engine.price_strikes(template, model, market, strikes)
            puts = (
                calls
                - spot * np.exp(-dividend_yield * float(T))
                + strikes * np.exp(-rate * float(T))
            )
        prices = np.where(quote_is_call, calls, puts)
        for j, (k, p) in enumerate(zip(strikes, prices)):
            is_call = bool(quote_is_call[j])
            true_p = float(max(p, 1e-6))
            if noise_std > 0.0:
                noisy = true_p * (1.0 + rng.normal(scale=noise_std))
                market_p = float(max(noisy, 1e-6))
            else:
                market_p = true_p
            iv = _safe_iv(
                market_p, spot, float(k), float(T), rate, dividend_yield,
                is_call=is_call,
            )
            iv_grid[i_T, j] = iv if iv is not None else np.nan
            quotes.append(
                OptionQuote(
                    strike=float(k),
                    maturity=float(T),
                    is_call=is_call,
                    market_price=market_p,
                    implied_vol=iv,
                )
            )

    md = OptionMarketData(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        quotes=tuple(quotes),
    )
    return SurfaceData(
        market_data=md,
        true_model=model,
        iv_grid=iv_grid,
        strikes=strikes,
        maturities=maturities,
    )


def _generate_gbm_surface(
    sigma: float,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_strikes: int,
    n_maturities: int,
    strike_min: float,
    strike_max: float,
    maturity_min: float,
    maturity_max: float,
    noise_std: float,
    seed: int,
) -> SurfaceData:
    """Closed-form Black-Scholes surface — used for the IV/GBM model."""
    from backend.utils.math import bs_price

    rng = np.random.default_rng(seed)
    strikes = np.linspace(strike_min, strike_max, n_strikes)
    maturities = np.linspace(maturity_min, maturity_max, n_maturities)
    iv_grid = np.full((len(maturities), len(strikes)), sigma)

    quotes: list[OptionQuote] = []
    for T in maturities:
        forward = spot * np.exp((rate - dividend_yield) * float(T))
        for k in strikes:
            is_call = bool(k >= forward)  # OTM: call above the forward, put below
            price = bs_price(spot, float(k), float(T), rate, sigma, is_call, dividend_yield)
            if noise_std > 0.0:
                price = max(price * (1.0 + rng.normal(scale=noise_std)), 1e-6)
            iv = _safe_iv(
                price, spot, float(k), float(T), rate, dividend_yield, is_call=is_call
            )
            quotes.append(
                OptionQuote(
                    strike=float(k),
                    maturity=float(T),
                    is_call=is_call,
                    market_price=float(price),
                    implied_vol=iv,
                )
            )

    md = OptionMarketData(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        quotes=tuple(quotes),
    )
    return SurfaceData(
        market_data=md,
        true_model=_GBMStandIn(sigma=float(sigma)),
        iv_grid=iv_grid,
        strikes=strikes,
        maturities=maturities,
    )


def _safe_iv(price, spot, k, T, rate, q, is_call: bool = True) -> float | None:
    """Invert Black-Scholes IV from a (positive) price.

    Pass ``is_call`` matching the option that was priced — invert the
    out-of-the-money side (call above the forward, put below) so the inversion
    stays well conditioned (an in-the-money price is dominated by intrinsic
    value, which the inversion cannot resolve).

    The backend's :func:`implied_volatility` returns ``nan`` on failure
    (non-convergence or a no-arbitrage violation) — we map that to ``None``
    so downstream code can treat it as a missing value uniformly.
    """
    if price <= 0 or T <= 0:
        return None
    try:
        iv = implied_volatility(
            price=float(price),
            spot=spot,
            strike=k,
            time_to_expiry=T,
            rate=rate,
            is_call=is_call,
            dividend_yield=q,
        )
    except (ValueError, RuntimeError):
        return None
    # Backend returns nan on failure / arbitrage violation; also reject
    # pathological values (nan comparisons are always False, so test it).
    if iv is None or not np.isfinite(iv) or iv <= 1e-3 or iv >= 4.99:
        return None
    return float(iv)


# --------------------------------------------------------------------- #
# Returns generation (GARCH family)
# --------------------------------------------------------------------- #


def generate_returns(
    garch_type: str,
    true_params: dict,
    n_periods: int,
    annualization_factor: int,
    spot: float,
    drift: float,
    seed: int,
) -> ReturnsData:
    """Simulate a single price path under the chosen GARCH variant.

    The backend ``GARCHSimulator`` works at *annualised* scale (its variance
    recursion uses annualised σ²), but the calibrator sees per-period log
    returns and recovers per-period ω/α/β.  The user enters parameters at
    the per-period scale (e.g. ω=2e-6 daily), so we convert
    ``ω_ann = ω_per × annualization_factor`` before passing to the
    simulator.  α and β are dimensionless and stay unchanged.
    """
    omega_per = float(true_params["omega"])
    alpha = float(true_params["alpha"])
    beta = float(true_params["beta"])

    persistence = alpha + beta
    if persistence >= 1.0:
        raise ValueError(
            f"GARCH stationarity violated (α+β={persistence:.3f} ≥ 1). "
            f"Lower α or β in the True Parameters panel."
        )

    # Per-period long-run variance, then annualise (variance scales linearly with time)
    long_run_var_per = omega_per / max(1e-12, 1.0 - persistence)
    long_run_var_ann = long_run_var_per * annualization_factor
    omega_ann = omega_per * annualization_factor
    sigma0_ann = float(np.sqrt(long_run_var_ann))

    if garch_type == "garch":
        sim = GARCHSimulator(
            sigma0=sigma0_ann,
            omega=omega_ann,
            alpha=alpha,
            beta=beta,
        )
    elif garch_type == "ngarch":
        sim = NGARCHSimulator(
            sigma0=sigma0_ann,
            omega=omega_ann,
            alpha=alpha,
            beta=beta,
            gamma=float(true_params.get("gamma", 0.5)),
        )
    else:  # gjr_garch
        sim = GJRGARCHSimulator(
            sigma0=sigma0_ann,
            omega=omega_ann,
            alpha=alpha,
            beta=beta,
            gamma=float(true_params.get("gamma", 0.04)),
        )

    t_years = n_periods / annualization_factor
    res = sim.simulate_paths(
        s0=float(spot),
        mu=float(drift),
        t=t_years,
        n_paths=1,
        n_steps=int(n_periods),
        seed=int(seed),
    )
    prices = np.asarray(res.price_paths[0, :])
    log_returns = np.diff(np.log(prices))

    md = HistoricalReturns(
        log_returns=log_returns,
        frequency="daily",
        annualization_factor=annualization_factor,
    )
    sample_vol = float(np.std(log_returns) * np.sqrt(annualization_factor))
    return ReturnsData(
        market_data=md,
        true_params=true_params,
        prices=prices,
        log_returns=log_returns,
        sample_volatility_ann=sample_vol,
    )
