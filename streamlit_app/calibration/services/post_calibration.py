"""
Post-calibration helpers
=========================

Reconstruct intermediate models from iteration history, price the option
surface under each, and project the resulting IVs onto the same (T, K)
grid as the market data so charts can compare market vs model and animate
the fit progression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.calibration.pricing_loop import price_surface, price_surface_mc
from backend.calibration.utils import model_prices_to_ivs
from backend.core.result_types import PricingCapability
from backend.engines.fft_engine import FFTEngine

from config.constants import MC_PATHS_SURFACE, MC_SEED

# Monte-Carlo defaults for the nonaffine / risk-neutral-GARCH pricing path
# (Duan NGARCH-Q surface model + physical-GARCH model-implied surfaces). Sourced
# from config.constants so the MC knobs live in one place.
_MC_PATHS_DEFAULT = MC_PATHS_SURFACE
_MC_SEED_DEFAULT = MC_SEED

_ENGINE: FFTEngine | None = None


def _engine() -> FFTEngine:
    """Lazy singleton — FFTEngine init does some warmup we don't want repeated."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FFTEngine()
    return _ENGINE


def _risk_neutral_simulator(model):
    """Build a risk-neutral GARCH MC simulator for a Monte-Carlo-only model.

    Dispatches on the model type — the surface-family Duan ``NGARCHRiskNeutralModel``
    is already risk-neutral (its own ``create_simulator``), while the physical GARCH
    family is converted via :meth:`GARCHRiskNeutralSimulator.from_physical_params`
    (which owns the annualised→per-period scaling and the Duan LRNVR λ = 0
    convention). Returns ``None`` for models with no GARCH MC mapping.
    """
    from backend.models.garch import GARCHModel, GJRGARCHModel, NGARCHModel
    from backend.models.ngarch_q import (
        GARCHRiskNeutralModel,
        GJRGARCHRiskNeutralModel,
        NGARCHRiskNeutralModel,
    )
    from backend.simulation.models.garch_q import GARCHRiskNeutralSimulator

    if isinstance(
        model,
        (GARCHRiskNeutralModel, NGARCHRiskNeutralModel, GJRGARCHRiskNeutralModel),
    ):
        return model.create_simulator()

    if isinstance(model, NGARCHModel):
        gtype, gamma = "ngarch", float(model.gamma)
    elif isinstance(model, GJRGARCHModel):
        gtype, gamma = "gjr_garch", float(model.gamma)
    elif isinstance(model, GARCHModel):
        gtype, gamma = "garch", 0.0
    else:
        return None

    return GARCHRiskNeutralSimulator.from_physical_params(
        gtype,
        omega_annualised=float(model.omega),
        alpha=float(model.alpha),
        beta=float(model.beta),
        gamma=gamma,
        sigma0=float(model.sigma0),
    )


def surface_model_prices(
    model,
    market_data,
    *,
    n_paths: int = _MC_PATHS_DEFAULT,
    mc_seed: int = _MC_SEED_DEFAULT,
) -> np.ndarray:
    """Model prices aligned to ``market_data.quotes`` — FFT for affine models,
    Monte-Carlo for the nonaffine / risk-neutral-GARCH family.

    Dispatches on ``model.supported_engines``: affine models (Heston, Bates,
    Merton, Heston-Nandi) keep the exact closed-form FFT path; nonaffine
    Monte-Carlo-only models (Duan NGARCH-Q, physical GARCH) go through the shared
    :func:`backend.calibration.pricing_loop.price_surface_mc`.
    """
    if PricingCapability.FFT in model.supported_engines:
        return price_surface(model, market_data, _engine())
    sim = _risk_neutral_simulator(model)
    if sim is None:
        # Generic SDE model (e.g. a user-defined custom model with no GARCH MC
        # mapping): Euler-price it via the same terminal simulator the custom
        # calibrator uses.
        if callable(getattr(model, "drift", None)) and callable(
            getattr(model, "diffusion", None)
        ):
            from backend.calibration.custom_calibrator import CustomTerminalSimulator

            sim = CustomTerminalSimulator(model)
        else:
            raise ValueError(
                f"No surface-pricing path for model {type(model).__name__}"
            )
    return price_surface_mc(sim, market_data, n_paths=n_paths, mc_seed=mc_seed)


def model_iv_grid(model, market_data, meta: dict) -> np.ndarray:
    """Return the model's implied-vol grid aligned on the market (T, K) mesh.

    Shape: ``(n_maturities, n_strikes)``. Cells without a matching quote
    stay NaN — same convention as ``meta["iv_grid"]``. Affine models price by
    FFT; nonaffine / risk-neutral-GARCH models price by Monte-Carlo.

    Inversion policy — LOAD-BEARING PRECONDITION: this inverts each quote with
    its own ``is_call`` flag, including deep-ITM quotes. That is only safe
    because every quote source here is **OTM by construction** (synthetic
    surfaces set ``is_call = strike >= forward``; the real-data loader picks the
    OTM side per strike). Deep-ITM inversion of an MC-priced surface is noisy —
    if a future quote source ever emits deep-ITM quotes, switch to the OTM-only
    rule used by :func:`iv_grid_from_simulator` (see its docstring). The two
    inversion policies are kept separate on purpose; do NOT unify them without
    revisiting this precondition.
    """
    model_prices = surface_model_prices(model, market_data)
    is_calls = np.array([qt.is_call for qt in market_data.quotes])
    flat_ivs = model_prices_to_ivs(
        model_prices=model_prices,
        spot=market_data.spot,
        strikes=market_data.strikes,
        maturities=market_data.maturities,
        rate=market_data.rate,
        is_calls=is_calls,
        dividend_yield=market_data.dividend_yield,
    )
    # implied_volatility() returns -1.0 as a non-convergence sentinel rather
    # than raising — without this filter the heatmap shows -100% IV cells.
    flat_ivs = np.where(flat_ivs <= 1e-4, np.nan, flat_ivs)

    grid_strikes = np.asarray(meta["strikes"], dtype=np.float64)  # (n_T, n_K)
    grid_maturities = np.asarray(meta["maturities"], dtype=np.float64)
    n_T, n_K = grid_strikes.shape
    grid = np.full((n_T, n_K), np.nan, dtype=np.float64)
    quote_T = market_data.maturities
    quote_K = market_data.strikes
    for q in range(flat_ivs.shape[0]):
        i_T = int(np.argmin(np.abs(grid_maturities - quote_T[q])))
        # strikes are per-maturity (adaptive moneyness grid) → match within the row.
        i_K = int(np.argmin(np.abs(grid_strikes[i_T] - quote_K[q])))
        grid[i_T, i_K] = flat_ivs[q]
    return grid


def rebuild_model(model_key: str, params_natural: dict):
    """Rebuild a surface-pricable model from its natural-scale parameters.

    Covers the affine FFT models, the MC-priced risk-neutral GARCH-Q trio,
    and the registered custom model. Returns ``None`` for model keys that do
    not price a surface from a parameter dict (physical GARCH family, IV/GBM
    closed-form) so the caller can short-circuit.
    """
    if model_key == "heston":
        from backend.models.heston import HestonModel

        return HestonModel(
            v0=float(params_natural["v0"]),
            kappa=float(params_natural["kappa"]),
            theta=float(params_natural["theta"]),
            alpha=float(params_natural["alpha"]),
            rho=float(params_natural["rho"]),
        )
    if model_key == "merton":
        from backend.models.merton import MertonModel

        return MertonModel(
            sigma=float(params_natural["sigma"]),
            lam=float(params_natural["lam"]),
            alpha_j=float(params_natural["alpha_j"]),
            sigma_j=float(params_natural["sigma_j"]),
        )
    if model_key == "bates":
        from backend.models.bates import BatesModel

        return BatesModel(
            v0=float(params_natural["v0"]),
            kappa=float(params_natural["kappa"]),
            theta=float(params_natural["theta"]),
            alpha=float(params_natural["alpha"]),
            rho=float(params_natural["rho"]),
            lam=float(params_natural["lam"]),
            alpha_j=float(params_natural["alpha_j"]),
            sigma_j=float(params_natural["sigma_j"]),
        )
    if model_key == "heston_nandi":
        from backend.models.heston_nandi import HestonNandiGARCHModel

        # steps_per_year defaults to HESTON_NANDI_STEPS_PER_YEAR (252) — the same
        # default the calibrator uses to build the final model, so re-pricing a
        # snapshot here matches the fitted surface exactly.
        return HestonNandiGARCHModel(
            omega=float(params_natural["omega"]),
            alpha=float(params_natural["alpha"]),
            beta=float(params_natural["beta"]),
            gamma=float(params_natural["gamma"]),
            h0=float(params_natural["h0"]),
        )
    if model_key in ("ngarch_q", "garch_q", "gjr_q"):
        from backend.models.ngarch_q import (
            GARCHRiskNeutralModel,
            GJRGARCHRiskNeutralModel,
            NGARCHRiskNeutralModel,
        )

        cls = {
            "ngarch_q": NGARCHRiskNeutralModel,
            "garch_q": GARCHRiskNeutralModel,
            "gjr_q": GJRGARCHRiskNeutralModel,
        }[model_key]
        # garch_q is symmetric — its model has no gamma even though the
        # calibrator's search vector pins one at (0, 0); filter to the
        # constructor's own parameter names.
        allowed = set(cls._PARAM_NAMES)
        return cls(**{k: float(v) for k, v in params_natural.items() if k in allowed})
    if model_key == "custom":
        # Rebuild the user-defined model from the registered class (session-scoped).
        from services.custom_model_service import get_custom_model_class

        cls = get_custom_model_class()
        if cls is None:
            return None
        return cls(**{k: float(v) for k, v in params_natural.items()})
    return None


def iv_grid_from_simulator(
    sim,
    *,
    spot: float,
    rate: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    dividend_yield: float = 0.0,
    n_paths: int = _MC_PATHS_DEFAULT,
    mc_seed: int = _MC_SEED_DEFAULT,
) -> np.ndarray:
    """Price a ``strikes × maturities`` grid with a risk-neutral GARCH simulator
    and invert each cell to Black-Scholes implied vol.

    Returns shape ``(n_maturities, n_strikes)`` with NaN where inversion fails.

    Inversion policy: OTM-only by forward (a call above the forward, a put
    below), which avoids the deep-ITM MC noise. This deliberately differs from
    :func:`model_iv_grid`'s per-quote ``is_call`` inversion — the two coincide
    today only because both consumers feed OTM quotes; see ``model_iv_grid``'s
    precondition. Kept separate on purpose.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    k_mesh, t_mesh = np.meshgrid(strikes, maturities)  # (n_T, n_K)

    # Invert *out-of-the-money* options (put below the forward, call at/above it).
    # OTM prices are pure time value, so the BS inversion is well-conditioned;
    # pricing calls everywhere instead makes deep-ITM cells almost all intrinsic,
    # where MC noise swamps the tiny time value and the inverted IV is garbage
    # (the spikes/holes that made these surfaces look strange). One simulation
    # feeds both payoff grids -> identical terminals -> put-call parity holds
    # exactly (and the MC cost is half that of two price_surface passes).
    calls, puts = sim.price_surface_call_put(
        spot, strikes, maturities, rate, n_paths=n_paths, seed=mc_seed
    )  # each (n_T, n_K)
    use_call = k_mesh >= spot * np.exp(rate * t_mesh)  # call OTM iff K >= forward
    price_grid = np.where(use_call, calls, puts)
    flat_ivs = model_prices_to_ivs(
        model_prices=price_grid.ravel(),
        spot=spot,
        strikes=k_mesh.ravel(),
        maturities=t_mesh.ravel(),
        rate=rate,
        is_calls=use_call.ravel(),
        dividend_yield=dividend_yield,
    )
    flat_ivs = np.where(flat_ivs <= 1e-4, np.nan, flat_ivs)
    return flat_ivs.reshape(price_grid.shape)


def garch_implied_iv_grid(
    model,
    *,
    spot: float,
    rate: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    dividend_yield: float = 0.0,
    n_paths: int = _MC_PATHS_DEFAULT,
    mc_seed: int = _MC_SEED_DEFAULT,
) -> np.ndarray:
    """Model-implied IV surface of a (risk-neutral or physical) GARCH model.

    Builds the risk-neutral simulator (Duan LRNVR) and inverts the priced grid
    to implied vol. Lets a GARCH fitted on returns show the *same* IV-surface
    charts as the surface-family models. Returns ``(n_maturities, n_strikes)``.
    """
    sim = _risk_neutral_simulator(model)
    if sim is None:
        raise ValueError(f"No GARCH MC mapping for model {type(model).__name__}")
    return iv_grid_from_simulator(
        sim,
        spot=spot,
        rate=rate,
        strikes=strikes,
        maturities=maturities,
        dividend_yield=dividend_yield,
        n_paths=n_paths,
        mc_seed=mc_seed,
    )


@dataclass(frozen=True)
class IVAnimationFrame:
    iter_index: int  # position within the source history
    objective: float  # objective at this snapshot
    iv_grid: np.ndarray  # (n_T, n_K), NaN for missing cells
    # Natural-scale parameters that priced this grid — lets a combined chart
    # show the parameter trajectory in lock-step with the morphing surface.
    params_natural: dict = field(default_factory=dict)


def iv_grid_animation_frames(
    model_key: str,
    history,
    market_data,
    meta: dict,
    *,
    max_frames: int = 12,
) -> list[IVAnimationFrame]:
    """Sub-sample the iteration history and price each picked snapshot.

    Prefers ``source='callback'`` snapshots (one per optimizer iteration)
    so the animation tracks the *running* best instead of every population
    evaluation. Falls back to all snapshots when too few callbacks exist.

    Snapshots whose params produce a non-pricable model (e.g. Feller-violating
    Heston) are silently skipped — the animation just shows fewer frames.
    """
    # Prefer ``source='callback'`` snapshots so the animation tracks the
    # optimizer's *major* iterations rather than every line-search probe.
    # Always prepend the first evaluation snapshot so the animation
    # starts from the user's initial guess — scipy's L-BFGS-B / NM /
    # DE callbacks only fire after iteration 1, so without this the
    # animation skipped the most visually informative frame (initial
    # surface, far from the optimum) and looked frozen for solvers
    # that converge in few iterations.
    callbacks = [s for s in history if getattr(s, "source", None) == "callback"]
    if len(callbacks) >= 2:
        first_eval = next(
            (s for s in history if getattr(s, "source", None) == "evaluation"),
            None,
        )
        source = ([first_eval] + callbacks) if first_eval is not None else callbacks
    else:
        source = list(history)
    n = len(source)
    if n == 0:
        return []

    # Evenly-spaced sampling produces a much smoother animation than the
    # previous step-based slicing, which left the playback stuttering
    # whenever ``n`` was not an exact multiple of ``max_frames``.
    indices = np.unique(np.linspace(0, n - 1, num=min(max_frames, n), dtype=int))

    frames: list[IVAnimationFrame] = []
    for idx in indices:
        snap = source[idx]
        try:
            model = rebuild_model(model_key, snap.params_natural)
            if model is None:
                continue
            grid = model_iv_grid(model, market_data, meta)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        frames.append(
            IVAnimationFrame(
                iter_index=idx,
                objective=float(snap.objective),
                iv_grid=grid,
                params_natural=dict(snap.params_natural),
            )
        )
    return frames


__all__ = [
    "model_iv_grid",
    "surface_model_prices",
    "garch_implied_iv_grid",
    "iv_grid_from_simulator",
    "rebuild_model",
    "iv_grid_animation_frames",
    "IVAnimationFrame",
]
