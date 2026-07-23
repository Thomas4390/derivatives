"""
Structured Product Monte Carlo Engine
=======================================

Full-path Monte Carlo engine for pricing structured products.

Reuses the existing simulation infrastructure (BaseSimulator, Model.create_simulator)
for path generation and adds structured-product-specific evaluation logic.

Features:
- price(): Full pricing with decomposition, probabilities, scenarios
- greeks(): Bump-and-reprice Greeks (delta, vega, rho, theta)
- scenario_analysis(): Price across a range of spot prices

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

from backend.utils.constants.monte_carlo import (
    DEFAULT_MC_PATHS,
    DEFAULT_MC_STEPS_PER_YEAR,
    DEFAULT_RATE_BUMP,
    DEFAULT_SPOT_BUMP,
    DEFAULT_VOL_BUMP,
)
from backend.utils.constants.time import CALENDAR_DAYS_PER_YEAR
from backend.core.interfaces import Instrument, Model, PricingEngine
from backend.core.market import EnrichedMarketEnvironment, MarketEnvironment, YieldCurve
from backend.core.result_types import (
    ExerciseStyle,
    GreeksResult,
    PricingCapability,
    PricingResult,
    StructuredProductResult,
)
from backend.core.structured_product import (
    EvaluationContext,
    ObservationSchedule,
    StructuredProduct,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.simulation.base import SimulationResult


@dataclass
class StructuredProductMCEngine(PricingEngine):
    """
    Monte Carlo engine for structured products.

    Implements PricingEngine so it integrates with EngineRegistry and
    GreeksCalculator.  The rich API (returning StructuredProductResult)
    is exposed via ``price_structured()``.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    n_steps_per_year : int
        Time steps per year (252 = daily).
    seed : int, optional
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates for variance reduction.
    """

    n_paths: int = DEFAULT_MC_PATHS
    n_steps_per_year: int = DEFAULT_MC_STEPS_PER_YEAR
    seed: int | None = None
    antithetic: bool = True

    # =========================================================================
    # PricingEngine interface
    # =========================================================================

    @property
    def capability(self) -> PricingCapability:
        return PricingCapability.MONTE_CARLO

    @property
    def supported_exercises(self) -> list[ExerciseStyle]:
        return [ExerciseStyle.EUROPEAN]

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        return isinstance(instrument, StructuredProduct)

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """
        PricingEngine-compatible pricing entry point.

        Delegates to ``price_structured()`` and wraps the result
        in a ``PricingResult`` with structured-product metadata.
        """
        if not isinstance(instrument, StructuredProduct):
            raise TypeError(
                f"StructuredProductMCEngine requires a StructuredProduct, "
                f"got {type(instrument).__name__}"
            )
        sp_result = self.price_structured(instrument, model, market)
        return PricingResult(
            price=sp_result.price,
            engine=sp_result.engine,
            model=sp_result.model,
            error=sp_result.error,
            metadata={
                "fair_value": sp_result.fair_value,
                "notional": sp_result.notional,
                "bond_floor": sp_result.bond_floor,
                "option_value": sp_result.option_value,
                "expected_coupon": sp_result.expected_coupon,
                "autocall_probability": sp_result.autocall_probability,
                "capital_loss_probability": sp_result.capital_loss_probability,
                "expected_return": sp_result.expected_return,
                "worst_case_return": sp_result.worst_case_return,
                "best_case_return": sp_result.best_case_return,
            },
        )

    # =========================================================================
    # Rich Structured-Product Pricing
    # =========================================================================

    def price_structured(
        self,
        product: StructuredProduct,
        model: Model,
        market: MarketEnvironment,
        s0_reference: float = -1.0,
        z_matrix: np.ndarray | None = None,
    ) -> StructuredProductResult:
        """
        Price a structured product.

        Parameters
        ----------
        product : StructuredProduct
            The structured product to price.
        model : Model
            Stochastic model for the underlying.
        market : MarketEnvironment
            Market conditions.
        z_matrix : np.ndarray, optional
            Pre-generated standard normal draws for deterministic simulation.
            Shape (n_half_paths, n_steps).  When provided, paths are built
            via vectorised numpy (bypassing the Numba kernel) so that
            bump-and-reprice Greeks use Common Random Numbers.

        Returns
        -------
        StructuredProductResult
        """
        # Simulate full paths
        sim_result = self._simulate(product, model, market, z_matrix=z_matrix)
        paths = sim_result.price_paths
        time_grid = sim_result.time_grid

        # Map observation dates to time grid indices
        schedule = product.observation_schedule
        obs_indices = self._map_obs_to_grid(schedule, time_grid)

        # Compute discount factors at observation dates
        obs_times = time_grid[obs_indices]
        discount_factors = self._get_discount_factors(market, obs_times)

        # Period lengths
        prev_times = np.concatenate([[0.0], obs_times[:-1]])
        obs_dt = obs_times - prev_times

        # Terminal discount factor
        df_terminal = self._get_discount_factor(market, product.maturity)

        # Polymorphic dispatch: let the product evaluate its own paths
        result = product.evaluate_paths(
            paths,
            obs_indices,
            discount_factors,
            obs_dt,
            df_terminal,
            s0_reference=s0_reference,
        )

        # Fallback to generic component-by-component evaluation
        if result is None:
            result = self._price_generic(
                product,
                paths,
                time_grid,
                obs_indices,
                discount_factors,
                obs_dt,
                df_terminal,
                s0_reference=s0_reference,
            )
            logger.debug(
                "Priced %s via generic fallback, %d paths", product.name, paths.shape[0]
            )
        else:
            logger.debug(
                "Priced %s via custom evaluate_paths, %d paths",
                product.name,
                paths.shape[0],
            )

        return self._build_result(
            product,
            model,
            market,
            **result,
        )

    def _price_generic(
        self,
        product: StructuredProduct,
        paths: np.ndarray,
        time_grid: np.ndarray,
        obs_indices: np.ndarray,
        discount_factors: np.ndarray,
        obs_dt: np.ndarray | None = None,
        df_terminal: float = 0.0,
        s0_reference: float = -1.0,
    ) -> dict[str, object]:
        """Generic fallback: evaluate each component via EvaluationContext."""
        from backend.instruments.structured.components import (
            AutocallTrigger,
            BondFloor,
            ConditionalCoupon,
            FixedCoupon,
            KnockInPut,
            SnowballCoupon,
        )

        n_paths = paths.shape[0]

        # Build obs_dt if not provided (backward compat)
        if obs_dt is None:
            obs_times = time_grid[obs_indices]
            prev_times = np.concatenate([[0.0], obs_times[:-1]])
            obs_dt = obs_times - prev_times
        if df_terminal == 0.0:
            df_terminal = float(discount_factors[-1])

        ctx = EvaluationContext.create(
            paths,
            time_grid,
            obs_indices,
            discount_factors,
            obs_dt,
            df_terminal,
            s0_reference,
        )

        pv = np.zeros(n_paths)
        bond_floor_pv = np.zeros(n_paths)
        option_pv = np.zeros(n_paths)
        coupon_pv = np.zeros(n_paths)

        for comp in product.components:
            contribution = comp.evaluate_in_context(ctx)
            pv += contribution

            # Tag by component type for decomposition
            if isinstance(comp, BondFloor):
                bond_floor_pv += contribution
            elif isinstance(comp, (FixedCoupon, ConditionalCoupon, SnowballCoupon)):
                coupon_pv += contribution
            elif isinstance(comp, (AutocallTrigger, KnockInPut)):
                bond_floor_pv += contribution
            else:
                option_pv += contribution

        autocall_prob = (
            float(np.mean(~ctx.alive)) if product.has_early_termination() else 0.0
        )

        return {
            "pv": pv,
            "bond_floor_pv": bond_floor_pv,
            "option_pv": option_pv,
            "coupon_pv": coupon_pv,
            "autocall_probability": autocall_prob,
        }

    # =========================================================================
    # Greeks
    # =========================================================================

    def greeks(
        self,
        product: StructuredProduct,
        model: Model,
        market: MarketEnvironment,
        spot_bump: float = DEFAULT_SPOT_BUMP,
        rate_bump: float = DEFAULT_RATE_BUMP,
        vol_bump: float = DEFAULT_VOL_BUMP,
    ) -> GreeksResult:
        """
        Calculate Greeks via bump-and-reprice.

        Uses pre-generated random draws (Common Random Numbers) so that all
        bumped simulations share the exact same Brownian paths.  This removes
        MC noise from finite-difference Greeks.

        Parameters
        ----------
        product : StructuredProduct
            The product.
        model : Model
            The model.
        market : MarketEnvironment
            Market conditions.
        spot_bump : float
            Relative spot bump (default 1%).
        rate_bump : float
            Absolute rate bump (default 1bp).
        vol_bump : float
            Absolute vol bump (default 1%).

        Returns
        -------
        GreeksResult
        """
        # Pre-generate random draws once — reused by every bumped scenario
        seed = self.seed if self.seed is not None else 42
        n_steps = max(1, int(product.maturity * self.n_steps_per_year))
        half = self.n_paths // 2 if self.antithetic else self.n_paths
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((half, n_steps))

        # Fix the reference level to the current spot so that bumped
        # scenarios correctly shift relative performance ratios.
        s0_ref = market.spot

        # Base price
        v0 = self.price_structured(
            product, model, market, s0_reference=s0_ref, z_matrix=z
        ).price

        # Delta (spot bump)
        h_s = market.spot * spot_bump
        v_up = self.price_structured(
            product, model, market.bump_spot(h_s), s0_reference=s0_ref, z_matrix=z
        ).price
        v_down = self.price_structured(
            product, model, market.bump_spot(-h_s), s0_reference=s0_ref, z_matrix=z
        ).price
        delta = (v_up - v_down) / (2 * h_s)
        gamma = (v_up - 2 * v0 + v_down) / (h_s**2)

        # Rho (rate bump)
        h_r = rate_bump
        v_r_up = self.price_structured(
            product, model, market.bump_rate(h_r), s0_reference=s0_ref, z_matrix=z
        ).price
        v_r_down = self.price_structured(
            product, model, market.bump_rate(-h_r), s0_reference=s0_ref, z_matrix=z
        ).price
        rho = (v_r_up - v_r_down) / (2 * h_r) / 100

        # Vega (vol bump)
        vega = self._compute_vega(
            product, model, market, vol_bump, s0_ref=s0_ref, z_matrix=z
        )

        # Theta (1-day decay)
        theta = self._compute_theta(
            product, model, market, v0, s0_ref=s0_ref, z_matrix=z
        )

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

    def _compute_vega(
        self,
        product: StructuredProduct,
        model: Model,
        market: MarketEnvironment,
        h: float,
        s0_ref: float = -1.0,
        z_matrix: np.ndarray | None = None,
    ) -> float:
        """Compute vega by bumping model volatility."""
        from backend.models.vol_bump import create_vol_bumped_pair

        model_up, model_down = create_vol_bumped_pair(model, h)
        if model_up is not None and model_down is not None:
            v_up = self.price_structured(
                product, model_up, market, s0_reference=s0_ref, z_matrix=z_matrix
            ).price
            v_down = self.price_structured(
                product, model_down, market, s0_reference=s0_ref, z_matrix=z_matrix
            ).price
            return (v_up - v_down) / (2 * h) / 100
        logger.warning(
            "Cannot compute vega for model %s: vol bump unsupported", model.name
        )
        return 0.0

    def _compute_theta(
        self,
        product: StructuredProduct,
        model: Model,
        market: MarketEnvironment,
        v0: float,
        s0_ref: float = -1.0,
        z_matrix: np.ndarray | None = None,
    ) -> float:
        """
        Compute theta via bump-and-reprice with maturity reduced by 1 day.
        """
        from dataclasses import replace

        dt = 1.0 / CALENDAR_DAYS_PER_YEAR
        new_mat = product.maturity - dt
        if new_mat < 2 * dt:
            return 0.0

        try:
            new_product = replace(product, maturity_=new_mat)
            # Truncate Z to match the shorter maturity
            z_theta = None
            if z_matrix is not None:
                new_steps = int(new_mat * self.n_steps_per_year)
                z_theta = z_matrix[:, :new_steps]
            v_shifted = self.price_structured(
                new_product, model, market, s0_reference=s0_ref, z_matrix=z_theta
            ).price
            return v_shifted - v0
        except (ValueError, TypeError):
            return 0.0

    # =========================================================================
    # Scenario Analysis
    # =========================================================================

    def scenario_analysis(
        self,
        product: StructuredProduct,
        model: Model,
        market: MarketEnvironment,
        spot_range: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Price the product across a range of spot prices.

        Parameters
        ----------
        product : StructuredProduct
            The product.
        model : Model
            The model.
        market : MarketEnvironment
            Base market conditions.
        spot_range : np.ndarray
            Array of spot prices to evaluate.

        Returns
        -------
        dict
            Keys: 'spots', 'fair_values', 'prices', 'deltas'.
        """
        fair_values = np.zeros(len(spot_range))
        prices = np.zeros(len(spot_range))
        deltas = np.zeros(len(spot_range))

        s0_ref = market.spot  # Fix reference level for scenario analysis
        for i, spot in enumerate(spot_range):
            bumped_market = market.with_spot(spot)
            result = self.price_structured(
                product, model, bumped_market, s0_reference=s0_ref
            )
            fair_values[i] = result.fair_value
            prices[i] = result.price

            # Numerical delta
            h = spot * 0.01
            v_up = self.price_structured(
                product, model, bumped_market.bump_spot(h), s0_reference=s0_ref
            ).price
            v_down = self.price_structured(
                product, model, bumped_market.bump_spot(-h), s0_reference=s0_ref
            ).price
            deltas[i] = (v_up - v_down) / (2 * h)

        return {
            "spots": spot_range,
            "fair_values": fair_values,
            "prices": prices,
            "deltas": deltas,
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _simulate(
        self,
        product: StructuredProduct,
        model: Model,
        market: MarketEnvironment,
        z_matrix: np.ndarray | None = None,
    ) -> SimulationResult:
        """Simulate full paths using the model's simulator or pre-generated Z."""
        if z_matrix is not None:
            return self._simulate_from_z(product, model, market, z_matrix)
        n_steps = max(1, int(product.maturity * self.n_steps_per_year))
        simulator = model.create_simulator(antithetic=self.antithetic)
        return simulator.simulate_paths(
            s0=market.spot,
            mu=market.rate - market.dividend_yield,
            t=product.maturity,
            n_paths=self.n_paths,
            n_steps=n_steps,
            seed=self.seed,
        )

    def _simulate_from_z(
        self,
        product: StructuredProduct,
        model: Model,
        market: MarketEnvironment,
        z_matrix: np.ndarray,
    ) -> SimulationResult:
        """Build GBM paths from pre-generated standard normal draws.

        This bypasses the Numba kernel (which is non-deterministic due to
        parallel thread scheduling) and uses pure numpy vectorisation instead.
        The same ``z_matrix`` reused across bumped scenarios guarantees
        Common Random Numbers for clean bump-and-reprice Greeks.

        Warning: This method uses a GBM approximation regardless of the actual
        model type. For stochastic volatility models (Heston, Bates), the CRN
        Greeks will be approximate since the volatility dynamics are collapsed
        to constant vol (sqrt(v0)).
        """
        params = model.get_parameters()
        if "v0" in params or "kappa" in params:
            import logging

            logging.getLogger(__name__).warning(
                "CRN Greeks for structured products use GBM approximation "
                "with sigma=sqrt(v0) for model '%s'. Stochastic volatility "
                "dynamics are not captured in bump-and-reprice scenarios.",
                model.name,
            )
        from backend.simulation.base import SimulationResult

        n_steps = max(1, int(product.maturity * self.n_steps_per_year))
        dt = product.maturity / n_steps
        mu = market.rate - market.dividend_yield
        sigma = self._extract_sigma(model)
        s0 = market.spot

        sqrt_dt = np.sqrt(dt)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * sqrt_dt

        half = z_matrix.shape[0]
        z = z_matrix[:, :n_steps]  # safety truncation

        if self.antithetic:
            n_paths = half * 2
            log_inc = drift + diffusion * z  # (half, n_steps)
            log_inc_anti = drift - diffusion * z

            cum = np.cumsum(log_inc, axis=1)
            cum_anti = np.cumsum(log_inc_anti, axis=1)

            paths = np.empty((n_paths, n_steps + 1))
            paths[:half, 0] = s0
            paths[half:, 0] = s0
            paths[:half, 1:] = s0 * np.exp(cum)
            paths[half:, 1:] = s0 * np.exp(cum_anti)
        else:
            n_paths = half
            log_inc = drift + diffusion * z
            cum = np.cumsum(log_inc, axis=1)
            paths = np.empty((n_paths, n_steps + 1))
            paths[:, 0] = s0
            paths[:, 1:] = s0 * np.exp(cum)

        time_grid = np.linspace(0, product.maturity, n_steps + 1)

        return SimulationResult(
            price_paths=paths,
            time_grid=time_grid,
            model_name=model.name,
            computation_time=0.0,
            n_paths=n_paths,
            n_steps=n_steps,
            volatility_paths=None,
            parameters={"s0": s0, "mu": mu, "sigma": sigma},
        )

    @staticmethod
    def _extract_sigma(model: Model) -> float:
        """Extract the diffusion volatility from a model."""
        if hasattr(model, "sigma"):
            return model.sigma
        params = model.get_parameters()
        if "sigma" in params:
            return params["sigma"]
        # Heston / Bates: use sqrt(v0) as instantaneous vol
        if "v0" in params:
            return float(np.sqrt(params["v0"]))
        raise ValueError(f"Cannot extract sigma from model {model.name}")

    @staticmethod
    def _map_obs_to_grid(
        schedule: ObservationSchedule, time_grid: np.ndarray
    ) -> np.ndarray:
        """Map observation times to nearest indices in time grid."""
        obs_times = np.array(schedule.times)
        # Use nearest-index instead of searchsorted insertion point
        indices = np.argmin(
            np.abs(time_grid[np.newaxis, :] - obs_times[:, np.newaxis]),
            axis=1,
        )
        return indices

    @staticmethod
    def _get_discount_factors(
        market: MarketEnvironment,
        times: np.ndarray,
    ) -> np.ndarray:
        """Get discount factors from market, using yield curve if available."""
        if isinstance(market, EnrichedMarketEnvironment):
            return market.discount_factors(times)
        return np.exp(-market.rate * times)

    @staticmethod
    def _get_discount_factor(
        market: MarketEnvironment,
        t: float,
    ) -> float:
        """Get single discount factor."""
        if isinstance(market, EnrichedMarketEnvironment):
            return market.discount_factor(t)
        return float(np.exp(-market.rate * t))

    def _build_result(
        self,
        product: StructuredProduct,
        model: Model,
        market: MarketEnvironment,
        pv: np.ndarray,
        bond_floor_pv: np.ndarray,
        option_pv: np.ndarray,
        coupon_pv: np.ndarray,
        autocall_probability: float,
    ) -> StructuredProductResult:
        """Build StructuredProductResult from per-path PVs."""
        from backend.engines.structured.kernels import aggregate_result_stats

        notional = product.notional
        df_terminal = self._get_discount_factor(market, product.maturity)

        (
            mean_pv,
            std_err,
            bond_floor_pct,
            option_pct,
            coupon_pct,
            capital_loss_prob,
            worst_case,
            best_case,
        ) = aggregate_result_stats(
            pv,
            bond_floor_pv,
            option_pv,
            coupon_pv,
            notional,
            df_terminal,
        )

        fair_value = mean_pv / notional * 100
        expected_return = mean_pv / notional - 1.0

        return StructuredProductResult.create(
            fair_value=fair_value,
            price=mean_pv,
            notional=notional,
            engine="StructuredProductMCEngine",
            model=model.name,
            error=std_err,
            bond_floor=bond_floor_pct,
            option_value=option_pct,
            expected_coupon=coupon_pct,
            autocall_probability=autocall_probability,
            capital_loss_probability=capital_loss_prob,
            expected_return=expected_return,
            worst_case_return=worst_case,
            best_case_return=best_case,
        )


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    from backend.core.market import (
        EnrichedMarketEnvironment,
        MarketEnvironment,
        YieldCurve,
    )
    from backend.instruments.structured.products import (
        Autocallable,
        CapitalProtectedNote,
        ReverseConvertible,
    )
    from backend.models.gbm import GBMModel

    print("=" * 60)
    print("Structured MC Engine Smoke Test")
    print("=" * 60)

    model = GBMModel(sigma=0.20)
    market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.0)
    engine = StructuredProductMCEngine(n_paths=50_000, seed=42)

    # --- CPN ---
    cpn = CapitalProtectedNote(
        notional_=1000,
        maturity_=3.0,
        participation_rate=0.80,
        cap=1.50,
    )
    result = engine.price_structured(cpn, model, market)
    print(f"\nCPN: fair_value={result.fair_value:.2f}%, price={result.price:.2f}")
    print(f"  Bond floor: {result.bond_floor:.2f}%, Option: {result.option_value:.2f}%")
    print(f"  Error: {result.error:.2f}")

    # --- Test PricingEngine.price() wrapper ---
    pr = engine.price(cpn, model, market)
    print(
        f"  PricingResult: price={pr.price:.2f}, metadata keys={list(pr.metadata.keys())}"
    )

    # --- Reverse Convertible ---
    rc = ReverseConvertible(
        notional_=1000,
        maturity_=1.0,
        coupon_rate=0.12,
        barrier=0.60,
    )
    result = engine.price_structured(rc, model, market)
    print(f"\nRC: fair_value={result.fair_value:.2f}%, price={result.price:.2f}")
    print(f"  Coupon: {result.expected_coupon:.2f}%")
    print(f"  Capital loss prob: {result.capital_loss_probability:.2%}")

    # --- Autocallable ---
    auto = Autocallable(
        notional_=1000,
        maturity_=3.0,
        coupon_rate=0.07,
        autocall_trigger=1.0,
        coupon_barrier=0.70,
        ki_barrier=0.60,
    )
    result = engine.price_structured(auto, model, market)
    print(
        f"\nAutocallable: fair_value={result.fair_value:.2f}%, price={result.price:.2f}"
    )
    print(f"  Autocall prob: {result.autocall_probability:.2%}")
    print(f"  Expected return: {result.expected_return:.2%}")
    print(f"  Worst case (5%): {result.worst_case_return:.2%}")
    print(f"  Best case (95%): {result.best_case_return:.2%}")

    # --- Test with YieldCurve ---
    print("\n--- With Yield Curve ---")
    curve = YieldCurve(
        tenors=np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0]),
        rates=np.array([0.03, 0.035, 0.04, 0.045, 0.05, 0.055]),
    )
    enriched_market = EnrichedMarketEnvironment(
        spot=100,
        rate=0.05,
        yield_curve=curve,
        credit_spread=0.01,
    )
    result_curve = engine.price_structured(cpn, model, enriched_market)
    print(f"CPN with curve: fair_value={result_curve.fair_value:.2f}%")

    print("\n" + "=" * 60)
    print("Structured MC Engine smoke test passed")
    print("=" * 60)
