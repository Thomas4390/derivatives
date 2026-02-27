"""
Monte Carlo Pricing Engine
==========================

Monte Carlo pricing engine wrapper for the new architecture.

This engine wraps the generic Monte Carlo machinery and implements the
PricingEngine interface, bridging Instrument/Model/Market to the
underlying simulation infrastructure.

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
import numpy as np

from backend.core.interfaces import PricingEngine, Instrument, Model
from backend.core.market import MarketEnvironment
from backend.core.result_types import (
    PricingResult, PricingCapability, ExerciseStyle, GreeksResult
)
from backend.instruments.options import VanillaOption
from backend.engines.monte_carlo.mc_base import (
    GenericMCEngine, MCConfig
)

# Import model types for type checking
from backend.models.gbm import GBMModel
from backend.models.heston import HestonModel
from backend.models.bates import BatesModel
from backend.models.merton import MertonModel


# =============================================================================
# MONTE CARLO ENGINE
# =============================================================================

@dataclass
class MonteCarloEngine(PricingEngine):
    """
    Monte Carlo pricing engine.

    Prices options using Monte Carlo simulation. Supports any model that
    provides drift/diffusion for SDE discretization.

    Supported Instruments
    ---------------------
    - European vanilla calls and puts (VanillaOption with ExerciseStyle.EUROPEAN)
    - Future: American options with LSM algorithm

    Supported Models
    ----------------
    - GBMModel, HestonModel, BatesModel, MertonModel
    - Any model that supports PricingCapability.MONTE_CARLO

    Parameters
    ----------
    n_paths : int
        Number of simulation paths (default 100,000)
    n_steps : int
        Number of time steps per path (default 252)
    seed : int, optional
        Random seed for reproducibility
    antithetic : bool
        Use antithetic variates (default True)

    Examples
    --------
    engine = MonteCarloEngine(n_paths=100000, seed=42)

    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    market = MarketEnvironment(spot=100, rate=0.05)

    result = engine.price(option, model, market)
    print(f"Price: ${result.price:.4f} ± ${result.error:.4f}")
    """

    n_paths: int = 100_000
    n_steps: int = 252
    seed: Optional[int] = None
    antithetic: bool = True

    def __post_init__(self):
        """Initialize the underlying MC engine."""
        self._config = MCConfig(
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            antithetic=self.antithetic,
            seed=self.seed
        )

    @property
    def capability(self) -> PricingCapability:
        """This is a Monte Carlo engine."""
        return PricingCapability.MONTE_CARLO

    @property
    def supported_exercises(self) -> List[ExerciseStyle]:
        """European and potentially American with LSM."""
        return [ExerciseStyle.EUROPEAN]

    def can_price(self, instrument: Instrument, model: Model) -> bool:
        """
        Check if this engine can price the given combination.

        MonteCarloEngine requires:
        - European exercise style (for now)
        - Vanilla options only (path-dependent exotics need full path simulation)
        - Model supports Monte Carlo
        """
        if instrument.exercise_style != ExerciseStyle.EUROPEAN:
            return False

        # Reject path-dependent exotic options — terminal simulation only
        from backend.instruments.options import (
            BarrierOption, AsianOption, DigitalOption, LookbackOption
        )
        if isinstance(instrument, (BarrierOption, AsianOption, DigitalOption, LookbackOption)):
            return False

        # Requires a fixed strike
        if not hasattr(instrument, 'strike'):
            return False

        if PricingCapability.MONTE_CARLO not in model.supported_engines:
            return False

        # Hardcoded models with optimized terminal simulators
        _SUPPORTED = (GBMModel, HestonModel, BatesModel, MertonModel)
        if isinstance(model, _SUPPORTED):
            return True

        # Accept custom models that have drift/diffusion for generic Euler
        try:
            model.drift(100.0, 0.04, 0.0, 0.05, 0.0)
            model.diffusion(100.0, 0.04, 0.0)
            return True
        except (NotImplementedError, TypeError, AttributeError):
            return False

    def price(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> PricingResult:
        """
        Price an option using Monte Carlo simulation.

        Parameters
        ----------
        instrument : VanillaOption
            The option to price
        model : Model
            The stochastic model
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        PricingResult
            MC price with standard error
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"MonteCarloEngine cannot price {type(instrument).__name__} "
                f"with {type(model).__name__}"
            )

        option = instrument  # type: VanillaOption
        s0 = market.spot
        k = option.strike
        t = option.maturity
        r = market.rate
        q = market.dividend_yield

        # Get terminal simulator for this model
        terminal_simulator = self._get_terminal_simulator(model, q)

        # Create generic MC engine and price
        mc_engine = GenericMCEngine(self._config)
        result = mc_engine.price(
            terminal_simulator=terminal_simulator,
            s0=s0, k=k, t=t, r=r,
            is_call=option.is_call,
            n_paths=self.n_paths, n_steps=self.n_steps, seed=self.seed
        )

        return PricingResult(
            price=result.price,
            engine="MonteCarloEngine",
            model=model.name,
            error=result.std_error
        )

    def greeks(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> GreeksResult:
        """
        Calculate Greeks via numerical finite differences.

        Note: MonteCarloEngine returns only first-order Greeks (delta, gamma, vega,
        theta, rho). For higher-order Greeks, use the GreeksCalculator class which
        can compute second-order (vanna, volga, charm, veta) and third-order
        (speed, zomma, color, ultima) Greeks numerically.

        Parameters
        ----------
        instrument : Instrument
            The option to calculate Greeks for
        model : Model
            The stochastic model
        market : MarketEnvironment
            Current market conditions

        Returns
        -------
        GreeksResult
            First-order Greeks only (delta, gamma, vega, theta, rho)
        """
        from backend.greeks.numerical import ModelNumericalGreeks

        calc = ModelNumericalGreeks()
        num_greeks = calc.calculate(self, instrument, model, market)

        return GreeksResult(
            delta=num_greeks.delta,
            gamma=num_greeks.gamma,
            vega=num_greeks.vega,
            theta=num_greeks.theta,
            rho=num_greeks.rho,
        )

    def price_with_paths(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
    ) -> Tuple[PricingResult, np.ndarray]:
        """
        Price an option and return the simulated terminal prices.

        Useful for diagnostics and Greeks computation via pathwise.

        Returns
        -------
        Tuple[PricingResult, np.ndarray]
            Pricing result and terminal prices array
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"MonteCarloEngine cannot price {type(instrument).__name__} "
                f"with {type(model).__name__}"
            )

        option = instrument  # type: VanillaOption
        s0 = market.spot
        k = option.strike
        t = option.maturity
        r = market.rate
        q = market.dividend_yield

        # Get terminal simulator
        terminal_simulator = self._get_terminal_simulator(model, q)

        # Simulate terminal prices
        terminals = terminal_simulator(
            s0=s0, t=t, r=r, n_paths=self.n_paths, n_steps=self.n_steps, seed=self.seed
        )

        # Compute payoffs
        if option.is_call:
            payoffs = np.maximum(terminals - k, 0.0)
        else:
            payoffs = np.maximum(k - terminals, 0.0)

        # Compute price and standard error
        discount = np.exp(-r * t)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))

        result = PricingResult(
            price=max(price, 0.0),
            engine="MonteCarloEngine",
            model=model.name,
            error=std_error
        )

        return result, terminals

    def price_strikes(
        self,
        instrument: Instrument,
        model: Model,
        market: MarketEnvironment,
        strikes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Price multiple strikes with single simulation.

        Parameters
        ----------
        instrument : VanillaOption
            Template option (strike ignored)
        model : Model
            The stochastic model
        market : MarketEnvironment
            Current market conditions
        strikes : np.ndarray
            Array of strikes to price

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Prices and standard errors
        """
        if not self.can_price(instrument, model):
            raise ValueError(
                f"MonteCarloEngine cannot price {type(instrument).__name__} "
                f"with {type(model).__name__}"
            )

        option = instrument  # type: VanillaOption
        s0 = market.spot
        t = option.maturity
        r = market.rate
        q = market.dividend_yield

        # Get terminal simulator
        terminal_simulator = self._get_terminal_simulator(model, q)

        # Create generic MC engine
        mc_engine = GenericMCEngine(self._config)

        # Price all strikes
        prices, std_errors = mc_engine.price_strikes(
            terminal_simulator=terminal_simulator,
            s0=s0, strikes=strikes, t=t, r=r,
            is_call=option.is_call,
            n_paths=self.n_paths, n_steps=self.n_steps, seed=self.seed
        )

        return prices, std_errors

    def _get_terminal_simulator(
        self, model: Model, q: float
    ) -> Callable[[float, float, float, int, int, Optional[int]], np.ndarray]:
        """
        Build a terminal price simulator for the given model.

        Returns a callable with signature:
            simulator(s0, t, r, n_paths, n_steps, seed) -> terminals

        Parameters
        ----------
        model : Model
            The stochastic model
        q : float
            Dividend yield (for drift adjustment)

        Returns
        -------
        Callable
            Terminal price simulator function
        """
        if isinstance(model, GBMModel):
            return self._make_gbm_simulator(model, q)
        elif isinstance(model, HestonModel):
            return self._make_heston_simulator(model, q)
        elif isinstance(model, BatesModel):
            return self._make_bates_simulator(model, q)
        elif isinstance(model, MertonModel):
            return self._make_merton_simulator(model, q)
        else:
            return self._make_generic_simulator(model, q)

    def _make_gbm_simulator(
        self, model: GBMModel, q: float
    ) -> Callable[[float, float, float, int, int, Optional[int]], np.ndarray]:
        """Create GBM terminal simulator."""
        from backend.simulation.models.gbm import GBMSimulator

        sigma = model.sigma
        antithetic = self.antithetic

        def simulator(s0, t, r, n_paths, n_steps, seed=None):
            sim = GBMSimulator(sigma=sigma, antithetic=antithetic)
            mu = r - q  # Risk-neutral drift
            return sim.simulate_terminal(s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)

        return simulator

    def _make_heston_simulator(
        self, model: HestonModel, q: float
    ) -> Callable[[float, float, float, int, int, Optional[int]], np.ndarray]:
        """Create Heston terminal simulator."""
        from backend.simulation.models.heston import HestonSimulator

        v0 = model.v0
        kappa = model.kappa
        theta = model.theta
        xi = model.xi
        rho = model.rho

        def simulator(s0, t, r, n_paths, n_steps, seed=None):
            sim = HestonSimulator(
                v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho
            )
            mu = r - q  # Risk-neutral drift
            return sim.simulate_terminal(s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)

        return simulator

    def _make_bates_simulator(
        self, model: BatesModel, q: float
    ) -> Callable[[float, float, float, int, int, Optional[int]], np.ndarray]:
        """Create Bates terminal simulator."""
        from backend.simulation.models.bates import BatesSimulator

        def simulator(s0, t, r, n_paths, n_steps, seed=None):
            sim = BatesSimulator(
                v0=model.v0, kappa=model.kappa, theta=model.theta,
                xi=model.xi, rho=model.rho,
                lambda_j=model.lambda_j, mu_j=model.mu_j, sigma_j=model.sigma_j
            )
            mu = r - q  # Risk-neutral drift (jump adjustment in simulator)
            return sim.simulate_terminal(s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)

        return simulator

    def _make_merton_simulator(
        self, model: MertonModel, q: float
    ) -> Callable[[float, float, float, int, int, Optional[int]], np.ndarray]:
        """Create Merton terminal simulator."""
        from backend.simulation.models.merton import MertonSimulator

        def simulator(s0, t, r, n_paths, n_steps, seed=None):
            sim = MertonSimulator(
                sigma=model.sigma,
                lambda_j=model.lambda_j, mu_j=model.mu_j, sigma_j=model.sigma_j
            )
            mu = r - q  # Risk-neutral drift (jump adjustment in simulator)
            return sim.simulate_terminal(s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)

        return simulator

    def _make_generic_simulator(
        self, model: Model, q: float
    ) -> Callable[[float, float, float, int, int, Optional[int]], np.ndarray]:
        """Create generic Euler-Maruyama terminal simulator for custom models."""
        from backend.simulation.models.generic_euler import GenericEulerSimulator

        sim = GenericEulerSimulator(model)

        def simulator(s0, t, r, n_paths, n_steps, seed=None):
            mu = r - q
            return sim.simulate_terminal(s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)

        return simulator


if __name__ == "__main__":
    from backend.instruments.options import VanillaOption
    from backend.models.gbm import GBMModel
    from backend.models.heston import HestonModel
    from backend.models.bates import BatesModel
    from backend.models.merton import MertonModel
    from backend.core.market import MarketEnvironment
    from backend.engines import BSAnalyticEngine, FFTEngine

    print("=" * 50)
    print("MonteCarloEngine Smoke Test")
    print("=" * 50)

    # Create components
    engine = MonteCarloEngine(n_paths=100_000, seed=42)
    option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    market = MarketEnvironment(spot=100, rate=0.05, dividend_yield=0.02)

    print("\n--- Setup ---")
    print(f"Option: K={option.strike}, T={option.maturity}, Call={option.is_call}")
    print(f"Market: S0={market.spot}, r={market.rate}, q={market.dividend_yield}")
    print(f"MC Config: n_paths={engine.n_paths}, seed={engine.seed}")

    # Test with GBM (compare to analytical)
    print("\n--- GBM Model (vs Analytical) ---")
    gbm = GBMModel(sigma=0.20)
    mc_result = engine.price(option, gbm, market)
    print(f"MC Price: ${mc_result.price:.4f} ± ${mc_result.error:.4f}")

    bs_engine = BSAnalyticEngine()
    bs_result = bs_engine.price(option, gbm, market)
    print(f"BS Price: ${bs_result.price:.4f}")
    print(f"Difference: ${abs(mc_result.price - bs_result.price):.4f} ({abs(mc_result.price - bs_result.price)/bs_result.price*100:.2f}%)")

    # Test with Heston (compare to FFT)
    print("\n--- Heston Model (vs FFT) ---")
    heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    mc_heston = engine.price(option, heston, market)
    print(f"MC Price: ${mc_heston.price:.4f} ± ${mc_heston.error:.4f}")

    fft_engine = FFTEngine()
    fft_result = fft_engine.price(option, heston, market)
    print(f"FFT Price: ${fft_result.price:.4f}")
    print(f"Difference: ${abs(mc_heston.price - fft_result.price):.4f}")

    # Test with Bates
    print("\n--- Bates Model (vs FFT) ---")
    bates = BatesModel(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
    )
    mc_bates = engine.price(option, bates, market)
    print(f"MC Price: ${mc_bates.price:.4f} ± ${mc_bates.error:.4f}")

    fft_bates = fft_engine.price(option, bates, market)
    print(f"FFT Price: ${fft_bates.price:.4f}")
    print(f"Difference: ${abs(mc_bates.price - fft_bates.price):.4f}")

    # Test with Merton
    print("\n--- Merton Model (vs FFT) ---")
    merton = MertonModel(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
    mc_merton = engine.price(option, merton, market)
    print(f"MC Price: ${mc_merton.price:.4f} ± ${mc_merton.error:.4f}")

    fft_merton = fft_engine.price(option, merton, market)
    print(f"FFT Price: ${fft_merton.price:.4f}")
    print(f"Difference: ${abs(mc_merton.price - fft_merton.price):.4f}")

    # Test put option
    print("\n--- Put Options ---")
    put_option = VanillaOption(strike=100, maturity=0.5, is_call=False)
    for model_name, model in [("GBM", gbm), ("Heston", heston)]:
        mc_put = engine.price(put_option, model, market)
        print(f"{model_name} Put: ${mc_put.price:.4f} ± ${mc_put.error:.4f}")

    # Test multiple strikes
    print("\n--- Multiple Strikes (Heston) ---")
    strikes = np.array([90, 95, 100, 105, 110])
    prices, errors = engine.price_strikes(option, heston, market, strikes)
    for k, p, e in zip(strikes, prices, errors):
        print(f"  K={k}: ${p:.4f} ± ${e:.4f}")

    # Test price_with_paths
    print("\n--- Price with Paths ---")
    result, terminals = engine.price_with_paths(option, gbm, market)
    print(f"Terminal prices: mean=${terminals.mean():.2f}, std=${terminals.std():.2f}")
    print(f"Price: ${result.price:.4f}")

    # Test can_price
    print("\n--- Compatibility Check ---")
    print(f"Can price European call with Heston: {engine.can_price(option, heston)}")
    print(f"Can price European call with GBM: {engine.can_price(option, gbm)}")

    print("\n" + "=" * 50)
    print("MonteCarloEngine smoke test passed")
    print("=" * 50)
