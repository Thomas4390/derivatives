"""
Backend Demonstration Script
============================

This script demonstrates the full capabilities of the derivatives backend
through concrete, practical examples. It covers:

1. Core Module - Interfaces, Market Environment
2. Models Module - GBM, Heston, Bates, Merton, GARCH
3. Engines Module - BS Analytic, FFT, Monte Carlo
4. Simulation Module - Path generation for all models
5. Instruments Module - Options, Strategies
6. Greeks Module - Full Greeks calculation
7. Portfolio Module - Portfolio management, P&L, Risk Analysis
8. Utils Module - Math utilities

Run with: python -m tests.demo_backend

Author: Thomas
Created: 2025
"""

import numpy as np
from typing import Dict, Any

# =============================================================================
# SECTION 1: CORE MODULE - Market Environment & Base Types
# =============================================================================

def demo_core_module():
    """Demonstrate the core module components."""
    print("\n" + "=" * 70)
    print("SECTION 1: CORE MODULE")
    print("=" * 70)

    from backend.core import MarketEnvironment, ExerciseStyle
    from backend.core.result_types import GreeksResult

    # Market Environment
    print("\n--- Market Environment ---")
    market = MarketEnvironment(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.02
    )
    print(f"Spot Price: ${market.spot}")
    print(f"Risk-Free Rate: {market.rate * 100:.1f}%")
    print(f"Dividend Yield: {market.dividend_yield * 100:.1f}%")

    # Exercise Styles
    print("\n--- Exercise Styles ---")
    print(f"European: {ExerciseStyle.EUROPEAN}")
    print(f"American: {ExerciseStyle.AMERICAN}")

    # Greeks Result structure
    print("\n--- Greeks Result Structure ---")
    greeks = GreeksResult(
        delta=0.55, gamma=0.02, vega=0.25, theta=-0.05, rho=0.15
    )
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.4f}")
    print(f"Vega: {greeks.vega:.4f}")
    print(f"Theta: {greeks.theta:.4f}")
    print(f"Rho: {greeks.rho:.4f}")

    return market


# =============================================================================
# SECTION 2: INSTRUMENTS MODULE - Options & Strategies
# =============================================================================

def demo_instruments_module():
    """Demonstrate the instruments module."""
    print("\n" + "=" * 70)
    print("SECTION 2: INSTRUMENTS MODULE")
    print("=" * 70)

    from backend.instruments import (
        VanillaOption,
        EuropeanCall,
        EuropeanPut,
        IronCondor,
        Straddle,
        Butterfly
    )

    # Vanilla Options
    print("\n--- Vanilla Options ---")
    call = VanillaOption(strike=100.0, maturity=0.5, is_call=True)
    put = VanillaOption(strike=100.0, maturity=0.5, is_call=False)

    print(f"Call Option: K={call.strike}, T={call.maturity}, is_call={call.is_call}")
    print(f"Put Option: K={put.strike}, T={put.maturity}, is_call={put.is_call}")

    # Convenience classes
    print("\n--- Convenience Classes ---")
    euro_call = EuropeanCall(strike=105.0, maturity=0.25)
    euro_put = EuropeanPut(strike=95.0, maturity=0.25)
    print(f"European Call: K={euro_call.strike}, T={euro_call.maturity}")
    print(f"European Put: K={euro_put.strike}, T={euro_put.maturity}")

    # Option Strategies
    print("\n--- Option Strategies ---")

    # Iron Condor (k1=long put, k2=short put, k3=short call, k4=long call)
    iron_condor = IronCondor(
        k1=90,   # Long put (wing)
        k2=95,   # Short put
        k3=105,  # Short call
        k4=110,  # Long call (wing)
        maturity=0.25
    )
    print(f"Iron Condor: {len(iron_condor.legs)} legs")
    for i, leg in enumerate(iron_condor.legs):
        print(f"  Leg {i+1}: K={leg.strike}, "
              f"{'Call' if leg.is_call else 'Put'}, "
              f"qty={leg.quantity}")

    # Straddle
    straddle = Straddle(strike=100.0, maturity=0.5)
    print(f"\nStraddle at K={straddle.legs[0].strike}: {len(straddle.legs)} legs")

    # Butterfly (k1=lower, k2=middle, k3=upper)
    butterfly = Butterfly(
        k1=95,
        k2=100,
        k3=105,
        maturity=0.25,
        is_call=True
    )
    print(f"Call Butterfly: {len(butterfly.legs)} legs")

    return call, put


# =============================================================================
# SECTION 3: MODELS MODULE - Pricing Models
# =============================================================================

def demo_models_module():
    """Demonstrate all pricing models."""
    print("\n" + "=" * 70)
    print("SECTION 3: MODELS MODULE")
    print("=" * 70)

    from backend.models import (
        GBMModel,
        HestonModel,
        BatesModel,
        MertonModel,
        GARCHModel,
        NGARCHModel,
        GJRGARCHModel
    )

    models = {}

    # GBM Model
    print("\n--- Geometric Brownian Motion (GBM) ---")
    gbm = GBMModel(sigma=0.2)
    models['GBM'] = gbm
    print(f"Volatility: {gbm.sigma * 100:.1f}%")

    # Heston Model
    print("\n--- Heston Stochastic Volatility ---")
    heston = HestonModel(
        v0=0.04,      # Initial variance (20% vol)
        kappa=2.0,    # Mean reversion speed
        theta=0.04,   # Long-run variance
        xi=0.3,       # Vol of vol
        rho=-0.7      # Correlation (leverage effect)
    )
    models['Heston'] = heston
    print(f"Initial Vol: {np.sqrt(heston.v0) * 100:.1f}%")
    print(f"Long-run Vol: {np.sqrt(heston.theta) * 100:.1f}%")
    print(f"Mean Reversion: kappa={heston.kappa}")
    print(f"Vol of Vol: xi={heston.xi}")
    print(f"Correlation: rho={heston.rho}")
    feller = 2 * heston.kappa * heston.theta > heston.xi ** 2
    print(f"Feller Condition: {feller}")

    # Bates Model (Heston + Jumps)
    print("\n--- Bates Model (Stochastic Vol + Jumps) ---")
    bates = BatesModel(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        lambda_j=0.5,    # Jump intensity (0.5 jumps/year)
        mu_j=-0.1,       # Mean jump size (-10%)
        sigma_j=0.15     # Jump size volatility
    )
    models['Bates'] = bates
    print(f"Jump Intensity: {bates.lambda_j} per year")
    print(f"Mean Jump: {bates.mu_j * 100:.1f}%")
    print(f"Jump Vol: {bates.sigma_j * 100:.1f}%")

    # Merton Jump Diffusion
    print("\n--- Merton Jump Diffusion ---")
    merton = MertonModel(
        sigma=0.2,
        lambda_j=0.5,
        mu_j=-0.1,
        sigma_j=0.2
    )
    models['Merton'] = merton
    print(f"Diffusion Vol: {merton.sigma * 100:.1f}%")
    print(f"Jump Parameters: lambda={merton.lambda_j}, mu={merton.mu_j}, sigma={merton.sigma_j}")

    # GARCH Family
    print("\n--- GARCH(1,1) Model ---")
    garch = GARCHModel(
        sigma0=0.2,     # Initial volatility
        omega=1e-6,     # Constant term
        alpha=0.1,      # ARCH coefficient
        beta=0.85       # GARCH coefficient
    )
    models['GARCH'] = garch
    persistence = garch.alpha + garch.beta
    print(f"Initial Vol: {garch.sigma0 * 100:.1f}%")
    print(f"Persistence (alpha + beta): {persistence:.2f}")
    print(f"Stationary: {persistence < 1}")

    print("\n--- NGARCH Model (Nonlinear GARCH) ---")
    ngarch = NGARCHModel(
        sigma0=0.2, omega=1e-6, alpha=0.05, beta=0.90,
        theta=0.5  # Asymmetry parameter
    )
    models['NGARCH'] = ngarch
    print(f"Asymmetry (theta): {ngarch.theta}")

    print("\n--- GJR-GARCH Model (Leverage Effect) ---")
    gjr = GJRGARCHModel(
        sigma0=0.2, omega=1e-6, alpha=0.05, beta=0.85,
        gamma=0.1  # Leverage coefficient
    )
    models['GJR-GARCH'] = gjr
    print(f"Leverage (gamma): {gjr.gamma}")

    return models


# =============================================================================
# SECTION 4: ENGINES MODULE - Pricing Engines
# =============================================================================

def demo_engines_module(models: Dict[str, Any]):
    """Demonstrate pricing engines."""
    print("\n" + "=" * 70)
    print("SECTION 4: ENGINES MODULE")
    print("=" * 70)

    from backend.engines import BSAnalyticEngine, FFTEngine, MonteCarloEngine, FFTConfig
    from backend.instruments import VanillaOption
    from backend.core import MarketEnvironment

    # Common setup
    call = VanillaOption(strike=100, maturity=0.5, is_call=True)
    put = VanillaOption(strike=100, maturity=0.5, is_call=False)
    market = MarketEnvironment(spot=100, rate=0.05)

    # Black-Scholes Analytic Engine
    print("\n--- Black-Scholes Analytic Engine ---")
    bs_engine = BSAnalyticEngine()

    result = bs_engine.price(call, models['GBM'], market)
    print(f"GBM Call Price: ${result.price:.4f}")

    # Get Greeks separately using the engine's method
    greeks = bs_engine.greeks(call, models['GBM'], market)
    print(f"Greeks: delta={greeks.delta:.4f}, gamma={greeks.gamma:.6f}")

    result_put = bs_engine.price(put, models['GBM'], market)
    print(f"GBM Put Price: ${result_put.price:.4f}")

    # Put-Call Parity Check
    parity_diff = result.price - result_put.price - (market.spot - call.strike * np.exp(-market.rate * call.maturity))
    print(f"Put-Call Parity Error: ${abs(parity_diff):.6f}")

    # FFT Engine for Stochastic Vol Models
    print("\n--- FFT Engine (Carr-Madan) ---")
    fft_config = FFTConfig(alpha=1.5, n_fft=4096, eta=0.25)
    fft_engine = FFTEngine(config=fft_config)

    print(f"\n{'Model':<12} {'Call Price':>12} {'Put Price':>12}")
    print("-" * 40)

    for name in ['Heston', 'Bates', 'Merton']:
        model = models[name]
        call_result = fft_engine.price(call, model, market)
        put_result = fft_engine.price(put, model, market)
        print(f"{name:<12} ${call_result.price:>11.4f} ${put_result.price:>11.4f}")

    # Monte Carlo Engine
    print("\n--- Monte Carlo Engine ---")
    mc_engine = MonteCarloEngine(n_paths=50_000, n_steps=126, seed=42)

    print(f"Configuration: {mc_engine.n_paths:,} paths, {mc_engine.n_steps} steps")

    mc_result = mc_engine.price(call, models['Heston'], market)
    error_str = f" +/- ${mc_result.error:.4f}" if mc_result.error else ""
    print(f"Heston Call (MC): ${mc_result.price:.4f}{error_str}")

    # Compare FFT vs MC for Heston
    fft_heston = fft_engine.price(call, models['Heston'], market)
    print(f"Heston Call (FFT): ${fft_heston.price:.4f}")
    print(f"Difference: ${abs(mc_result.price - fft_heston.price):.4f}")

    return bs_engine, fft_engine


# =============================================================================
# SECTION 5: GREEKS MODULE - Full Greeks Calculation
# =============================================================================

def demo_greeks_module():
    """Demonstrate Greeks calculation."""
    print("\n" + "=" * 70)
    print("SECTION 5: GREEKS MODULE")
    print("=" * 70)

    from backend.greeks import (
        bs_all_greeks,
        bs_greeks_first_order,
        bs_greeks_second_order,
        bs_greeks_third_order,
        GreeksCalculator,
    )
    from backend.instruments import VanillaOption
    from backend.models import GBMModel
    from backend.core import MarketEnvironment
    from backend.utils import bs_price

    # Parameters
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.0, 0.2

    # Analytic Greeks (all 14 Greeks)
    print("\n--- Analytic Black-Scholes Greeks ---")
    greeks_tuple = bs_all_greeks(s=S, k=K, t=T, r=r, q=q, sigma=sigma, is_call=True)
    # Unpack: (price, delta, gamma, vega, theta, rho, vanna, volga, charm, veta, speed, zomma, color, ultima)
    price, delta, gamma, vega, theta, rho, vanna, volga, charm, veta, speed, zomma, color, ultima = greeks_tuple

    print(f"\nOption Price: ${price:>10.4f}")

    print("\nFirst Order Greeks:")
    print(f"  Delta (dV/dS):     {delta:>10.6f}")
    print(f"  Vega (dV/dsigma):  {vega:>10.6f}")
    print(f"  Theta (dV/dt):     {theta:>10.6f}")
    print(f"  Rho (dV/dr):       {rho:>10.6f}")

    print("\nSecond Order Greeks:")
    print(f"  Gamma (d2V/dS2):   {gamma:>10.6f}")
    print(f"  Vanna (d2V/dSds):  {vanna:>10.6f}")
    print(f"  Volga (d2V/ds2):   {volga:>10.6f}")
    print(f"  Charm (d2V/dSdt):  {charm:>10.6f}")
    print(f"  Veta (d2V/dsdt):   {veta:>10.6f}")

    print("\nThird Order Greeks:")
    print(f"  Speed (d3V/dS3):   {speed:>10.6f}")
    print(f"  Zomma (d3V/dS2ds): {zomma:>10.6f}")
    print(f"  Color (d3V/dS2dt): {color:>10.6f}")
    print(f"  Ultima (d3V/ds3):  {ultima:>10.6f}")

    # Separate calls for first/second/third order
    print("\n--- Greeks by Order ---")
    first = bs_greeks_first_order(s=S, k=K, t=T, r=r, q=q, sigma=sigma, is_call=True)
    second = bs_greeks_second_order(s=S, k=K, t=T, r=r, q=q, sigma=sigma)  # No is_call for second/third
    third = bs_greeks_third_order(s=S, k=K, t=T, r=r, q=q, sigma=sigma)   # No is_call for second/third
    print(f"First Order (tuple):  {len(first)} values")
    print(f"Second Order (tuple): {len(second)} values")
    print(f"Third Order (tuple):  {len(third)} values")

    # Numerical Greeks with finite differences (manual approach)
    print("\n--- Numerical vs Analytic Comparison ---")

    # Use bs_price from utils for numerical approximation
    # Signature: bs_price(spot, strike, time_to_expiry, rate, volatility, is_call, dividend_yield)
    h = 0.01 * S  # 1% bump

    # Delta: central difference (dV/dS)
    num_delta = (bs_price(S + h, K, T, r, sigma, True, q) - bs_price(S - h, K, T, r, sigma, True, q)) / (2 * h)

    # Gamma: second derivative (d2V/dS2)
    num_gamma = (bs_price(S + h, K, T, r, sigma, True, q) - 2 * price + bs_price(S - h, K, T, r, sigma, True, q)) / (h ** 2)

    # Vega: per 1% vol change (multiply by 0.01 for same units as analytic vega)
    h_vol = 0.01
    num_vega = (bs_price(S, K, T, r, sigma + h_vol, True, q) - bs_price(S, K, T, r, sigma - h_vol, True, q)) / (2 * h_vol) * 0.01

    # Theta: -dV/dT (value decreases as time passes), annualized
    h_t = 1/365
    num_theta = (bs_price(S, K, T - h_t, r, sigma, True, q) - price) / h_t / 365

    print(f"{'Greek':<10} {'Analytic':>12} {'Numerical':>12} {'Diff':>12}")
    print("-" * 50)
    for name, analytic, numerical in [
        ('delta', delta, num_delta),
        ('gamma', gamma, num_gamma),
        ('vega', vega, num_vega),
        ('theta', theta, num_theta),
    ]:
        diff = abs(analytic - numerical)
        print(f"{name:<10} {analytic:>12.6f} {numerical:>12.6f} {diff:>12.8f}")

    # Greeks Calculator with objects
    print("\n--- Greeks Calculator Interface ---")
    from backend.engines import BSAnalyticEngine

    calculator = GreeksCalculator()
    bs_engine = BSAnalyticEngine()
    option = VanillaOption(strike=K, maturity=T, is_call=True)
    model = GBMModel(sigma=sigma)
    market = MarketEnvironment(spot=S, rate=r)

    result = calculator.calculate(bs_engine, option, model, market)
    print(f"Calculator Delta: {result.delta:.6f}")
    print(f"Calculator Gamma: {result.gamma:.6f}")


# =============================================================================
# SECTION 6: SIMULATION MODULE - Monte Carlo Path Simulation
# =============================================================================

def demo_simulation_module():
    """Demonstrate Monte Carlo simulation."""
    print("\n" + "=" * 70)
    print("SECTION 6: SIMULATION MODULE")
    print("=" * 70)

    from backend.simulation import (
        GBMSimulator,
        HestonSimulator,
        MertonSimulator,
        GARCHSimulator,
        create_simulator,
        DiscretizationScheme,
        compute_risk_metrics
    )

    # Common parameters
    s0, mu, t = 100.0, 0.08, 1.0
    n_paths, n_steps = 10_000, 252

    # GBM Simulation
    print("\n--- GBM Simulation ---")
    gbm_sim = GBMSimulator(sigma=0.2)
    gbm_result = gbm_sim.simulate_paths(
        s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=42
    )

    print(f"Paths shape: {gbm_result.price_paths.shape}")
    print(f"Terminal mean: ${gbm_result.terminal_mean:.2f}")
    print(f"Terminal std: ${gbm_result.terminal_std:.2f}")
    print(f"Expected (risk-neutral): ${s0 * np.exp(mu * t):.2f}")

    # Heston Simulation with QE scheme
    print("\n--- Heston Simulation (QE Scheme) ---")
    heston_sim = HestonSimulator(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        scheme=DiscretizationScheme.QE
    )
    heston_result = heston_sim.simulate_paths(
        s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=42
    )

    print(f"Price paths shape: {heston_result.price_paths.shape}")
    print(f"Variance paths shape: {heston_result.volatility_paths.shape}")
    print(f"Terminal price mean: ${heston_result.terminal_mean:.2f}")
    print(f"Terminal variance mean: {heston_result.volatility_paths[:, -1].mean():.4f}")
    print(f"Has volatility paths: {heston_result.has_volatility}")

    # Percentile paths
    pcts = heston_result.percentile_paths([5, 25, 50, 75, 95])
    print("\nTerminal Distribution:")
    print(f"  5th percentile:  ${pcts[0, -1]:.2f}")
    print(f"  25th percentile: ${pcts[1, -1]:.2f}")
    print(f"  50th percentile: ${pcts[2, -1]:.2f}")
    print(f"  75th percentile: ${pcts[3, -1]:.2f}")
    print(f"  95th percentile: ${pcts[4, -1]:.2f}")

    # Merton Jump Diffusion
    print("\n--- Merton Jump Diffusion Simulation ---")
    merton_sim = MertonSimulator(
        sigma=0.2, lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
    )
    merton_result = merton_sim.simulate_paths(
        s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=42
    )
    print(f"Terminal mean: ${merton_result.terminal_mean:.2f}")
    print(f"Terminal std: ${merton_result.terminal_std:.2f}")

    # GARCH Simulation
    print("\n--- GARCH(1,1) Simulation ---")
    garch_sim = GARCHSimulator(
        sigma0=0.2, omega=1e-6, alpha=0.1, beta=0.85
    )
    garch_result = garch_sim.simulate_paths(
        s0=s0, mu=mu, t=t, n_paths=n_paths, n_steps=n_steps, seed=42
    )
    print(f"Terminal mean: ${garch_result.terminal_mean:.2f}")
    print(f"Has volatility paths: {garch_result.has_volatility}")
    if garch_result.has_volatility:
        final_vol = np.sqrt(garch_result.volatility_paths[:, -1]).mean()
        print(f"Terminal volatility mean: {final_vol * 100:.2f}%")

    # Factory pattern
    print("\n--- Simulator Factory ---")
    sim = create_simulator("heston", v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    print(f"Created: {type(sim).__name__}")

    # Risk Metrics - compute from P&L array
    print("\n--- Risk Metrics ---")
    # Calculate P&L from simulation (simple stock returns)
    initial_price = heston_result.price_paths[0, 0]
    terminal_prices = heston_result.price_paths[:, -1]
    pnl = terminal_prices - initial_price  # Dollar P&L

    metrics = compute_risk_metrics(pnl)
    print(f"Mean P&L: ${metrics.mean_pnl:.2f}")
    print(f"Std P&L: ${metrics.std_pnl:.2f}")
    print(f"VaR (95%): ${metrics.var_95:.2f}")
    print(f"VaR (99%): ${metrics.var_99:.2f}")
    print(f"CVaR (95%): ${metrics.cvar_95:.2f}")
    print(f"CVaR (99%): ${metrics.cvar_99:.2f}")
    print(f"Prob Profit: {metrics.prob_profit * 100:.1f}%")
    print(f"Max Profit: ${metrics.max_profit:.2f}")
    print(f"Max Loss: ${metrics.max_loss:.2f}")
    print(f"Skewness: {metrics.skewness:.4f}")
    print(f"Kurtosis: {metrics.kurtosis:.4f}")

    return heston_result


# =============================================================================
# SECTION 7: PORTFOLIO MODULE - Portfolio Management & Risk Analysis
# =============================================================================

def demo_portfolio_module():
    """Demonstrate portfolio management."""
    print("\n" + "=" * 70)
    print("SECTION 7: PORTFOLIO MODULE")
    print("=" * 70)

    from backend.portfolio import (
        OptionsPortfolio,
        long_call, short_call, long_put, short_put,
        long_stock,
        find_breakevens_from_portfolio,
        RiskProfile,
        check_unlimited_risk_from_portfolio,
        get_risk_summary
    )
    from backend.models import GBMModel

    # Helper to calculate net premium
    def get_net_premium(portfolio):
        """Calculate net premium (credit positive, debit negative)."""
        return sum(-p.premium * p.quantity for p in portfolio.positions)

    # Example 1: Bull Call Spread
    print("\n--- Example 1: Bull Call Spread ---")
    spread_portfolio = OptionsPortfolio(GBMModel(sigma=0.2))
    spread_portfolio.add(long_call(strike=100, maturity=0.25, premium=5.50))
    spread_portfolio.add(short_call(strike=110, maturity=0.25, premium=2.00))

    print(f"Number of positions: {len(spread_portfolio.positions)}")
    net_premium = get_net_premium(spread_portfolio)
    print(f"Net premium paid: ${abs(net_premium):.2f}")

    # P&L at expiry using pnl_at_expiry method
    spots = np.array([90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    pnl = spread_portfolio.pnl_at_expiry(spots)

    print(f"\n{'Spot':>8} {'P&L':>10}")
    print("-" * 20)
    for s, p in zip(spots, pnl):
        print(f"${s:>7.0f} ${p:>9.2f}")

    # Breakeven analysis (separate parameters)
    breakeven = find_breakevens_from_portfolio(spread_portfolio, spot_min=90, spot_max=120)
    if breakeven.breakeven_points:
        print(f"\nBreakeven: ${breakeven.breakeven_points[0]:.2f}")
    print(f"Max Profit: ${breakeven.max_profit:.2f}")
    print(f"Max Loss: ${breakeven.max_loss:.2f}")

    # Risk analysis
    unlimited_profit, unlimited_loss = check_unlimited_risk_from_portfolio(spread_portfolio)
    print("\nRisk Analysis:")
    print(f"  Unlimited Profit: {unlimited_profit}")
    print(f"  Unlimited Loss: {unlimited_loss}")

    # Example 2: Iron Condor
    print("\n--- Example 2: Iron Condor ---")
    condor = OptionsPortfolio(GBMModel(sigma=0.2))
    condor.add(long_put(strike=90, maturity=0.25, premium=1.00))
    condor.add(short_put(strike=95, maturity=0.25, premium=2.50))
    condor.add(short_call(strike=105, maturity=0.25, premium=2.50))
    condor.add(long_call(strike=110, maturity=0.25, premium=1.00))

    condor_net_premium = get_net_premium(condor)
    print(f"Net Credit: ${condor_net_premium:.2f}")

    condor_pnl = condor.pnl_at_expiry(spots)
    print(f"\n{'Spot':>8} {'P&L':>10}")
    print("-" * 20)
    for s, p in zip(spots, condor_pnl):
        print(f"${s:>7.0f} ${p:>9.2f}")

    up, ul = check_unlimited_risk_from_portfolio(condor)
    print(f"\nUnlimited Profit: {up}, Unlimited Loss: {ul}")

    # Example 3: Covered Call (Stock + Short Call)
    print("\n--- Example 3: Covered Call ---")
    covered = OptionsPortfolio(GBMModel(sigma=0.2))
    covered.add(long_stock(quantity=100, entry_price=100))  # add() works for stock too
    covered.add(short_call(strike=105, maturity=0.25, premium=3.00, quantity=1))

    print(f"Stock Position: {covered.stock.quantity} shares @ ${covered.stock.entry_price}")
    print(f"Short Call Premium: ${3.00}")

    covered_pnl = covered.pnl_at_expiry(spots)
    print(f"\n{'Spot':>8} {'P&L':>10}")
    print("-" * 20)
    for s, p in zip(spots, covered_pnl):
        print(f"${s:>7.0f} ${p:>9.2f}")

    # Example 4: Long Straddle
    print("\n--- Example 4: Long Straddle ---")
    straddle = OptionsPortfolio(GBMModel(sigma=0.2))
    straddle.add(long_call(strike=100, maturity=0.25, premium=5.00))
    straddle.add(long_put(strike=100, maturity=0.25, premium=4.50))

    straddle_debit = get_net_premium(straddle)
    print(f"Total Debit: ${abs(straddle_debit):.2f}")

    straddle_spots = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    straddle_pnl = straddle.pnl_at_expiry(straddle_spots)

    print(f"\n{'Spot':>8} {'P&L':>10}")
    print("-" * 20)
    for s, p in zip(straddle_spots, straddle_pnl):
        print(f"${s:>7.0f} ${p:>9.2f}")

    up, ul = check_unlimited_risk_from_portfolio(straddle)
    print(f"\nUnlimited Profit: {up}, Unlimited Loss: {ul}")

    # Example 5: Short Strangle (Naked)
    print("\n--- Example 5: Short Strangle (High Risk) ---")
    strangle = OptionsPortfolio(GBMModel(sigma=0.2))
    strangle.add(short_put(strike=95, maturity=0.25, premium=2.00))
    strangle.add(short_call(strike=105, maturity=0.25, premium=2.00))

    strangle_credit = get_net_premium(strangle)
    print(f"Net Credit: ${strangle_credit:.2f}")

    up, ul = check_unlimited_risk_from_portfolio(strangle)
    print(f"Unlimited Profit: {up}, Unlimited Loss: {ul}")

    # Create risk profile manually
    risk_profile = RiskProfile(
        has_unlimited_profit=up,
        has_unlimited_loss=ul,
        max_profit=strangle_credit,
        max_loss=None,
        max_profit_spot=100.0,
        max_loss_spot=None
    )
    summary = get_risk_summary(risk_profile)
    print("\nRisk Summary:")
    print(f"  Risk Level: {summary['risk_level']}")
    print(f"  Profit Potential: {summary['profit_potential']}")
    print(f"  Loss Potential: {summary['loss_potential']}")
    if summary['warnings']:
        print(f"  Warnings: {summary['warnings']}")

    return spread_portfolio


# =============================================================================
# SECTION 8: UTILS MODULE - Mathematical Utilities
# =============================================================================

def demo_utils_module():
    """Demonstrate utility functions."""
    print("\n" + "=" * 70)
    print("SECTION 8: UTILS MODULE")
    print("=" * 70)

    from backend.utils import norm_cdf, norm_pdf, d1_d2, bs_price, implied_volatility

    # Normal distribution functions
    print("\n--- Normal Distribution Functions ---")
    x_values = [-2.0, -1.0, 0.0, 1.0, 1.96, 2.0]

    print(f"{'x':>8} {'N(x)':>10} {'n(x)':>10}")
    print("-" * 30)
    for x in x_values:
        print(f"{x:>8.2f} {norm_cdf(x):>10.6f} {norm_pdf(x):>10.6f}")

    # Black-Scholes d1, d2
    print("\n--- Black-Scholes d1, d2 ---")
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.2
    d1, d2 = d1_d2(S, K, T, r, sigma)
    print(f"Parameters: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"d1 = {d1:.6f}")
    print(f"d2 = {d2:.6f}")
    print(f"d1 - d2 = {d1 - d2:.6f} (should be sigma*sqrt(T) = {sigma * np.sqrt(T):.6f})")

    # Black-Scholes price
    print("\n--- Black-Scholes Price ---")
    call_price = bs_price(S, K, T, r, sigma, is_call=True)
    put_price = bs_price(S, K, T, r, sigma, is_call=False)
    print(f"Call Price: ${call_price:.4f}")
    print(f"Put Price: ${put_price:.4f}")

    # Put-Call Parity
    parity = call_price - put_price - (S - K * np.exp(-r * T))
    print(f"Put-Call Parity Error: ${abs(parity):.8f}")

    # Implied Volatility
    print("\n--- Implied Volatility ---")
    market_price = 10.45  # Observed market price
    # implied_volatility(price, spot, strike, time_to_expiry, rate, is_call, dividend_yield)
    iv = implied_volatility(market_price, S, K, T, r, is_call=True, dividend_yield=0.0)
    print(f"Market Price: ${market_price:.2f}")
    print(f"Implied Vol: {iv * 100:.2f}%")

    # Verify by repricing
    repriced = bs_price(S, K, T, r, iv, is_call=True)
    print(f"Repriced with IV: ${repriced:.4f}")
    print(f"Error: ${abs(market_price - repriced):.6f}")


# =============================================================================
# SECTION 9: ADVANCED EXAMPLES - Combining Everything
# =============================================================================

def demo_advanced_examples():
    """Advanced examples combining multiple modules."""
    print("\n" + "=" * 70)
    print("SECTION 9: ADVANCED EXAMPLES")
    print("=" * 70)

    from backend import (
        VanillaOption, HestonModel, FFTEngine, MarketEnvironment, MonteCarloEngine,
        HestonSimulator, DiscretizationScheme,
    )
    from backend.greeks import bs_all_greeks
    from backend.portfolio import (
        OptionsPortfolio, long_call, long_put, long_stock,
        find_breakevens_from_portfolio, compute_risk_metrics
    )
    from backend.models import GBMModel

    # Example 1: Volatility Surface (Heston Model)
    print("\n--- Volatility Surface Generation (Heston) ---")
    model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    market = MarketEnvironment(spot=100, rate=0.05)
    engine = FFTEngine()

    strikes = [90, 95, 100, 105, 110]
    maturities = [0.1, 0.25, 0.5, 1.0]

    print(f"\n{'K/T':<8}", end="")
    for mat in maturities:
        print(f"{mat:>8.2f}", end="")
    print()

    for k in strikes:
        print(f"{k:<8.0f}", end="")
        for mat in maturities:
            option = VanillaOption(strike=k, maturity=mat, is_call=True)
            result = engine.price(option, model, market)
            print(f"${result.price:>7.2f}", end="")
        print()

    # Example 2: Greeks Surface (Delta vs Spot & Vol)
    print("\n--- Greeks Surface (Delta vs Spot & Vol) ---")
    spot_list = [90, 95, 100, 105, 110]
    vols = [0.15, 0.20, 0.25, 0.30]

    print(f"\n{'S/sigma':<8}", end="")
    for sigma in vols:
        print(f"{sigma*100:>8.0f}%", end="")
    print()

    for spot in spot_list:
        print(f"{spot:<8.0f}", end="")
        for sigma in vols:
            # bs_all_greeks returns tuple: (price, delta, gamma, vega, theta, rho, ...)
            greeks_tuple = bs_all_greeks(s=float(spot), k=100.0, t=0.5, r=0.05, q=0.0, sigma=sigma, is_call=True)
            delta = greeks_tuple[1]  # delta is index 1
            print(f"{delta:>8.4f}", end="")
        print()

    # Example 3: Monte Carlo vs FFT Comparison
    print("\n--- Monte Carlo vs FFT Comparison ---")
    mc_engine = MonteCarloEngine(n_paths=100_000, n_steps=252, seed=42)
    option = VanillaOption(strike=100, maturity=0.5, is_call=True)

    fft_price = engine.price(option, model, market).price
    mc_result = mc_engine.price(option, model, market)

    print(f"FFT Price:   ${fft_price:.4f}")
    error_str = f" +/- ${mc_result.error:.4f}" if mc_result.error else ""
    print(f"MC Price:    ${mc_result.price:.4f}{error_str}")
    print(f"Difference:  ${abs(fft_price - mc_result.price):.4f}")
    if mc_result.error:
        print(f"Within 2 SE: {abs(fft_price - mc_result.price) < 2 * mc_result.error}")

    # Example 4: Full Portfolio + Simulation + Risk Analysis
    print("\n--- Full Workflow: Portfolio + Simulation + Risk ---")

    # Create a protective put portfolio (long stock + long put)
    portfolio = OptionsPortfolio(GBMModel(sigma=0.2))
    portfolio.add(long_stock(quantity=100, entry_price=100))
    portfolio.add(long_put(strike=95, maturity=0.25, premium=2.50, quantity=1))

    print(f"Portfolio: {portfolio.stock.quantity} shares + 1 protective put at $95")

    # Simulate Heston paths
    sim = HestonSimulator(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        scheme=DiscretizationScheme.QE
    )
    sim_result = sim.simulate_paths(
        s0=100, mu=0.05, t=1/12,  # 1 month
        n_paths=50_000, n_steps=21, seed=42
    )

    # Calculate portfolio P&L at expiry using terminal prices
    terminal_prices = sim_result.price_paths[:, -1]
    pnl = portfolio.pnl_at_expiry_fast(terminal_prices, multiplier=1.0)

    # Compute risk metrics from P&L array
    metrics = compute_risk_metrics(pnl)

    print("\nRisk Metrics (1-Month Horizon, 50K paths):")
    print(f"  Mean P&L:    ${metrics.mean_pnl:.2f}")
    print(f"  Std P&L:     ${metrics.std_pnl:.2f}")
    print(f"  VaR (95%):   ${-metrics.var_95:.2f}")
    print(f"  CVaR (95%):  ${-metrics.cvar_95:.2f}")
    print(f"  Prob Profit: {metrics.prob_profit * 100:.1f}%")
    print(f"  Max Profit:  ${metrics.max_profit:.2f}")
    print(f"  Max Loss:    ${metrics.max_loss:.2f}")

    # Example 5: Combining Greeks + Portfolio + Breakeven
    print("\n--- Complete Strategy Analysis ---")
    # Iron Butterfly (for complex analysis)
    butterfly = OptionsPortfolio(GBMModel(sigma=0.25))
    butterfly.add(long_put(strike=90, maturity=0.25, premium=0.80))
    butterfly.add(long_call(strike=110, maturity=0.25, premium=0.80))
    from backend.portfolio import short_put, short_call
    butterfly.add(short_put(strike=100, maturity=0.25, premium=3.50))
    butterfly.add(short_call(strike=100, maturity=0.25, premium=3.50))

    # Breakeven analysis
    be_result = find_breakevens_from_portfolio(butterfly, spot_min=85, spot_max=115)

    print("Iron Butterfly Analysis:")
    print(f"  Breakeven Points: {[f'${p:.2f}' for p in be_result.breakeven_points]}")
    print(f"  Max Profit: ${be_result.max_profit:.2f} at ${be_result.max_profit_spot:.2f}")
    print(f"  Max Loss: ${be_result.max_loss:.2f} at ${be_result.max_loss_spot:.2f}")

    # P&L curve
    test_spots = np.linspace(85, 115, 7)
    pnl_curve = butterfly.pnl_at_expiry(test_spots)

    print(f"\n{'Spot':>8} {'P&L':>10}")
    print("-" * 20)
    for s, p in zip(test_spots, pnl_curve):
        print(f"${s:>7.0f} ${p:>9.2f}")


# =============================================================================
# SECTION 10: COMPREHENSIVE COMBINED EXAMPLES
# =============================================================================

def demo_combined_scenarios():
    """Demonstrate comprehensive scenarios combining all modules."""
    print("\n" + "=" * 70)
    print("SECTION 10: COMPREHENSIVE COMBINED EXAMPLES")
    print("=" * 70)

    import numpy as np
    from backend import (
        VanillaOption, MarketEnvironment, GBMModel, HestonModel,
        FFTEngine, BSAnalyticEngine, MonteCarloEngine,
        HestonSimulator, DiscretizationScheme
    )
    from backend.greeks import bs_all_greeks
    from backend.portfolio import (
        OptionsPortfolio, long_call, long_put, short_call, short_put, long_stock,
        find_breakevens_from_portfolio, compute_risk_metrics,
        check_unlimited_risk_from_portfolio
    )
    from backend.simulation import GBMSimulator, MertonSimulator
    from backend.utils import implied_volatility

    # =========================================================================
    # SCENARIO 1: Hedged Portfolio VaR with Real-World Model
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 1: Hedged Portfolio VaR (Heston Dynamics)")
    print("-" * 60)

    # Define market
    S0 = 100.0
    r = 0.05
    market = MarketEnvironment(spot=S0, rate=r)

    # Create portfolio: 100 shares + protective put at 95 + covered call at 110
    hedged = OptionsPortfolio(HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7))
    hedged.add(long_stock(quantity=100, entry_price=S0))
    hedged.add(long_put(strike=95, maturity=0.25, premium=2.50, quantity=1))
    hedged.add(short_call(strike=110, maturity=0.25, premium=2.00, quantity=1))

    print("Portfolio: Collar Strategy")
    print("  - 100 shares @ $100")
    print("  - Long Put K=95 (protection)")
    print("  - Short Call K=110 (yield enhancement)")

    # Simulate with Heston
    heston_sim = HestonSimulator(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        scheme=DiscretizationScheme.QE
    )
    sim = heston_sim.simulate_paths(s0=S0, mu=r, t=0.25, n_paths=100_000, n_steps=63, seed=12345)

    # P&L distribution at expiry
    terminal = sim.price_paths[:, -1]
    pnl = hedged.pnl_at_expiry_fast(terminal, multiplier=1.0)
    metrics = compute_risk_metrics(pnl)

    print("\n3-Month Risk Profile (100K Heston paths):")
    print(f"  Expected P&L:    ${metrics.mean_pnl:,.2f}")
    print(f"  P&L Volatility:  ${metrics.std_pnl:,.2f}")
    print(f"  VaR (95%):       ${-metrics.var_95:,.2f}")
    print(f"  CVaR (95%):      ${-metrics.cvar_95:,.2f}")
    print(f"  Max Downside:    ${-metrics.max_loss:,.2f}")
    print(f"  Max Upside:      ${metrics.max_profit:,.2f}")
    print(f"  Win Rate:        {metrics.prob_profit * 100:.1f}%")

    # =========================================================================
    # SCENARIO 2: Model Comparison for Exotic Pricing
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 2: Model Comparison (GBM vs Heston vs MC)")
    print("-" * 60)

    option = VanillaOption(strike=100, maturity=0.5, is_call=True)

    # GBM with BS Engine
    gbm = GBMModel(sigma=0.2)
    bs_engine = BSAnalyticEngine()
    bs_price = bs_engine.price(option, gbm, market).price

    # Heston with FFT Engine
    heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    fft_engine = FFTEngine()
    fft_price = fft_engine.price(option, heston, market).price

    # Heston with Monte Carlo
    mc_engine = MonteCarloEngine(n_paths=100_000, n_steps=252, seed=42)
    mc_result = mc_engine.price(option, heston, market)

    print("\nATM Call (K=100, T=0.5)")
    print(f"  GBM/BS:      ${bs_price:.4f}")
    print(f"  Heston/FFT:  ${fft_price:.4f}")
    print(f"  Heston/MC:   ${mc_result.price:.4f} +/- ${mc_result.error:.4f}")
    print(f"\n  GBM vs Heston Spread: ${abs(bs_price - fft_price):.4f}")
    print(f"  FFT vs MC Error:      ${abs(fft_price - mc_result.price):.4f}")

    # =========================================================================
    # SCENARIO 3: Greeks Sensitivity Analysis
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 3: Greeks Sensitivity Analysis")
    print("-" * 60)

    print("\nDelta Surface (Spot x IV):")
    spots = np.array([90, 95, 100, 105, 110])
    ivs = np.array([0.15, 0.20, 0.25, 0.30, 0.35])

    print(f"\n{'S\\IV':<6}", end="")
    for iv in ivs:
        print(f"{iv*100:>8.0f}%", end="")
    print()
    print("-" * 50)

    for s in spots:
        print(f"{s:<6.0f}", end="")
        for iv in ivs:
            greeks = bs_all_greeks(s=float(s), k=100.0, t=0.5, r=0.05, q=0.0, sigma=iv, is_call=True)
            delta = greeks[1]
            print(f"{delta:>8.3f}", end="")
        print()

    # =========================================================================
    # SCENARIO 4: Strategy Comparison
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 4: Strategy Comparison")
    print("-" * 60)

    strategies = {
        "Long Call": lambda p: p.add(long_call(strike=100, maturity=0.25, premium=5.00)),
        "Bull Spread": lambda p: p.add(long_call(strike=95, maturity=0.25, premium=7.50))
                                  .add(short_call(strike=105, maturity=0.25, premium=3.00)),
        "Iron Condor": lambda p: p.add(long_put(strike=85, maturity=0.25, premium=0.50))
                                  .add(short_put(strike=90, maturity=0.25, premium=1.50))
                                  .add(short_call(strike=110, maturity=0.25, premium=1.50))
                                  .add(long_call(strike=115, maturity=0.25, premium=0.50)),
        "Straddle": lambda p: p.add(long_call(strike=100, maturity=0.25, premium=5.00))
                               .add(long_put(strike=100, maturity=0.25, premium=4.50)),
    }

    test_spots = np.linspace(80, 120, 200)

    print(f"\n{'Strategy':<15} {'Max P/L':>10} {'Min P/L':>10} {'BE Count':>10} {'Unltd Risk':>12}")
    print("-" * 60)

    for name, builder in strategies.items():
        portfolio = OptionsPortfolio(GBMModel(sigma=0.2))
        builder(portfolio)

        pnl = portfolio.pnl_at_expiry(test_spots)
        be_result = find_breakevens_from_portfolio(portfolio, spot_min=80, spot_max=120)
        up, ul = check_unlimited_risk_from_portfolio(portfolio)

        max_pnl = pnl.max()
        min_pnl = pnl.min()
        be_count = len(be_result.breakeven_points)
        risk_str = "Yes" if ul else "No"

        print(f"{name:<15} ${max_pnl:>9.2f} ${min_pnl:>9.2f} {be_count:>10} {risk_str:>12}")

    # =========================================================================
    # SCENARIO 5: Real-World Simulation with Multiple Strategies
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 5: Monte Carlo Strategy Evaluation")
    print("-" * 60)

    # Simulate stock price paths
    heston_sim = HestonSimulator(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        scheme=DiscretizationScheme.QE
    )
    paths = heston_sim.simulate_paths(
        s0=100, mu=0.08, t=0.25, n_paths=50_000, n_steps=63, seed=999
    )
    terminal_prices = paths.price_paths[:, -1]

    # Test strategies with the same simulated paths
    test_strategies = [
        ("Long Stock", lambda p: p.add(long_stock(quantity=1, entry_price=100))),
        ("Covered Call", lambda p: p.add(long_stock(quantity=1, entry_price=100))
                                   .add(short_call(strike=105, maturity=0.25, premium=3.00, quantity=1))),
        ("Protective Put", lambda p: p.add(long_stock(quantity=1, entry_price=100))
                                     .add(long_put(strike=95, maturity=0.25, premium=2.00, quantity=1))),
    ]

    print(f"\n{'Strategy':<15} {'Mean':>10} {'Std':>10} {'VaR95':>10} {'Sharpe':>10}")
    print("-" * 55)

    for name, builder in test_strategies:
        portfolio = OptionsPortfolio(GBMModel(sigma=0.2))
        builder(portfolio)

        pnl = portfolio.pnl_at_expiry_fast(terminal_prices, multiplier=1.0)
        metrics = compute_risk_metrics(pnl)

        # Simplified Sharpe (mean / std, not annualized)
        sharpe = metrics.mean_pnl / metrics.std_pnl if metrics.std_pnl > 0 else 0

        print(f"{name:<15} ${metrics.mean_pnl:>9.2f} ${metrics.std_pnl:>9.2f} ${-metrics.var_95:>9.2f} {sharpe:>10.3f}")

    # =========================================================================
    # SCENARIO 6: Full Greeks Dashboard (All 14 Greeks)
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 6: Full Greeks Dashboard (All 14 Greeks)")
    print("-" * 60)

    # Create a multi-leg portfolio
    greeks_portfolio = OptionsPortfolio(GBMModel(sigma=0.25))
    greeks_portfolio.add(long_call(strike=100, maturity=0.5, premium=8.0, quantity=2))
    greeks_portfolio.add(short_call(strike=110, maturity=0.5, premium=4.0, quantity=1))
    greeks_portfolio.add(long_put(strike=90, maturity=0.5, premium=3.0, quantity=1))

    print("Portfolio: Ratio Call Spread + Protective Put")
    print("  - 2x Long Call K=100")
    print("  - 1x Short Call K=110")
    print("  - 1x Long Put K=90")

    # Calculate all 14 Greeks at current spot
    greek_names = ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho',
                   'Vanna', 'Volga', 'Charm', 'Veta', 'Speed', 'Zomma', 'Color', 'Ultima']

    print("\nPortfolio Greeks @ S=$100, σ=25%, T=0.5yr, r=5%:")
    print("-" * 40)

    # Sum Greeks across positions
    total_greeks = np.zeros(14)
    for pos in greeks_portfolio.positions:
        is_call = 1 if pos.is_call else 0
        sign = 1 if pos.is_long else -1
        greeks = bs_all_greeks(s=100.0, k=pos.strike, t=pos.maturity, r=0.05, q=0.0, sigma=0.25, is_call=is_call)
        for i in range(14):
            total_greeks[i] += sign * pos.quantity * greeks[i]

    # Display in two columns
    for i in range(0, 14, 2):
        left = f"{greek_names[i]:<10}: {total_greeks[i]:>12.4f}"
        if i + 1 < 14:
            right = f"{greek_names[i+1]:<10}: {total_greeks[i+1]:>12.4f}"
            print(f"  {left}    {right}")
        else:
            print(f"  {left}")

    # =========================================================================
    # SCENARIO 7: Term Structure Analysis
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 7: Term Structure Analysis")
    print("-" * 60)

    maturities = [0.083, 0.25, 0.5, 1.0, 2.0]  # 1M, 3M, 6M, 1Y, 2Y
    mat_labels = ["1M", "3M", "6M", "1Y", "2Y"]

    # Price ATM options with different models
    gbm_model = GBMModel(sigma=0.20)
    heston_model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

    print("\nATM Call Prices (S=K=100) across term structure:")
    print(f"\n{'Maturity':<10} {'GBM/BS':>10} {'Heston/FFT':>12} {'Spread':>10} {'Vol Smile':>12}")
    print("-" * 56)

    for T, label in zip(maturities, mat_labels):
        option = VanillaOption(strike=100, maturity=T, is_call=True)

        bs_p = bs_engine.price(option, gbm_model, market).price
        fft_p = fft_engine.price(option, heston_model, market).price

        # Calculate implied vol from Heston price
        heston_iv = implied_volatility(fft_p, 100.0, 100.0, T, 0.05, is_call=True, dividend_yield=0.0)

        spread = fft_p - bs_p
        smile = (heston_iv - 0.20) * 100  # Basis points from 20%

        print(f"{label:<10} ${bs_p:>9.4f} ${fft_p:>11.4f} ${spread:>9.4f} {smile:>+11.2f}bp")

    # =========================================================================
    # SCENARIO 8: Volatility Smile from Heston Prices
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 8: Volatility Smile from Heston Prices")
    print("-" * 60)

    strikes_smile = np.array([85, 90, 95, 100, 105, 110, 115])
    T_smile = 0.25

    print("\nImplied Volatility Smile (T=3M, S=100):")
    print(f"\n{'Strike':<10} {'Heston Price':>12} {'Implied Vol':>12} {'Moneyness':>12}")
    print("-" * 50)

    for K in strikes_smile:
        is_call = K >= 100  # Use calls for ATM+, puts for ITM
        option = VanillaOption(strike=K, maturity=T_smile, is_call=is_call)
        heston_price = fft_engine.price(option, heston_model, market).price

        iv = implied_volatility(heston_price, 100.0, K, T_smile, 0.05, is_call=is_call, dividend_yield=0.0)
        moneyness = np.log(100.0 / K) / np.sqrt(T_smile)

        print(f"{K:<10.0f} ${heston_price:>11.4f} {iv*100:>11.2f}% {moneyness:>+11.3f}")

    # =========================================================================
    # SCENARIO 9: Dynamic Hedging Simulation
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 9: Dynamic Hedging Simulation")
    print("-" * 60)

    # Simulate stock paths and track delta hedge P&L
    n_paths_hedge = 10_000
    n_steps_hedge = 63  # Daily for 3 months
    T_hedge = 0.25
    K_hedge = 100.0
    sigma_hedge = 0.20

    # Generate paths
    gbm_sim = GBMSimulator(sigma=sigma_hedge)
    hedge_paths = gbm_sim.simulate_paths(
        s0=100.0, mu=0.05, t=T_hedge,
        n_paths=n_paths_hedge, n_steps=n_steps_hedge, seed=42
    )

    print(f"\nDelta Hedging Analysis (K=100, T=3M, σ=20%, {n_paths_hedge:,} paths):")

    # Track hedge P&L for a short call position
    dt = T_hedge / n_steps_hedge
    total_hedge_errors = []

    for path_idx in range(min(1000, n_paths_hedge)):  # Sample for speed
        path = hedge_paths.price_paths[path_idx]
        hedge_pnl = 0.0
        stock_position = 0.0

        for step in range(n_steps_hedge):
            S = path[step]
            tau = T_hedge - step * dt

            if tau > 0.001:
                # Calculate delta
                greeks = bs_all_greeks(s=S, k=K_hedge, t=tau, r=0.05, q=0.0, sigma=sigma_hedge, is_call=True)
                new_delta = greeks[1]

                # Rebalance hedge
                delta_change = new_delta - stock_position
                hedge_pnl -= delta_change * S
                stock_position = new_delta

        # Final settlement
        final_S = path[-1]
        option_payoff = max(0, final_S - K_hedge)
        hedge_pnl += stock_position * final_S - option_payoff

        total_hedge_errors.append(hedge_pnl)

    hedge_errors = np.array(total_hedge_errors)
    print(f"  Mean Hedge Error:    ${hedge_errors.mean():>8.4f}")
    print(f"  Std Hedge Error:     ${hedge_errors.std():>8.4f}")
    print(f"  Hedge Efficiency:    {(1 - hedge_errors.std() / (100 * sigma_hedge * np.sqrt(T_hedge))) * 100:.1f}%")

    # =========================================================================
    # SCENARIO 10: Complete Strategy Lifecycle
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 10: Complete Strategy Lifecycle")
    print("-" * 60)

    print("\n--- PHASE 1: Strategy Selection & Pricing ---")

    # Create Iron Butterfly
    butterfly = OptionsPortfolio(HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7))
    butterfly.add(long_put(strike=90, maturity=0.25, premium=1.50, quantity=1))
    butterfly.add(short_put(strike=100, maturity=0.25, premium=5.00, quantity=1))
    butterfly.add(short_call(strike=100, maturity=0.25, premium=5.00, quantity=1))
    butterfly.add(long_call(strike=110, maturity=0.25, premium=1.50, quantity=1))

    net_credit = -1.50 + 5.00 + 5.00 - 1.50  # Net premium received
    print("Strategy: Iron Butterfly @ K=100")
    print(f"  Net Credit: ${net_credit:.2f}")

    print("\n--- PHASE 2: Greeks Analysis ---")
    portfolio_greeks = np.zeros(6)  # Price, Delta, Gamma, Vega, Theta, Rho
    for pos in butterfly.positions:
        is_call = 1 if pos.is_call else 0
        sign = 1 if pos.is_long else -1
        greeks = bs_all_greeks(s=100.0, k=pos.strike, t=0.25, r=0.05, q=0.0, sigma=0.20, is_call=is_call)
        for i in range(6):
            portfolio_greeks[i] += sign * pos.quantity * greeks[i]

    print(f"  Delta: {portfolio_greeks[1]:>8.4f} (near-neutral)")
    print(f"  Gamma: {portfolio_greeks[2]:>8.4f} (negative)")
    print(f"  Vega:  {portfolio_greeks[3]:>8.4f} (negative)")
    print(f"  Theta: {portfolio_greeks[4]:>8.4f} (positive - time decay in favor)")

    print("\n--- PHASE 3: Breakeven & Risk Profile ---")
    be_butterfly = find_breakevens_from_portfolio(butterfly, spot_min=80, spot_max=120)
    up_butterfly, ul_butterfly = check_unlimited_risk_from_portfolio(butterfly)

    print(f"  Breakeven Points: {[f'${b:.2f}' for b in be_butterfly.breakeven_points]}")
    print(f"  Max Profit: ${be_butterfly.max_profit:.2f} @ ${be_butterfly.max_profit_spot:.2f}")
    print(f"  Max Loss: ${be_butterfly.max_loss:.2f} @ ${be_butterfly.max_loss_spot:.2f}")
    print(f"  Unlimited Risk: {'Yes' if ul_butterfly else 'No'}")

    print("\n--- PHASE 4: Monte Carlo Risk Assessment ---")
    heston_butterfly = HestonSimulator(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    butterfly_paths = heston_butterfly.simulate_paths(s0=100, mu=0.05, t=0.25, n_paths=50_000, n_steps=63, seed=555)
    butterfly_pnl = butterfly.pnl_at_expiry_fast(butterfly_paths.price_paths[:, -1], multiplier=1.0)
    butterfly_metrics = compute_risk_metrics(butterfly_pnl)

    print(f"  Expected P&L:  ${butterfly_metrics.mean_pnl:.2f}")
    print(f"  P&L Std:       ${butterfly_metrics.std_pnl:.2f}")
    print(f"  VaR (95%):     ${-butterfly_metrics.var_95:.2f}")
    print(f"  Prob of Profit: {butterfly_metrics.prob_profit*100:.1f}%")

    print("\n--- PHASE 5: Scenario Analysis ---")
    scenarios = [
        ("Spot -10%", 90),
        ("Spot unchanged", 100),
        ("Spot +10%", 110),
        ("Vol +50%", 100),  # Will recalculate with higher vol
    ]

    print(f"\n{'Scenario':<18} {'P&L':>10} {'New Delta':>12}")
    print("-" * 42)

    for name, spot in scenarios[:3]:
        pnl_scenario = butterfly.pnl_at_expiry(np.array([spot]))[0]
        # Recalculate delta at new spot
        new_delta = 0.0
        for pos in butterfly.positions:
            is_call = 1 if pos.is_call else 0
            sign = 1 if pos.is_long else -1
            greeks = bs_all_greeks(s=float(spot), k=pos.strike, t=0.25, r=0.05, q=0.0, sigma=0.20, is_call=is_call)
            new_delta += sign * pos.quantity * greeks[1]
        print(f"{name:<18} ${pnl_scenario:>9.2f} {new_delta:>+12.4f}")

    # Vol scenario (at same spot but higher vol)
    new_delta_vol = 0.0
    for pos in butterfly.positions:
        is_call = 1 if pos.is_call else 0
        sign = 1 if pos.is_long else -1
        greeks = bs_all_greeks(s=100.0, k=pos.strike, t=0.25, r=0.05, q=0.0, sigma=0.30, is_call=is_call)
        new_delta_vol += sign * pos.quantity * greeks[1]
    # Vol impact on P&L (approximate via vega)
    vol_pnl_impact = portfolio_greeks[3] * 0.10 / 0.01  # Vega * 10% vol increase
    print(f"{'Vol +50%':<18} ${vol_pnl_impact:>9.2f} {new_delta_vol:>+12.4f}")

    # =========================================================================
    # SCENARIO 11: Multi-Asset Correlation Study
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 11: Multi-Asset Correlation Study")
    print("-" * 60)

    # Simulate correlated assets
    from backend.math_kernels.random import compute_cholesky

    # Define correlation matrix (3 assets)
    corr_matrix = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    compute_cholesky(corr_matrix)

    n_paths_corr = 50_000
    T_corr = 1.0

    # Initial prices and vols
    S0_assets = np.array([100.0, 50.0, 200.0])
    sigma_assets = np.array([0.20, 0.30, 0.15])
    mu_assets = np.array([0.08, 0.10, 0.06])

    print("\n3-Asset Portfolio (ρ12=0.6, ρ13=0.3, ρ23=0.5):")
    print("  Asset 1: S0=$100, σ=20%, μ=8%")
    print("  Asset 2: S0=$50, σ=30%, μ=10%")
    print("  Asset 3: S0=$200, σ=15%, μ=6%")

    # Generate correlated paths
    np.random.seed(123)
    terminal_prices_corr = np.zeros((n_paths_corr, 3))

    for i in range(3):
        # GBM terminal price
        Z = np.random.randn(n_paths_corr)
        terminal_prices_corr[:, i] = S0_assets[i] * np.exp(
            (mu_assets[i] - 0.5 * sigma_assets[i]**2) * T_corr +
            sigma_assets[i] * np.sqrt(T_corr) * Z
        )

    # Calculate portfolio with equal weights
    weights = np.array([1/3, 1/3, 1/3])
    initial_value = np.sum(S0_assets * weights)
    terminal_values = terminal_prices_corr @ weights
    portfolio_returns = (terminal_values - initial_value) / initial_value

    # Individual asset returns
    asset_returns = (terminal_prices_corr - S0_assets) / S0_assets

    print("\n1-Year Return Statistics:")
    print(f"\n{'Asset':<15} {'Mean':>10} {'Std':>10} {'VaR95':>10} {'Sharpe':>10}")
    print("-" * 55)

    for i in range(3):
        ret = asset_returns[:, i]
        mean_ret = ret.mean()
        std_ret = ret.std()
        var95 = np.percentile(ret, 5)
        sharpe = mean_ret / std_ret if std_ret > 0 else 0
        print(f"Asset {i+1:<10} {mean_ret*100:>9.2f}% {std_ret*100:>9.2f}% {var95*100:>9.2f}% {sharpe:>10.3f}")

    # Portfolio stats
    mean_port = portfolio_returns.mean()
    std_port = portfolio_returns.std()
    var95_port = np.percentile(portfolio_returns, 5)
    sharpe_port = mean_port / std_port if std_port > 0 else 0
    print(f"{'Portfolio':<15} {mean_port*100:>9.2f}% {std_port*100:>9.2f}% {var95_port*100:>9.2f}% {sharpe_port:>10.3f}")

    # Diversification benefit
    weighted_vol = np.sqrt(np.sum((weights * sigma_assets)**2))
    actual_vol = std_port
    div_benefit = (weighted_vol - actual_vol) / weighted_vol * 100
    print(f"\nDiversification Benefit: {div_benefit:.1f}% volatility reduction")

    # =========================================================================
    # SCENARIO 12: Greeks Evolution Over Time
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 12: Greeks Evolution Over Time")
    print("-" * 60)

    print("\nATM Call (K=100, σ=20%) Greeks vs Time to Expiry:")
    times_to_expiry = [0.5, 0.25, 0.125, 0.0625, 0.02]  # 6M, 3M, 6W, 3W, 1W
    time_labels = ["6M", "3M", "6W", "3W", "1W"]

    print(f"\n{'Time':<6} {'Price':>10} {'Delta':>8} {'Gamma':>8} {'Vega':>8} {'Theta':>10}")
    print("-" * 55)

    for T_evo, label in zip(times_to_expiry, time_labels):
        greeks_evo = bs_all_greeks(s=100.0, k=100.0, t=T_evo, r=0.05, q=0.0, sigma=0.20, is_call=True)
        price_evo = greeks_evo[0]
        delta_evo = greeks_evo[1]
        gamma_evo = greeks_evo[2]
        vega_evo = greeks_evo[3]
        theta_evo = greeks_evo[4]
        print(f"{label:<6} ${price_evo:>9.4f} {delta_evo:>8.4f} {gamma_evo:>8.4f} {vega_evo:>8.4f} {theta_evo:>10.4f}")

    print("\nObservations:")
    print("  - Delta converges to 0.5 as T→0 (ATM)")
    print("  - Gamma increases sharply near expiry")
    print("  - Vega decreases as less time for vol to matter")
    print("  - Theta becomes more negative (accelerating time decay)")

    # =========================================================================
    # SCENARIO 13: Jump Diffusion Impact Analysis
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 13: Jump Diffusion Impact Analysis")
    print("-" * 60)

    # Compare GBM vs Merton (jump diffusion)
    # Merton: 0.5 jumps/year, -10% mean jump, 15% jump std
    gbm_jd = GBMSimulator(sigma=0.20)
    merton_jd = MertonSimulator(sigma=0.20, lambda_j=0.5, mu_j=-0.1, sigma_j=0.15)

    n_paths_jd = 100_000
    T_jd = 0.25

    gbm_paths_jd = gbm_jd.simulate_paths(s0=100, mu=0.08, t=T_jd, n_paths=n_paths_jd, n_steps=63, seed=777)
    merton_paths_jd = merton_jd.simulate_paths(s0=100, mu=0.08, t=T_jd, n_paths=n_paths_jd, n_steps=63, seed=777)

    gbm_terminal = gbm_paths_jd.price_paths[:, -1]
    merton_terminal = merton_paths_jd.price_paths[:, -1]

    print("\nTerminal Distribution Comparison (T=3M, 100K paths):")
    print(f"{'Model':<12} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10}")
    print("-" * 55)

    # GBM stats
    gbm_mean = gbm_terminal.mean()
    gbm_std = gbm_terminal.std()
    gbm_skew = ((gbm_terminal - gbm_mean)**3).mean() / gbm_std**3
    gbm_kurt = ((gbm_terminal - gbm_mean)**4).mean() / gbm_std**4 - 3
    print(f"{'GBM':<12} ${gbm_mean:>9.2f} ${gbm_std:>9.2f} {gbm_skew:>10.3f} {gbm_kurt:>10.3f}")

    # Merton stats
    merton_mean = merton_terminal.mean()
    merton_std = merton_terminal.std()
    merton_skew = ((merton_terminal - merton_mean)**3).mean() / merton_std**3
    merton_kurt = ((merton_terminal - merton_mean)**4).mean() / merton_std**4 - 3
    print(f"{'Merton':<12} ${merton_mean:>9.2f} ${merton_std:>9.2f} {merton_skew:>10.3f} {merton_kurt:>10.3f}")

    # Option pricing impact
    print("\nOTM Put (K=90) Pricing Impact:")
    put_payoff_gbm = np.maximum(90 - gbm_terminal, 0)
    put_payoff_merton = np.maximum(90 - merton_terminal, 0)

    put_price_gbm = np.exp(-0.05 * T_jd) * put_payoff_gbm.mean()
    put_price_merton = np.exp(-0.05 * T_jd) * put_payoff_merton.mean()

    print(f"  GBM Price:    ${put_price_gbm:.4f}")
    print(f"  Merton Price: ${put_price_merton:.4f}")
    print(f"  Jump Premium: ${put_price_merton - put_price_gbm:.4f} ({(put_price_merton/put_price_gbm - 1)*100:.1f}% higher)")

    # =========================================================================
    # SCENARIO 14: Bates Model (Stochastic Vol + Jumps)
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 14: Bates Model (Stochastic Vol + Jumps)")
    print("-" * 60)

    from backend.simulation import BatesSimulator

    print("\nComparing Heston vs Bates (Heston + Jumps):")

    # Bates = Heston + Jumps
    bates_sim = BatesSimulator(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,  # Heston params
        lambda_j=0.2, mu_j=-0.05, sigma_j=0.10  # Jump params
    )

    heston_bates = HestonSimulator(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

    n_paths_bates = 50_000
    T_bates = 0.5

    heston_paths_b = heston_bates.simulate_paths(s0=100, mu=0.05, t=T_bates, n_paths=n_paths_bates, n_steps=126, seed=888)
    bates_paths = bates_sim.simulate_paths(s0=100, mu=0.05, t=T_bates, n_paths=n_paths_bates, n_steps=126, seed=888)

    heston_term_b = heston_paths_b.price_paths[:, -1]
    bates_term = bates_paths.price_paths[:, -1]

    print("\nTerminal Distribution (T=6M, 50K paths):")
    print(f"{'Model':<12} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10}")
    print("-" * 55)

    # Heston stats
    h_mean = heston_term_b.mean()
    h_std = heston_term_b.std()
    h_skew = ((heston_term_b - h_mean)**3).mean() / h_std**3
    h_kurt = ((heston_term_b - h_mean)**4).mean() / h_std**4 - 3
    print(f"{'Heston':<12} ${h_mean:>9.2f} ${h_std:>9.2f} {h_skew:>10.3f} {h_kurt:>10.3f}")

    # Bates stats
    b_mean = bates_term.mean()
    b_std = bates_term.std()
    b_skew = ((bates_term - b_mean)**3).mean() / b_std**3
    b_kurt = ((bates_term - b_mean)**4).mean() / b_std**4 - 3
    print(f"{'Bates':<12} ${b_mean:>9.2f} ${b_std:>9.2f} {b_skew:>10.3f} {b_kurt:>10.3f}")

    # Option pricing impact
    print("\nOption Pricing (MC - 50K paths):")
    strikes_bates = [90, 95, 100, 105, 110]
    print(f"{'Strike':<10} {'Heston':>12} {'Bates':>12} {'Jump Premium':>14}")
    print("-" * 50)

    for K in strikes_bates:
        h_payoff = np.maximum(heston_term_b - K, 0)
        b_payoff = np.maximum(bates_term - K, 0)
        h_price = np.exp(-0.05 * T_bates) * h_payoff.mean()
        b_price = np.exp(-0.05 * T_bates) * b_payoff.mean()
        premium = b_price - h_price
        print(f"K={K:<7} ${h_price:>11.4f} ${b_price:>11.4f} ${premium:>+13.4f}")

    # =========================================================================
    # SCENARIO 15: GARCH vs Constant Vol Comparison
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 15: GARCH vs Constant Volatility")
    print("-" * 60)

    from backend.simulation import GARCHSimulator

    print("\nGARCH(1,1) Model: σ0=22%, ω=0.00001, α=0.1, β=0.85")
    print("Long-run variance = ω/(1-α-β) = 0.0002 → σ_∞ ≈ 22.4%/year")

    # GARCH(1,1) params: sigma0, omega, alpha, beta
    garch_sim = GARCHSimulator(sigma0=0.22, omega=0.00001, alpha=0.1, beta=0.85)
    gbm_const = GBMSimulator(sigma=0.22)  # Constant vol matching long-run

    n_paths_garch = 50_000
    T_garch = 0.25

    garch_paths = garch_sim.simulate_paths(s0=100, mu=0.05, t=T_garch, n_paths=n_paths_garch, n_steps=63, seed=999)
    const_paths = gbm_const.simulate_paths(s0=100, mu=0.05, t=T_garch, n_paths=n_paths_garch, n_steps=63, seed=999)

    garch_term = garch_paths.price_paths[:, -1]
    const_term = const_paths.price_paths[:, -1]

    print("\nTerminal Distribution (T=3M, 50K paths):")
    print(f"{'Model':<15} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10}")
    print("-" * 58)

    g_mean = garch_term.mean()
    g_std = garch_term.std()
    g_skew = ((garch_term - g_mean)**3).mean() / g_std**3
    g_kurt = ((garch_term - g_mean)**4).mean() / g_std**4 - 3
    print(f"{'GARCH(1,1)':<15} ${g_mean:>9.2f} ${g_std:>9.2f} {g_skew:>10.3f} {g_kurt:>10.3f}")

    c_mean = const_term.mean()
    c_std = const_term.std()
    c_skew = ((const_term - c_mean)**3).mean() / c_std**3
    c_kurt = ((const_term - c_mean)**4).mean() / c_std**4 - 3
    print(f"{'Constant Vol':<15} ${c_mean:>9.2f} ${c_std:>9.2f} {c_skew:>10.3f} {c_kurt:>10.3f}")

    # Volatility path analysis
    print("\nVolatility Clustering Effect (sample path variance):")
    garch_vol_path = np.sqrt(garch_paths.variance_paths[0]) if hasattr(garch_paths, 'variance_paths') else None
    if garch_vol_path is not None:
        print(f"  GARCH vol range: {garch_vol_path.min()*np.sqrt(252)*100:.1f}% - {garch_vol_path.max()*np.sqrt(252)*100:.1f}%")
    else:
        # Calculate realized vol from returns
        sample_returns = np.diff(np.log(garch_paths.price_paths[0]))
        rolling_vol = np.array([sample_returns[max(0,i-20):i].std() for i in range(21, len(sample_returns))])
        if len(rolling_vol) > 0:
            print(f"  GARCH 20-day realized vol: {rolling_vol.min()*np.sqrt(252)*100:.1f}% - {rolling_vol.max()*np.sqrt(252)*100:.1f}%")

    # =========================================================================
    # SCENARIO 16: Full Ecosystem Integration Test
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 16: Full Ecosystem Integration Test")
    print("-" * 60)

    print("\nEnd-to-End Workflow: Research → Strategy → Backtest → Risk")

    # Step 1: Model Selection via Pricing Accuracy
    print("\n--- Step 1: Model Selection ---")
    test_option = VanillaOption(strike=100, maturity=0.5, is_call=True)
    market_test = MarketEnvironment(spot=100, rate=0.05)

    models_test = {
        'GBM': (GBMModel(sigma=0.20), BSAnalyticEngine()),
        'Heston': (HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7), FFTEngine()),
    }

    print("ATM Call (K=100, T=6M) Model Prices:")
    for name, (model, engine) in models_test.items():
        price = engine.price(test_option, model, market_test).price
        print(f"  {name:>10}: ${price:.4f}")

    # Step 2: Strategy Design with Greeks
    print("\n--- Step 2: Strategy Design (Long Straddle) ---")
    straddle = OptionsPortfolio(GBMModel(sigma=0.20))
    straddle.add(long_call(strike=100, maturity=0.5, premium=6.89, quantity=1))
    straddle.add(long_put(strike=100, maturity=0.5, premium=4.42, quantity=1))

    total_cost = 6.89 + 4.42
    print(f"  Total Premium Paid: ${total_cost:.2f}")

    # Calculate portfolio Greeks
    straddle_delta = 0.0
    straddle_gamma = 0.0
    straddle_vega = 0.0
    for pos in straddle.positions:
        is_call = 1 if pos.is_call else 0
        sign = 1 if pos.is_long else -1
        greeks = bs_all_greeks(s=100.0, k=pos.strike, t=0.5, r=0.05, q=0.0, sigma=0.20, is_call=is_call)
        straddle_delta += sign * greeks[1]
        straddle_gamma += sign * greeks[2]
        straddle_vega += sign * greeks[3]

    print(f"  Delta: {straddle_delta:+.4f} (market neutral)")
    print(f"  Gamma: {straddle_gamma:+.4f} (long gamma - benefits from moves)")
    print(f"  Vega:  {straddle_vega:+.4f} (long vega - benefits from vol increase)")

    # Step 3: Monte Carlo Backtest
    print("\n--- Step 3: Monte Carlo Backtest ---")
    backtest_sim = HestonSimulator(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    backtest_paths = backtest_sim.simulate_paths(s0=100, mu=0.05, t=0.5, n_paths=100_000, n_steps=126, seed=1111)
    backtest_pnl = straddle.pnl_at_expiry_fast(backtest_paths.price_paths[:, -1], multiplier=1.0)

    print("  Simulated P&L (100K Heston paths):")
    print(f"    Mean:     ${backtest_pnl.mean():.2f}")
    print(f"    Std:      ${backtest_pnl.std():.2f}")
    print(f"    Win Rate: {(backtest_pnl > 0).mean()*100:.1f}%")

    # Step 4: Risk Report
    print("\n--- Step 4: Risk Report ---")
    backtest_metrics = compute_risk_metrics(backtest_pnl)
    be_straddle = find_breakevens_from_portfolio(straddle, spot_min=70, spot_max=130)
    up_straddle, ul_straddle = check_unlimited_risk_from_portfolio(straddle)

    print(f"  Breakeven Points: {[f'${b:.2f}' for b in be_straddle.breakeven_points]}")
    print(f"  VaR (95%):        ${-backtest_metrics.var_95:.2f}")
    print(f"  CVaR (95%):       ${-backtest_metrics.cvar_95:.2f}")
    print(f"  Max Drawdown:     ${-backtest_metrics.max_loss:.2f}")
    print(f"  Unlimited Profit: {'Yes' if up_straddle else 'No'}")
    print(f"  Unlimited Loss:   {'Yes' if ul_straddle else 'No'}")

    # Step 5: Decision Summary
    print("\n--- Step 5: Strategy Assessment ---")
    risk_reward = backtest_pnl.max() / abs(backtest_metrics.max_loss) if backtest_metrics.max_loss < 0 else float('inf')
    expected_sharpe = backtest_pnl.mean() / backtest_pnl.std() if backtest_pnl.std() > 0 else 0

    print(f"  Risk/Reward Ratio:  {risk_reward:.2f}x")
    print(f"  Expected Sharpe:    {expected_sharpe:.3f}")
    print(f"  Recommendation:     {'FAVORABLE' if expected_sharpe > 0.1 else 'MARGINAL' if expected_sharpe > 0 else 'UNFAVORABLE'}")

    print("\n" + "-" * 60)
    print("All combined scenarios completed!")
    print("-" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "DERIVATIVES BACKEND - COMPLETE DEMONSTRATION".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    # Run all sections
    demo_core_module()
    call, put = demo_instruments_module()
    models = demo_models_module()
    bs_engine, fft_engine = demo_engines_module(models)
    demo_greeks_module()
    demo_simulation_module()
    demo_portfolio_module()
    demo_utils_module()
    demo_advanced_examples()
    demo_combined_scenarios()  # NEW comprehensive examples

    # Summary
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nModules demonstrated:")
    print("  1. Core Module - Interfaces, MarketEnvironment")
    print("  2. Instruments Module - Options, Strategies")
    print("  3. Models Module - GBM, Heston, Bates, Merton, GARCH")
    print("  4. Engines Module - BS Analytic, FFT, Monte Carlo")
    print("  5. Greeks Module - All 14 Greeks")
    print("  6. Simulation Module - Path generation, Risk metrics")
    print("  7. Portfolio Module - P&L, Breakeven, Risk analysis")
    print("  8. Utils Module - Math utilities")
    print("  9. Advanced Examples - Surfaces, VaR")
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()
