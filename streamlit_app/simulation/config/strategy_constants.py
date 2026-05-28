"""
Strategy constants for Monte Carlo Simulation Explorer.

Self-contained strategy definitions for the simulation app.
The simulation supports all exotic types including path-dependent ones
(barrier knock-in, lookback floating) that are computed via MC on actual paths.
"""

# =============================================================================
# CONTRACT
# =============================================================================

CONTRACT_MULTIPLIER = 100

# =============================================================================
# STRATEGIES WITH STOCK LEGS
# =============================================================================

STRATEGIES_WITH_STOCK = ["covered_call", "protective_put", "collar"]

STRATEGY_STOCK_POSITION = {
    "covered_call": "long",
    "protective_put": "long",
    "collar": "long",
}

# =============================================================================
# INSTRUMENT CLASSES (exotic types available as portfolio legs)
# =============================================================================

INSTRUMENT_CLASSES = {
    "barrier": "Barrier",
    "asian": "Asian (Geo)",
    "digital": "Digital",
    "lookback_floating": "Lookback (Floating)",
    "lookback_fixed": "Lookback (Fixed)",
    "chooser": "Chooser",
    "asset_or_nothing": "Asset-or-Nothing",
    "power": "Power",
    "gap": "Gap",
}

# =============================================================================
# STRATEGY DISPLAY NAMES
# =============================================================================

STRATEGY_DISPLAY_NAMES = {
    "": "",
    # Vanilla
    "long_call": "Long Call",
    "short_call": "Short Call",
    "long_put": "Long Put",
    "short_put": "Short Put",
    # Spreads
    "bull_call_spread": "Bull Call Spread",
    "bear_put_spread": "Bear Put Spread",
    "bull_put_spread": "Bull Put Spread",
    "bear_call_spread": "Bear Call Spread",
    # Volatility
    "long_straddle": "Long Straddle",
    "short_straddle": "Short Straddle",
    "long_strangle": "Long Strangle",
    "short_strangle": "Short Strangle",
    # Complex
    "iron_condor": "Iron Condor",
    "butterfly": "Butterfly",
    "covered_call": "Covered Call",
    "protective_put": "Protective Put",
    "collar": "Collar",
    # Exotic
    "barrier_up_out_call": "Up-and-Out Call",
    "barrier_down_out_put": "Down-and-Out Put",
    "barrier_knock_in_call": "Down-and-In Call",
    "digital_call": "Digital Call",
    "digital_put": "Digital Put",
    "digital_range_bet": "Digital Range Bet",
    "asian_call": "Asian Call (Geometric)",
    "lookback_floating_call": "Lookback Call (Floating Strike)",
    "lookback_fixed_call": "Lookback Call (Fixed Strike)",
    "chooser": "Chooser Option",
    "asset_or_nothing_call": "Asset-or-Nothing Call",
    "asset_or_nothing_put": "Asset-or-Nothing Put",
    "power_call": "Power Call (n=2)",
    "gap_call": "Gap Call",
}

# =============================================================================
# STRATEGY DESCRIPTIONS
# =============================================================================

STRATEGY_DESCRIPTIONS = {
    # Directional
    "long_call": "Bullish bet: unlimited upside, max loss = premium paid.",
    "short_call": "Bearish/neutral: collect premium, unlimited risk if stock rises.",
    "long_put": "Bearish bet: profits when stock falls, max loss = premium paid.",
    "short_put": "Bullish/neutral: collect premium, risk if stock drops below strike.",
    # Spreads
    "bull_call_spread": "Moderately bullish: buy lower-strike call, sell higher-strike call. Capped profit & loss.",
    "bear_put_spread": "Moderately bearish: buy higher-strike put, sell lower-strike put. Capped profit & loss.",
    "bull_put_spread": "Bullish credit spread: sell higher-strike put, buy lower-strike put for protection.",
    "bear_call_spread": "Bearish credit spread: sell lower-strike call, buy higher-strike call for protection.",
    # Volatility
    "long_straddle": "Bet on high volatility: buy ATM call + put. Profits from large moves in either direction.",
    "short_straddle": "Bet on low volatility: sell ATM call + put. Profits if stock stays near strike.",
    "long_strangle": "Cheaper volatility bet: buy OTM call + put. Needs a larger move than straddle.",
    "short_strangle": "Wider neutral zone: sell OTM call + put. Profits if stock stays between strikes.",
    # Advanced
    "iron_condor": "Neutral strategy: sell inner strangle + buy outer wings. Max profit if stock stays in range.",
    "butterfly": "Neutral pinning bet: profits if stock finishes near middle strike at expiry.",
    # Stock + Options
    "covered_call": "Income strategy: long stock + short OTM call. Caps upside for premium income.",
    "protective_put": "Insurance: long stock + long OTM put. Protects against downside.",
    "collar": "Hedged position: long stock + protective put + covered call. Limits both upside and downside.",
    # Exotic — Barrier
    "barrier_up_out_call": "Cheaper call that dies if stock rises above the barrier. Knock-In + Knock-Out = Vanilla.",
    "barrier_down_out_put": "Cheaper put that dies if stock drops below the barrier.",
    "barrier_knock_in_call": "Call that activates only if stock first dips below the barrier. Path-dependent payoff.",
    # Exotic — Digital
    "digital_call": "Binary payoff: pays a fixed amount if stock > strike at expiry, else zero.",
    "digital_put": "Binary payoff: pays a fixed amount if stock < strike at expiry, else zero.",
    "digital_range_bet": "Pays if stock finishes between two strikes. Built from two digital calls.",
    # Exotic — Path-dependent
    "asian_call": "Payoff based on geometric average price. Cheaper than vanilla (averaging reduces vol).",
    "lookback_floating_call": "Floating strike: payoff = S_T - min(S). Strike set to path minimum, always ITM.",
    "lookback_fixed_call": "Payoff based on the maximum price observed vs fixed strike.",
    # Exotic — Other
    "chooser": "Choose call or put at time t_c. Price >= max(call, put). Decomposes via put-call parity.",
    "asset_or_nothing_call": "Pays the asset price S if S > K at expiry, else zero.",
    "asset_or_nothing_put": "Pays the asset price S if S < K at expiry, else zero.",
    "power_call": "Payoff on S^n instead of S. n=2 gives quadratic payoff. Adjusted vol = n * sigma.",
    "gap_call": "Two strikes: K1 (payment) and K2 (trigger). Pays (S-K1) if S > K2.",
}

# =============================================================================
# STRATEGY LEG DEFINITIONS
# =============================================================================

STRATEGY_LEGS = {
    # ── Vanilla ──
    "long_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
        }
    ],
    "short_call": [
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.0,
            "quantity": 1,
        }
    ],
    "long_put": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
        }
    ],
    "short_put": [
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 1.0,
            "quantity": 1,
        }
    ],
    # ── Vertical Spreads ──
    "bull_call_spread": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 0.975,
            "quantity": 1,
        },
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.025,
            "quantity": 1,
        },
    ],
    "bear_put_spread": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.025,
            "quantity": 1,
        },
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 0.975,
            "quantity": 1,
        },
    ],
    "bull_put_spread": [
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 0.975,
            "quantity": 1,
        },
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.925,
            "quantity": 1,
        },
    ],
    "bear_call_spread": [
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.025,
            "quantity": 1,
        },
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.075,
            "quantity": 1,
        },
    ],
    # ── Volatility ──
    "long_straddle": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
        },
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
        },
    ],
    "short_straddle": [
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.0,
            "quantity": 1,
        },
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 1.0,
            "quantity": 1,
        },
    ],
    "long_strangle": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
        },
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.05,
            "quantity": 1,
        },
    ],
    "short_strangle": [
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 0.95,
            "quantity": 1,
        },
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
        },
    ],
    # ── Complex ──
    "iron_condor": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.90,
            "quantity": 1,
        },
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 0.95,
            "quantity": 1,
        },
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
        },
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.10,
            "quantity": 1,
        },
    ],
    "butterfly": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
        },
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.0,
            "quantity": 2,
        },
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.05,
            "quantity": 1,
        },
    ],
    # ── Stock + Options ──
    "covered_call": [
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
        }
    ],
    "protective_put": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
        }
    ],
    "collar": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
        },
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
        },
    ],
    # ── Exotic — Barrier ──
    "barrier_up_out_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "barrier",
            "barrier_factor": 1.10,
            "is_up": True,
            "is_knock_in": False,
        },
    ],
    "barrier_down_out_put": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "barrier",
            "barrier_factor": 0.90,
            "is_up": False,
            "is_knock_in": False,
        },
    ],
    "barrier_knock_in_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "barrier",
            "barrier_factor": 0.90,
            "is_up": False,
            "is_knock_in": True,
        },
    ],
    # ── Exotic — Digital ──
    "digital_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "digital",
            "payout": 1.0,
        },
    ],
    "digital_put": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "digital",
            "payout": 1.0,
        },
    ],
    "digital_range_bet": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
            "instrument_class": "digital",
            "payout": 1.0,
        },
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
            "instrument_class": "digital",
            "payout": 1.0,
        },
    ],
    # ── Exotic — Path-Dependent ──
    "asian_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "asian",
        },
    ],
    "lookback_floating_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "lookback_floating",
        },
    ],
    "lookback_fixed_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "lookback_fixed",
        },
    ],
    # ── Exotic — Other ──
    "chooser": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "chooser",
            "choice_time_pct": 0.5,
        },
    ],
    "asset_or_nothing_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "asset_or_nothing",
        },
    ],
    "asset_or_nothing_put": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "asset_or_nothing",
        },
    ],
    "power_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "power",
            "power_n": 2.0,
        },
    ],
    "gap_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "gap",
            "gap_trigger_factor": 1.05,
        },
    ],
}
