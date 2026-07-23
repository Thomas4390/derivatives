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
    # Haug advanced catalog (registry-priced, model-consistent in this app)
    "powered": "Powered (Esser)",
    "capped_power": "Capped Power (Esser)",
    "log_contract": "Log Contract",
    "log_option": "Log Option",
    "supershare": "Supershare",
    "double_barrier": "Double Barrier",
    "discrete_barrier": "Discrete Barrier",
    "partial_barrier": "Partial-Time Barrier",
    "binary_barrier": "Binary Barrier",
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
    # Haug advanced catalog
    "powered_call": "Powered Call (Esser)",
    "capped_power_call": "Capped Power Call (Esser)",
    "log_contract": "Log Contract (Neuberger)",
    "log_option": "Log Option (Wilmott)",
    "supershare": "Supershare (Hakansson)",
    "double_barrier_call": "Double Barrier Call (Ikeda-Kunitomo)",
    "discrete_barrier_call": "Discrete Barrier Call (BGK)",
    "partial_barrier_call": "Partial-Time Barrier Call (Heynen-Kat)",
    "binary_barrier_call": "Binary Barrier (Reiner-Rubinstein)",
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
    # Exotic — Haug advanced
    "powered_call": "Esser powered payoff max(S-K, 0)^i. Convex, leverages deep-ITM moves.",
    "capped_power_call": "Esser power payoff min(max(S^i - K, 0), cap). Capped upside.",
    "log_contract": "Neuberger log contract: pays ln(S/K). Building block for variance swaps.",
    "log_option": "Wilmott log option: pays max(ln(S/K), 0). Convex in log-moneyness.",
    "supershare": "Hakansson supershare: pays S/X_L if X_L < S < X_H at expiry, else zero.",
    "double_barrier_call": "Ikeda-Kunitomo corridor: knocks out if the path leaves [L, U].",
    "discrete_barrier_call": "Barrier monitored at discrete dates only (BGK correction).",
    "partial_barrier_call": "Heynen-Kat: barrier live only over a partial window [0, t1] (or [t1, T]).",
    "binary_barrier_call": "Reiner-Rubinstein binary barrier: cash/asset gated by a path hit (28 types).",
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
    # ── Exotic — Haug Power (Esser) ──
    "powered_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "powered",
            "power_n": 2,
        },
    ],
    "capped_power_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "capped_power",
            "power_n": 2.0,
            "cap": 50.0,
        },
    ],
    # ── Exotic — Haug Analytic ──
    "log_contract": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "log_contract",
        },
    ],
    "log_option": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "log_option",
        },
    ],
    "supershare": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "supershare",
        },
    ],
    # ── Exotic — Haug Advanced Barriers ──
    "double_barrier_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "double_barrier",
        },
    ],
    "discrete_barrier_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "discrete_barrier",
        },
    ],
    "partial_barrier_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "partial_barrier",
        },
    ],
    "binary_barrier_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "binary_barrier",
        },
    ],
}
