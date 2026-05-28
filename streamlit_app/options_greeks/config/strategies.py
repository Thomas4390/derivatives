"""
Strategy definitions for Options Greeks Explorer.

Contains predefined strategy configurations including display names,
descriptions, leg definitions, and stock position mappings.
"""

# =============================================================================
# AVAILABLE STRATEGIES
# =============================================================================

AVAILABLE_STRATEGIES = [
    "",  # Empty option
    # Vanilla strategies
    "long_call",
    "short_call",
    "long_put",
    "short_put",
    # Spread strategies
    "bull_call_spread",
    "bear_put_spread",
    "bull_put_spread",
    "bear_call_spread",
    # Volatility strategies
    "long_straddle",
    "short_straddle",
    "long_strangle",
    "short_strangle",
    # Complex strategies
    "iron_condor",
    "butterfly",
    "covered_call",
    "protective_put",
    "collar",
    # Exotic strategies
    "barrier_up_out_call",
    "barrier_down_out_put",
    "digital_call",
    "digital_put",
    "digital_range_bet",
    "asian_call",
    "lookback_fixed_call",
    "chooser",
    "asset_or_nothing_call",
    "asset_or_nothing_put",
    "power_call",
    "gap_call",
]

# =============================================================================
# DISPLAY NAMES
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
    "digital_call": "Digital Call",
    "digital_put": "Digital Put",
    "digital_range_bet": "Digital Range Bet",
    "asian_call": "Asian Call (Geometric)",
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

STRATEGY_DESCRIPTIONS: dict[str, str] = {
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
    # Exotic -- Barrier
    "barrier_up_out_call": "Cheaper call that dies if stock rises above the barrier. Knock-In + Knock-Out = Vanilla.",
    "barrier_down_out_put": "Cheaper put that dies if stock drops below the barrier.",
    # Exotic -- Digital
    "digital_call": "Binary payoff: pays a fixed amount if stock > strike at expiry, else zero.",
    "digital_put": "Binary payoff: pays a fixed amount if stock < strike at expiry, else zero.",
    "digital_range_bet": "Pays if stock finishes between two strikes. Built from two digital calls.",
    # Exotic -- Path-dependent
    "asian_call": "Payoff based on geometric average price. Cheaper than vanilla (averaging reduces vol).",
    "lookback_fixed_call": "Payoff based on the maximum price observed vs fixed strike.",
    # Exotic -- Other
    "chooser": "Choose call or put at time t_c. Price >= max(call, put). Decomposes via put-call parity.",
    "asset_or_nothing_call": "Pays the asset price S if S > K at expiry, else zero.",
    "asset_or_nothing_put": "Pays the asset price S if S < K at expiry, else zero.",
    "power_call": "Payoff on S^n instead of S. n=2 gives quadratic payoff. Adjusted vol = n * sigma.",
    "gap_call": "Two strikes: K1 (payment) and K2 (trigger). Pays (S-K1) if S > K2.",
    # Structured Products
    "sp_cpn": "Capital Protected Note: bond floor guarantees capital at maturity, upside via participation in the underlying with optional cap.",
    "sp_reverse_convertible": "Reverse Convertible: high coupon in exchange for downside risk — if underlying breaches barrier, investor receives shares instead of notional.",
    "sp_autocallable": "Autocallable: pays periodic coupons if underlying stays above coupon barrier; auto-redeems early if above autocall trigger; knock-in put risk at maturity.",
}

# =============================================================================
# STRATEGY LEG DEFINITIONS
# =============================================================================

# Strategy leg definitions (option_type, position_type, strike_offset from spot)
# Standard strike intervals for clear payoff diagrams:
# - ATM = 1.0 (100% of spot)
# - 5% OTM/ITM intervals for spreads and strangles
# - 10% intervals for iron condor wings
# - Butterfly uses 5% equal spacing for symmetry
STRATEGY_LEGS = {
    # ==========================================================================
    # VANILLA STRATEGIES (Single leg, ATM)
    # ==========================================================================
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
    # ==========================================================================
    # VERTICAL SPREADS (5% width between strikes)
    # ==========================================================================
    # Bull Call Spread: Buy lower strike call, Sell higher strike call
    # Profit when underlying rises, limited risk/reward
    "bull_call_spread": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 0.975,
            "quantity": 1,
        },  # Slightly ITM
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.025,
            "quantity": 1,
        },  # Slightly OTM
    ],
    # Bear Put Spread: Buy higher strike put, Sell lower strike put
    # Profit when underlying falls, limited risk/reward
    "bear_put_spread": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.025,
            "quantity": 1,
        },  # Slightly ITM
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 0.975,
            "quantity": 1,
        },  # Slightly OTM
    ],
    # Bull Put Spread (Credit Spread): Sell higher strike put, Buy lower strike put
    # Profit when underlying stays above short strike
    "bull_put_spread": [
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 0.975,
            "quantity": 1,
        },  # ATM-ish
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.925,
            "quantity": 1,
        },  # OTM protection
    ],
    # Bear Call Spread (Credit Spread): Sell lower strike call, Buy higher strike call
    # Profit when underlying stays below short strike
    "bear_call_spread": [
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.025,
            "quantity": 1,
        },  # ATM-ish
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.075,
            "quantity": 1,
        },  # OTM protection
    ],
    # ==========================================================================
    # VOLATILITY STRATEGIES
    # ==========================================================================
    # Straddle: ATM Call + ATM Put (same strike)
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
    # Strangle: OTM Call + OTM Put (different strikes, 5% OTM each side)
    "long_strangle": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
        },  # 5% OTM put
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.05,
            "quantity": 1,
        },  # 5% OTM call
    ],
    "short_strangle": [
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 0.95,
            "quantity": 1,
        },  # 5% OTM put
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
        },  # 5% OTM call
    ],
    # ==========================================================================
    # COMPLEX STRATEGIES
    # ==========================================================================
    # Iron Condor: Short strangle + long wings for protection
    # Symmetric around spot: 90/95 put spread + 105/110 call spread
    "iron_condor": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.90,
            "quantity": 1,
        },  # Long put wing
        {
            "option_type": "put",
            "position_type": "short",
            "strike_factor": 0.95,
            "quantity": 1,
        },  # Short put
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
        },  # Short call
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.10,
            "quantity": 1,
        },  # Long call wing
    ],
    # Butterfly: Long wings + 2x short middle (5% equal spacing)
    # Maximum profit at middle strike at expiration
    "butterfly": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
        },  # Lower wing
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.0,
            "quantity": 2,
        },  # Body (ATM)
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.05,
            "quantity": 1,
        },  # Upper wing
    ],
    # ==========================================================================
    # STOCK + OPTIONS STRATEGIES
    # ==========================================================================
    # Covered Call: Long stock + Short OTM call
    "covered_call": [
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
        }  # 5% OTM
    ],
    # Protective Put: Long stock + Long OTM put
    "protective_put": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
        }  # 5% OTM
    ],
    # Collar: Long stock + Long OTM put + Short OTM call
    "collar": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 0.95,
            "quantity": 1,
        },  # 5% OTM put
        {
            "option_type": "call",
            "position_type": "short",
            "strike_factor": 1.05,
            "quantity": 1,
        },  # 5% OTM call
    ],
    # ==========================================================================
    # EXOTIC STRATEGIES
    # ==========================================================================
    # Up-and-Out Call: cheap directional bet, knocked out if spot rises too high
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
        }
    ],
    # Down-and-Out Put: cheap downside protection, knocked out if spot drops too far
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
        }
    ],
    # Digital Call: pays fixed amount if spot > strike at expiry
    "digital_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "digital",
            "payout": 1.0,
        }
    ],
    # Digital Put: pays fixed amount if spot < strike at expiry
    "digital_put": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "digital",
            "payout": 1.0,
        }
    ],
    # Digital Range Bet: pays if spot ends between two strikes
    # Long digital call at lower strike + Short digital call at upper strike
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
    # Asian Call (Geometric): payoff based on geometric average price
    "asian_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "asian",
        }
    ],
    # Lookback Fixed Call: payoff based on max price vs fixed strike
    "lookback_fixed_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "lookback_fixed",
        }
    ],
    # Chooser: holder chooses call or put at choice time
    "chooser": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "chooser",
            "choice_time_pct": 0.5,
        }
    ],
    # Asset-or-Nothing Call: pays S if S > K at expiry
    "asset_or_nothing_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "asset_or_nothing",
        }
    ],
    # Asset-or-Nothing Put: pays S if S < K at expiry
    "asset_or_nothing_put": [
        {
            "option_type": "put",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "asset_or_nothing",
        }
    ],
    # Power Call (n=2): payoff = max(S^2 - K, 0)
    "power_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "power",
            "power_n": 2.0,
        }
    ],
    # Gap Call: trigger at K2, pays (S - K1) if S > K2
    "gap_call": [
        {
            "option_type": "call",
            "position_type": "long",
            "strike_factor": 1.0,
            "quantity": 1,
            "instrument_class": "gap",
            "gap_trigger_factor": 1.05,
        }
    ],
}

# =============================================================================
# STOCK POSITION MAPPINGS
# =============================================================================

# Strategies that include a stock position
STRATEGIES_WITH_STOCK = ["covered_call", "protective_put", "collar"]

# Stock position type for each strategy (default is 'long')
STRATEGY_STOCK_POSITION = {
    "covered_call": "long",
    "protective_put": "long",
    "collar": "long",
}
