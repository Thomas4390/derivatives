"""
Exotic options configuration for Options Greeks Explorer.

Contains instrument class definitions, barrier subtypes, educational
descriptions, and default values for exotic option types.
"""

# =============================================================================
# INSTRUMENT CLASSES
# =============================================================================

# Instrument classes available as portfolio legs
INSTRUMENT_CLASSES = {
    "barrier": "Barrier",
    "asian": "Asian (Geo)",
    "digital": "Digital",
    "lookback_fixed": "Lookback (Fixed)",
    "chooser": "Chooser",
    "asset_or_nothing": "Asset-or-Nothing",
    "power": "Power",
    "gap": "Gap",
}

# Exotic type display names
EXOTIC_TYPE_NAMES = {
    "barrier": "Barrier Option",
    "asian": "Asian Option (Geometric)",
    "digital": "Digital Option (Cash-or-Nothing)",
    "lookback_fixed": "Lookback (Fixed Strike)",
    "chooser": "Chooser Option",
    "asset_or_nothing": "Asset-or-Nothing Option",
    "power": "Power Option",
    "gap": "Gap Option",
}

# =============================================================================
# BARRIER SUBTYPES
# =============================================================================

# Barrier subtypes: all 8 combinations of up/down, in/out, call/put
BARRIER_SUBTYPES = {
    "up_out_call": {
        "label": "Up-and-Out Call",
        "is_call": True,
        "is_up": True,
        "is_knock_in": False,
    },
    "down_out_call": {
        "label": "Down-and-Out Call",
        "is_call": True,
        "is_up": False,
        "is_knock_in": False,
    },
    "up_out_put": {
        "label": "Up-and-Out Put",
        "is_call": False,
        "is_up": True,
        "is_knock_in": False,
    },
    "down_out_put": {
        "label": "Down-and-Out Put",
        "is_call": False,
        "is_up": False,
        "is_knock_in": False,
    },
}

# =============================================================================
# EDUCATIONAL DESCRIPTIONS
# =============================================================================

EXOTIC_DESCRIPTIONS = {
    "barrier": (
        "**Barrier options** activate (knock-in) or deactivate (knock-out) when the underlying "
        "hits a specified barrier level. **Key identity:** Knock-In + Knock-Out = Vanilla option "
        "(for same strike and barrier). Barriers are cheaper than vanilla because they can become "
        "worthless (knock-out) or may never activate (knock-in)."
    ),
    "asian": (
        "**Asian (Geometric) options** have a payoff based on the geometric average price of the "
        "underlying over the option's life, rather than the terminal price. They are **always cheaper** "
        "than vanilla options because averaging reduces the effective volatility "
        "(by a factor of 1/sqrt(3) for geometric averaging)."
    ),
    "digital": (
        "**Digital (Cash-or-Nothing) options** pay a fixed amount if the option expires in-the-money, "
        "and zero otherwise. The price equals PV(probability of ITM) x Payout. "
        "**Key identity:** Digital Call + Digital Put = e^(-rT) x Payout."
    ),
    "lookback_floating": (
        "**Floating-strike lookback options** (Goldman-Sosin-Gatto 1979) have a strike set by "
        "the path extremum. Call payoff: S_T - min(S). Put payoff: max(S) - S_T. "
        "They are always in-the-money and more expensive than fixed-strike lookbacks."
    ),
    "lookback_fixed": (
        "**Fixed-strike lookback options** have payoff based on the maximum (call) or minimum (put) "
        "of the underlying price vs a fixed strike. Call payoff: max(M_max - K, 0). "
        "Put payoff: max(K - M_min, 0). They are more expensive than vanilla since the "
        "lookback feature can only improve the payoff."
    ),
    "chooser": (
        "**Chooser options** (Rubinstein 1991) let the holder choose at a predetermined time t_c "
        "whether the option becomes a call or put. The price is always >= max(call, put). "
        "**Decomposition:** V = BS_call(S,K,T) + BS_put(S, K*exp(-(r-q)*(T-t_c)), t_c)."
    ),
    "asset_or_nothing": (
        "**Asset-or-Nothing options** pay the asset price S if the option expires ITM, "
        "and zero otherwise. Call: S*exp(-qT)*N(d1). Put: S*exp(-qT)*N(-d1). "
        "**Key identity:** Asset-or-Nothing Call + Digital Call*K = Vanilla Call + K*exp(-rT)."
    ),
    "power": (
        "**Power options** have payoff based on S^n instead of S. Call: max(S_T^n - K, 0). "
        "The adjusted volatility is n*sigma and the drift includes an extra n*(n-1)*sigma^2/2 term. "
        "For n=1, the power option reduces to a vanilla option."
    ),
    "gap": (
        "**Gap options** have two strikes: K1 (payment) and K2 (trigger). "
        "Call pays (S-K1) if S>K2. The gap option price **can be negative** when K1 > K2. "
        "When K1 = K2, the gap option equals a vanilla option."
    ),
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_EXOTIC_DTE = 90
DEFAULT_DIGITAL_PAYOUT = 1.0
DEFAULT_BARRIER_UP_FACTOR = 1.10  # Barrier at 110% of spot for up-barriers
DEFAULT_BARRIER_DOWN_FACTOR = 0.90  # Barrier at 90% of spot for down-barriers
