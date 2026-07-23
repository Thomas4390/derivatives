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
    "digital": "Digital",
    "chooser": "Chooser",
    "asset_or_nothing": "Asset-or-Nothing",
    "power": "Power",
    "gap": "Gap",
    "powered": "Powered",
    "capped_power": "Capped Power",
    "log_contract": "Log Contract",
    "log_option": "Log Option",
    "supershare": "Supershare",
    "double_barrier": "Double Barrier",
    "discrete_barrier": "Discrete Barrier",
    "partial_barrier": "Partial-Time Barrier",
    "binary_barrier": "Binary Barrier",
    "arithmetic_asian": "Asian (Arithmetic)",
}

# Exotic type display names
EXOTIC_TYPE_NAMES = {
    "barrier": "Barrier Option",
    "digital": "Digital Option (Cash-or-Nothing)",
    "chooser": "Chooser Option",
    "asset_or_nothing": "Asset-or-Nothing Option",
    "power": "Power Option",
    "gap": "Gap Option",
    "powered": "Powered Option (Esser)",
    "capped_power": "Capped Power Option (Esser)",
    "log_contract": "Log Contract (Neuberger)",
    "log_option": "Log Option (Wilmott)",
    "supershare": "Supershare Option (Hakansson)",
    "double_barrier": "Double Barrier Option (Ikeda-Kunitomo)",
    "discrete_barrier": "Discrete Barrier Option (BGK correction)",
    "partial_barrier": "Partial-Time Barrier Option (Heynen-Kat)",
    "binary_barrier": "Binary Barrier Option (Reiner-Rubinstein)",
    "arithmetic_asian": "Arithmetic Asian (Turnbull-Wakeman)",
}

# Every exotic key a leg dict may carry (written by the sidebar leg editor).
# ``prepare_portfolio_data`` must pass ALL of them through: the base expiry
# curve, the P&L metrics and the premium have to describe the same instrument
# as the raw leg used by the scenario overlay. All but ``scenario`` (transient
# overlay state) are also persisted by ``services.portfolio_io`` — a sync test
# locks the two lists together.
EXOTIC_LEG_KEYS: tuple[str, ...] = (
    "barrier",
    "cap",
    "choice_time_pct",
    "extra1",
    "gap_trigger",
    "is_knock_in",
    "is_up",
    "payout",
    "power_n",
    "rebate",
    "scenario",
    "lower_strike",
    "upper_strike",
    "dbl_lower",
    "dbl_upper",
    "adv_barrier",
    "adv_is_up",
    "adv_in",
    "monitoring_points",
    "t1_pct",
    "partial_type",
    "cash",
    "binary_type",
    "avg_elapsed_pct",
    "avg_realized",
)

# =============================================================================
# PAYOFF CLASSIFICATION (single source of truth)
# =============================================================================
# Each exotic the picker exposes is either TERMINAL (payoff = f(S_T), exact
# diagram) or DISCRETE_EVENT (payoff = f(S_T) conditional on a binary path
# event -> a scenario control draws the conditional branch). Continuous /
# counter families (asian, lookback, time-switch, soft-barrier) are not in the
# picker; they live in the Simulation app.
PAYOFF_SCENARIOS: dict[str, dict] = {
    # --- terminal-spot ---
    "digital": {"kind": "terminal"},
    "asset_or_nothing": {"kind": "terminal"},
    "power": {"kind": "terminal"},
    "gap": {"kind": "terminal"},
    "powered": {"kind": "terminal"},
    "capped_power": {"kind": "terminal"},
    "log_contract": {"kind": "terminal"},
    "log_option": {"kind": "terminal"},
    "supershare": {"kind": "terminal"},
    # Payoff convention lives in the adapter: A_T = w*SA + (1-w)*S_T.
    "arithmetic_asian": {"kind": "terminal"},
    # --- discrete-event (scenarios[0] = base case) ---
    "barrier": {
        "kind": "discrete_event",
        "scenarios": [
            ("not_touched", "Barrier not touched"),
            ("touched", "Barrier touched"),
        ],
    },
    "discrete_barrier": {
        "kind": "discrete_event",
        "scenarios": [("not_touched", "Not touched"), ("touched", "Touched")],
    },
    "double_barrier": {
        "kind": "discrete_event",
        "scenarios": [
            ("not_touched", "Stayed in corridor"),
            ("touched", "Touched a barrier"),
        ],
    },
    "partial_barrier": {
        "kind": "discrete_event",
        "scenarios": [
            ("not_touched", "Not touched in window"),
            ("touched", "Touched in window"),
        ],
    },
    "binary_barrier": {
        "kind": "discrete_event",
        "scenarios": [("not_touched", "Barrier not hit"), ("touched", "Barrier hit")],
    },
    "chooser": {
        "kind": "discrete_event",
        "scenarios": [("call", "Chose call"), ("put", "Chose put")],
    },
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
    "powered": (
        "**Powered options** (Esser 2003) raise the *standard* payoff to an integer power i: "
        "a call pays max(S_T - K, 0)^i, a put max(K - S_T, 0)^i. The exponent sharply amplifies "
        "in-the-money payoffs. For i=1 it reduces to a vanilla option. Distinct from the Power "
        "option, which raises the *asset* itself (S^n - K)."
    ),
    "capped_power": (
        "**Capped power options** (Esser 2003) are power options whose payoff is capped at C: "
        "a call pays min(max(S_T^i - K, 0), C). The cap bounds the otherwise unbounded power "
        "payoff and lowers the premium. Relaxing C -> infinity recovers the standard power option. "
        "For a put, the cap must be below the strike."
    ),
    "log_contract": (
        "**Log contracts** (Neuberger 1994) pay ln(S_T / K) at maturity — not strictly an option "
        "(the payoff can be negative), but the building block of variance/volatility swaps. "
        "K=1 gives the pure log(S) contract. Call/put is irrelevant here."
    ),
    "log_option": (
        "**Log options** (Wilmott 2000) pay max(ln(S_T / K), 0) — an option on the asset's "
        "log-return, struck at ln(K). The payoff grows only logarithmically in S, so it is far "
        "cheaper than a vanilla call. Call/put is irrelevant here."
    ),
    "supershare": (
        "**Supershare options** (Hakansson 1976) pay S_T / X_L if X_L < S_T < X_H at maturity, "
        "and zero otherwise — a normalised asset-or-nothing band. Portfolios of supershares form "
        "the classic 'superfund'. Priced as a scaled difference of two asset-or-nothing calls."
    ),
    "double_barrier": (
        "**Double-barrier options** (Ikeda-Kunitomo 1992) knock out (or in) if the underlying "
        "touches *either* a lower barrier L or an upper barrier U before expiry. A double "
        "knock-out is the cheapest barrier — it survives only if the asset stays inside the "
        "(L, U) corridor for the whole life."
    ),
    "discrete_barrier": (
        "**Discrete-barrier options** are single barriers monitored only at a finite set of dates "
        "(e.g. daily), not continuously. Priced by the Broadie-Glasserman-Kou (1997) continuity "
        "correction: the barrier is shifted by exp(±0.5826·σ·√(T/m)). Fewer monitoring points "
        "make a knock-out more valuable (harder to knock)."
    ),
    "partial_barrier": (
        "**Partial-time barrier options** (Heynen-Kat 1994) monitor the barrier only over part "
        "of the life: start types (*_A) watch [0, t1]; end types (B1/B2) watch [t1, T]. Because "
        "the barrier is live for less time, they are more valuable (knock-out) than the "
        "fully-monitored barrier."
    ),
    "binary_barrier": (
        "**Binary-barrier options** (Reiner-Rubinstein 1991) are barrier digitals: they pay a "
        "fixed cash amount or one unit of the asset, contingent on the barrier being hit (in) or "
        "not (out) — 28 variants in Haug's catalogue. Building blocks of one-touch / no-touch "
        "and barrier rebate structures."
    ),
    "arithmetic_asian": (
        "Average-rate option on the arithmetic mean of the underlying over the "
        "averaging window (Turnbull-Wakeman). The realized average so far acts "
        "as a dead weight: once a fraction w of the window has elapsed, the "
        "terminal payoff responds to the remaining path with slope (1-w) only."
    ),
}

# =============================================================================
# ADVANCED-BARRIER TYPE SELECTORS
# =============================================================================

# Heynen-Kat partial-time monitoring types (key -> UI label). Order matches the
# adapter's _PARTIAL_BARRIER_TYPES_UI.
PARTIAL_BARRIER_TYPES = {
    "down_out_A": "Down-out, start window [0, t1]",
    "up_out_A": "Up-out, start window [0, t1]",
    "out_B1": "Knock-out (any touch), end window [t1, T]",
    "down_out_B2": "Down-out, end window [t1, T]",
    "up_out_B2": "Up-out, end window [t1, T]",
}

# Reiner-Rubinstein binary-barrier types (1..28, Haug 4.19.5) -> UI label.
BINARY_BARRIER_TYPES = {
    1: "1 — Down-in cash (at hit)",
    2: "2 — Up-in cash (at hit)",
    3: "3 — Down-in asset (at hit)",
    4: "4 — Up-in asset (at hit)",
    5: "5 — Down-in cash (at expiry)",
    6: "6 — Up-in cash (at expiry)",
    7: "7 — Down-in asset (at expiry)",
    8: "8 — Up-in asset (at expiry)",
    9: "9 — Down-out cash",
    10: "10 — Up-out cash",
    11: "11 — Down-out asset",
    12: "12 — Up-out asset",
    13: "13 — Down-in cash call",
    14: "14 — Up-in cash call",
    15: "15 — Down-in asset call",
    16: "16 — Up-in asset call",
    17: "17 — Down-in cash put",
    18: "18 — Up-in cash put",
    19: "19 — Down-in asset put",
    20: "20 — Up-in asset put",
    21: "21 — Down-out cash call",
    22: "22 — Up-out cash call",
    23: "23 — Down-out asset call",
    24: "24 — Up-out asset call",
    25: "25 — Down-out cash put",
    26: "26 — Up-out cash put",
    27: "27 — Down-out asset put",
    28: "28 — Up-out asset put",
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_EXOTIC_DTE = 90
DEFAULT_DIGITAL_PAYOUT = 1.0
DEFAULT_BARRIER_UP_FACTOR = 1.10  # Barrier at 110% of spot for up-barriers
DEFAULT_BARRIER_DOWN_FACTOR = 0.90  # Barrier at 90% of spot for down-barriers


# =============================================================================
# GREEK AVAILABILITY
# =============================================================================

# Which Greeks each exotic instrument family exposes on the aggregate spot-axis
# Greeks surface. Every exotic family is priced there through
# ``calculate_exotic_greeks_surface``, which returns first-order Greeks only, so
# second- and third-order Greeks are NOT defined for exotic legs on this view.
# Vanilla legs (the default in the Greeks tabs' availability helper) expose the
# full analytic set. The higher-order tabs intersect these sets over the legs
# present and hide any Greek not available for every leg, so an exotic position
# shows an explanatory note instead of a misleading flat-zero curve.
_EXOTIC_FIRST_ORDER_GREEKS = frozenset(
    {"price", "delta", "gamma", "vega", "theta", "rho"}
)
GREEK_AVAILABILITY: dict[str, frozenset] = {
    cls: _EXOTIC_FIRST_ORDER_GREEKS for cls in INSTRUMENT_CLASSES
}


# =============================================================================
# EXOTIC PARAMETER SWEEP (payoff explorer)
# =============================================================================

# Terminal exotic families whose primary shape parameter can be swept from a
# slider under the payoff chart to see its effect on the terminal payoff. Kept
# to strike-independent params with a direct payoff-path override
# (``calculate_exotic_payoff_at_expiry_vec`` reads the leg key straight off the
# position dict), so the sweep needs no strike-relative range logic and never
# collides with the discrete-event outcome overlay (those families are
# path-dependent, not terminal).
EXOTIC_SWEEP_PARAM: dict[str, dict] = {
    "power": {
        "label": "Sweep power exponent n",
        "short": "n",
        "key": "power_n",
        "default": 2.0,
        "lo": 1.0,
        "hi": 3.0,
        "step": 0.25,
    },
    "powered": {
        "label": "Sweep power exponent n",
        "short": "n",
        "key": "power_n",
        "default": 2.0,
        "lo": 1.0,
        "hi": 3.0,
        "step": 1.0,
    },
    "capped_power": {
        "label": "Sweep power exponent n",
        "short": "n",
        "key": "power_n",
        "default": 2.0,
        "lo": 1.0,
        "hi": 3.0,
        "step": 0.25,
    },
    "digital": {
        "label": "Sweep cash payout",
        "short": "payout",
        "key": "payout",
        "default": 1.0,
        "lo": 0.5,
        "hi": 3.0,
        "step": 0.25,
    },
    "arithmetic_asian": {
        "label": "Sweep realized average SA (what-if)",
        "short": "SA",
        "key": "avg_realized",
        "default": 100.0,
        "lo": 10.0,
        "hi": 500.0,
        "step": 1.0,
    },
}
