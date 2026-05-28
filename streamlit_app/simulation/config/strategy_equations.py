"""
Option strategy equations for the Simulation page.

Each strategy returns {name, description, equations: [{label, latex}]}
displayed in a st.expander() at the bottom of the page.
"""

# ── Shared BS building blocks ──────────────────────────────────────────

_BS_D1D2 = {
    "label": "Black-Scholes d₁ / d₂",
    "latex": (
        r"d_1 = \frac{\ln(S/K) + (r - q + \tfrac{\sigma^2}{2})\,T}"
        r"{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}"
    ),
}

_BS_CALL_PRICE = {
    "label": "Call Price (Black-Scholes)",
    "latex": r"C = S\,e^{-qT}\,N(d_1) - K\,e^{-rT}\,N(d_2)",
}

_BS_PUT_PRICE = {
    "label": "Put Price (Black-Scholes)",
    "latex": r"P = K\,e^{-rT}\,N(-d_2) - S\,e^{-qT}\,N(-d_1)",
}


def _vanilla(direction: str, option_type: str) -> dict:
    """Vanilla option equations (long/short call/put)."""
    is_call = option_type == "call"
    sign = "+" if direction == "long" else "-"
    name = f"{direction.title()} {option_type.title()}"

    payoff_inner = r"S_T - K" if is_call else r"K - S_T"
    payoff_latex = rf"\text{{Payoff}} = {sign}\max({payoff_inner},\, 0)"

    price_block = _BS_CALL_PRICE if is_call else _BS_PUT_PRICE

    profit_latex = (
        rf"\text{{P\&L}} = {sign}\bigl[\max({payoff_inner},\, 0) - "
        rf"{'C' if is_call else 'P'}\bigr]"
    )

    return {
        "name": name,
        "description": f"{'Buy' if direction == 'long' else 'Sell'} a European {option_type} option.",
        "equations": [
            _BS_D1D2,
            price_block,
            {"label": "Payoff at Maturity", "latex": payoff_latex},
            {"label": "Profit / Loss", "latex": profit_latex},
        ],
    }


# ── Vertical Spreads ───────────────────────────────────────────────────


def _bull_call_spread() -> dict:
    return {
        "name": "Bull Call Spread",
        "description": "Buy a call at K₁ and sell a call at K₂ > K₁ (net debit).",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = \max(S_T - K_1,\, 0) - \max(S_T - K_2,\, 0)",
            },
            {
                "label": "Max Profit / Max Loss",
                "latex": (
                    r"\text{Max Profit} = K_2 - K_1 - \text{Net Premium}, \qquad "
                    r"\text{Max Loss} = \text{Net Premium}"
                ),
            },
            {
                "label": "Breakeven",
                "latex": r"S^* = K_1 + \text{Net Premium}",
            },
        ],
    }


def _bear_put_spread() -> dict:
    return {
        "name": "Bear Put Spread",
        "description": "Buy a put at K₂ and sell a put at K₁ < K₂ (net debit).",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = \max(K_2 - S_T,\, 0) - \max(K_1 - S_T,\, 0)",
            },
            {
                "label": "Max Profit / Max Loss",
                "latex": (
                    r"\text{Max Profit} = K_2 - K_1 - \text{Net Premium}, \qquad "
                    r"\text{Max Loss} = \text{Net Premium}"
                ),
            },
            {
                "label": "Breakeven",
                "latex": r"S^* = K_2 - \text{Net Premium}",
            },
        ],
    }


def _bull_put_spread() -> dict:
    return {
        "name": "Bull Put Spread",
        "description": "Sell a put at K₂ and buy a put at K₁ < K₂ (net credit).",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = -\max(K_2 - S_T,\, 0) + \max(K_1 - S_T,\, 0)",
            },
            {
                "label": "Max Profit / Max Loss",
                "latex": (
                    r"\text{Max Profit} = \text{Net Premium}, \qquad "
                    r"\text{Max Loss} = K_2 - K_1 - \text{Net Premium}"
                ),
            },
            {
                "label": "Breakeven",
                "latex": r"S^* = K_2 - \text{Net Premium}",
            },
        ],
    }


def _bear_call_spread() -> dict:
    return {
        "name": "Bear Call Spread",
        "description": "Sell a call at K₁ and buy a call at K₂ > K₁ (net credit).",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = -\max(S_T - K_1,\, 0) + \max(S_T - K_2,\, 0)",
            },
            {
                "label": "Max Profit / Max Loss",
                "latex": (
                    r"\text{Max Profit} = \text{Net Premium}, \qquad "
                    r"\text{Max Loss} = K_2 - K_1 - \text{Net Premium}"
                ),
            },
            {
                "label": "Breakeven",
                "latex": r"S^* = K_1 + \text{Net Premium}",
            },
        ],
    }


# ── Volatility ─────────────────────────────────────────────────────────


def _long_straddle() -> dict:
    return {
        "name": "Long Straddle",
        "description": "Buy a call and a put at the same strike K (profit from large moves).",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = |S_T - K| = \max(S_T - K,\, 0) + \max(K - S_T,\, 0)",
            },
            {
                "label": "Cost",
                "latex": r"\text{Premium} = C(K) + P(K)",
            },
            {
                "label": "Breakeven Points",
                "latex": r"S^*_{\text{down}} = K - (C + P), \qquad S^*_{\text{up}} = K + (C + P)",
            },
        ],
    }


def _short_straddle() -> dict:
    return {
        "name": "Short Straddle",
        "description": "Sell a call and a put at the same strike K (profit from low volatility).",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = -|S_T - K| = -\max(S_T - K,\, 0) - \max(K - S_T,\, 0)",
            },
            {
                "label": "Credit Received",
                "latex": r"\text{Premium} = C(K) + P(K)",
            },
            {
                "label": "Breakeven Points",
                "latex": r"S^*_{\text{down}} = K - (C + P), \qquad S^*_{\text{up}} = K + (C + P)",
            },
        ],
    }


def _long_strangle() -> dict:
    return {
        "name": "Long Strangle",
        "description": "Buy an OTM put at K₁ and an OTM call at K₂ > K₁.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = \max(S_T - K_2,\, 0) + \max(K_1 - S_T,\, 0)",
            },
            {
                "label": "Cost",
                "latex": r"\text{Premium} = C(K_2) + P(K_1)",
            },
            {
                "label": "Breakeven Points",
                "latex": r"S^*_{\text{down}} = K_1 - (C + P), \qquad S^*_{\text{up}} = K_2 + (C + P)",
            },
        ],
    }


def _short_strangle() -> dict:
    return {
        "name": "Short Strangle",
        "description": "Sell an OTM put at K₁ and an OTM call at K₂ > K₁.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = -\max(S_T - K_2,\, 0) - \max(K_1 - S_T,\, 0)",
            },
            {
                "label": "Credit Received",
                "latex": r"\text{Premium} = C(K_2) + P(K_1)",
            },
            {
                "label": "Breakeven Points",
                "latex": r"S^*_{\text{down}} = K_1 - (C + P), \qquad S^*_{\text{up}} = K_2 + (C + P)",
            },
        ],
    }


# ── Advanced ───────────────────────────────────────────────────────────


def _iron_condor() -> dict:
    return {
        "name": "Iron Condor",
        "description": "Combine a bull put spread (K₁, K₂) and a bear call spread (K₃, K₄) for a range-bound bet.",
        "equations": [
            {
                "label": "Payoff at Maturity",
                "latex": (
                    r"\text{Payoff} = \begin{cases}"
                    r" -(K_2 - K_1) & S_T \leq K_1 \\"
                    r" S_T - K_2 & K_1 < S_T < K_2 \\"
                    r" 0 & K_2 \leq S_T \leq K_3 \\"
                    r" K_3 - S_T & K_3 < S_T < K_4 \\"
                    r" -(K_4 - K_3) & S_T \geq K_4"
                    r"\end{cases}"
                ),
            },
            {
                "label": "Max Profit / Max Loss",
                "latex": (
                    r"\text{Max Profit} = \text{Net Premium}, \qquad "
                    r"\text{Max Loss} = \max(K_2 - K_1,\; K_4 - K_3) - \text{Net Premium}"
                ),
            },
            {
                "label": "Breakeven Points",
                "latex": r"S^*_{\text{down}} = K_2 - \text{Net Premium}, \qquad S^*_{\text{up}} = K_3 + \text{Net Premium}",
            },
        ],
    }


def _butterfly() -> dict:
    return {
        "name": "Butterfly Spread",
        "description": "Buy 1 call at K₁, sell 2 calls at K₂, buy 1 call at K₃ with K₂ = (K₁+K₃)/2.",
        "equations": [
            {
                "label": "Payoff at Maturity",
                "latex": (
                    r"\text{Payoff} = \begin{cases}"
                    r" 0 & S_T \leq K_1 \\"
                    r" S_T - K_1 & K_1 < S_T \leq K_2 \\"
                    r" K_3 - S_T & K_2 < S_T < K_3 \\"
                    r" 0 & S_T \geq K_3"
                    r"\end{cases}"
                ),
            },
            {
                "label": "Max Profit / Max Loss",
                "latex": (
                    r"\text{Max Profit} = K_2 - K_1 - \text{Net Premium}, \qquad "
                    r"\text{Max Loss} = \text{Net Premium}"
                ),
            },
            {
                "label": "Breakeven Points",
                "latex": r"S^*_1 = K_1 + \text{Net Premium}, \qquad S^*_2 = K_3 - \text{Net Premium}",
            },
        ],
    }


# ── Stock + Options ────────────────────────────────────────────────────


def _covered_call() -> dict:
    return {
        "name": "Covered Call",
        "description": "Hold the underlying stock and sell a call option.",
        "equations": [
            {
                "label": "Combined Payoff",
                "latex": r"\text{Payoff} = (S_T - S_0) - \max(S_T - K,\, 0) + C",
            },
            {
                "label": "Simplified Payoff",
                "latex": (
                    r"\text{P\&L} = \begin{cases}"
                    r" S_T - S_0 + C & S_T \leq K \\"
                    r" K - S_0 + C & S_T > K"
                    r"\end{cases}"
                ),
            },
            {
                "label": "Max Profit / Breakeven",
                "latex": (
                    r"\text{Max Profit} = K - S_0 + C, \qquad "
                    r"S^* = S_0 - C"
                ),
            },
        ],
    }


def _protective_put() -> dict:
    return {
        "name": "Protective Put",
        "description": "Hold the underlying stock and buy a put option (portfolio insurance).",
        "equations": [
            {
                "label": "Combined Payoff",
                "latex": r"\text{Payoff} = (S_T - S_0) + \max(K - S_T,\, 0) - P",
            },
            {
                "label": "Simplified Payoff",
                "latex": (
                    r"\text{P\&L} = \begin{cases}"
                    r" K - S_0 - P & S_T \leq K \\"
                    r" S_T - S_0 - P & S_T > K"
                    r"\end{cases}"
                ),
            },
            {
                "label": "Max Loss / Breakeven",
                "latex": (
                    r"\text{Max Loss} = S_0 - K + P, \qquad "
                    r"S^* = S_0 + P"
                ),
            },
        ],
    }


def _collar() -> dict:
    return {
        "name": "Collar",
        "description": "Hold the stock, buy a protective put at K₁, sell a covered call at K₂ > K₁.",
        "equations": [
            {
                "label": "Combined Payoff",
                "latex": r"\text{Payoff} = (S_T - S_0) + \max(K_1 - S_T,\, 0) - \max(S_T - K_2,\, 0) - P + C",
            },
            {
                "label": "Simplified Payoff",
                "latex": (
                    r"\text{P\&L} = \begin{cases}"
                    r" K_1 - S_0 - P + C & S_T \leq K_1 \\"
                    r" S_T - S_0 - P + C & K_1 < S_T < K_2 \\"
                    r" K_2 - S_0 - P + C & S_T \geq K_2"
                    r"\end{cases}"
                ),
            },
            {
                "label": "Range",
                "latex": (
                    r"\text{Floor} = K_1 - S_0 - \text{Net Cost}, \qquad "
                    r"\text{Cap} = K_2 - S_0 - \text{Net Cost}"
                ),
            },
        ],
    }


# ── Barrier ────────────────────────────────────────────────────────────

_BARRIER_PARITY = {
    "label": "Knock-In / Knock-Out Parity",
    "latex": r"V_{\text{KI}} + V_{\text{KO}} = V_{\text{Vanilla}} \qquad \text{(Reiner \& Rubinstein, 1991)}",
}


def _barrier_up_out_call() -> dict:
    return {
        "name": "Up-and-Out Call",
        "description": "European call that is extinguished if the spot reaches H > K from below.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff",
                "latex": r"\text{Payoff} = \max(S_T - K,\, 0)\;\cdot\;\mathbf{1}_{\{\max_{0 \leq t \leq T} S_t < H\}}",
            },
            {
                "label": "Key Parameters",
                "latex": (
                    r"\mu = \frac{r - q - \sigma^2/2}{\sigma^2}, \qquad "
                    r"\lambda = \sqrt{\mu^2 + \frac{2r}{\sigma^2}}"
                ),
            },
            _BARRIER_PARITY,
        ],
    }


def _barrier_down_out_put() -> dict:
    return {
        "name": "Down-and-Out Put",
        "description": "European put that is extinguished if the spot reaches H < K from above.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff",
                "latex": r"\text{Payoff} = \max(K - S_T,\, 0)\;\cdot\;\mathbf{1}_{\{\min_{0 \leq t \leq T} S_t > H\}}",
            },
            {
                "label": "Key Parameters",
                "latex": (
                    r"\mu = \frac{r - q - \sigma^2/2}{\sigma^2}, \qquad "
                    r"\lambda = \sqrt{\mu^2 + \frac{2r}{\sigma^2}}"
                ),
            },
            _BARRIER_PARITY,
        ],
    }


def _barrier_knock_in_call() -> dict:
    return {
        "name": "Down-and-In Call",
        "description": "European call that activates only if the spot falls to H < S₀.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff",
                "latex": r"\text{Payoff} = \max(S_T - K,\, 0)\;\cdot\;\mathbf{1}_{\{\min_{0 \leq t \leq T} S_t \leq H\}}",
            },
            {
                "label": "Parity (used for pricing)",
                "latex": r"V_{\text{DI Call}} = V_{\text{Vanilla Call}} - V_{\text{DO Call}}",
            },
            _BARRIER_PARITY,
        ],
    }


# ── Digital ────────────────────────────────────────────────────────────


def _digital_call() -> dict:
    return {
        "name": "Digital Call (Cash-or-Nothing)",
        "description": "Pays a fixed amount P if S_T > K, zero otherwise.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Price",
                "latex": r"V = P \cdot e^{-rT} \cdot N(d_2)",
            },
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = P \cdot \mathbf{1}_{\{S_T > K\}}",
            },
        ],
    }


def _digital_put() -> dict:
    return {
        "name": "Digital Put (Cash-or-Nothing)",
        "description": "Pays a fixed amount P if S_T < K, zero otherwise.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Price",
                "latex": r"V = P \cdot e^{-rT} \cdot N(-d_2)",
            },
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = P \cdot \mathbf{1}_{\{S_T < K\}}",
            },
        ],
    }


def _digital_range_bet() -> dict:
    return {
        "name": "Digital Range Bet",
        "description": "Pays P if K₁ < S_T < K₂ — combination of two digital calls.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Decomposition",
                "latex": r"V = \text{Digital Call}(K_1) - \text{Digital Call}(K_2)",
            },
            {
                "label": "Price",
                "latex": r"V = P \cdot e^{-rT}\bigl[N(d_2(K_1)) - N(d_2(K_2))\bigr]",
            },
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = P \cdot \mathbf{1}_{\{K_1 < S_T < K_2\}}",
            },
        ],
    }


# ── Asian ──────────────────────────────────────────────────────────────


def _asian_call() -> dict:
    return {
        "name": "Asian Call (Geometric Average)",
        "description": "Payoff based on the geometric average price — Kemna & Vorst (1990) closed-form.",
        "equations": [
            {
                "label": "Payoff at Maturity",
                "latex": (
                    r"\text{Payoff} = \max\!\left(G_T - K,\; 0\right), \qquad "
                    r"G_T = \left(\prod_{i=1}^{n} S_{t_i}\right)^{1/n}"
                ),
            },
            {
                "label": "Adjusted Volatility",
                "latex": r"\sigma_{\text{adj}} = \frac{\sigma}{\sqrt{3}}",
            },
            {
                "label": "Adjusted Cost of Carry",
                "latex": r"b_{\text{adj}} = \frac{r - q}{2} - \frac{\sigma^2}{12}",
            },
            {
                "label": "BS-like Price with Adjusted Parameters",
                "latex": (
                    r"V = S\,e^{(b_{\text{adj}} - r)T}\,N(d_1^*) - K\,e^{-rT}\,N(d_2^*)"
                ),
            },
            {
                "label": "Adjusted d₁ / d₂",
                "latex": (
                    r"d_1^* = \frac{\ln(S/K) + (b_{\text{adj}} + \tfrac{\sigma_{\text{adj}}^2}{2})\,T}"
                    r"{\sigma_{\text{adj}}\sqrt{T}}, \qquad d_2^* = d_1^* - \sigma_{\text{adj}}\sqrt{T}"
                ),
            },
        ],
    }


# ── Lookback ───────────────────────────────────────────────────────────


def _lookback_floating_call() -> dict:
    return {
        "name": "Lookback Call (Floating Strike)",
        "description": "Strike set at the minimum price over the life — Goldman, Sosin & Gatto (1979).",
        "equations": [
            {
                "label": "Payoff",
                "latex": r"\text{Payoff} = S_T - \min_{0 \leq t \leq T} S_t",
            },
            {
                "label": "Analytical Price (b = r − q)",
                "latex": (
                    r"V = S\,e^{-qT}\,N(a_1) - S_{\min}\,e^{-rT}\,N(a_2)"
                    r" - S\,e^{-rT}\,\frac{\sigma^2}{2b}"
                    r"\left[e^{bT}\,N(-a_1) - \left(\frac{S_{\min}}{S}\right)^{2b/\sigma^2}\!N(-a_3)\right]"
                ),
            },
            {
                "label": "Key Terms",
                "latex": (
                    r"a_1 = \frac{\ln(S/S_{\min}) + (b + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \qquad "
                    r"a_2 = a_1 - \sigma\sqrt{T}"
                ),
            },
        ],
    }


def _lookback_fixed_call() -> dict:
    return {
        "name": "Lookback Call (Fixed Strike)",
        "description": "Payoff based on the maximum price over the life — Conze & Viswanathan (1991).",
        "equations": [
            {
                "label": "Payoff",
                "latex": r"\text{Payoff} = \max\!\left(\max_{0 \leq t \leq T} S_t - K,\; 0\right)",
            },
            {
                "label": "Analytical Price (b = r − q, M = max so far)",
                "latex": (
                    r"V = M\,e^{-rT}\,N(b_1) - K\,e^{-rT}\,N(b_2)"
                    r" + S\,e^{-rT}\,\frac{\sigma^2}{2b}"
                    r"\left[\left(\frac{S}{M}\right)^{-2b/\sigma^2}\!N(-b_3) - e^{bT}\,N(-b_1')\right]"
                ),
            },
            {
                "label": "Key Terms",
                "latex": (
                    r"b_1 = \frac{\ln(M/S) - (b - \sigma^2/2)\,T}{\sigma\sqrt{T}}, \qquad "
                    r"M = \max\!\left(S_{\max},\, K\right)"
                ),
            },
        ],
    }


# ── Chooser ────────────────────────────────────────────────────────────


def _chooser() -> dict:
    return {
        "name": "Chooser Option",
        "description": "Holder chooses at tₓ whether the option is a call or a put — Rubinstein (1991).",
        "equations": [
            {
                "label": "Value at Choosing Time tₓ",
                "latex": r"V(t_c) = \max\!\bigl(C(S,K,T-t_c),\; P(S,K,T-t_c)\bigr)",
            },
            {
                "label": "Closed-Form Price (at t = 0)",
                "latex": r"V_0 = C(S,\,K,\,T) + P\!\left(S,\; K\,e^{-(r-q)(T - t_c)},\; t_c\right)",
            },
            {
                "label": "Intuition",
                "latex": (
                    r"\text{Put-call parity at } t_c: \quad "
                    r"P = C - S\,e^{-q(T-t_c)} + K\,e^{-r(T-t_c)}"
                ),
            },
        ],
    }


# ── Asset-or-Nothing ──────────────────────────────────────────────────


def _asset_or_nothing_call() -> dict:
    return {
        "name": "Asset-or-Nothing Call",
        "description": "Pays the asset S_T if S_T > K, zero otherwise.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Price",
                "latex": r"V = S\,e^{-qT}\,N(d_1)",
            },
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = S_T \cdot \mathbf{1}_{\{S_T > K\}}",
            },
        ],
    }


def _asset_or_nothing_put() -> dict:
    return {
        "name": "Asset-or-Nothing Put",
        "description": "Pays the asset S_T if S_T < K, zero otherwise.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Price",
                "latex": r"V = S\,e^{-qT}\,N(-d_1)",
            },
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = S_T \cdot \mathbf{1}_{\{S_T < K\}}",
            },
        ],
    }


# ── Power ──────────────────────────────────────────────────────────────


def _power_call() -> dict:
    return {
        "name": "Power Call (n = 2)",
        "description": "Payoff is a power function of the spot — Esser (2003).",
        "equations": [
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = \max(S_T^{\,n} - K,\; 0)",
            },
            {
                "label": "Equivalent Spot (for BS mapping)",
                "latex": (
                    r"F_n = S^n \exp\!\left[\left(n(r - q) + \tfrac{n(n-1)\sigma^2}{2}\right)T\right], "
                    r"\qquad \sigma_n = n\,\sigma"
                ),
            },
            {
                "label": "Price via Adjusted BS",
                "latex": r"V = e^{-rT}\!\left[F_n\,N(d_1^*) - K\,N(d_2^*)\right]",
            },
        ],
    }


# ── Gap ────────────────────────────────────────────────────────────────


def _gap_call() -> dict:
    return {
        "name": "Gap Call",
        "description": "Two strikes: K₁ triggers the payoff, K₂ determines the amount paid.",
        "equations": [
            _BS_D1D2,
            {
                "label": "Payoff at Maturity",
                "latex": r"\text{Payoff} = (S_T - K_2)\;\cdot\;\mathbf{1}_{\{S_T > K_1\}}",
            },
            {
                "label": "Price",
                "latex": r"V = S\,e^{-qT}\,N(d_1(K_1)) - K_2\,e^{-rT}\,N(d_2(K_1))",
            },
            {
                "label": "d₁ / d₂ (computed with trigger strike K₁)",
                "latex": (
                    r"d_1(K_1) = \frac{\ln(S/K_1) + (r - q + \sigma^2/2)\,T}{\sigma\sqrt{T}}, "
                    r"\qquad d_2 = d_1 - \sigma\sqrt{T}"
                ),
            },
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

_STRATEGY_EQUATIONS = {
    # Directional
    "long_call": lambda: _vanilla("long", "call"),
    "short_call": lambda: _vanilla("short", "call"),
    "long_put": lambda: _vanilla("long", "put"),
    "short_put": lambda: _vanilla("short", "put"),
    # Vertical Spreads
    "bull_call_spread": _bull_call_spread,
    "bear_put_spread": _bear_put_spread,
    "bull_put_spread": _bull_put_spread,
    "bear_call_spread": _bear_call_spread,
    # Volatility
    "long_straddle": _long_straddle,
    "short_straddle": _short_straddle,
    "long_strangle": _long_strangle,
    "short_strangle": _short_strangle,
    # Advanced
    "iron_condor": _iron_condor,
    "butterfly": _butterfly,
    # Stock + Options
    "covered_call": _covered_call,
    "protective_put": _protective_put,
    "collar": _collar,
    # Barrier
    "barrier_up_out_call": _barrier_up_out_call,
    "barrier_down_out_put": _barrier_down_out_put,
    "barrier_knock_in_call": _barrier_knock_in_call,
    # Digital
    "digital_call": _digital_call,
    "digital_put": _digital_put,
    "digital_range_bet": _digital_range_bet,
    # Path-Dependent
    "asian_call": _asian_call,
    "lookback_floating_call": _lookback_floating_call,
    "lookback_fixed_call": _lookback_fixed_call,
    # Other Exotic
    "chooser": _chooser,
    "asset_or_nothing_call": _asset_or_nothing_call,
    "asset_or_nothing_put": _asset_or_nothing_put,
    "power_call": _power_call,
    "gap_call": _gap_call,
}


def get_option_strategy_equations(strategy_key: str) -> dict | None:
    """Return equations dict for a given strategy key, or None if not found."""
    factory = _STRATEGY_EQUATIONS.get(strategy_key)
    if factory is None:
        return None
    return factory()
