"""
Guide module for Options Greeks Explorer
Comprehensive formulas and trading strategies for all Greeks in English
Enriched with practical applications and advanced insights
"""

import streamlit as st


def render_guide_section():
    """Display the comprehensive guide and formulas section with dropdown selection"""

    st.subheader("🎓 Complete Guide to Black-Scholes and Options Greeks")

    # Dropdown selector for Greek selection
    selected_topic = st.selectbox(
        "📚 Select a topic to explore in depth",
        ["Black-Scholes Model", "Delta", "Gamma", "Vega", "Theta", "Rho",
         "Vanna", "Volga (Vomma)", "Charm", "Veta", "Speed", "Zomma", "Color", "Ultima",
         "Greeks Trading Strategies", "P&L Attribution", "Risk Management"],
        index=0
    )

    st.markdown("---")

    # Display content based on selection
    if selected_topic == "Black-Scholes Model":
        render_black_scholes()
    elif selected_topic == "Delta":
        render_delta()
    elif selected_topic == "Gamma":
        render_gamma()
    elif selected_topic == "Vega":
        render_vega()
    elif selected_topic == "Theta":
        render_theta()
    elif selected_topic == "Rho":
        render_rho()
    elif selected_topic == "Vanna":
        render_vanna()
    elif selected_topic == "Volga (Vomma)":
        render_volga()
    elif selected_topic == "Charm":
        render_charm()
    elif selected_topic == "Veta":
        render_veta()
    elif selected_topic == "Speed":
        render_speed()
    elif selected_topic == "Zomma":
        render_zomma()
    elif selected_topic == "Color":
        render_color()
    elif selected_topic == "Ultima":
        render_ultima()
    elif selected_topic == "Greeks Trading Strategies":
        render_trading_strategies()
    elif selected_topic == "P&L Attribution":
        render_pnl_attribution()
    elif selected_topic == "Risk Management":
        render_risk_management()


def render_black_scholes():
    """Display Black-Scholes model content"""
    st.markdown("### 📚 The Black-Scholes Model")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### European Call Option Price:")
        st.latex(r"C = S_0 \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)")

        st.markdown("#### European Put Option Price:")
        st.latex(r"P = K \cdot e^{-rT} \cdot N(-d_2) - S_0 \cdot N(-d_1)")

    with col2:
        st.markdown("#### Where:")
        st.latex(r"d_1 = \frac{\ln(\frac{S_0}{K}) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")

    st.markdown("""
    #### Parameters:
    - **S₀**: Current spot price of the underlying asset
    - **K**: Strike price (exercise price)
    - **r**: Risk-free interest rate (annualized)
    - **T**: Time to maturity in years
    - **σ**: Implied volatility (annualized)
    - **N(x)**: Cumulative standard normal distribution function
    """)

    st.info("""
    💡 **Key Assumptions:**
    1. No transaction costs or taxes
    2. Risk-free rate is constant and known
    3. Underlying follows geometric Brownian motion
    4. No dividends during option life
    5. Markets are perfectly efficient
    6. European-style exercise only
    """)

    st.success("""
    📊 **Practical Applications:**
    - **Market Making**: Used as baseline for quoting options, adjusted for volatility smile
    - **Portfolio Hedging**: Calculate hedge ratios for delta-neutral portfolios
    - **Volatility Arbitrage**: Compare implied vs realized volatility for trading opportunities
    - **Risk Management**: Estimate portfolio exposure to market movements
    """)

    st.warning("""
    ⚠️ **Model Limitations:**
    - **Volatility Smile**: Real markets show varying IV across strikes
    - **Jump Risk**: Doesn't capture discontinuous price movements
    - **Early Exercise**: Not applicable to American options
    - **Transaction Costs**: Ignores real-world frictions
    - **Stochastic Volatility**: Assumes constant volatility
    """)


def render_delta():
    """Display Delta content with advanced insights"""
    st.markdown("### Delta (Δ) - The Directional Greek")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Formula:")
        st.latex(r"\Delta = \frac{\partial V}{\partial S}")
        st.markdown("**Call Delta:**")
        st.latex(r"\Delta_{call} = N(d_1)")
        st.markdown("**Put Delta:**")
        st.latex(r"\Delta_{put} = N(d_1) - 1")

    with col2:
        st.markdown("#### Properties:")
        st.markdown("""
        - **Call Range**: [0, 1]
        - **Put Range**: [-1, 0]
        - **ATM Call**: ≈ 0.5
        - **ATM Put**: ≈ -0.5
        - **Deep ITM**: → ±1
        - **Deep OTM**: → 0
        """)

    st.markdown("#### Advanced Delta Concepts:")

    tab1, tab2, tab3 = st.tabs(["Trading Applications", "Risk Management", "Market Dynamics"])

    with tab1:
        st.markdown("""
        **Delta-Neutral Strategies:**
        - **Straddle/Strangle**: Combine calls and puts for net delta ≈ 0
        - **Ratio Spreads**: Use different quantities to achieve neutrality
        - **Dynamic Hedging**: Continuously rebalance to maintain delta neutrality

        **Directional Trading:**
        - **25-Delta Options**: Popular for risk reversals in FX markets
        - **Delta Ladders**: Scale into positions at different delta levels
        - **Delta Targeting**: Maintain specific portfolio delta exposure
        """)

    with tab2:
        st.markdown("""
        **Hedging Applications:**
        - **Share Equivalency**: Delta × 100 = equivalent share position
        - **Portfolio Protection**: Use put deltas to offset long stock exposure
        - **Cross-Hedging**: Use correlated assets when direct hedges unavailable

        **Risk Metrics:**
        - **Delta-Adjusted Notional**: Position size × Delta
        - **Portfolio Delta**: Sum of all position deltas
        - **Delta Decay**: Monitor charm for overnight changes
        """)

    with tab3:
        st.markdown("""
        **Market Microstructure:**
        - **Pin Risk**: Delta jumps near expiration at strike prices
        - **Sticky Strike vs Sticky Delta**: Different assumptions for vol dynamics
        - **Dealer Positioning**: Aggregate delta drives market flows

        **Volatility Smile Impact:**
        - **Risk Reversal**: 25-delta call IV minus 25-delta put IV
        - **Butterfly**: Measures smile curvature at different deltas
        - **Delta Surface**: 3D visualization of delta across strikes and expiries
        """)

    st.info("""
    💡 **Pro Tip**: Delta can be interpreted as:
    1. Hedge ratio (shares needed to hedge)
    2. Probability of finishing ITM (risk-neutral)
    3. Rate of change of option value
    4. Equivalent stock position
    """)


def render_gamma():
    """Display Gamma content with practical insights"""
    st.markdown("### Gamma (Γ) - The Acceleration Greek")

    st.markdown("#### Formula:")
    st.latex(r"\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\partial \Delta}{\partial S}")
    st.latex(r"\Gamma = \frac{n(d_1)}{S_0 \sigma \sqrt{T}}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Key Properties:
        - **Always Positive** for long options
        - **Identical** for calls and puts (same strike/expiry)
        - **Maximum** at ATM
        - **Increases** as expiration approaches
        - **Decreases** with higher volatility
        """)

    with col2:
        st.markdown("""
        #### Gamma Trading Metrics:
        - **Gamma Rent**: Daily P&L from gamma hedging
        - **Gamma Scalping**: P&L = ½ × Gamma × (ΔS)²
        - **Realized Vol**: Extractable through dynamic hedging
        - **Gamma Trap**: Concentrated near strikes at expiry
        """)

    st.markdown("#### Advanced Gamma Strategies:")

    tab1, tab2, tab3 = st.tabs(["Long Gamma", "Short Gamma", "Gamma Risk"])

    with tab1:
        st.success("""
        **Long Gamma Strategies (Volatility Long):**

        **Straddle/Strangle:**
        - Maximum gamma exposure at ATM
        - Profit from large moves in either direction
        - Cost: High theta decay

        **Backspread:**
        - Long more OTM options than short ATM
        - Positive gamma with limited risk
        - Benefits from volatility expansion

        **Gamma Scalping:**
        - Delta-hedge frequently to capture realized vol
        - P&L proportional to (RV² - IV²)
        - Requires low transaction costs
        """)

    with tab2:
        st.warning("""
        **Short Gamma Strategies (Premium Collection):**

        **Iron Condor/Butterfly:**
        - Defined risk with negative gamma
        - Profit from range-bound markets
        - Maximum profit at expiration

        **Covered Call/Cash-Secured Put:**
        - Generate income with limited gamma risk
        - Popular for dividend capture
        - Limited upside potential

        **Calendar Spreads:**
        - Short near-term gamma
        - Long longer-term protection
        - Benefits from time decay differential
        """)

    with tab3:
        st.error("""
        **Gamma Risk Management:**

        **Gamma Squeeze:**
        - Dealer hedging amplifies price moves
        - Common at major strikes near expiry
        - Can cause violent price action

        **Weekend Gamma:**
        - 3 days of gamma accumulation
        - Gap risk on Monday open
        - Consider Friday afternoon adjustments

        **Gamma Concentration:**
        - Monitor open interest by strike
        - Identify potential "magnets"
        - Adjust positions before events
        """)

    st.info("""
    💡 **Gamma vs Theta Relationship**:
    Being long gamma means being short theta - you pay time decay for the right to profit from volatility.
    The key is whether realized volatility will exceed implied volatility by enough to cover theta costs.
    """)


def render_vega():
    """Display Vega content with volatility trading insights"""
    st.markdown("### Vega (ν) - The Volatility Greek")

    st.markdown("#### Formula:")
    st.latex(r"\nu = \frac{\partial V}{\partial \sigma}")
    st.latex(r"\nu = S_0 \cdot n(d_1) \cdot \sqrt{T} / 100")

    st.markdown("#### Volatility Trading Framework:")

    tab1, tab2, tab3, tab4 = st.tabs(["Vega Basics", "Vol Trading", "Term Structure", "Advanced Strategies"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Properties:**
            - Always positive for long options
            - Maximum at ATM
            - Increases with time to expiry
            - Expressed per 1% vol change
            - Identical for calls and puts
            """)

        with col2:
            st.markdown("""
            **Vega by Expiry:**
            - Weekly options: Low vega, high gamma
            - Monthly options: Balanced vega/gamma
            - LEAPS: High vega, low gamma
            - Optimal: 30-45 DTE for vol trading
            """)

    with tab2:
        st.markdown("""
        **Volatility Arbitrage Strategies:**

        **Implied vs Realized:**
        - Compare IV to historical/expected RV
        - Long vega if RV > IV expected
        - Short vega if IV too rich

        **Volatility Pairs Trading:**
        - Relative value between related underlyings
        - Cross-asset volatility spreads
        - Index vs component volatility (dispersion)

        **Event Volatility:**
        - Pre-earnings IV expansion
        - Post-announcement IV crush
        - Binary event trading strategies
        """)

    with tab3:
        st.markdown("""
        **Term Structure Trading:**

        **Calendar Spreads:**
        - Exploit term structure differences
        - Long back month, short front month
        - Positive vega, positive theta possible

        **Diagonal Spreads:**
        - Different strikes and expiries
        - Fine-tune vega exposure
        - Adapt to volatility surface

        **Volatility Carry:**
        - Systematic short vol in contango
        - Risk management crucial
        - Popular with volatility funds
        """)

    with tab4:
        st.success("""
        **Advanced Vega Strategies:**

        **Vanna-Volga Trading:**
        - Hedge both vanna and volga risks
        - Used in exotic option pricing
        - Important for FX markets

        **Volatility Surface Arbitrage:**
        - Trade discrepancies in vol surface
        - Butterfly arbitrage
        - Risk reversal strategies

        **Cross-Asset Volatility:**
        - Correlation trading
        - Volatility beta strategies
        - Macro volatility themes
        """)

    st.info("""
    💡 **The Greek Trinity (Vega, Vanna, Volga)**:
    These three Greeks form the core of volatility trading. Together they capture:
    - First-order volatility sensitivity (Vega)
    - Cross-sensitivity to spot and vol (Vanna)
    - Volatility convexity (Volga)
    """)


def render_theta():
    """Display Theta content with time decay strategies"""
    st.markdown("### Theta (Θ) - The Time Decay Greek")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Call Theta:")
        st.latex(r"\Theta_{call} = -\frac{S_0 n(d_1) \sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)")

    with col2:
        st.markdown("#### Put Theta:")
        st.latex(r"\Theta_{put} = -\frac{S_0 n(d_1) \sigma}{2\sqrt{T}} + rKe^{-rT}N(-d_2)")

    st.markdown("#### Time Decay Characteristics:")

    tab1, tab2, tab3 = st.tabs(["Decay Patterns", "Income Strategies", "Risk Management"])

    with tab1:
        st.markdown("""
        **Non-Linear Decay:**
        - **45-30 DTE**: ~20% of premium decays
        - **30-15 DTE**: ~30% of premium decays
        - **15-7 DTE**: ~25% of premium decays
        - **7-0 DTE**: ~25% of premium decays (accelerating)

        **Weekend Effect:**
        - Friday positions carry 3 days of theta
        - Market makers adjust IV for weekends
        - Monday morning IV typically drops

        **Theta by Moneyness:**
        - **ATM**: Maximum theta (highest time value)
        - **Deep ITM**: Low theta (mostly intrinsic)
        - **Deep OTM**: Low theta (little value left)
        """)

    with tab2:
        st.success("""
        **Premium Selling Strategies:**

        **The Wheel Strategy:**
        1. Sell cash-secured puts
        2. If assigned, hold stock
        3. Sell covered calls
        4. If called away, restart
        - Target: 15-30% annual returns

        **Iron Condor Management:**
        - Enter at 45 DTE
        - Close at 50% profit or 21 DTE
        - Adjust untested side if breached
        - Monthly income generation

        **Poor Man's Covered Call:**
        - Long LEAPS call (low theta)
        - Short monthly calls (high theta)
        - Synthetic covered call with less capital
        """)

    with tab3:
        st.warning("""
        **Theta Risk Factors:**

        **Gamma Risk:**
        - High theta comes with high gamma
        - Large moves can overwhelm theta gains
        - Position sizing crucial

        **Assignment Risk:**
        - Early assignment on short options
        - Dividend risk for calls
        - Interest rate risk for puts

        **Volatility Expansion:**
        - Theta strategies vulnerable to IV spikes
        - Hedge with long tail options
        - Monitor volatility regime changes
        """)

    st.info("""
    💡 **Optimal Theta Harvesting**:
    Research shows entering trades at 45 DTE and closing at 21 DTE captures the best risk-adjusted returns.
    This avoids the gamma risk of the final weeks while capturing steady time decay.
    """)


def render_rho():
    """Display Rho content"""
    st.markdown("### Rho (ρ) - The Interest Rate Greek")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Call Rho:")
        st.latex(r"\rho_{call} = KTe^{-rT}N(d_2) / 100")

    with col2:
        st.markdown("#### Put Rho:")
        st.latex(r"\rho_{put} = -KTe^{-rT}N(-d_2) / 100")

    st.markdown("""
    #### Rho Characteristics:
    - **Positive for calls**, negative for puts
    - **Increases with time to expiry**
    - **More significant for LEAPS**
    - **Expressed per 1% rate change**
    - **Often overlooked but important in rate cycles**

    #### When Rho Matters:
    - **LEAPS positions** (>1 year expiry)
    - **Deep ITM options** (high intrinsic value)
    - **Rate cycle transitions** (Fed policy changes)
    - **Currency options** (rate differentials)
    - **Dividend arbitrage** strategies
    """)


def render_vanna():
    """Display Vanna content with advanced applications"""
    st.markdown("### Vanna - The Spot-Volatility Greek")

    st.markdown("#### Formula:")
    st.latex(
        r"Vanna = \frac{\partial^2 V}{\partial S \partial \sigma} = \frac{\partial \Delta}{\partial \sigma} = \frac{\partial \nu}{\partial S}")
    st.latex(r"Vanna = -\frac{n(d_1) \cdot d_2}{\sigma} / 100")

    st.markdown("#### Understanding Vanna:")

    tab1, tab2, tab3 = st.tabs(["Concept", "Trading Applications", "Market Impact"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Properties:**
            - Changes sign around ATM
            - Positive for OTM options
            - Negative for ITM options
            - Maximum around 25-delta
            - Critical for skew trading
            """)

        with col2:
            st.markdown("""
            **Interpretation:**
            - How delta changes with volatility
            - How vega changes with spot
            - Measures vol-spot correlation risk
            - Key for volatility smile dynamics
            """)

    with tab2:
        st.success("""
        **Vanna Trading Strategies:**

        **Risk Reversal:**
        - Long OTM call, short OTM put
        - Maximum vanna exposure
        - Trades directional view with vol
        - Popular in FX markets

        **25-Delta Strangle:**
        - Both sides have high vanna
        - Benefits from vol-spot decorrelation
        - Used for event trading

        **Vanna Spread:**
        - Long positive vanna (OTM)
        - Short negative vanna (ITM)
        - Delta-neutral but vanna exposed
        """)

    with tab3:
        st.info("""
        **Market Dynamics:**

        **Dealer Vanna Impact:**
        When dealers are net short puts and long calls (typical positioning):
        - Spot ↑ → IV ↓ → Dealers buy (stabilizing)
        - Spot ↓ → IV ↑ → Dealers sell (destabilizing)

        **Volatility Smile Evolution:**
        Vanna explains how the smile moves with spot:
        - Sticky strike: Smile fixed by strike
        - Sticky delta: Smile moves with spot
        - Reality: Combination of both
        """)

    st.warning("""
    ⚠️ **Vanna Risk**: 
    Vanna can create feedback loops in volatile markets. During crashes, negative spot-vol correlation 
    combined with dealer vanna exposure can accelerate moves. Understanding vanna is crucial for tail risk management.
    """)


def render_volga():
    """Display Volga content with vol-of-vol insights"""
    st.markdown("### Volga (Vomma) - The Volatility Convexity Greek")

    st.markdown("#### Formula:")
    st.latex(r"Volga = \frac{\partial^2 V}{\partial \sigma^2} = \frac{\partial \nu}{\partial \sigma}")
    st.latex(r"Volga = \nu \cdot \frac{d_1 d_2}{\sigma} / 10000")

    st.markdown("#### Volga Concepts:")

    tab1, tab2, tab3 = st.tabs(["Understanding Volga", "Trading Applications", "Risk Management"])

    with tab1:
        st.markdown("""
        **What Volga Measures:**
        - **Vega convexity**: How vega accelerates with volatility
        - **Vol-of-vol exposure**: Sensitivity to volatility regime changes
        - **Tail risk**: Higher-order exposure to extreme moves

        **Properties:**
        - Always positive for vanilla options
        - Maximum for ~25-delta options
        - Minimum at ATM (where vega is max)
        - Increases with time to expiry
        """)

    with tab2:
        st.success("""
        **Volga Trading Strategies:**

        **Wing Strangle:**
        - Long far OTM options (10-delta)
        - Maximum volga, minimum vega
        - Black swan protection
        - High theta cost

        **Volatility Convexity Trade:**
        - Long volga during calm markets
        - Anticipate volatility regime shifts
        - Use VIX options for pure play

        **Butterfly vs Condor:**
        - Butterfly: Short volga at body
        - Condor: Long volga at wings
        - Choose based on vol outlook
        """)

    with tab3:
        st.warning("""
        **Volga Risk Considerations:**

        **Short Volga Danger:**
        - Selling strangles = short volga
        - Losses accelerate if IV explodes
        - 2018 "Volmageddon" example
        - Size positions conservatively

        **Model Risk:**
        - Black-Scholes underestimates tail risk
        - Real volga > theoretical volga
        - Jump risk not captured
        - Use advanced models for accuracy
        """)

    st.info("""
    💡 **Vanna-Volga Method**:
    Used extensively in FX markets for pricing first-generation exotic options. 
    The method adjusts Black-Scholes prices by the cost of hedging vanna and volga risks,
    providing more accurate prices for options with smile and skew effects.
    """)


def render_charm():
    """Display Charm content"""
    st.markdown("### Charm - Delta Decay")

    st.markdown("#### Formula:")
    st.latex(r"Charm = \frac{\partial^2 V}{\partial S \partial t} = \frac{\partial \Delta}{\partial t}")
    st.latex(r"Charm = -n(d_1) \frac{2rt - d_2 \sigma \sqrt{t}}{2t \sigma \sqrt{t}} / 365")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Properties:**
        - Measures delta drift over time
        - Critical near expiration
        - Maximum impact at ATM
        - Important for overnight risk
        - Also called "DdeltaDtime"
        """)

    with col2:
        st.markdown("""
        **Practical Impact:**
        - Delta changes without spot move
        - Weekend effect (3 days charm)
        - Pin risk amplification
        - Hedging frequency requirements
        """)

    st.info("""
    💡 **Charm Trading Tip**:
    If your position has delta 0.30 and charm 0.05, tomorrow it will have delta 0.25 even if spot doesn't move.
    Market makers adjust hedges Friday afternoon to account for weekend charm.
    """)


def render_veta():
    """Display Veta content"""
    st.markdown("### Veta - Vega Decay")

    st.markdown("#### Formula:")
    st.latex(r"Veta = \frac{\partial^2 V}{\partial \sigma \partial t} = \frac{\partial \nu}{\partial t}")

    st.markdown("""
    **Understanding Veta:**
    - Measures how vega changes with time
    - Important for calendar spreads
    - Usually negative (vega decays)
    - Expressed per day per 1% volatility

    **Applications:**
    - Calendar spread optimization
    - Vega decay management
    - Term structure trading
    - Volatility carry strategies
    """)


def render_speed():
    """Display Speed content"""
    st.markdown("### Speed - Gamma Acceleration")

    st.markdown("#### Formula:")
    st.latex(r"Speed = \frac{\partial^3 V}{\partial S^3} = \frac{\partial \Gamma}{\partial S}")
    st.latex(r"Speed = -\frac{\Gamma}{S_0} \left( \frac{d_1}{\sigma \sqrt{T}} + 1 \right)")

    st.markdown("""
    **Speed Characteristics:**
    - Third-order Greek
    - Changes sign around ATM
    - Measures gamma instability
    - Critical for large moves
    - Maximum near 25-30 delta

    **Trading Implications:**
    - Gamma becomes unstable with large moves
    - Important for stress testing
    - Affects rehedging costs
    - Critical for market making algorithms
    """)


def render_zomma():
    """Display Zomma content with practical applications"""
    st.markdown("### Zomma (DgammaDvol) - Gamma Volatility Sensitivity")

    st.markdown("#### Formula:")
    st.latex(r"Zomma = \frac{\partial^3 V}{\partial S^2 \partial \sigma} = \frac{\partial \Gamma}{\partial \sigma}")
    st.latex(r"Zomma = \frac{\Gamma \cdot (d_1 d_2 - 1)}{\sigma} / 100")

    tab1, tab2 = st.tabs(["Concept", "Applications"])

    with tab1:
        st.markdown("""
        **What Zomma Tells Us:**
        - How gamma changes with volatility
        - Important for dynamic hedging
        - Changes sign based on moneyness
        - Positive for ~25-delta options
        - Critical during volatility regime changes
        """)

    with tab2:
        st.markdown("""
        **Practical Applications:**

        **Gamma Stability:**
        - Positive zomma: Gamma increases with vol
        - Negative zomma: Gamma decreases with vol
        - Plan hedging adjustments accordingly

        **Volatility Events:**
        - Earnings announcements
        - Economic data releases
        - Central bank decisions
        - Zomma affects hedging effectiveness

        **Dispersion Trading:**
        - Different zomma for index vs components
        - Arbitrage opportunities
        - Correlation breakdown scenarios
        """)

    st.info("""
    💡 **Zomma in Practice**:
    During volatile markets, positions with high zomma require frequent rehedging as gamma becomes unstable.
    Market makers price this instability into spreads, making options with high zomma more expensive to trade.
    """)


def render_color():
    """Display Color content"""
    st.markdown("### Color - Gamma Decay")

    st.markdown("#### Formula:")
    st.latex(r"Color = \frac{\partial^3 V}{\partial S^2 \partial t} = \frac{\partial \Gamma}{\partial t}")

    st.markdown("""
    **Understanding Color:**
    - Measures gamma time decay
    - Third-order Greek
    - Critical near expiration
    - Usually negative
    - Creates "gamma holes" at strikes

    **Expiration Week Dynamics:**
    - Color effect intensifies
    - Gamma concentrates at strikes
    - Can cause price "magnets"
    - Important for 0DTE options

    **Risk Management:**
    - Monitor color exposure
    - Roll positions before color spikes
    - Avoid gamma traps
    - Essential for weekly options
    """)


def render_ultima():
    """Display Ultima content"""
    st.markdown("### Ultima - Third-Order Volatility Sensitivity")

    st.markdown("#### Formula:")
    st.latex(r"Ultima = \frac{\partial^3 V}{\partial \sigma^3}")
    st.latex(r"Ultima = -\frac{\nu}{\sigma^3} \left[ d_1 d_2(1-d_1 d_2) + d_1^2 + d_2^2 \right] / 1000000")

    st.markdown("""
    **Ultima Characteristics:**
    - Volga convexity
    - Third-order vol sensitivity
    - Important for exotic options
    - Used in advanced models

    **Applications:**
    - Extreme volatility scenarios
    - Exotic option pricing
    - Vol-of-vol-of-vol exposure
    - Tail risk assessment
    """)


def render_trading_strategies():
    """Display comprehensive trading strategies"""
    st.markdown("### 🎯 Advanced Greeks Trading Strategies")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Volatility Strategies", "Directional Strategies", "Income Strategies", "Exotic Strategies"])

    with tab1:
        st.markdown("""
        #### Volatility Trading Framework

        **Pure Volatility Plays:**

        **1. Vega-Vanna-Volga Trading:**
        - Hedge all three volatility Greeks
        - Used by market makers
        - Capture volatility risk premium
        - Requires sophisticated modeling

        **2. Volatility Arbitrage:**
        - **Statistical Arb**: Trade IV vs realized vol
        - **Calendar Arb**: Exploit term structure
        - **Cross-Asset**: Related volatilities
        - **Dispersion**: Index vs components

        **3. Event Volatility:**
        - Pre-earnings IV ramp
        - Post-announcement crush
        - Binary event strategies
        - Straddle/strangle management
        """)

    with tab2:
        st.markdown("""
        #### Directional Strategies with Greeks

        **Delta-Based Strategies:**

        **1. Delta Neutral Adjustments:**
        - Start with directional view
        - Add options to neutralize delta
        - Profit from theta or vega
        - Reduce directional risk

        **2. Dynamic Delta Trading:**
        - Scale in/out based on delta levels
        - 25-delta for risk reversals
        - 10-delta for tail hedges
        - Delta targeting strategies

        **3. Charm Trading:**
        - Anticipate delta changes
        - Position for weekend charm
        - Exploit Friday effects
        - Pin risk management
        """)

    with tab3:
        st.markdown("""
        #### Income Generation Strategies

        **Theta Harvesting:**

        **1. The Wheel Strategy:**
        - Systematic CSP → CC cycle
        - 30-45 DTE optimal entry
        - 50% profit target
        - 15-30% annual returns typical

        **2. Iron Condor Optimization:**
        - 16-delta short strikes
        - 45 DTE entry, 21 DTE exit
        - Adjust untested side at 30 delta
        - Monthly income stream

        **3. Ratio Spreads:**
        - 1:2 or 1:3 ratios common
        - Collect credit upfront
        - Manage at 25% loss
        - Works in range-bound markets
        """)

    with tab4:
        st.markdown("""
        #### Exotic and Advanced Strategies

        **Higher-Order Greeks Trading:**

        **1. Speed Trading:**
        - Position for gamma instability
        - Exploit near-expiry dynamics
        - Market maker edge required
        - High-frequency adjustments

        **2. Zomma Arbitrage:**
        - Trade gamma-vol correlation
        - Dispersion opportunities
        - Requires real-time modeling
        - Institutional strategy

        **3. Color Management:**
        - Gamma decay trading
        - Expiration week focus
        - 0DTE specialist strategy
        - Requires automation
        """)


def render_pnl_attribution():
    """Display P&L attribution concepts"""
    st.markdown("### 📊 P&L Attribution and Greeks")

    st.markdown("""
    #### Taylor Series Expansion for Options P&L

    The change in option value can be decomposed using Greeks:
    """)

    st.latex(r"""
    dV = \Delta \cdot dS + \frac{1}{2}\Gamma \cdot dS^2 + \nu \cdot d\sigma + \Theta \cdot dt + \text{higher order terms}
    """)

    tab1, tab2, tab3 = st.tabs(["First Order", "Second Order", "Complete Attribution"])

    with tab1:
        st.markdown("""
        **First-Order P&L Components:**

        1. **Delta P&L**: Direction profit/loss
           - P&L = Δ × ΔS × contract size

        2. **Vega P&L**: Volatility profit/loss
           - P&L = ν × Δσ × contract size

        3. **Theta P&L**: Time decay
           - P&L = Θ × Δt × contract size

        4. **Rho P&L**: Interest rate impact
           - P&L = ρ × Δr × contract size
        """)

    with tab2:
        st.markdown("""
        **Second-Order P&L Components:**

        1. **Gamma P&L**: Convexity profit
           - P&L = ½ × Γ × (ΔS)² × contract size

        2. **Vanna P&L**: Cross effect
           - P&L = Vanna × ΔS × Δσ × contract size

        3. **Volga P&L**: Vol convexity
           - P&L = ½ × Volga × (Δσ)² × contract size

        4. **Charm P&L**: Delta decay
           - P&L = Charm × ΔS × Δt × contract size
        """)

    with tab3:
        st.markdown("""
        **Complete P&L Attribution:**

        ```
        Total P&L = Market Value Change - Initial Premium

        Explained P&L = Delta P&L + Gamma P&L + Vega P&L + 
                       Theta P&L + Vanna P&L + Volga P&L + 
                       Charm P&L + Other Greeks

        Unexplained P&L = Total P&L - Explained P&L
        ```

        **Quality Metrics:**
        - R² should be > 95% for good attribution
        - Residual < 5% indicates good model
        - Large residuals suggest model risk
        """)

    st.info("""
    💡 **P&L Attribution Best Practices**:
    1. Calculate Greeks at mid-point of move for accuracy
    2. Include cross-Greeks for large moves
    3. Monitor unexplained P&L for model validation
    4. Use for risk limit monitoring and performance analysis
    """)


def render_risk_management():
    """Display risk management with Greeks"""
    st.markdown("### 🛡️ Risk Management with Greeks")

    tab1, tab2, tab3, tab4 = st.tabs(["Risk Limits", "Stress Testing", "Dynamic Hedging", "Portfolio Greeks"])

    with tab1:
        st.markdown("""
        #### Greeks-Based Risk Limits

        **Typical Institutional Limits:**

        **Delta Limits:**
        - Net delta < 10% of portfolio NAV
        - Gross delta < 50% of NAV
        - Single name delta < 5% of NAV

        **Gamma Limits:**
        - 1% spot move impact < 2% of NAV
        - Gamma/theta ratio monitored
        - Concentration limits by strike

        **Vega Limits:**
        - 1 vol point impact < 1% of NAV
        - Vega by expiry bucket
        - Vega/theta ratio tracked

        **Higher-Order Limits:**
        - Vanna exposure monitored
        - Volga for tail risk
        - Speed for stability
        """)

    with tab2:
        st.markdown("""
        #### Stress Testing Framework

        **Standard Scenarios:**

        **Market Shocks:**
        - Spot: ±10%, ±20%, ±30%
        - Volatility: +50%, +100%, -30%
        - Combined: Crash scenarios

        **Greeks Evolution:**
        - Delta flip risk
        - Gamma explosion
        - Vega regime change
        - Charm over weekend

        **Historical Scenarios:**
        - 1987 Black Monday
        - 2008 Financial Crisis
        - 2020 COVID Crash
        - 2018 Volmageddon
        """)

    with tab3:
        st.markdown("""
        #### Dynamic Hedging Strategies

        **Delta Hedging:**
        - Frequency: Based on gamma
        - Threshold: 1% portfolio delta
        - Costs: Include in P&L
        - Effectiveness: Track hedge error

        **Gamma Hedging:**
        - Use options not stock
        - ATM options most efficient
        - Rebalance weekly minimum
        - Monitor gamma concentration

        **Vega Hedging:**
        - Match vega by expiry
        - Use VIX for systematic
        - Calendar spreads for term
        - Monitor vanna risk
        """)

    with tab4:
        st.markdown("""
        #### Portfolio-Level Greeks

        **Aggregation Methods:**

        **Simple Sum:**
        - Add all position Greeks
        - Quick and dirty
        - Ignores correlations
        - Overestimates risk

        **Weighted by Correlation:**
        - Account for correlations
        - More accurate risk
        - Computationally intensive
        - Industry standard

        **Scenario-Based:**
        - Greeks under scenarios
        - Capture non-linearities
        - Most accurate
        - Requires full revaluation

        **Reporting:**
        - Daily Greeks report
        - Limit utilization
        - Attribution analysis
        - Risk dashboard
        """)

    st.success("""
    ✅ **Risk Management Golden Rules**:
    1. Never exceed gamma limits near expiry
    2. Monitor vanna during volatile markets
    3. Check charm before weekends
    4. Stress test with all Greeks
    5. Document limit breaches
    6. Automate Greeks calculation
    7. Independent risk verification
    """)