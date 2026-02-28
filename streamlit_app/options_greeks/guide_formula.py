"""
Option Pricing and Greeks: Complete Mathematical Guide
======================================================

A comprehensive, pedagogically-structured guide covering:
- Black-Scholes-Merton option pricing theory
- First, second, and third-order Greeks
- Practical trading applications and risk management

Version: 2.0 - Enhanced pedagogical edition
"""

import streamlit as st


def render_guide_section():
    """Main entry point for the mathematical guide."""

    # Title with visual styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h1 style="color: #ffffff; margin: 0; font-size: 2.5rem;">
            The Complete Guide to Option Pricing
        </h1>
        <p style="color: #a8c8e8; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Black-Scholes Theory & Greeks Encyclopedia
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation with visual tabs
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "Foundation",
        "First-Order Greeks",
        "Second-Order Greeks",
        "Third-Order Greeks",
        "Trading Applications"
    ])

    with tabs[0]:
        render_foundation_section()
    with tabs[1]:
        render_first_order_greeks()
    with tabs[2]:
        render_second_order_greeks()
    with tabs[3]:
        render_third_order_greeks()
    with tabs[4]:
        render_trading_applications()


# ==============================================================================
# FOUNDATION SECTION
# ==============================================================================

def render_foundation_section():
    """Render the Black-Scholes foundation section."""

    st.markdown("## The Black-Scholes-Merton Framework")

    # Historical context box
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                border-left: 5px solid #1e3a5f;">
        <h4 style="color: #1e3a5f; margin-top: 0;">Historical Context</h4>
        <p style="color: #333; margin-bottom: 0;">
            Published in 1973 by Fischer Black, Myron Scholes, and Robert Merton, the Black-Scholes model
            revolutionized financial markets by providing the first closed-form solution for pricing
            European options. This breakthrough earned Scholes and Merton the 1997 Nobel Prize in Economics
            (Black had passed away in 1995).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Core assumptions
    st.markdown("### Model Assumptions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #ffffff; padding: 1.2rem; border-radius: 10px;
                    border: 1px solid #e0e0e0; height: 100%;">
            <h5 style="color: #1e3a5f;">Market Assumptions</h5>
            <ul style="color: #444; margin-bottom: 0;">
                <li><strong>No arbitrage:</strong> Risk-free profits impossible</li>
                <li><strong>Frictionless markets:</strong> No transaction costs or taxes</li>
                <li><strong>Continuous trading:</strong> Assets trade without interruption</li>
                <li><strong>Short selling:</strong> Allowed without restrictions</li>
                <li><strong>Divisibility:</strong> Assets infinitely divisible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #ffffff; padding: 1.2rem; border-radius: 10px;
                    border: 1px solid #e0e0e0; height: 100%;">
            <h5 style="color: #1e3a5f;">Asset Assumptions</h5>
            <ul style="color: #444; margin-bottom: 0;">
                <li><strong>Log-normal prices:</strong> S follows geometric Brownian motion</li>
                <li><strong>Constant volatility:</strong> sigma fixed over option life</li>
                <li><strong>Constant rates:</strong> Risk-free rate r fixed</li>
                <li><strong>No dividends:</strong> Or known continuous dividend yield</li>
                <li><strong>European exercise:</strong> Exercise only at expiry</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### The Pricing Formula")

    # Main formula display
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                padding: 2rem; border-radius: 15px; margin: 1.5rem 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
        <h4 style="color: #ffffff; text-align: center; margin-top: 0;">Black-Scholes Pricing Equations</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Call Option")
        st.latex(r"C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)")

    with col2:
        st.markdown("#### Put Option")
        st.latex(r"P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)")

    # d1 and d2 formulas
    st.markdown("#### Key Components")

    st.latex(r"d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}")
    st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")

    # Variable definitions
    render_variable_table()

    # Intuition section
    st.markdown("### Understanding the Formula")

    st.markdown("""
    <div style="background: #e8f4f8; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <h5 style="color: #1e3a5f; margin-top: 0;">Intuitive Interpretation</h5>
        <p>The Black-Scholes formula can be understood as an expected payoff calculation:</p>
        <ul>
            <li><strong>N(d2)</strong> = Probability the option expires in-the-money (risk-neutral)</li>
            <li><strong>N(d1)</strong> = Delta-weighted probability (accounts for expected stock movement)</li>
            <li><strong>Ke^(-rT)N(d2)</strong> = Present value of expected strike payment</li>
            <li><strong>S0e^(-qT)N(d1)</strong> = Present value of expected stock receipt</li>
        </ul>
        <p style="margin-bottom: 0;"><em>The option price is the difference between what you expect to receive
        and what you expect to pay, discounted to present value.</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Put-Call Parity
    render_put_call_parity()

    # Normal distribution properties
    render_normal_distribution()


def render_variable_table():
    """Render the variable definitions table."""

    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.2rem; border-radius: 10px; margin: 1rem 0;">
        <h5 style="color: #1e3a5f; margin-top: 0;">Variable Definitions</h5>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 8px; width: 15%;"><strong>S0</strong></td>
                <td style="padding: 8px;">Current underlying asset price (spot price)</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 8px;"><strong>K</strong></td>
                <td style="padding: 8px;">Strike price (exercise price)</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 8px;"><strong>T</strong></td>
                <td style="padding: 8px;">Time to expiration (in years)</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 8px;"><strong>r</strong></td>
                <td style="padding: 8px;">Risk-free interest rate (continuously compounded)</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 8px;"><strong>q</strong></td>
                <td style="padding: 8px;">Continuous dividend yield</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 8px;"><strong>sigma</strong></td>
                <td style="padding: 8px;">Volatility (annualized standard deviation of returns)</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><strong>N(x)</strong></td>
                <td style="padding: 8px;">Cumulative standard normal distribution function</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_put_call_parity():
    """Render the put-call parity section."""

    st.markdown("### Put-Call Parity")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #2d5a87 0%, #3d7ab7 100%);
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0; color: white;">
        <h5 style="margin-top: 0;">Fundamental Relationship</h5>
        <p>Put-call parity is a no-arbitrage relationship that must hold for European options:</p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"C - P = S_0 e^{-qT} - K e^{-rT}")

    st.markdown("""
    This relationship implies:
    - A call and put with the same strike and expiry are intimately connected
    - You can synthetically create any option from the others
    - Violations create arbitrage opportunities (rare and short-lived)
    """)

    with st.expander("Synthetic Position Construction"):
        st.markdown("""
        | **Target** | **Construction** |
        |-----------|------------------|
        | Synthetic Long Call | Buy Put + Buy Stock + Borrow PV(K) |
        | Synthetic Long Put | Buy Call + Short Stock + Lend PV(K) |
        | Synthetic Long Stock | Buy Call + Sell Put + Lend PV(K) |
        | Synthetic Short Stock | Sell Call + Buy Put + Borrow PV(K) |
        """)


def render_normal_distribution():
    """Render normal distribution properties."""

    with st.expander("Standard Normal Distribution Properties"):
        st.markdown("#### Key Relationships")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Symmetry Properties:**")
            st.latex(r"N(-x) = 1 - N(x)")
            st.latex(r"n(-x) = n(x)")
            st.latex(r"n'(x) = -x \cdot n(x)")

        with col2:
            st.markdown("**Density Function (PDF):**")
            st.latex(r"n(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}")
            st.markdown("**CDF Relationship:**")
            st.latex(r"N'(x) = n(x)")

        st.markdown("""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <strong>Key Identity for Greeks:</strong> The relationship between N(d1) and N(d2)
            appears frequently in Greek calculations:
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"S_0 e^{-qT} n(d_1) = K e^{-rT} n(d_2)")


# ==============================================================================
# FIRST-ORDER GREEKS
# ==============================================================================

def render_first_order_greeks():
    """Render first-order Greeks section."""

    st.markdown("## First-Order Greeks")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <p style="margin: 0;">
            First-order Greeks measure the sensitivity of option price to changes in a single
            underlying variable. They form the foundation of options risk management and are
            essential for delta hedging and portfolio construction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Greek overview cards
    render_greek_overview_cards()

    # Detailed sections
    render_delta_section()
    render_gamma_section()
    render_vega_section()
    render_theta_section()
    render_rho_section()


def render_greek_overview_cards():
    """Render overview cards for first-order Greeks."""

    st.markdown("### Quick Reference")

    cols = st.columns(5)

    greeks_data = [
        ("Delta", r"\frac{\partial V}{\partial S}", "#28a745", "Price sensitivity"),
        ("Gamma", r"\frac{\partial^2 V}{\partial S^2}", "#17a2b8", "Delta sensitivity"),
        ("Vega", r"\frac{\partial V}{\partial \sigma}", "#6f42c1", "Vol sensitivity"),
        ("Theta", r"\frac{\partial V}{\partial t}", "#dc3545", "Time decay"),
        ("Rho", r"\frac{\partial V}{\partial r}", "#fd7e14", "Rate sensitivity"),
    ]

    for col, (name, formula, color, desc) in zip(cols, greeks_data):
        with col:
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 10px;
                        border-top: 4px solid {color}; text-align: center;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h5 style="color: {color}; margin: 0;">{name}</h5>
            </div>
            """, unsafe_allow_html=True)
            st.latex(formula)
            st.caption(desc)


def render_delta_section():
    """Render comprehensive Delta section."""

    st.markdown("---")
    st.markdown("### Delta - Directional Exposure")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Delta measures the rate of change of option value with respect to changes in the
        underlying asset price. It is the most fundamental Greek and serves multiple purposes:

        1. **Hedge ratio:** Number of shares needed to delta-hedge the option
        2. **Probability proxy:** Approximate probability of expiring ITM
        3. **Equivalent position:** Dollar exposure in terms of stock shares
        """)

    with col2:
        st.markdown("""
        <div style="background: #d4edda; padding: 1rem; border-radius: 8px;">
            <strong>Range:</strong><br>
            Call: [0, 1]<br>
            Put: [-1, 0]
        </div>
        """, unsafe_allow_html=True)

    # Formulas
    st.markdown("#### Mathematical Definition")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Call Delta:**")
        st.latex(r"\Delta_{call} = e^{-qT} N(d_1)")

    with col2:
        st.markdown("**Put Delta:**")
        st.latex(r"\Delta_{put} = -e^{-qT} N(-d_1) = e^{-qT}[N(d_1) - 1]")

    # Delta behavior
    st.markdown("#### Delta Behavior by Moneyness")

    st.markdown("""
    | **Moneyness** | **Call Delta** | **Put Delta** | **Interpretation** |
    |--------------|----------------|---------------|-------------------|
    | Deep ITM | approaches 1.0 | approaches -1.0 | Behaves like stock |
    | ATM | approximately 0.5 | approximately -0.5 | 50/50 probability |
    | Deep OTM | approaches 0.0 | approaches 0.0 | Minimal exposure |
    """)

    # Key insights
    with st.expander("Delta Trading Insights"):
        st.markdown("""
        **Delta Hedging:**
        - To delta-hedge a long call, short Delta shares of stock
        - To delta-hedge a long put, buy |Delta| shares of stock
        - Hedge must be rebalanced as delta changes (gamma risk)

        **Delta as Probability:**
        - N(d2) is the exact risk-neutral probability of ITM expiry
        - Delta approximately equals N(d1) is slightly higher due to stock drift adjustment
        - The difference becomes significant for longer-dated options

        **Position Delta:**
        - Long calls have positive delta (bullish)
        - Long puts have negative delta (bearish)
        - Portfolio delta = sum of individual position deltas x quantity

        **Dollar Delta:**
        - Dollar Delta = Delta x S x Position Size
        - Represents the dollar P&L for a $1 move in the underlying
        """)


def render_gamma_section():
    """Render comprehensive Gamma section."""

    st.markdown("---")
    st.markdown("### Gamma - Convexity")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Gamma measures the rate of change of delta with respect to changes in the underlying
        price. It represents the **convexity** or **curvature** of the option's price profile
        and is identical for calls and puts with the same parameters.

        High gamma means delta changes rapidly, requiring frequent hedge adjustments.
        """)

    with col2:
        st.markdown("""
        <div style="background: #cce5ff; padding: 1rem; border-radius: 8px;">
            <strong>Always positive</strong> for long options<br>
            <strong>Maximum at ATM</strong><br>
            <strong>Increases</strong> near expiry
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\Gamma = \frac{\partial \Delta}{\partial S} = \frac{\partial^2 V}{\partial S^2} = \frac{e^{-qT} n(d_1)}{S_0 \sigma \sqrt{T}}")

    # Gamma characteristics
    st.markdown("#### Gamma Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
            <h6>ATM Options</h6>
            <p>Highest gamma - delta changes most rapidly around the strike price</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
            <h6>Near Expiry</h6>
            <p>Gamma spikes for ATM options as time compresses uncertainty</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
            <h6>ITM/OTM Options</h6>
            <p>Lower gamma - delta is relatively stable</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Gamma Trading Insights"):
        st.markdown("""
        **Gamma Scalping:**
        - Long gamma positions profit from realized volatility > implied volatility
        - Delta-hedge frequently to "scalp" profits from price movements
        - P&L from gamma: 0.5 x Gamma x (Delta S)^2 per rebalance

        **Gamma Risk:**
        - Short gamma positions (sold options) face unlimited risk from large moves
        - Near expiry, short gamma on ATM options is extremely dangerous
        - Known as "gamma risk" or "pin risk"

        **Dollar Gamma:**
        - Dollar Gamma = 0.5 x Gamma x S^2 x 0.01
        - P&L for a 1% move in the underlying (approximately)

        **Gamma-Theta Relationship:**
        - Long gamma comes with negative theta (time decay cost)
        - This is the fundamental gamma-theta tradeoff
        - You pay theta to be long gamma (insurance premium)
        """)


def render_vega_section():
    """Render comprehensive Vega section."""

    st.markdown("---")
    st.markdown("### Vega - Volatility Sensitivity")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Vega measures the sensitivity of option value to changes in implied volatility.
        Unlike the Greek letter name suggests, vega is not actually a Greek letter - it was
        invented by options traders.

        Vega is identical for calls and puts with the same parameters, and is always positive
        for long option positions.
        """)

    with col2:
        st.markdown("""
        <div style="background: #e2d5f1; padding: 1rem; border-radius: 8px;">
            <strong>Expressed per 1%</strong> change in IV<br>
            <strong>Maximum at ATM</strong><br>
            <strong>Increases</strong> with time to expiry
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\mathcal{V} = \frac{\partial V}{\partial \sigma} = S_0 e^{-qT} n(d_1) \sqrt{T}")

    st.markdown("#### Vega Characteristics")

    st.markdown("""
    | **Factor** | **Effect on Vega** | **Intuition** |
    |-----------|-------------------|---------------|
    | Time to Expiry up | Vega up | More time for volatility to matter |
    | ATM vs OTM | ATM highest | Most uncertainty about final payoff |
    | Volatility up | Complex | Can increase or decrease vega |
    """)

    with st.expander("Vega Trading Insights"):
        st.markdown("""
        **Volatility Trading:**
        - Long vega = betting that realized or implied volatility will increase
        - Short vega = betting that volatility will decrease
        - Straddles and strangles are common vega plays

        **Vega Decay:**
        - Vega decreases as expiration approaches (unlike gamma)
        - Long-dated options have more vega exposure
        - Use LEAPS for maximum vega exposure

        **Weighted Vega:**
        - Adjust vega by time: Weighted Vega = Vega x sqrt(30/DTE)
        - Normalizes vega across different expirations
        - Useful for calendar spread analysis

        **Volatility Term Structure:**
        - Different expirations have different implied volatilities
        - Calendar spreads trade the vol term structure
        - Vega exposure depends on which expiration you're in
        """)


def render_theta_section():
    """Render comprehensive Theta section."""

    st.markdown("---")
    st.markdown("### Theta - Time Decay")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Theta measures the rate of change of option value with respect to the passage of time,
        holding all else constant. It represents the **time decay** or **time erosion** of
        an option's extrinsic value.

        Theta is typically expressed as the dollar amount the option loses per day.
        """)

    with col2:
        st.markdown("""
        <div style="background: #f8d7da; padding: 1rem; border-radius: 8px;">
            <strong>Usually negative</strong> for long options<br>
            <strong>Accelerates</strong> near expiry<br>
            <strong>ATM options</strong> decay fastest
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.markdown("**Call Theta:**")
    st.latex(r"\Theta_{call} = -\frac{S_0 e^{-qT} n(d_1) \sigma}{2\sqrt{T}} + q S_0 e^{-qT} N(d_1) - r K e^{-rT} N(d_2)")

    st.markdown("**Put Theta:**")
    st.latex(r"\Theta_{put} = -\frac{S_0 e^{-qT} n(d_1) \sigma}{2\sqrt{T}} - q S_0 e^{-qT} N(-d_1) + r K e^{-rT} N(-d_2)")

    st.markdown("#### Time Decay Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
            <h6>Theta Acceleration</h6>
            <p>Time decay is not linear - it accelerates as expiration approaches.
            The "square root of time" relationship means an option loses more value
            in its final weeks than in earlier periods.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
            <h6>Weekend Decay</h6>
            <p>Markets price in weekend decay over the week. Some models suggest
            Friday afternoons have higher theta, while others spread it evenly.
            Earnings and events create similar dynamics.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Theta Trading Insights"):
        st.markdown("""
        **Theta Collection Strategies:**
        - Covered calls: Collect theta while owning stock
        - Cash-secured puts: Collect theta while waiting to buy
        - Iron condors: Collect theta from both sides
        - Calendar spreads: Long far-dated, short near-dated

        **Theta-Gamma Tradeoff:**
        - Theta approximately equals -0.5 x Gamma x S^2 x sigma^2 (approximately, for ATM options)
        - This relationship shows theta "pays" for gamma
        - You cannot be long gamma without paying theta

        **Daily Theta:**
        - Divide annual theta by 365 for daily theta
        - Some use 252 (trading days) for more accuracy
        - Adjust for weekends and holidays

        **Theta Burn Rate:**
        - ATM options: lose approximately 1/sqrt(T) of their value per day
        - Weekend/holiday adjustment may be needed
        - Implied volatility changes can offset theta
        """)


def render_rho_section():
    """Render comprehensive Rho section."""

    st.markdown("---")
    st.markdown("### Rho - Interest Rate Sensitivity")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Rho measures the sensitivity of option value to changes in the risk-free interest rate.
        While often considered the "forgotten Greek" due to typically stable interest rates,
        rho becomes important for long-dated options and in changing rate environments.
        """)

    with col2:
        st.markdown("""
        <div style="background: #ffe5d0; padding: 1rem; border-radius: 8px;">
            <strong>Calls:</strong> Positive rho<br>
            <strong>Puts:</strong> Negative rho<br>
            <strong>Larger</strong> for longer maturities
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Call Rho:**")
        st.latex(r"\rho_{call} = K T e^{-rT} N(d_2)")

    with col2:
        st.markdown("**Put Rho:**")
        st.latex(r"\rho_{put} = -K T e^{-rT} N(-d_2)")

    with st.expander("Rho Trading Insights"):
        st.markdown("""
        **Why Calls Have Positive Rho:**
        - Higher rates -> higher forward price -> calls worth more
        - Higher rates -> lower PV of strike -> calls worth more

        **Why Puts Have Negative Rho:**
        - Higher rates -> lower PV of strike -> puts worth less
        - Higher rates -> higher cost of carry on stock -> offsetting effect

        **When Rho Matters:**
        - LEAPS and other long-dated options
        - Periods of Fed rate changes
        - Cross-currency options
        - Fixed income derivatives

        **Dividend Rho (phi):**
        - Sensitivity to dividend yield changes
        - phi_call = -T x S x e^(-qT) x N(d1)
        - phi_put = T x S x e^(-qT) x N(-d1)
        """)


# ==============================================================================
# SECOND-ORDER GREEKS
# ==============================================================================

def render_second_order_greeks():
    """Render second-order Greeks section."""

    st.markdown("## Second-Order Greeks")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <p style="margin: 0;">
            Second-order Greeks measure how first-order Greeks change with respect to various
            factors. They are crucial for understanding the stability of hedges and the risks
            of large market moves. These are essential for sophisticated risk management.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Overview cards
    cols = st.columns(4)

    second_order = [
        ("Vanna", r"\frac{\partial \Delta}{\partial \sigma}", "#9b59b6", "Delta-Vol cross"),
        ("Volga", r"\frac{\partial^2 V}{\partial \sigma^2}", "#e74c3c", "Vega convexity"),
        ("Charm", r"\frac{\partial \Delta}{\partial t}", "#3498db", "Delta decay"),
        ("Veta", r"\frac{\partial \nu}{\partial t}", "#2ecc71", "Vega decay"),
    ]

    for col, (name, formula, color, desc) in zip(cols, second_order):
        with col:
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 10px;
                        border-top: 4px solid {color}; text-align: center;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h5 style="color: {color}; margin: 0;">{name}</h5>
            </div>
            """, unsafe_allow_html=True)
            st.latex(formula)
            st.caption(desc)

    render_vanna_section()
    render_volga_section()
    render_charm_section()
    render_veta_section()


def render_vanna_section():
    """Render Vanna section."""

    st.markdown("---")
    st.markdown("### Vanna - Delta-Volatility Cross Sensitivity")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Vanna measures how delta changes when volatility changes, or equivalently, how vega
        changes when the underlying price changes. This cross-sensitivity is crucial for
        understanding how your delta hedge will behave in a volatility shock.

        Vanna is particularly important for exotic options and volatility trading strategies.
        """)

    with col2:
        st.markdown("""
        <div style="background: #e8daef; padding: 1rem; border-radius: 8px;">
            <strong>Equivalent definitions:</strong><br>
            ∂Δ/∂σ = ∂ν/∂S<br>
            <strong>Zero at ATM</strong><br>
            <strong>Positive</strong> for OTM calls
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = \frac{\partial \mathcal{V}}{\partial S} = -e^{-qT} n(d_1) \frac{d_2}{\sigma}")

    st.markdown("*Alternative form:*")
    st.latex(r"\text{Vanna} = \frac{\mathcal{V}}{S} \left(1 - \frac{d_1}{\sigma\sqrt{T}}\right)")

    with st.expander("Vanna Trading Insights"):
        st.markdown("""
        **Vanna Sign Convention:**
        | Position | d2 > 0 (ITM call/OTM put) | d2 < 0 (OTM call/ITM put) |
        |----------|---------------------------|---------------------------|
        | Vanna | Negative | Positive |

        **Practical Implications:**
        - If vol rises and you're long OTM calls, your delta increases (vanna effect)
        - Market makers adjust hedges not just for gamma but also for vanna
        - Vanna flows can amplify or dampen market moves

        **Vanna Exposure:**
        - Long risk reversals (long OTM calls, short OTM puts) have significant vanna
        - Vanna can cause delta hedges to fail in vol spikes
        - Important for understanding "vol-of-vol" risk
        """)


def render_volga_section():
    """Render Volga/Vomma section."""

    st.markdown("---")
    st.markdown("### Volga (Vomma) - Vega Convexity")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Volga (also called Vomma) measures the rate of change of vega with respect to changes
        in implied volatility. It represents the **convexity of the vega profile** and is
        important for volatility-of-volatility exposure.

        A positive volga position benefits from large volatility moves in either direction.
        """)

    with col2:
        st.markdown("""
        <div style="background: #fadbd8; padding: 1rem; border-radius: 8px;">
            <strong>Maximum</strong> for OTM options<br>
            <strong>Positive</strong> for long options<br>
            <strong>Key for</strong> vol-of-vol
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\text{Volga} = \frac{\partial^2 V}{\partial \sigma^2} = \frac{\partial \mathcal{V}}{\partial \sigma} = \mathcal{V} \frac{d_1 d_2}{\sigma}")

    st.markdown("*Alternative form:*")
    st.latex(r"\text{Volga} = S_0 e^{-qT} n(d_1) \sqrt{T} \frac{d_1 d_2}{\sigma}")

    with st.expander("Volga Trading Insights"):
        st.markdown("""
        **Volga Characteristics:**
        - ATM options have near-zero volga (d1 approximately 0 or d2 approximately 0)
        - OTM options have positive volga
        - Volga is symmetric around ATM

        **Volga in Practice:**
        - Long straddles/strangles have positive volga
        - Beneficial when you expect vol to move significantly
        - Important for pricing exotic options

        **Volatility Smile:**
        - Volga helps explain the volatility smile
        - OTM options need higher IV to compensate for volga
        - This creates the characteristic "smile" shape

        **Volga Hedging:**
        - Dealers must hedge volga exposure
        - Creates feedback loops in the options market
        - Contributes to "volatility clustering"
        """)


def render_charm_section():
    """Render Charm section."""

    st.markdown("---")
    st.markdown("### Charm (Delta Bleed) - Delta Decay")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Charm measures the rate of change of delta with respect to time, showing how delta
        "bleeds" or decays as time passes. This is crucial for understanding how your delta
        hedge will drift even if the underlying price doesn't move.

        Also known as **DdeltaDtime** or **delta decay**.
        """)

    with col2:
        st.markdown("""
        <div style="background: #d6eaf8; padding: 1rem; border-radius: 8px;">
            <strong>Critical near expiry</strong><br>
            <strong>Hedging drift</strong><br>
            <strong>∂Δ/∂t = -∂Θ/∂S</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.markdown("**Call Charm:**")
    st.latex(r"\text{Charm}_{call} = -e^{-qT} n(d_1) \left[ q + \frac{(r-q)d_2 - \frac{d_1}{2T}}{\sigma\sqrt{T}} \right]")

    st.markdown("**Put Charm:**")
    st.latex(r"\text{Charm}_{put} = e^{-qT} n(d_1) \left[ q - \frac{(r-q)d_2 - \frac{d_1}{2T}}{\sigma\sqrt{T}} \right]")

    with st.expander("Charm Trading Insights"):
        st.markdown("""
        **Why Charm Matters:**
        - Your delta hedge drifts over time even without price movement
        - Near expiry, charm can cause rapid delta changes
        - Weekend and overnight charm effects can be significant

        **Charm Behavior:**
        | Moneyness | Effect on Call Delta | Effect on Put Delta |
        |-----------|---------------------|---------------------|
        | ITM | Delta -> 1 over time | Delta -> -1 over time |
        | ATM | Relatively stable | Relatively stable |
        | OTM | Delta -> 0 over time | Delta -> 0 over time |

        **Overnight Risk:**
        - Charm tells you how much to adjust hedges overnight
        - Important for market makers holding positions
        - Particularly relevant over weekends/holidays
        """)


def render_veta_section():
    """Render Veta section."""

    st.markdown("---")
    st.markdown("### Veta (DvegaDtime) - Vega Decay")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Veta measures how vega changes with the passage of time. As expiration approaches,
        vega typically decreases, but the rate of this decrease varies with moneyness and
        other factors.

        Important for calendar spread traders and long-dated option positions.
        """)

    with col2:
        st.markdown("""
        <div style="background: #d5f5e3; padding: 1rem; border-radius: 8px;">
            <strong>Usually negative</strong><br>
            <strong>Calendar spreads</strong><br>
            <strong>Term structure</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\text{Veta} = \frac{\partial \mathcal{V}}{\partial T} = S_0 e^{-qT} n(d_1) \sqrt{T} \left[ q + \frac{(r-q)d_1}{\sigma\sqrt{T}} - \frac{1 + d_1 d_2}{2T} \right]")

    with st.expander("Veta Trading Insights"):
        st.markdown("""
        **Veta in Calendar Spreads:**
        - Long calendar: long veta (benefit from vega decay difference)
        - Short calendar: short veta
        - Veta helps predict how calendar P&L evolves

        **Veta Characteristics:**
        - Generally negative: vega decreases over time
        - Effect is larger for ATM options
        - Important for LEAPS and long-dated strategies

        **Practical Application:**
        - Adjust vega hedges for time decay
        - Understand calendar spread "theta-like" behavior
        - Plan for vega exposure changes over time
        """)


# ==============================================================================
# THIRD-ORDER GREEKS
# ==============================================================================

def render_third_order_greeks():
    """Render third-order Greeks section."""

    st.markdown("## Third-Order Greeks")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <p style="margin: 0;">
            Third-order Greeks measure how second-order Greeks change with respect to various
            factors. They are primarily used by sophisticated market makers, exotic option
            traders, and for stress testing large portfolios. While rarely hedged directly,
            understanding them helps anticipate risk during extreme market conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Overview
    cols = st.columns(4)

    third_order = [
        ("Speed", r"\frac{\partial \Gamma}{\partial S}", "#1abc9c", "Gamma sensitivity"),
        ("Zomma", r"\frac{\partial \Gamma}{\partial \sigma}", "#e67e22", "Gamma-vol cross"),
        ("Color", r"\frac{\partial \Gamma}{\partial t}", "#9b59b6", "Gamma decay"),
        ("Ultima", r"\frac{\partial^3 V}{\partial \sigma^3}", "#34495e", "Vol cubed sensitivity"),
    ]

    for col, (name, formula, color, desc) in zip(cols, third_order):
        with col:
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 10px;
                        border-top: 4px solid {color}; text-align: center;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h5 style="color: {color}; margin: 0;">{name}</h5>
            </div>
            """, unsafe_allow_html=True)
            st.latex(formula)
            st.caption(desc)

    render_speed_section()
    render_zomma_section()
    render_color_section()
    render_ultima_section()


def render_speed_section():
    """Render Speed section."""

    st.markdown("---")
    st.markdown("### Speed (DgammaDspot) - Gamma Sensitivity to Price")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Speed measures how gamma changes as the underlying price moves. It indicates whether
        gamma is increasing or decreasing as the stock moves, which is important for
        understanding how rapidly your hedge effectiveness changes.

        Also known as **DgammaDspot** or the third derivative of price with respect to spot.
        """)

    with col2:
        st.markdown("""
        <div style="background: #d1f2eb; padding: 1rem; border-radius: 8px;">
            <strong>Third derivative</strong><br>
            <strong>Gamma acceleration</strong><br>
            <strong>Large move risk</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\text{Speed} = \frac{\partial \Gamma}{\partial S} = \frac{\partial^3 V}{\partial S^3} = -\frac{\Gamma}{S} \left(1 + \frac{d_1}{\sigma\sqrt{T}}\right)")

    st.markdown("*Alternative form:*")
    st.latex(r"\text{Speed} = -\frac{e^{-qT} n(d_1)}{S^2 \sigma \sqrt{T}} \left(\frac{d_1}{\sigma\sqrt{T}} + 1\right)")

    with st.expander("Speed Trading Insights"):
        st.markdown("""
        **Speed Characteristics:**
        - Negative below ATM, positive above ATM (for standard options)
        - Tells you whether gamma is increasing or decreasing as price moves
        - Important for predicting hedge adjustments

        **Practical Use:**
        - Anticipate gamma changes during trending markets
        - Understand convexity risk for large positions
        - Important for stress testing

        **Speed vs. Gamma:**
        - Gamma tells you current convexity
        - Speed tells you how convexity will change
        - Together they give a complete picture of price risk
        """)


def render_zomma_section():
    """Render Zomma section."""

    st.markdown("---")
    st.markdown("### Zomma (DgammaDvol) - Gamma-Volatility Cross")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Zomma measures how gamma changes when implied volatility changes. This cross-Greek
        is important during volatility shocks, as it tells you whether your gamma exposure
        will increase or decrease when the market becomes more volatile.
        """)

    with col2:
        st.markdown("""
        <div style="background: #fdebd0; padding: 1rem; border-radius: 8px;">
            <strong>Vol shock risk</strong><br>
            <strong>∂Γ/∂σ = ∂Vanna/∂S</strong><br>
            <strong>Cross-gamma</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\text{Zomma} = \frac{\partial \Gamma}{\partial \sigma} = \Gamma \frac{d_1 d_2 - 1}{\sigma}")

    st.markdown("*Alternative form:*")
    st.latex(r"\text{Zomma} = \frac{e^{-qT} n(d_1)(d_1 d_2 - 1)}{S \sigma^2 \sqrt{T}}")

    with st.expander("Zomma Trading Insights"):
        st.markdown("""
        **Zomma Behavior:**
        - Positive when |d1*d2| > 1 (far from ATM)
        - Negative when |d1*d2| < 1 (near ATM)
        - Zero when d1*d2 = 1

        **Vol Spike Scenarios:**
        - When vol spikes, your gamma exposure changes
        - Zomma tells you the direction and magnitude
        - Critical for large option portfolios

        **Risk Management:**
        - Include zomma in stress tests
        - Understand how hedges behave in crises
        - Important for exotic option pricing
        """)


def render_color_section():
    """Render Color section."""

    st.markdown("---")
    st.markdown("### Color (Gamma Decay) - Gamma Change Over Time")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Color measures how gamma changes as time passes. Near expiration, gamma can change
        dramatically, and color quantifies this rate of change. Critical for understanding
        the "pin risk" as options approach expiration.
        """)

    with col2:
        st.markdown("""
        <div style="background: #e8daef; padding: 1rem; border-radius: 8px;">
            <strong>Gamma evolution</strong><br>
            <strong>Expiration dynamics</strong><br>
            <strong>Pin risk</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\text{Color} = \frac{\partial \Gamma}{\partial T} = -\frac{e^{-qT} n(d_1)}{2 S \sigma T \sqrt{T}} \left[ 2qT + 1 + d_1 \frac{2(r-q)T - d_2 \sigma \sqrt{T}}{\sigma\sqrt{T}} \right]")

    with st.expander("Color Trading Insights"):
        st.markdown("""
        **Color Characteristics:**
        - Usually negative: gamma increases as expiry approaches (for ATM)
        - Accelerates dramatically in final days
        - Important for weekly options

        **Pin Risk:**
        - ATM gamma spikes near expiry
        - Color quantifies this spike rate
        - Critical for market makers

        **Weekend/Overnight:**
        - Color tells you how much gamma changes overnight
        - Important for position sizing
        - Helps anticipate Monday opening risk
        """)


def render_ultima_section():
    """Render Ultima section."""

    st.markdown("---")
    st.markdown("### Ultima (DvommaDvol) - Third-Order Volatility Sensitivity")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Ultima measures how volga (vomma) changes with respect to volatility. It represents
        the "vol of vol of vol" exposure and is important for understanding tail risk in
        volatility itself.

        Primarily used in exotic options pricing and extreme scenario analysis.
        """)

    with col2:
        st.markdown("""
        <div style="background: #d5d8dc; padding: 1rem; border-radius: 8px;">
            <strong>∂³V/∂σ³</strong><br>
            <strong>Tail vol risk</strong><br>
            <strong>Exotic pricing</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Mathematical Definition")

    st.latex(r"\text{Ultima} = \frac{\partial \text{Volga}}{\partial \sigma} = \frac{\mathcal{V}}{\sigma^2} \left[ d_1 d_2 (1 - d_1 d_2) + d_1^2 + d_2^2 \right]")

    with st.expander("Ultima Trading Insights"):
        st.markdown("""
        **Ultima Characteristics:**
        - Highly complex behavior
        - Important for variance swaps and vol derivatives
        - Rarely hedged directly

        **When Ultima Matters:**
        - Extreme volatility scenarios
        - Variance swap pricing
        - Vol surface modeling

        **Practical Application:**
        - Stress testing vol exposure
        - Understanding vol surface dynamics
        - Exotic option risk assessment
        """)


# ==============================================================================
# TRADING APPLICATIONS
# ==============================================================================

def render_trading_applications():
    """Render trading applications section."""

    st.markdown("## Trading Applications")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0; color: white;">
        <p style="margin: 0;">
            Understanding Greeks in isolation is only the first step. This section covers
            how to apply Greek analysis to real trading scenarios, portfolio construction,
            and risk management frameworks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_hedging_strategies()
    render_portfolio_management()
    render_greek_relationships()
    render_risk_scenarios()


def render_hedging_strategies():
    """Render hedging strategies section."""

    st.markdown("### Hedging Strategies")

    st.markdown("#### Delta Hedging Framework")

    st.markdown("""
    Delta hedging is the foundation of options market making and risk management. The goal
    is to neutralize directional exposure while maintaining other Greek exposures.
    """)

    st.latex(r"\text{Shares to hedge} = -\Delta \times \text{Option Contracts} \times 100")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #d4edda; padding: 1rem; border-radius: 8px;">
            <h6>Continuous Hedging</h6>
            <ul>
                <li>Rebalance as delta changes</li>
                <li>Higher frequency = lower gamma P&L variance</li>
                <li>Trade-off: more transaction costs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #cce5ff; padding: 1rem; border-radius: 8px;">
            <h6>Discrete Hedging</h6>
            <ul>
                <li>Rebalance at fixed intervals</li>
                <li>Lower costs but more variance</li>
                <li>Common: daily or weekly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Gamma Hedging")

    st.markdown("""
    To hedge gamma, you need other options (the underlying only provides delta).
    The process involves:

    1. Calculate portfolio gamma
    2. Find offsetting option position
    3. Adjust delta hedge for the new options
    """)

    st.latex(r"\text{Options needed} = -\frac{\Gamma_{\text{portfolio}}}{\Gamma_{\text{hedge option}}}")

    st.markdown("#### Vega Hedging")

    st.markdown("""
    Vega hedging protects against implied volatility changes:
    """)

    st.latex(r"\text{Options needed} = -\frac{\mathcal{V}_{\text{portfolio}}}{\mathcal{V}_{\text{hedge option}}}")

    st.markdown("""
    **Key considerations:**
    - Different expirations have different vegas (term structure)
    - ATM options have highest vega per dollar
    - Calendar spreads can provide vega without gamma
    """)


def render_portfolio_management():
    """Render portfolio management section."""

    st.markdown("### Portfolio Greek Management")

    st.markdown("#### Aggregating Greeks")

    st.markdown("""
    Portfolio Greeks are the weighted sum of individual position Greeks:
    """)

    st.latex(r"\text{Portfolio } \Gamma = \sum_{i} n_i \times \Gamma_i")

    st.markdown("""
    Where n_i is the position size (positive for long, negative for short).
    """)

    st.markdown("#### Greek Limits and Targets")

    st.markdown("""
    | Greek | Typical Limits | Management Approach |
    |-------|---------------|---------------------|
    | Delta | +/- 100 delta per $1M | Continuous hedging |
    | Gamma | Position-specific | Option spreads |
    | Vega | +/- 10 vega per $1M | Time spreads |
    | Theta | Monitor daily P&L | Accept or offset |
    """)

    st.markdown("#### P&L Attribution")

    st.markdown("""
    Daily P&L can be decomposed into Greek contributions:
    """)

    st.latex(r"\Delta P\&L \approx \Delta \cdot dS + \frac{1}{2}\Gamma \cdot dS^2 + \mathcal{V} \cdot d\sigma + \Theta \cdot dt + \rho \cdot dr + \epsilon")

    st.markdown("""
    Where epsilon represents unexplained P&L (model error, cross-Greeks, etc.).
    """)


def render_greek_relationships():
    """Render Greek relationships section."""

    st.markdown("### Key Greek Relationships")

    st.markdown("#### The Black-Scholes PDE")

    st.markdown("""
    The fundamental relationship between Greeks comes from the Black-Scholes partial
    differential equation:
    """)

    st.latex(r"\Theta + (r-q)S\Delta + \frac{1}{2}\sigma^2 S^2 \Gamma = rV")

    st.markdown("""
    This equation must hold for any correctly priced option and connects theta,
    delta, and gamma.
    """)

    st.markdown("#### Gamma-Theta Tradeoff")

    st.markdown("""
    For ATM options, there's an approximate relationship:
    """)

    st.latex(r"\Theta \approx -\frac{1}{2}\Gamma S^2 \sigma^2")

    st.markdown("""
    This shows that:
    - Long gamma costs theta (you pay for convexity)
    - Short gamma earns theta (you collect for taking convexity risk)
    - The ratio depends on volatility
    """)

    st.markdown("#### Cross-Greek Symmetries")

    st.markdown("""
    | Relationship | Implication |
    |-------------|-------------|
    | Vanna = ∂Δ/∂σ = ∂ν/∂S | Delta and vega are connected |
    | Charm = ∂Δ/∂t = -∂Θ/∂S | Delta decay equals theta sensitivity |
    | Zomma = ∂Γ/∂σ = ∂Vanna/∂S | Gamma and vanna are connected |
    | Veta = ∂ν/∂t = ∂Θ/∂σ | Vega decay equals theta-vol sensitivity |
    """)


def render_risk_scenarios():
    """Render risk scenarios section."""

    st.markdown("### Risk Scenarios")

    st.markdown("#### Scenario Analysis Framework")

    st.markdown("""
    Use Greeks for first-order approximations, but run full revaluation for accuracy:
    """)

    st.latex(r"V_{new} \approx V + \Delta \cdot \Delta S + \frac{1}{2}\Gamma \cdot (\Delta S)^2 + \mathcal{V} \cdot \Delta\sigma + ...")

    st.markdown("#### Common Stress Scenarios")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #f8d7da; padding: 1rem; border-radius: 8px;">
            <h6>Market Crash (-20%)</h6>
            <ul>
                <li>Test delta and gamma impact</li>
                <li>Vol typically spikes (test vega)</li>
                <li>Correlation goes to 1</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px;">
            <h6>Vol Spike (VIX +50%)</h6>
            <ul>
                <li>Test vega and volga impact</li>
                <li>Vanna effects on delta</li>
                <li>Margin requirements increase</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Greek Limits by Strategy")

    st.markdown("""
    | Strategy | Key Greeks | Risk Focus |
    |----------|-----------|------------|
    | Delta Hedging | Gamma, Vanna | Rebalancing frequency |
    | Vol Trading | Vega, Volga | Term structure |
    | Theta Collection | Theta, Gamma | Tail risk |
    | Dispersion | Correlation | Index vs components |
    """)

    st.markdown("""
    <div style="background: #e8f4f8; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;">
        <h5 style="color: #1e3a5f; margin-top: 0;">Final Thoughts</h5>
        <p style="margin-bottom: 0;">
            Greeks are approximations that work well for small moves but break down for large
            moves. Always combine Greek analysis with full scenario revaluation, stress testing,
            and common sense. The best risk management combines quantitative tools with qualitative
            judgment about market conditions and regime changes.
        </p>
    </div>
    """, unsafe_allow_html=True)


# Entry point for the guide
if __name__ == "__main__":
    render_guide_section()
