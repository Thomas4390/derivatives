"""Data-family toggle (Surface ↔ Returns).

Surface family → IV-grid market data → Heston/Merton/Bates/iv_gbm.
Returns family → log-returns time series → GARCH/NGARCH/GJR-GARCH.

The two families consume different data structures so they cannot share
a single calibration run; the toggle resets the generator + candidate
selections to family defaults when flipped.
"""

from __future__ import annotations

import streamlit as st

from config.constants import FAMILY_DEFAULT_MODEL
from services import state_manager

# Button label ↔ internal family key. The labels lead with the measure
# in proper blackboard-bold notation (ℚ for risk-neutral, ℙ for physical) —
# the same symbols the popover and the textbook use, so the buttons read
# as the math itself rather than as an ASCII shorthand.
_OPTION_TO_FAMILY: dict[str, str] = {
    "ℚ · Surface": "surface",
    "ℙ · Returns": "returns",
}
_FAMILY_TO_OPTION: dict[str, str] = {v: k for k, v in _OPTION_TO_FAMILY.items()}

# The P-measure (ℙ · Returns) family is hidden from the UI for now (per request):
# the app defaults to the Q / Surface family and the picker + its explainer below are
# not rendered. Everything in this module is kept intact so the toggle can be restored
# by flipping this flag back to True. Tests still pre-seed ``calib_data_family`` to
# exercise the Returns path — the early-return in ``render`` reads that state.
_SHOW_FAMILY_PICKER = False


def _render_measure_explainer() -> None:
    """Pedagogical popover detailing the P vs Q distinction.

    Adopts the same ``📖 About …`` / ``📖 Glossary`` shape used by the
    other info affordances in the calibration sidebar (cf.
    ``constraints_panel._render_feller_pedagogy`` and
    ``components.glossary.render``) so the help surface stays visually
    consistent across panels.
    """
    with st.popover(
        "📖 About the P and Q measures",
        help="What is the difference between the P and Q measures?",
    ):
        st.markdown(
            r"""
### Two probability measures, one asset

In financial modelling, an asset's dynamics are described under a
**probability measure** — a rule that assigns probabilities to future
scenarios. The *same* underlying can be modelled under different
measures, each one tailored to a different question.

#### 📈 **P — the physical (real-world) measure**

Under **P** the stock drifts at its *true* expected return $\mu$:

$$
\mathrm{d}S_t \,=\, \mu\, S_t\,\mathrm{d}t \,+\, \sigma\, S_t\,\mathrm{d}W_t^{\mathbb{P}}.
$$

- **What it answers** — *What will actually happen?*
  Forecasting, risk management, VaR, expected P&L.
- **How we estimate it** — fit to the **historical return series**
  (e.g. a GARCH likelihood). Whatever drift the data exhibit is what $\mu$ is.

#### 💰 **Q — the risk-neutral measure**

Under **Q** the drift is *replaced* by the risk-free rate $r$ (net of
the dividend yield $q$):

$$
\mathrm{d}S_t \,=\, (r - q)\, S_t\,\mathrm{d}t \,+\, \sigma\, S_t\,\mathrm{d}W_t^{\mathbb{Q}}.
$$

- **What it answers** — *What is a derivative worth today?*
  Option pricing, hedging, calibration to traded quotes.
- **How we estimate it** — fit so the **model reproduces today's option
  prices** (the implied-volatility surface). The real-world drift $\mu$
  is irrelevant: no-arbitrage forces it to $r - q$.

#### 🔄 The bridge — Girsanov's theorem

P and Q are linked by a **change of measure**: the *volatility
structure* is preserved, only the *drift* changes. That is also why a
P-fit GARCH can be mapped to a Q surface via the **Duan (1995)
LRNVR** — used here for the *Returns* family's Diagnostics surface with
risk premium $\lambda = 0$.

#### How this app uses them

| Family | Measure | Data source | Estimation |
|---|---|---|---|
| **Surface (options)** | $\mathbb{Q}$ | implied-vol grid | least squares on option prices |
| **Returns (GARCH)** | $\mathbb{P}$ | log-return time series | maximum likelihood |

The two consume **different data** and answer **different questions**,
so they cannot share a single calibration run. Switching family here
resets the model dropdowns to that family's defaults.
"""
        )


def render() -> str:
    if not _SHOW_FAMILY_PICKER:
        # UI hidden — default to Q / Surface; honour any pre-seeded family (tests
        # set ``calib_data_family`` directly to exercise the Returns path).
        family = state_manager.get("calib_data_family") or "surface"
        state_manager.set("calib_data_family", family)
        return family

    st.subheader("🗂️ Data Family")
    _render_measure_explainer()

    options = tuple(_OPTION_TO_FAMILY.keys())
    current = state_manager.get("calib_data_family") or "surface"
    default = _FAMILY_TO_OPTION[current]
    pick = st.segmented_control(
        "family",
        options=options,
        default=default,
        selection_mode="single",
        label_visibility="collapsed",
        key="calib_data_family_picker",
    )
    if pick is None:
        pick = default
    family = _OPTION_TO_FAMILY[pick]

    previous = state_manager.get("calib_data_family")
    if previous != family:
        # Family switched — reset generator and candidates to the family
        # default so the sidebar never displays a model from the other
        # family in the dropdowns below.
        default_model = FAMILY_DEFAULT_MODEL[family]
        state_manager.update(
            calib_data_family=family,
            calib_generator_model=default_model,
            calib_candidate_models=(default_model,),
        )
        state_manager.reset_results()
    else:
        state_manager.set("calib_data_family", family)

    return family
