"""Partial-time floating/fixed lookbacks and extreme-spread options (Heynen-Kat, Bermin).

Part of the Haug exotic-options catalog; see the package overview in
``backend.instruments.exotic_advanced``.

Author: Thomas Vaudescal
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import AnalyticalOnlyPayoff

__all__ = [
    "PartialFloatLookbackOption",
    "PartialFixedLookbackOption",
    "ExtremeSpreadOption",
]


# =============================================================================
# PARTIAL-TIME FLOATING-STRIKE LOOKBACK OPTION (Heynen-Kat 1994c)
# =============================================================================


class PartialFloatLookbackOption(FrozenInstrument, Instrument):
    """
    Partial-time floating-strike lookback option (Heynen-Kat 1994c).

    A floating-strike lookback whose lookback window covers only the *start* of
    the option's life, ``[0, t1]``; after ``t1`` the strike is locked at the
    extreme seen so far (minimum for a call, maximum for a put). As ``t1`` grows
    toward ``maturity`` the option approaches the standard floating-strike
    lookback (Goldman-Sosin-Gatto); a shorter window makes it cheaper.

    Freshly struck: the observed extreme equals the spot at pricing, so the
    pricer uses the current spot as both running min and max. The ``weight``
    (lambda) sets the strike at a fraction of the extreme for "fractional"
    lookbacks (``>= 1`` for calls, ``in (0, 1]`` for puts); ``1.0`` is standard.

    Immutable after construction.

    Parameters
    ----------
    t1 : float
        End of the lookback window ``[0, t1]`` (``0 < t1 <= maturity``).
    maturity : float
        Time to expiration T2 in years (> 0).
    is_call : bool
        True for call (lookback on the minimum), False for put (maximum).
    weight : float
        Fractional-lookback factor lambda (> 0; default 1.0 = standard).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).
    """

    __slots__ = (
        "_t1",
        "_maturity",
        "_is_call",
        "_weight",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        t1: float,
        maturity: float,
        is_call: bool = True,
        weight: float = 1.0,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (0 < t1 <= maturity):
            raise ValueError(f"t1 must satisfy 0 < t1 <= maturity, got {t1}")
        if weight <= 0:
            raise ValueError(f"weight (lambda) must be positive, got {weight}")

        object.__setattr__(self, "_t1", t1)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_weight", float(weight))
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "PartialFloatLookbackOption",
                "partial-time floating lookback uses the Heynen-Kat closed-form",
            ),
        )

    @property
    def t1(self) -> float:
        """End of the lookback window."""
        return self._t1

    @property
    def maturity(self) -> float:
        """Time to expiration T2 in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def weight(self) -> float:
        """Fractional-lookback factor lambda."""
        return self._weight

    @property
    def payoff(self) -> Payoff:
        """An :class:`AnalyticalOnlyPayoff` (closed-form pricing only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        opt = "call" if self._is_call else "put"
        return f"partial_float_lookback_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        wt = f", lambda={self._weight}" if self._weight != 1.0 else ""
        return (
            f"PartialFloatLookbackOption({opt_type}, t1={self._t1}, "
            f"T2={self._maturity}{wt})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PartialFloatLookbackOption):
            return NotImplemented
        return (
            self._t1 == other._t1
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._weight == other._weight
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._t1,
                self._maturity,
                self._is_call,
                self._weight,
                self._exercise,
            )
        )


# =============================================================================
# PARTIAL-TIME FIXED-STRIKE LOOKBACK OPTION (Heynen-Kat 1994c)
# =============================================================================


class PartialFixedLookbackOption(FrozenInstrument, Instrument):
    """
    Partial-time fixed-strike lookback option (Heynen-Kat 1994c).

    A fixed-strike lookback whose lookback period starts only at a predetermined
    date ``t1`` *after* the contract is initiated, and runs to expiry
    ``maturity``. The call pays ``max(S_max - strike, 0)`` and the put
    ``max(strike - S_min, 0)``, where the extreme is observed over ``[t1,
    maturity]`` only. Because that window is shorter than a standard
    fixed-strike lookback, this option is cheaper; as ``t1 -> 0`` the window
    spans the whole life and the price approaches the standard fixed-strike
    lookback (Conze-Viswanathan). A larger ``t1`` (shorter window) is worth
    less.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Fixed strike price X (> 0).
    t1 : float
        Start of the lookback window ``[t1, maturity]`` (``0 < t1 < maturity``).
    maturity : float
        Time to expiration T2 in years (> 0).
    is_call : bool
        True for call (lookback on the maximum), False for put (minimum).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).
    """

    __slots__ = (
        "_strike",
        "_t1",
        "_maturity",
        "_is_call",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        t1: float,
        maturity: float,
        is_call: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (0 < t1 < maturity):
            raise ValueError(f"t1 must satisfy 0 < t1 < maturity, got {t1}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_t1", t1)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "PartialFixedLookbackOption",
                "partial-time fixed lookback uses the Heynen-Kat closed-form",
            ),
        )

    @property
    def strike(self) -> float:
        """Fixed strike price X."""
        return self._strike

    @property
    def t1(self) -> float:
        """Start of the lookback window."""
        return self._t1

    @property
    def maturity(self) -> float:
        """Time to expiration T2 in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def payoff(self) -> Payoff:
        """An :class:`AnalyticalOnlyPayoff` (closed-form pricing only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        opt = "call" if self._is_call else "put"
        return f"partial_fixed_lookback_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        return (
            f"PartialFixedLookbackOption({opt_type}, K={self._strike}, "
            f"t1={self._t1}, T2={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PartialFixedLookbackOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._t1 == other._t1
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._t1,
                self._maturity,
                self._is_call,
                self._exercise,
            )
        )


# =============================================================================
# EXTREME-SPREAD OPTION (Bermin 1996b)
# =============================================================================


class ExtremeSpreadOption(FrozenInstrument, Instrument):
    """
    Extreme-spread (and reverse extreme-spread) option on a single asset.

    Priced by the Bermin (1996b) closed-form. The life is split into a first
    period ``[0, t1]`` and a second period ``[t1, maturity]``. Writing
    ``Smax_i`` / ``Smin_i`` for the extremes of period ``i``:

    * extreme-spread call pays ``max(Smax_2 - Smax_1, 0)``;
    * extreme-spread put  pays ``max(Smin_1 - Smin_2, 0)``;
    * reverse extreme-spread call pays ``max(Smin_2 - Smin_1, 0)``;
    * reverse extreme-spread put  pays ``max(Smax_1 - Smax_2, 0)``.

    The relevant first-period extreme is seeded by the observed running extreme
    carried into the contract (``Smax`` for the call / ``Smin`` for the put);
    a freshly issued option carries the spot as that extreme.

    Immutable after construction.

    Parameters
    ----------
    t1 : float
        End of the first period / start of the second period (``0 < t1 <
        maturity``).
    maturity : float
        Time to expiration T2 in years (> 0).
    is_call : bool
        True for call, False for put.
    is_reverse : bool
        True for the reverse extreme-spread variant (default False).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    es = ExtremeSpreadOption(t1=0.5, maturity=1.0, is_call=True)
    """

    __slots__ = (
        "_t1",
        "_maturity",
        "_is_call",
        "_is_reverse",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        t1: float,
        maturity: float,
        is_call: bool = True,
        is_reverse: bool = False,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (0 < t1 < maturity):
            raise ValueError(f"t1 must satisfy 0 < t1 < maturity, got {t1}")

        object.__setattr__(self, "_t1", t1)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_is_reverse", is_reverse)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "ExtremeSpreadOption",
                "extreme-spread option uses the Bermin closed-form",
            ),
        )

    @property
    def t1(self) -> float:
        """End of the first period / start of the second period."""
        return self._t1

    @property
    def maturity(self) -> float:
        """Time to expiration T2 in years."""
        return self._maturity

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def is_reverse(self) -> bool:
        """True for the reverse extreme-spread variant."""
        return self._is_reverse

    @property
    def payoff(self) -> Payoff:
        """An :class:`AnalyticalOnlyPayoff` (closed-form pricing only)."""
        return self._payoff

    @property
    def exercise_style(self) -> ExerciseStyle:
        """Exercise style (Instrument interface)."""
        return self._exercise

    @property
    def option_type(self) -> str:
        """String representation of option type."""
        opt = "call" if self._is_call else "put"
        kind = "reverse_extreme_spread" if self._is_reverse else "extreme_spread"
        return f"{kind}_{opt}"

    def __repr__(self) -> str:
        opt_type = "Call" if self._is_call else "Put"
        kind = "Reverse" if self._is_reverse else ""
        return (
            f"ExtremeSpreadOption({kind}{opt_type}, t1={self._t1}, T2={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtremeSpreadOption):
            return NotImplemented
        return (
            self._t1 == other._t1
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._is_reverse == other._is_reverse
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._t1,
                self._maturity,
                self._is_call,
                self._is_reverse,
                self._exercise,
            )
        )
