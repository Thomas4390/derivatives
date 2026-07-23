"""Extendible-maturity option (Longstaff 1990).

Part of the Haug exotic-options catalog; see the package overview in
``backend.instruments.exotic_advanced``.

Author: Thomas Vaudescal
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import AnalyticalOnlyPayoff

__all__ = ["ExtendibleOption"]


class ExtendibleOption(FrozenInstrument, Instrument):
    """
    Extendible-maturity option (Longstaff 1990) on a single asset.

    The option can be exercised at its initial expiry ``t1`` but its life may
    also be extended to ``maturity`` (``T2``) with the strike adjusted from
    ``strike1`` to ``strike2``. Two styles:

    * holder-extendible (``holder_extendible=True``): the holder may extend at
      ``t1`` by paying the writer the fee ``extension_fee`` (``A``). Payoff at
      ``t1``: call ``max(S - strike1, c_BSM(S, strike2, T2 - t1) - A, 0)``.
    * writer-extendible (``holder_extendible=False``): the writer extends
      automatically (no fee) if the option is out-of-the-money at ``t1``. Payoff
      at ``t1``: call ``S - strike1`` if ``S >= strike1`` else ``c_BSM(S,
      strike2, T2 - t1)``.

    Priced by the Longstaff (1990) closed form (Haug 4.14). The holder variant's
    two critical asset prices at ``t1`` are found by safeguarded bisection.

    Immutable after construction.

    Parameters
    ----------
    strike1 : float
        Initial strike ``X1`` (> 0).
    strike2 : float
        Extended strike ``X2`` (> 0).
    t1 : float
        Initial expiry (``0 < t1 < maturity``).
    maturity : float
        Extended expiry ``T2`` (> 0).
    extension_fee : float
        Holder extension fee ``A`` (>= 0); ignored for the writer variant.
    is_call : bool
        True for call, False for put.
    holder_extendible : bool
        True for holder-extendible (default), False for writer-extendible.
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    ext = ExtendibleOption(100, 105, 0.5, 0.75, extension_fee=1.0, is_call=True)
    """

    __slots__ = (
        "_strike1",
        "_strike2",
        "_t1",
        "_maturity",
        "_extension_fee",
        "_is_call",
        "_holder_extendible",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike1: float,
        strike2: float,
        t1: float,
        maturity: float,
        extension_fee: float = 0.0,
        is_call: bool = True,
        holder_extendible: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike1 <= 0:
            raise ValueError(f"Initial strike must be positive, got {strike1}")
        if strike2 <= 0:
            raise ValueError(f"Extended strike must be positive, got {strike2}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (0 < t1 < maturity):
            raise ValueError(f"t1 must satisfy 0 < t1 < maturity, got {t1}")
        if extension_fee < 0:
            raise ValueError(f"Extension fee must be non-negative, got {extension_fee}")

        object.__setattr__(self, "_strike1", strike1)
        object.__setattr__(self, "_strike2", strike2)
        object.__setattr__(self, "_t1", t1)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_extension_fee", extension_fee)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_holder_extendible", holder_extendible)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "ExtendibleOption",
                "extendible option uses the Longstaff closed-form",
            ),
        )

    @property
    def strike1(self) -> float:
        """Initial strike X1."""
        return self._strike1

    @property
    def strike2(self) -> float:
        """Extended strike X2."""
        return self._strike2

    @property
    def t1(self) -> float:
        """Initial expiry."""
        return self._t1

    @property
    def maturity(self) -> float:
        """Extended expiry T2 -- Instrument interface."""
        return self._maturity

    @property
    def extension_fee(self) -> float:
        """Holder extension fee A (ignored for the writer variant)."""
        return self._extension_fee

    @property
    def is_call(self) -> bool:
        """True for call, False for put."""
        return self._is_call

    @property
    def holder_extendible(self) -> bool:
        """True for holder-extendible, False for writer-extendible."""
        return self._holder_extendible

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
        style = "holder" if self._holder_extendible else "writer"
        opt = "call" if self._is_call else "put"
        return f"extendible_{style}_{opt}"

    def __repr__(self) -> str:
        style = "Holder" if self._holder_extendible else "Writer"
        opt = "Call" if self._is_call else "Put"
        return (
            f"ExtendibleOption({style}{opt}, X1={self._strike1}, "
            f"X2={self._strike2}, t1={self._t1}, T2={self._maturity}, "
            f"A={self._extension_fee})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtendibleOption):
            return NotImplemented
        return (
            self._strike1 == other._strike1
            and self._strike2 == other._strike2
            and self._t1 == other._t1
            and self._maturity == other._maturity
            and self._extension_fee == other._extension_fee
            and self._is_call == other._is_call
            and self._holder_extendible == other._holder_extendible
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike1,
                self._strike2,
                self._t1,
                self._maturity,
                self._extension_fee,
                self._is_call,
                self._holder_extendible,
                self._exercise,
            )
        )
