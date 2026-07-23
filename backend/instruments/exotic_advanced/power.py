"""Powered and capped power options (Esser 2003).

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
    "PoweredOption",
    "CappedPowerOption",
]


# =============================================================================
# POWERED OPTION (Esser 2003, Haug 4.4.4)
# =============================================================================


class PoweredOption(FrozenInstrument, Instrument):
    """
    Powered option (Esser 2003) on a single asset (Haug 4.4.4).

    The standard payoff is raised to an integer power: a call pays
    ``max(S_T - strike, 0) ** power``, a put ``max(strike - S_T, 0) ** power``.
    Priced by Haug's 4.10/4.11 binomial sum (distinct from the legacy
    :class:`~backend.instruments.exotic_options.PowerOption`, which powers the
    *asset* as ``max(S^i - X, 0)``).

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike ``X`` (> 0).
    maturity : float
        Time to expiration ``T`` (> 0).
    power : int
        Power ``i`` applied to the payoff (positive integer).
    is_call : bool
        True for call, False for put.
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    po = PoweredOption(strike=100.0, maturity=0.5, power=2)
    """

    __slots__ = ("_strike", "_maturity", "_power", "_is_call", "_exercise", "_payoff")

    def __init__(
        self,
        strike: float,
        maturity: float,
        power: int,
        is_call: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if int(power) != power or power < 1:
            raise ValueError(f"power must be a positive integer, got {power}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_power", int(power))
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "PoweredOption",
                "powered option uses the Esser/Haug 4.10-4.11 binomial sum",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike X."""
        return self._strike

    @property
    def maturity(self) -> float:
        """Time to expiration T -- Instrument interface."""
        return self._maturity

    @property
    def power(self) -> int:
        """Power i applied to the payoff."""
        return self._power

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
        return f"powered_{opt}"

    def __repr__(self) -> str:
        opt = "Call" if self._is_call else "Put"
        return (
            f"PoweredOption({opt}, X={self._strike}, T={self._maturity}, "
            f"i={self._power})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoweredOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._power == other._power
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._maturity,
                self._power,
                self._is_call,
                self._exercise,
            )
        )


# =============================================================================
# CAPPED POWER OPTION (Esser 2003, Haug 4.4.3)
# =============================================================================


class CappedPowerOption(FrozenInstrument, Instrument):
    """
    Capped power option (Esser 2003) on a single asset (Haug 4.4.3).

    A standard power option whose payoff is capped at ``cap``: a call pays
    ``min(max(S_T ** power - strike, 0), cap)``, a put
    ``min(max(strike - S_T ** power, 0), cap)``. Relaxing ``cap -> inf`` recovers
    the standard power option (Haug 4.4.2). Priced by Haug's 4.8/4.9 closed form.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike ``X`` (> 0).
    maturity : float
        Time to expiration ``T`` (> 0).
    power : float
        Power ``i`` applied to the asset (> 0).
    cap : float
        Maximum payoff ``C`` (> 0; for a put, ``cap < strike``).
    is_call : bool
        True for call, False for put.
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    cp = CappedPowerOption(strike=100.0, maturity=0.5, power=2.0, cap=30.0)
    """

    __slots__ = (
        "_strike",
        "_maturity",
        "_power",
        "_cap",
        "_is_call",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        maturity: float,
        power: float,
        cap: float,
        is_call: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")
        if cap <= 0:
            raise ValueError(f"cap must be positive, got {cap}")
        if not is_call and cap >= strike:
            raise ValueError(
                f"for a capped power put, cap must be < strike, got {cap} >= {strike}"
            )

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_power", power)
        object.__setattr__(self, "_cap", cap)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "CappedPowerOption",
                "capped power option uses the Esser/Haug 4.8-4.9 closed-form",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike X."""
        return self._strike

    @property
    def maturity(self) -> float:
        """Time to expiration T -- Instrument interface."""
        return self._maturity

    @property
    def power(self) -> float:
        """Power i applied to the asset."""
        return self._power

    @property
    def cap(self) -> float:
        """Maximum payoff C."""
        return self._cap

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
        return f"capped_power_{opt}"

    def __repr__(self) -> str:
        opt = "Call" if self._is_call else "Put"
        return (
            f"CappedPowerOption({opt}, X={self._strike}, T={self._maturity}, "
            f"i={self._power}, C={self._cap})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CappedPowerOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._power == other._power
            and self._cap == other._cap
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._maturity,
                self._power,
                self._cap,
                self._is_call,
                self._exercise,
            )
        )
