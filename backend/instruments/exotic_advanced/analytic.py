"""Closed-form univariate exotics: log contract/option, time-switch, supershare, arithmetic Asian.

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
    "LogContract",
    "LogOption",
    "TimeSwitchOption",
    "SupershareOption",
    "ArithmeticAsianOption",
]


# =============================================================================
# LOG CONTRACT (Neuberger 1994) / LOG(S) CONTRACT (Haug 4.14/4.15)
# =============================================================================


class LogContract(FrozenInstrument, Instrument):
    """
    Log contract (Neuberger 1994/1996) on a single asset (Haug 4.14).

    Pays ``ln(S_T / strike)`` at maturity -- not strictly an option, but a key
    building block for variance/volatility derivatives. The ``strike = 1`` case
    is the log(S) contract (Haug 4.15), whose payoff is simply ``ln(S_T)``.
    Priced by the closed form ``e^{-rT}[ln(S/X) + (b - sigma^2/2)T]``.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike ``X`` (> 0); ``strike = 1`` recovers the log(S) contract.
    maturity : float
        Time to expiration ``T`` (> 0).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    lc = LogContract(strike=80.0, maturity=0.25)
    """

    __slots__ = ("_strike", "_maturity", "_exercise", "_payoff")

    def __init__(
        self,
        strike: float,
        maturity: float,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "LogContract",
                "log contract uses the Neuberger/Haug 4.14 closed-form",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike X (= 1 for the log(S) contract)."""
        return self._strike

    @property
    def maturity(self) -> float:
        """Time to expiration T -- Instrument interface."""
        return self._maturity

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
        """String representation of contract type."""
        return "log_contract"

    def __repr__(self) -> str:
        return f"LogContract(X={self._strike}, T={self._maturity})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogContract):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._exercise))


# =============================================================================
# LOG OPTION (Wilmott 2000, Haug 4.16)
# =============================================================================


class LogOption(FrozenInstrument, Instrument):
    """
    Log option (Wilmott 2000) on a single asset (Haug 4.16).

    Pays ``max(ln(S_T / strike), 0)`` at maturity -- an option on the asset's
    log-return, struck at ``ln(strike)``. Priced by Haug's 4.16 closed form.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike ``X`` (> 0); the option is on the log-return struck at ln(X).
    maturity : float
        Time to expiration ``T`` (> 0).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    lo = LogOption(strike=100.0, maturity=0.75)
    """

    __slots__ = ("_strike", "_maturity", "_exercise", "_payoff")

    def __init__(
        self,
        strike: float,
        maturity: float,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "LogOption",
                "log option uses the Wilmott/Haug 4.16 closed-form",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike X (the option is on the log-return struck at ln(X))."""
        return self._strike

    @property
    def maturity(self) -> float:
        """Time to expiration T -- Instrument interface."""
        return self._maturity

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
        return "log_option"

    def __repr__(self) -> str:
        return f"LogOption(X={self._strike}, T={self._maturity})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash((self._strike, self._maturity, self._exercise))


# =============================================================================
# DISCRETE TIME-SWITCH OPTION (Pechtl 1995, Haug 4.11)
# =============================================================================


class TimeSwitchOption(FrozenInstrument, Instrument):
    """
    Discrete time-switch option (Pechtl 1995) on a single asset (Haug 4.11).

    Pays a fixed amount ``accrual * step`` at maturity for every monitoring
    instant ``i * step`` (``i = 1..T/step``) at which the asset is on the
    in-condition side of ``strike`` -- above it for a call, below it for a put.
    Priced by Haug's 4.24/4.25 discounted digital sum. Accrual swaps in rates
    markets price as a sum of these.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike ``X`` (> 0); accrual is conditioned on ``S_t`` vs ``X``.
    accrual : float
        Accumulated amount per time unit ``A`` (> 0); each in-condition instant
        pays ``A * step``.
    maturity : float
        Time to expiration ``T`` (> 0).
    step : float
        Monitoring time step ``Delta t`` (``0 < step <= maturity``), e.g.
        ``1/365`` for daily accrual.
    is_call : bool
        True accrues while ``S > X`` (call), False while ``S < X`` (put).
    units_filled : int
        Number of past time units already fulfilling the condition (seasoned
        option); ``0`` for a fresh option.
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    ts = TimeSwitchOption(strike=110.0, accrual=5.0, maturity=1.0, step=1 / 365)
    """

    __slots__ = (
        "_strike",
        "_accrual",
        "_maturity",
        "_step",
        "_is_call",
        "_units_filled",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        accrual: float,
        maturity: float,
        step: float,
        is_call: bool = True,
        units_filled: int = 0,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if accrual <= 0:
            raise ValueError(f"accrual must be positive, got {accrual}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (0 < step <= maturity):
            raise ValueError(f"step must satisfy 0 < step <= maturity, got {step}")
        if units_filled < 0:
            raise ValueError(f"units_filled must be >= 0, got {units_filled}")

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_accrual", accrual)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_step", step)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_units_filled", units_filled)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "TimeSwitchOption",
                "time-switch uses the Pechtl/Haug 4.24-4.25 digital sum",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike X."""
        return self._strike

    @property
    def accrual(self) -> float:
        """Accumulated amount per time unit A."""
        return self._accrual

    @property
    def maturity(self) -> float:
        """Time to expiration T -- Instrument interface."""
        return self._maturity

    @property
    def step(self) -> float:
        """Monitoring time step Delta t."""
        return self._step

    @property
    def is_call(self) -> bool:
        """True accrues while S > X (call), False while S < X (put)."""
        return self._is_call

    @property
    def units_filled(self) -> int:
        """Past time units already fulfilling the condition (m)."""
        return self._units_filled

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
        return f"time_switch_{opt}"

    def __repr__(self) -> str:
        opt = "Call" if self._is_call else "Put"
        return (
            f"TimeSwitchOption({opt}, X={self._strike}, A={self._accrual}, "
            f"T={self._maturity}, dt={self._step}, m={self._units_filled})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSwitchOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._accrual == other._accrual
            and self._maturity == other._maturity
            and self._step == other._step
            and self._is_call == other._is_call
            and self._units_filled == other._units_filled
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._accrual,
                self._maturity,
                self._step,
                self._is_call,
                self._units_filled,
                self._exercise,
            )
        )


# =============================================================================
# SUPERSHARE OPTION (Hakansson 1976, Haug 4.19.4)
# =============================================================================


class SupershareOption(FrozenInstrument, Instrument):
    """
    Supershare option (Hakansson 1976) on a single asset (Haug 4.19.4).

    Pays ``S_T / lower_strike`` at maturity if ``lower_strike < S_T <
    upper_strike`` and 0 otherwise -- a normalised asset-or-nothing band.
    Portfolios of supershares build the "superfund" (SuperShares / SuperUnits)
    traded products. Priced by Haug's 4.88 closed form (it is ``1/lower_strike``
    times the difference of two asset-or-nothing calls struck at the boundaries).

    Immutable after construction. Neither call nor put -- it is its own contract.

    Parameters
    ----------
    lower_strike : float
        Lower boundary ``X_L`` (> 0).
    upper_strike : float
        Upper boundary ``X_H`` (> ``lower_strike``).
    maturity : float
        Time to expiration ``T`` (> 0).
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    ss = SupershareOption(lower_strike=90.0, upper_strike=110.0, maturity=0.25)
    """

    __slots__ = ("_lower_strike", "_upper_strike", "_maturity", "_exercise", "_payoff")

    def __init__(
        self,
        lower_strike: float,
        upper_strike: float,
        maturity: float,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if lower_strike <= 0:
            raise ValueError(f"lower_strike must be positive, got {lower_strike}")
        if upper_strike <= lower_strike:
            raise ValueError(
                f"upper_strike must exceed lower_strike, got "
                f"{upper_strike} <= {lower_strike}"
            )
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        object.__setattr__(self, "_lower_strike", lower_strike)
        object.__setattr__(self, "_upper_strike", upper_strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "SupershareOption",
                "supershare uses the Hakansson/Haug 4.88 closed-form",
            ),
        )

    @property
    def lower_strike(self) -> float:
        """Lower boundary X_L."""
        return self._lower_strike

    @property
    def upper_strike(self) -> float:
        """Upper boundary X_H."""
        return self._upper_strike

    @property
    def maturity(self) -> float:
        """Time to expiration T -- Instrument interface."""
        return self._maturity

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
        return "supershare"

    def __repr__(self) -> str:
        return (
            f"SupershareOption(X_L={self._lower_strike}, "
            f"X_H={self._upper_strike}, T={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SupershareOption):
            return NotImplemented
        return (
            self._lower_strike == other._lower_strike
            and self._upper_strike == other._upper_strike
            and self._maturity == other._maturity
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._lower_strike,
                self._upper_strike,
                self._maturity,
                self._exercise,
            )
        )


# =============================================================================
# ARITHMETIC AVERAGE-RATE OPTION (Turnbull-Wakeman 1991, Haug 4.20.2)
# =============================================================================


class ArithmeticAsianOption(FrozenInstrument, Instrument):
    """
    Arithmetic average-rate (Asian) option (Turnbull-Wakeman 1991, Haug 4.20.2).

    Payoff on the *arithmetic* average of the asset over the averaging window:
    ``max(A - strike, 0)`` (call) or ``max(strike - A, 0)`` (put). Priced by the
    Turnbull-Wakeman moment-matching approximation (Haug 4.97/4.98); the legacy
    :func:`~backend.engines.exotic.asian.asian_geometric_price` prices the
    distinct *geometric* average and is left untouched.

    Immutable after construction.

    Parameters
    ----------
    strike : float
        Strike ``X`` (> 0).
    maturity : float
        Remaining time to maturity ``T`` in years (> 0).
    average_period : float
        Length of the averaging window ``T2`` in years (> 0; constant over the
        option's life). ``average_period > maturity`` means the option is already
        into the average period (seasoned).
    realized_average : float
        Arithmetic average ``SA`` realized so far. Required (> 0) for a seasoned
        option (``average_period > maturity``); ignored otherwise (a fresh
        option's average is taken as the spot at pricing time).
    is_call : bool
        True for call, False for put.
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    aa = ArithmeticAsianOption(strike=95.0, maturity=0.25, average_period=0.25)
    """

    __slots__ = (
        "_strike",
        "_maturity",
        "_average_period",
        "_realized_average",
        "_is_call",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        strike: float,
        maturity: float,
        average_period: float,
        realized_average: float = 0.0,
        is_call: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if average_period <= 0:
            raise ValueError(f"average_period must be positive, got {average_period}")
        if average_period > maturity and realized_average <= 0:
            raise ValueError(
                "a seasoned option (average_period > maturity) requires "
                f"realized_average > 0, got {realized_average}"
            )

        object.__setattr__(self, "_strike", strike)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_average_period", average_period)
        object.__setattr__(self, "_realized_average", realized_average)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "ArithmeticAsianOption",
                "arithmetic Asian uses the Turnbull-Wakeman/Haug 4.97-4.98 form",
            ),
        )

    @property
    def strike(self) -> float:
        """Strike X."""
        return self._strike

    @property
    def maturity(self) -> float:
        """Remaining time to maturity T -- Instrument interface."""
        return self._maturity

    @property
    def average_period(self) -> float:
        """Length of the averaging window T2."""
        return self._average_period

    @property
    def realized_average(self) -> float:
        """Arithmetic average SA realized so far (0 => fresh, use spot)."""
        return self._realized_average

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
        return f"arithmetic_asian_{opt}"

    def __repr__(self) -> str:
        opt = "Call" if self._is_call else "Put"
        return (
            f"ArithmeticAsianOption({opt}, X={self._strike}, T={self._maturity}, "
            f"T2={self._average_period}, SA={self._realized_average})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArithmeticAsianOption):
            return NotImplemented
        return (
            self._strike == other._strike
            and self._maturity == other._maturity
            and self._average_period == other._average_period
            and self._realized_average == other._realized_average
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._strike,
                self._maturity,
                self._average_period,
                self._realized_average,
                self._is_call,
                self._exercise,
            )
        )
