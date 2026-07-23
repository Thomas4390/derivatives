"""Forward-start option (Rubinstein 1990).

Part of the Haug exotic-options catalog; see the package overview in
``backend.instruments.exotic_advanced``.

Author: Thomas Vaudescal
"""

from __future__ import annotations

from backend.core.interfaces import Instrument, Payoff
from backend.instruments._frozen import FrozenInstrument
from backend.core.result_types import ExerciseStyle
from backend.instruments.payoffs import AnalyticalOnlyPayoff

__all__ = ["ForwardStartOption"]


class ForwardStartOption(FrozenInstrument, Instrument):
    """
    Forward-start option (Rubinstein 1990) on a single asset.

    Paid for now but only "starts" at the grant date ``grant_time``: the strike
    is then set to ``alpha * S(grant_time)`` and the option runs to expiry
    ``maturity``. ``alpha < 1`` grants the call in-the-money, ``alpha = 1``
    at-the-money, ``alpha > 1`` out-of-the-money. Priced by the Rubinstein (1990)
    closed form (Haug 4.6); ratchet/cliquet options are sums of these.

    Immutable after construction.

    Parameters
    ----------
    alpha : float
        Forward-strike moneyness (> 0); strike = ``alpha * S(grant_time)``.
    grant_time : float
        Grant date ``t1`` (``0 < grant_time < maturity``).
    maturity : float
        Time to expiration ``T`` (> 0).
    is_call : bool
        True for call, False for put.
    exercise : ExerciseStyle
        Exercise style (default EUROPEAN).

    Examples
    --------
    fs = ForwardStartOption(alpha=1.1, grant_time=0.25, maturity=1.0)
    """

    __slots__ = (
        "_alpha",
        "_grant_time",
        "_maturity",
        "_is_call",
        "_exercise",
        "_payoff",
    )

    def __init__(
        self,
        alpha: float,
        grant_time: float,
        maturity: float,
        is_call: bool = True,
        exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")
        if not (0 < grant_time < maturity):
            raise ValueError(
                f"grant_time must satisfy 0 < grant_time < maturity, got {grant_time}"
            )

        object.__setattr__(self, "_alpha", alpha)
        object.__setattr__(self, "_grant_time", grant_time)
        object.__setattr__(self, "_maturity", maturity)
        object.__setattr__(self, "_is_call", is_call)
        object.__setattr__(self, "_exercise", exercise)
        object.__setattr__(
            self,
            "_payoff",
            AnalyticalOnlyPayoff(
                "ForwardStartOption",
                "forward-start uses the Rubinstein closed-form",
            ),
        )

    @property
    def alpha(self) -> float:
        """Forward-strike moneyness."""
        return self._alpha

    @property
    def grant_time(self) -> float:
        """Grant date t1."""
        return self._grant_time

    @property
    def maturity(self) -> float:
        """Time to expiration T -- Instrument interface."""
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
        return f"forward_start_{opt}"

    def __repr__(self) -> str:
        opt = "Call" if self._is_call else "Put"
        return (
            f"ForwardStartOption({opt}, alpha={self._alpha}, "
            f"t1={self._grant_time}, T={self._maturity})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForwardStartOption):
            return NotImplemented
        return (
            self._alpha == other._alpha
            and self._grant_time == other._grant_time
            and self._maturity == other._maturity
            and self._is_call == other._is_call
            and self._exercise == other._exercise
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._alpha,
                self._grant_time,
                self._maturity,
                self._is_call,
                self._exercise,
            )
        )
