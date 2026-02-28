"""
Exercise Schedules
==================

Exercise type definitions and schedules for options.

This module provides:
- EuropeanExercise: Exercise only at maturity
- AmericanExercise: Exercise any time up to maturity
- BermudanExercise: Exercise at discrete dates

Note: ExerciseStyle enum is defined in backend.core.result_types
      ExerciseType is an alias for backward compatibility.

Author: Thomas
Created: 2025
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from backend.core.result_types import ExerciseStyle

# Backward compatibility alias
ExerciseType = ExerciseStyle


# =============================================================================
# Exercise Schedules
# =============================================================================

@dataclass
class EuropeanExercise:
    """
    European exercise: only at maturity.

    Parameters
    ----------
    maturity : float
        Time to maturity in years
    """
    maturity: float

    @property
    def exercise_type(self) -> ExerciseStyle:
        return ExerciseStyle.EUROPEAN

    def can_exercise(self, t: float, tol: float = 1e-8) -> bool:
        """Check if exercise is allowed at time t."""
        return abs(t - self.maturity) < tol

    def get_exercise_times(self) -> np.ndarray:
        """Get all exercise times."""
        return np.array([self.maturity])


@dataclass
class AmericanExercise:
    """
    American exercise: any time up to maturity.

    Parameters
    ----------
    maturity : float
        Time to maturity in years
    start_time : float
        Earliest exercise time (default 0)
    """
    maturity: float
    start_time: float = 0.0

    @property
    def exercise_type(self) -> ExerciseStyle:
        return ExerciseStyle.AMERICAN

    def can_exercise(self, t: float, tol: float = 1e-8) -> bool:
        """Check if exercise is allowed at time t."""
        return self.start_time - tol <= t <= self.maturity + tol

    def get_exercise_times(self, n_steps: int = 100) -> np.ndarray:
        """
        Get discretized exercise times for numerical methods.

        Parameters
        ----------
        n_steps : int
            Number of time steps

        Returns
        -------
        np.ndarray
            Array of exercise times
        """
        return np.linspace(self.start_time, self.maturity, n_steps + 1)


@dataclass
class BermudanExercise:
    """
    Bermudan exercise: at discrete dates.

    Parameters
    ----------
    exercise_dates : List[float]
        List of exercise times in years
    """
    exercise_dates: list[float]

    def __post_init__(self):
        """Sort exercise dates."""
        self.exercise_dates = sorted(self.exercise_dates)

    @property
    def exercise_type(self) -> ExerciseStyle:
        return ExerciseStyle.BERMUDAN

    @property
    def maturity(self) -> float:
        """Final exercise date."""
        return self.exercise_dates[-1]

    def can_exercise(self, t: float, tol: float = 1e-8) -> bool:
        """Check if exercise is allowed at time t."""
        for date in self.exercise_dates:
            if abs(t - date) < tol:
                return True
        return False

    def get_exercise_times(self) -> np.ndarray:
        """Get all exercise times."""
        return np.array(self.exercise_dates)

    @classmethod
    def from_schedule(
        cls,
        start: float,
        end: float,
        frequency: str = "monthly"
    ) -> "BermudanExercise":
        """
        Create Bermudan exercise from schedule.

        Parameters
        ----------
        start : float
            First exercise date (years)
        end : float
            Last exercise date (years)
        frequency : str
            Exercise frequency: 'daily', 'weekly', 'monthly', 'quarterly'

        Returns
        -------
        BermudanExercise
            Bermudan exercise schedule
        """
        freq_days = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "quarterly": 91,
            "semiannual": 182,
            "annual": 365
        }

        if frequency not in freq_days:
            raise ValueError(f"Unknown frequency: {frequency}")

        days = freq_days[frequency]
        n_dates = int((end - start) * 365 / days) + 1
        dates = [start + i * days / 365.0 for i in range(n_dates)]

        return cls(exercise_dates=dates)


# =============================================================================
# Type Alias
# =============================================================================

ExerciseSchedule = EuropeanExercise | AmericanExercise | BermudanExercise


# =============================================================================
# Factory Function
# =============================================================================

def create_exercise(
    exercise_type: str | ExerciseStyle,
    maturity: float,
    exercise_dates: list[float] | None = None,
    start_time: float = 0.0
) -> ExerciseSchedule:
    """
    Factory function to create exercise schedule.

    Parameters
    ----------
    exercise_type : str or ExerciseStyle
        'european', 'american', or 'bermudan'
    maturity : float
        Option maturity in years
    exercise_dates : List[float], optional
        Bermudan exercise dates (required for bermudan)
    start_time : float
        American exercise start time (default 0)

    Returns
    -------
    ExerciseSchedule
        Appropriate exercise schedule object
    """
    if isinstance(exercise_type, str):
        # Map string to ExerciseStyle enum
        style_map = {
            "european": ExerciseStyle.EUROPEAN,
            "american": ExerciseStyle.AMERICAN,
            "bermudan": ExerciseStyle.BERMUDAN,
        }
        exercise_type = style_map.get(exercise_type.lower())
        if exercise_type is None:
            raise ValueError(f"Unknown exercise type: {exercise_type}")

    if exercise_type == ExerciseStyle.EUROPEAN:
        return EuropeanExercise(maturity=maturity)

    if exercise_type == ExerciseStyle.AMERICAN:
        return AmericanExercise(maturity=maturity, start_time=start_time)

    if exercise_type == ExerciseStyle.BERMUDAN:
        if exercise_dates is None:
            raise ValueError("exercise_dates required for Bermudan exercise")
        return BermudanExercise(exercise_dates=exercise_dates)

    raise ValueError(f"Unknown exercise type: {exercise_type}")


# =============================================================================
# Exercise Decision Helper
# =============================================================================

class ExerciseDecision(NamedTuple):
    """Result of exercise decision at a time step."""
    should_exercise: bool
    intrinsic_value: float
    continuation_value: float
    time: float


def optimal_exercise_boundary(
    exercise: AmericanExercise,
    strike: float,
    is_call: bool,
    spot_grid: np.ndarray,
    time_grid: np.ndarray,
    option_values: np.ndarray
) -> np.ndarray:
    """
    Extract optimal exercise boundary from option value grid.

    For American options, finds the critical spot price at each
    time step where exercise becomes optimal.

    Parameters
    ----------
    exercise : AmericanExercise
        Exercise schedule
    strike : float
        Strike price
    is_call : bool
        True for call, False for put
    spot_grid : np.ndarray
        Array of spot prices
    time_grid : np.ndarray
        Array of time points
    option_values : np.ndarray
        Option values on grid, shape (n_spots, n_times)

    Returns
    -------
    np.ndarray
        Critical spot prices at each time, shape (n_times,)

    Notes
    -----
    For calls: exercise boundary is the minimum spot where exercise is optimal
    For puts: exercise boundary is the maximum spot where exercise is optimal
    """
    n_times = len(time_grid)
    boundary = np.zeros(n_times)

    for t_idx in range(n_times):
        # Compute intrinsic values
        if is_call:
            intrinsic = np.maximum(spot_grid - strike, 0)
        else:
            intrinsic = np.maximum(strike - spot_grid, 0)

        # Find where exercise is optimal (intrinsic >= option value)
        option_vals = option_values[:, t_idx]
        exercise_optimal = intrinsic >= option_vals - 1e-8

        if is_call:
            # For call, find lowest spot where exercise is optimal
            exercise_spots = spot_grid[exercise_optimal & (intrinsic > 0)]
            boundary[t_idx] = exercise_spots.min() if len(exercise_spots) > 0 else np.inf
        else:
            # For put, find highest spot where exercise is optimal
            exercise_spots = spot_grid[exercise_optimal & (intrinsic > 0)]
            boundary[t_idx] = exercise_spots.max() if len(exercise_spots) > 0 else 0.0

    return boundary


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Exercise Schedules Smoke Test")
    print("=" * 50)

    # Test European
    european = create_exercise('european', maturity=1.0)
    print("\nEuropean Exercise:")
    print(f"  Maturity: {european.maturity}")
    print(f"  Can exercise at t=0.5: {european.can_exercise(0.5)}")
    print(f"  Can exercise at t=1.0: {european.can_exercise(1.0)}")
    print(f"  Exercise times: {european.get_exercise_times()}")

    # Test American
    american = create_exercise('american', maturity=1.0)
    print("\nAmerican Exercise:")
    print(f"  Maturity: {american.maturity}")
    print(f"  Can exercise at t=0.0: {american.can_exercise(0.0)}")
    print(f"  Can exercise at t=0.5: {american.can_exercise(0.5)}")
    print(f"  Can exercise at t=1.0: {american.can_exercise(1.0)}")
    print(f"  Discretized (5 steps): {american.get_exercise_times(5)}")

    # Test Bermudan
    bermudan = create_exercise(
        'bermudan',
        maturity=1.0,
        exercise_dates=[0.25, 0.5, 0.75, 1.0]
    )
    print("\nBermudan Exercise:")
    print(f"  Maturity: {bermudan.maturity}")
    print(f"  Exercise dates: {bermudan.exercise_dates}")
    print(f"  Can exercise at t=0.25: {bermudan.can_exercise(0.25)}")
    print(f"  Can exercise at t=0.30: {bermudan.can_exercise(0.30)}")

    # Test Bermudan from schedule
    bermudan_monthly = BermudanExercise.from_schedule(
        start=0.25, end=1.0, frequency='monthly'
    )
    print("\nBermudan (monthly from t=0.25 to t=1.0):")
    print(f"  N dates: {len(bermudan_monthly.exercise_dates)}")
    print(f"  First: {bermudan_monthly.exercise_dates[0]:.4f}")
    print(f"  Last: {bermudan_monthly.exercise_dates[-1]:.4f}")

    print("\n" + "=" * 50)
    print("Exercise Schedules smoke test passed")
    print("=" * 50)
