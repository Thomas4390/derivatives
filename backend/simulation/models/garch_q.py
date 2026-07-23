"""
Risk-neutral GARCH simulators (discrete-time, daily grid)
=========================================================

Monte-Carlo simulator for the nonaffine GARCH option-pricing family under the
risk-neutral measure Q, following Duan (1995)'s local risk-neutral valuation
relationship (LRNVR) and the discrete-time formulation of Dorion & François
(ch. 7.2). One trading-day step, per-period (daily) conditional variance
``h_t``:

    R_t      = r_step - 0.5 h_t + sqrt(h_t) z_t,    z_t ~ N(0, 1)
    r_step   = r / steps_per_year

so that ``E^Q[S_T] = S_0 e^{r T}`` holds by construction. The variance
recursion depends on the GARCH variant (``gamma`` already carries the Duan
risk-neutral shift ``gamma* = gamma + lambda`` for the asymmetric variants):

    garch  : h_{t+1} = omega + alpha h_t z_t^2 + beta h_t
    ngarch : h_{t+1} = omega + alpha h_t (z_t - gamma)^2 + beta h_t   (Duan 1995)
    gjr    : h_{t+1} = omega + (alpha + gamma 1{z_t<0}) h_t z_t^2 + beta h_t

Unlike the affine Heston-Nandi model (which has a closed-form characteristic
function and prices by FFT), these nonaffine recursions admit **no** analytical
option formula — European options price by Monte-Carlo only. Simulation is
vectorised numpy seeded through a ``np.random.Generator`` so that re-pricing
under a fixed ``seed`` reuses identical innovations (common random numbers),
giving a smooth surface objective for the calibrator.

Author: Thomas Vaudescal
Created: 2026
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numba import njit, prange

from backend.simulation.base import (
    BaseSimulator,
    SimulationResult,
    StochasticVolatilityMixin,
)
from backend.utils.constants.calibration import VALID_GARCH_TYPES
from backend.utils.constants.numerical import (
    GARCH_CALIBRATION_VARIANCE_FLOOR as _VAR_FLOOR,
)
from backend.utils.constants.time import TRADING_DAYS_PER_YEAR

# Variant codes consumed by the Numba kernel (strings don't JIT cleanly).
_VARIANT_CODES: dict[str, int] = {"garch": 0, "ngarch": 1, "gjr_garch": 2}


# Per-period risk-neutral GARCH recursion, path-by-path, in parallel — one
# specialized kernel per variant so the variance-update branch (constant for a
# given simulator) stays out of the per-step hot loop.
#
# Innovations ``z`` are pre-drawn by a seeded NumPy ``Generator`` so each kernel
# is *pure deterministic arithmetic* — that keeps common random numbers
# bit-reproducible across optimizer evaluations (and across thread counts) while
# still parallelising the recursion over paths with Numba. ``z`` is laid out
# ``(n_base, n_steps)`` so each path's innovations are contiguous. Antithetic
# mirroring happens in-kernel: paths ``i >= n_base`` consume ``-z[i - n_base]``
# (IEEE negation is exact), which avoids materialising the mirrored
# ``(n_paths, n_steps)`` copy that ``np.concatenate`` used to allocate per call.


@njit(parallel=True, cache=True, fastmath=True)
def _garch_q_terminal_log_garch(
    z: np.ndarray,  # (n_base, n_steps) innovations; mirrored for i >= n_base
    n_out: int,
    log_s0: float,
    r_step: float,
    omega: float,
    alpha: float,
    beta: float,
    h0: float,
) -> np.ndarray:
    """Symmetric GARCH: ``h' = omega + alpha*h*z^2 + beta*h``."""
    n_base = z.shape[0]
    n_steps = z.shape[1]
    out = np.empty(n_out)
    for i in prange(n_out):
        # np.int64 casts: the parfor index is unsigned, and mixing it with
        # signed n_base would promote `row` to float64 (NumPy promotion rule).
        if i >= n_base:
            mirror = True
            row = np.int64(i) - n_base
        else:
            mirror = False
            row = np.int64(i)
        log_s = log_s0
        h = h0
        for j in range(n_steps):
            zz = z[row, j]
            if mirror:
                zz = -zz
            if h < _VAR_FLOOR:
                h = _VAR_FLOOR
            sqrt_h = np.sqrt(h)
            log_s += r_step - 0.5 * h + sqrt_h * zz
            h = omega + alpha * h * zz * zz + beta * h
        out[i] = log_s
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _garch_q_terminal_log_ngarch(
    z: np.ndarray,
    n_out: int,
    log_s0: float,
    r_step: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    h0: float,
) -> np.ndarray:
    """Duan NGARCH: ``h' = omega + alpha*h*(z - gamma)^2 + beta*h``."""
    n_base = z.shape[0]
    n_steps = z.shape[1]
    out = np.empty(n_out)
    for i in prange(n_out):
        # np.int64 casts: the parfor index is unsigned, and mixing it with
        # signed n_base would promote `row` to float64 (NumPy promotion rule).
        if i >= n_base:
            mirror = True
            row = np.int64(i) - n_base
        else:
            mirror = False
            row = np.int64(i)
        log_s = log_s0
        h = h0
        for j in range(n_steps):
            zz = z[row, j]
            if mirror:
                zz = -zz
            if h < _VAR_FLOOR:
                h = _VAR_FLOOR
            sqrt_h = np.sqrt(h)
            log_s += r_step - 0.5 * h + sqrt_h * zz
            d = zz - gamma
            h = omega + alpha * h * d * d + beta * h
        out[i] = log_s
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _garch_q_terminal_log_gjr(
    z: np.ndarray,
    n_out: int,
    log_s0: float,
    r_step: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    h0: float,
) -> np.ndarray:
    """GJR: ``h' = omega + (alpha + gamma*1{z<0})*h*z^2 + beta*h``."""
    n_base = z.shape[0]
    n_steps = z.shape[1]
    out = np.empty(n_out)
    for i in prange(n_out):
        # np.int64 casts: the parfor index is unsigned, and mixing it with
        # signed n_base would promote `row` to float64 (NumPy promotion rule).
        if i >= n_base:
            mirror = True
            row = np.int64(i) - n_base
        else:
            mirror = False
            row = np.int64(i)
        log_s = log_s0
        h = h0
        for j in range(n_steps):
            zz = z[row, j]
            if mirror:
                zz = -zz
            if h < _VAR_FLOOR:
                h = _VAR_FLOOR
            sqrt_h = np.sqrt(h)
            log_s += r_step - 0.5 * h + sqrt_h * zz
            arch = alpha + gamma if zz < 0.0 else alpha
            h = omega + arch * h * zz * zz + beta * h
        out[i] = log_s
    return out


class GARCHRiskNeutralSimulator(BaseSimulator, StochasticVolatilityMixin):
    """Risk-neutral MC simulator for the nonaffine GARCH(1,1) family.

    Parameters
    ----------
    garch_type : str
        One of ``"garch"``, ``"ngarch"``, ``"gjr_garch"``.
    omega, alpha, beta : float
        Per-period variance-recursion coefficients (non-negative).
    gamma : float
        Risk-neutral leverage / asymmetry (``gamma*`` under Q). Ignored for
        plain ``"garch"`` (symmetric).
    h0 : float
        Initial conditional variance ``h_1`` (per period).
    steps_per_year : int
        Trading-day discretization (default 252).
    """

    def __init__(
        self,
        garch_type: str,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float = 0.0,
        h0: float = 1e-4,
        steps_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__()
        gt = str(garch_type).lower()
        if gt not in VALID_GARCH_TYPES:
            raise ValueError(
                f"garch_type must be one of {VALID_GARCH_TYPES}, got '{garch_type}'"
            )
        self.garch_type = gt
        self._variant_code = _VARIANT_CODES[gt]
        self._model_name = f"{gt.upper()} (risk-neutral MC)"
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.h0 = float(h0)
        self.steps_per_year = int(steps_per_year)

        if self.omega < 0 or self.alpha < 0 or self.beta < 0 or self.h0 < 0:
            raise ValueError("omega, alpha, beta, h0 must all be non-negative")
        if self.garch_type == "gjr_garch" and self.gamma < 0:
            raise ValueError(f"GJR gamma must be non-negative, got {self.gamma}")
        if self.steps_per_year <= 0:
            raise ValueError(
                f"steps_per_year must be positive, got {self.steps_per_year}"
            )

    @classmethod
    def from_physical_params(
        cls,
        garch_type: str,
        *,
        omega_annualised: float,
        alpha: float,
        beta: float,
        sigma0: float,
        gamma: float = 0.0,
        steps_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> GARCHRiskNeutralSimulator:
        """Build a risk-neutral simulator from a *physical-measure* GARCH fit.

        The physical GARCH family (``GARCHModel`` / ``NGARCHModel`` /
        ``GJRGARCHModel``) stores ω on the **annualised** variance scale and
        ``sigma0`` as the annualised initial vol, whereas the discrete daily Q
        recursion needs per-period units. This applies the scale conversion
        (``omega_per = omega_annualised / steps_per_year``,
        ``h0 = sigma0**2 / steps_per_year``) and the Duan (1995) LRNVR with unit
        market price of risk **λ = 0** (the fitted asymmetry γ is used directly as
        the risk-neutral γ* = γ + λ). α, β, γ are dimensionless and unchanged.
        """
        spy = int(steps_per_year)
        return cls(
            garch_type,
            omega=float(omega_annualised) / spy,
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            h0=float(sigma0) ** 2 / spy,
            steps_per_year=spy,
        )

    # ------------------------------------------------------------------ #
    # Variance recursion (used by the full-path simulator)
    # ------------------------------------------------------------------ #

    def _variance_update(self, h: np.ndarray, z: np.ndarray) -> np.ndarray:
        """One-step risk-neutral variance recursion for the chosen variant."""
        if self.garch_type == "ngarch":
            shock = self.alpha * h * (z - self.gamma) ** 2
        elif self.garch_type == "gjr_garch":
            arch = self.alpha + self.gamma * (z < 0.0)
            shock = arch * h * z * z
        else:  # garch (symmetric)
            shock = self.alpha * h * z * z
        return self.omega + shock + self.beta * h

    @staticmethod
    def _draw_noise(
        rng: np.random.Generator, n_paths: int, n_steps: int, antithetic: bool
    ) -> np.ndarray:
        """Innovations consumed by the terminal kernels.

        Shape is ``(n_paths // 2, n_steps)`` when antithetic mirroring applies
        (the kernels mirror in place) and ``(n_paths, n_steps)`` otherwise.
        """
        if antithetic and n_paths % 2 == 0:
            return rng.standard_normal((n_paths // 2, n_steps))
        return rng.standard_normal((n_paths, n_steps))

    def draw_terminal_noise(
        self,
        t: float,
        *,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Pre-draw the innovations :meth:`terminals` would consume for ``t``.

        The innovations are parameter-independent, so a caller evaluating many
        parameter sets under common random numbers (a calibration objective)
        can draw them once and replay them through ``terminals(..., noise=...)``
        — bit-identical to letting ``terminals`` draw from the same ``rng``.
        """
        return self._draw_noise(
            rng, int(n_paths), self._calendar_steps(float(t)), antithetic
        )

    def _simulate_log_terminal(
        self,
        log_s0: float,
        r_step: float,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
        antithetic: bool,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Risk-neutral GARCH recursion -> terminal log-prices (Numba kernel).

        Innovations are drawn here by the seeded ``rng`` (so common random
        numbers are preserved) unless pre-drawn ``noise`` is supplied; the
        sequential per-path recursion runs in the per-variant parallel Numba
        kernel (antithetic paths are mirrored in-kernel, no concatenated copy).
        """
        if noise is None:
            noise = self._draw_noise(rng, n_paths, n_steps, antithetic)
        else:
            valid_shapes = {(n_paths, n_steps)}
            if n_paths % 2 == 0:  # half layout mirrors in-kernel — even only
                valid_shapes.add((n_paths // 2, n_steps))
            if noise.shape not in valid_shapes:
                raise ValueError(
                    f"noise shape {noise.shape} does not match n_paths={n_paths}, "
                    f"n_steps={n_steps} (expected full or antithetic-half layout)"
                )
        z = np.ascontiguousarray(noise, dtype=np.float64)
        args = (z, int(n_paths), float(log_s0), float(r_step))
        if self._variant_code == 1:
            log_terminal = _garch_q_terminal_log_ngarch(
                *args, self.omega, self.alpha, self.beta, self.gamma, self.h0
            )
        elif self._variant_code == 2:
            log_terminal = _garch_q_terminal_log_gjr(
                *args, self.omega, self.alpha, self.beta, self.gamma, self.h0
            )
        else:
            log_terminal = _garch_q_terminal_log_garch(
                *args, self.omega, self.alpha, self.beta, self.h0
            )
        return np.asarray(log_terminal, dtype=float)

    # ------------------------------------------------------------------ #
    # Risk-neutral European pricers
    # ------------------------------------------------------------------ #

    def _calendar_steps(self, t: float) -> int:
        """Calendar-consistent step count ``N = round(t * steps_per_year)``."""
        return max(int(round(t * self.steps_per_year)), 1)

    def terminals(
        self,
        s0: float,
        r: float,
        t: float,
        *,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = True,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calendar-consistent terminal prices S(T) from a shared ``rng``.

        Passing one ``Generator`` across maturities/strikes gives surface-wide
        common random numbers (reproducible objective across optimizer steps).
        Pre-drawn ``noise`` (from :meth:`draw_terminal_noise`) skips the draw —
        the parameter-independent innovations can then be reused across
        objective evaluations without re-drawing them each time.
        """
        return np.asarray(
            np.exp(
                self._simulate_log_terminal(
                    np.log(s0),
                    r / self.steps_per_year,
                    int(n_paths),
                    self._calendar_steps(t),
                    rng,
                    antithetic,
                    noise=noise,
                )
            ),
            dtype=float,
        )

    def price_european_call(
        self,
        s0: float,
        strike: float,
        r: float,
        t: float,
        n_paths: int = 200_000,
        seed: int | None = None,
        antithetic: bool = True,
    ) -> float:
        """Discounted risk-neutral MC price of a European call (cross-check)."""
        rng = np.random.default_rng(seed)
        log_terminal = self._simulate_log_terminal(
            np.log(s0),
            r / self.steps_per_year,
            n_paths,
            self._calendar_steps(t),
            rng,
            antithetic,
        )
        payoff = np.maximum(np.exp(log_terminal) - strike, 0.0)
        return float(np.exp(-r * t) * np.mean(payoff))

    def price_strikes(
        self,
        s0: float,
        strikes: np.ndarray,
        r: float,
        t: float,
        *,
        n_paths: int = 50_000,
        seed: int | None = 0,
        antithetic: bool = True,
        is_call: bool = True,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Price a vector of strikes at a single maturity from one simulation.

        Terminals are simulated once and reused across strikes. Pass an explicit
        ``rng`` (instead of ``seed``) to share one random stream across maturities
        for surface-wide common random numbers.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        terminals = self.terminals(
            s0, r, t, n_paths=int(n_paths), rng=rng, antithetic=antithetic
        )
        discount = np.exp(-r * t)
        strikes = np.asarray(strikes, dtype=float)
        if is_call:
            payoffs = np.maximum(terminals[:, None] - strikes[None, :], 0.0)
        else:
            payoffs = np.maximum(strikes[None, :] - terminals[:, None], 0.0)
        return np.asarray(discount * payoffs.mean(axis=0), dtype=float)

    def price_surface(
        self,
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        *,
        n_paths: int = 50_000,
        seed: int = 0,
        antithetic: bool = True,
        is_call: bool = True,
    ) -> np.ndarray:
        """Price a strike x maturity grid; returns shape ``(n_maturities, n_strikes)``.

        A single ``Generator(seed)`` drives every maturity so the whole surface
        is reproducible across optimizer evaluations (common random numbers).
        """
        rng = np.random.default_rng(seed)
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        grid = np.empty((maturities.size, strikes.size), dtype=float)
        for i, t in enumerate(maturities):
            grid[i, :] = self.price_strikes(
                s0,
                strikes,
                r,
                float(t),
                n_paths=n_paths,
                antithetic=antithetic,
                is_call=is_call,
                rng=rng,
            )
        return grid

    def price_surface_call_put(
        self,
        s0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        *,
        n_paths: int = 50_000,
        seed: int = 0,
        antithetic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Call **and** put grids from one simulation per maturity.

        Bit-identical to two :meth:`price_surface` calls with the same
        ``seed`` (those re-simulate identical terminals, since calls and puts
        share the random stream) at half the Monte-Carlo cost: each
        maturity's terminals are simulated once and feed both payoffs.
        Returns ``(calls, puts)``, each shaped ``(n_maturities, n_strikes)``.
        """
        rng = np.random.default_rng(seed)
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        calls = np.empty((maturities.size, strikes.size), dtype=float)
        puts = np.empty((maturities.size, strikes.size), dtype=float)
        for i, t in enumerate(maturities):
            terminals = self.terminals(
                s0,
                r,
                float(t),
                n_paths=int(n_paths),
                rng=rng,
                antithetic=antithetic,
            )
            discount = np.exp(-r * float(t))
            calls[i, :] = discount * np.maximum(
                terminals[:, None] - strikes[None, :], 0.0
            ).mean(axis=0)
            puts[i, :] = discount * np.maximum(
                strikes[None, :] - terminals[:, None], 0.0
            ).mean(axis=0)
        return calls, puts

    # ------------------------------------------------------------------ #
    # BaseSimulator contract
    # ------------------------------------------------------------------ #

    def simulate_terminal(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Terminal prices S(T). ``mu`` is the annual drift (pass ``r`` for Q)."""
        self.validate_inputs(s0, mu, t, n_paths, n_steps)
        rng = np.random.default_rng(seed)
        log_terminal = self._simulate_log_terminal(
            np.log(s0), mu * t / n_steps, n_paths, n_steps, rng, antithetic=False
        )
        return np.asarray(np.exp(log_terminal), dtype=float)

    def simulate_paths(
        self,
        s0: float,
        mu: float,
        t: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> SimulationResult:
        """Full risk-neutral price + volatility paths."""
        self.validate_inputs(s0, mu, t, n_paths, n_steps)
        r_step = mu * t / n_steps
        rng = np.random.default_rng(seed)

        start = time.perf_counter()
        log_s = np.empty((n_paths, n_steps + 1), dtype=float)
        vol = np.empty((n_paths, n_steps + 1), dtype=float)
        log_s[:, 0] = np.log(s0)
        h = np.full(n_paths, self.h0, dtype=float)
        vol[:, 0] = np.sqrt(np.maximum(h, 0.0))
        for step in range(n_steps):
            z = rng.standard_normal(n_paths)
            h = np.maximum(h, _VAR_FLOOR)
            sqrt_h = np.sqrt(h)
            log_s[:, step + 1] = log_s[:, step] + r_step - 0.5 * h + sqrt_h * z
            h = self._variance_update(h, z)
            vol[:, step + 1] = np.sqrt(np.maximum(h, 0.0))
        elapsed = time.perf_counter() - start

        return SimulationResult(
            price_paths=np.exp(log_s),
            time_grid=np.linspace(0.0, t, n_steps + 1),
            model_name=self._model_name,
            computation_time=elapsed,
            n_paths=n_paths,
            n_steps=n_steps,
            volatility_paths=vol,
            parameters=self.get_parameters(),
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "h0": self.h0,
            "steps_per_year": self.steps_per_year,
        }

    # ------------------------------------------------------------------ #
    # StochasticVolatilityMixin / diagnostics
    # ------------------------------------------------------------------ #

    @property
    def persistence(self) -> float:
        """Variance persistence (< 1 ⇒ finite unconditional variance)."""
        if self.garch_type == "ngarch":
            return self.beta + self.alpha * (1.0 + self.gamma**2)
        if self.garch_type == "gjr_garch":
            return self.beta + self.alpha + 0.5 * self.gamma
        return self.beta + self.alpha

    def long_run_variance(self) -> float:
        """Per-period unconditional variance (``inf`` if non-stationary)."""
        gap = 1.0 - self.persistence
        if gap <= 0.0:
            return float("inf")
        return self.omega / gap
