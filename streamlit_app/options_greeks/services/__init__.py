"""Services module for Options Greeks Explorer."""

from .portfolio_calculator import (
    calculate_all_surfaces,
    get_portfolio_hash,
    prepare_portfolio_data,
)
from .risk_analyzer import analyze_portfolio_risk, check_unlimited_risk
from .state_manager import (
    add_position,
    clear_positions,
    get_positions,
    get_stock_position,
    init_session_state,
    set_stock_position,
)

__all__ = [
    # Portfolio Calculator
    "calculate_all_surfaces",
    "prepare_portfolio_data",
    "get_portfolio_hash",
    # Risk Analyzer
    "check_unlimited_risk",
    "analyze_portfolio_risk",
    # State Manager
    "init_session_state",
    "get_positions",
    "get_stock_position",
    "add_position",
    "clear_positions",
    "set_stock_position",
]
