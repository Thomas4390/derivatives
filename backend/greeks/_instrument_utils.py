"""
Instrument utility functions for Greeks calculations.

Provides helpers to create modified copies of immutable instruments
(e.g., with bumped maturity for theta/charm/color/veta calculations).
"""

from typing import Optional

from backend.core.interfaces import Instrument


def create_decayed_instrument(instrument: Instrument, new_maturity: float) -> Optional[Instrument]:
    """
    Create a copy of the instrument with a different maturity.

    Handles all instrument types (vanilla + exotic). Returns None if the
    instrument type is not recognized.

    Parameters
    ----------
    instrument : Instrument
        Original instrument
    new_maturity : float
        New maturity value (must be > 0)

    Returns
    -------
    Instrument or None
        New instrument with modified maturity, or None if unsupported
    """
    from backend.instruments.options import (
        VanillaOption, BarrierOption, AsianOption, DigitalOption, LookbackOption
    )

    if isinstance(instrument, BarrierOption):
        return BarrierOption(
            strike=instrument.strike,
            barrier=instrument.barrier,
            maturity=new_maturity,
            is_call=instrument.is_call,
            is_up=instrument.is_up,
            is_knock_in=instrument.is_knock_in,
            rebate=instrument.rebate,
        )
    elif isinstance(instrument, AsianOption):
        return AsianOption(
            strike=instrument.strike,
            maturity=new_maturity,
            is_call=instrument.is_call,
            average_type=instrument.average_type,
        )
    elif isinstance(instrument, DigitalOption):
        return DigitalOption(
            strike=instrument.strike,
            maturity=new_maturity,
            is_call=instrument.is_call,
            payout=instrument.payout,
        )
    elif isinstance(instrument, LookbackOption):
        return LookbackOption(
            maturity=new_maturity,
            is_call=instrument.is_call,
            strike=instrument.strike,
            lookback_type=instrument.lookback_type,
        )
    elif isinstance(instrument, VanillaOption):
        return VanillaOption(
            strike=instrument.strike,
            maturity=new_maturity,
            is_call=instrument.is_call,
        )
    else:
        return None
