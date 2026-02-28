"""
Instruments Module
==================

Financial contracts and payoffs for option pricing.

Quick Start
-----------
    from backend.instruments import (
        # Factory functions (recommended)
        EuropeanCall, EuropeanPut,
        AmericanCall, AmericanPut,
        BermudanCall, BermudanPut,

        # Option classes
        VanillaOption, DigitalOption,

        # Payoff classes
        VanillaCallPayoff, VanillaPutPayoff,
        DigitalCallPayoff, DigitalPutPayoff,
        CompositePayoff,

        # Strategies
        Straddle, Strangle, Butterfly,
        IronCondor, IronButterfly,
        CallSpread, PutSpread,
        StrategyLeg,
    )

Examples
--------
    # European call option
    call = EuropeanCall(strike=100, maturity=0.5)
    payoffs = call.payoff(np.array([90, 100, 110]))  # [0, 0, 10]

    # Iron Condor strategy
    ic = IronCondor(k1=85, k2=95, k3=105, k4=115, maturity=0.5)

    # Butterfly spread
    butterfly = Butterfly(k1=90, k2=100, k3=110, maturity=0.5)

Author: Thomas
Created: 2025
"""

# Payoffs
from backend.core.result_types import ExerciseStyle

# Exercise schedules
from backend.instruments.exercise import (
    AmericanExercise,
    BermudanExercise,
    EuropeanExercise,
    ExerciseType,  # Backward compatibility alias for ExerciseStyle
    create_exercise,
)

# Options
from backend.instruments.options import (
    AmericanCall,
    AmericanPut,
    # Exotic factories
    AsianCall,
    AsianGeometricCall,
    AsianGeometricPut,
    # Exotic options
    AsianOption,
    AsianPut,
    AssetOrNothingCall,
    AssetOrNothingOption,
    AssetOrNothingPut,
    BarrierDownInCall,
    BarrierDownInPut,
    BarrierDownOutCall,
    BarrierDownOutPut,
    BarrierOption,
    BarrierUpInCall,
    BarrierUpInPut,
    BarrierUpOutCall,
    BarrierUpOutPut,
    BermudanCall,
    BermudanPut,
    Chooser,
    ChooserOption,
    DigitalOption,
    # Convenience factories
    EuropeanCall,
    EuropeanPut,
    GapCall,
    GapOption,
    GapPut,
    LookbackCall,
    LookbackFixedCall,
    LookbackFixedPut,
    LookbackOption,
    LookbackPut,
    PowerCall,
    PowerOption,
    PowerPut,
    VanillaOption,
    # Generic factory
    create_vanilla_option,
)
from backend.instruments.payoffs import (
    # Exotic payoffs
    AsianCallPayoff,
    AsianPutPayoff,
    BarrierDownOutPutPayoff,
    BarrierUpOutCallPayoff,
    CompositePayoff,
    DigitalCallPayoff,
    DigitalPutPayoff,
    LookbackFloatingCallPayoff,
    LookbackFloatingPutPayoff,
    VanillaCallPayoff,
    VanillaPutPayoff,
)

# Strategies
from backend.instruments.strategies import (
    Butterfly,
    CallSpread,
    IronButterfly,
    IronCondor,
    OptionStrategy,
    PutSpread,
    Straddle,
    Strangle,
    StrategyLeg,
)

__all__ = [
    # Payoffs
    "VanillaCallPayoff",
    "VanillaPutPayoff",
    "DigitalCallPayoff",
    "DigitalPutPayoff",
    "CompositePayoff",
    # Exotic payoffs
    "AsianCallPayoff",
    "AsianPutPayoff",
    "BarrierUpOutCallPayoff",
    "BarrierDownOutPutPayoff",
    "LookbackFloatingCallPayoff",
    "LookbackFloatingPutPayoff",
    # Options
    "VanillaOption",
    "DigitalOption",
    # Generic factory
    "create_vanilla_option",
    # Convenience factories
    "EuropeanCall",
    "EuropeanPut",
    "AmericanCall",
    "AmericanPut",
    "BermudanCall",
    "BermudanPut",
    # Exotic options
    "AsianOption",
    "BarrierOption",
    "LookbackOption",
    "ChooserOption",
    "AssetOrNothingOption",
    "PowerOption",
    "GapOption",
    # Exotic factories
    "AsianCall",
    "AsianPut",
    "AsianGeometricCall",
    "AsianGeometricPut",
    "BarrierUpOutCall",
    "BarrierUpInCall",
    "BarrierDownOutCall",
    "BarrierDownInCall",
    "BarrierUpOutPut",
    "BarrierUpInPut",
    "BarrierDownOutPut",
    "BarrierDownInPut",
    "LookbackCall",
    "LookbackPut",
    "LookbackFixedCall",
    "LookbackFixedPut",
    "Chooser",
    "AssetOrNothingCall",
    "AssetOrNothingPut",
    "PowerCall",
    "PowerPut",
    "GapCall",
    "GapPut",
    # Strategies
    "StrategyLeg",
    "OptionStrategy",
    "Straddle",
    "Strangle",
    "Butterfly",
    "IronCondor",
    "IronButterfly",
    "CallSpread",
    "PutSpread",
    # Exercise schedules
    "ExerciseStyle",  # Primary enum
    "ExerciseType",   # Backward compatibility alias
    "EuropeanExercise",
    "AmericanExercise",
    "BermudanExercise",
    "create_exercise",
]


if __name__ == "__main__":
    import numpy as np

    print("=" * 50)
    print("Instruments Module Smoke Test")
    print("=" * 50)

    spots = np.array([90.0, 100.0, 110.0])

    # Test factory functions
    print("\n--- Factory Functions ---")
    euro_call = EuropeanCall(strike=100, maturity=0.5)
    euro_put = EuropeanPut(strike=100, maturity=0.5)
    amer_call = AmericanCall(strike=100, maturity=0.5)
    amer_put = AmericanPut(strike=100, maturity=0.5)

    print(f"European Call: {euro_call}")
    print(f"European Put: {euro_put}")
    print(f"American Call: {amer_call}")
    print(f"American Put: {amer_put}")

    # Test payoffs
    print("\n--- Payoffs ---")
    print(f"Euro Call payoffs at {spots}: {euro_call.payoff(spots)}")
    print(f"Euro Put payoffs at {spots}: {euro_put.payoff(spots)}")

    # Test strategies
    print("\n--- Strategies ---")
    straddle = Straddle(strike=100, maturity=0.5)
    ic = IronCondor(k1=85, k2=95, k3=105, k4=115, maturity=0.5)
    butterfly = Butterfly(k1=90, k2=100, k3=110, maturity=0.5)

    print(f"Straddle: {straddle}")
    print(f"Iron Condor: {ic}")
    print(f"Butterfly: {butterfly}")

    # Test interface compliance
    print("\n--- Interface Compliance ---")
    print(f"Euro Call is Instrument: {isinstance(euro_call, VanillaOption)}")
    print(f"Iron Condor is Instrument: {isinstance(ic, OptionStrategy)}")
    print(f"Iron Condor exercise: {ic.exercise_style}")
    print(f"Iron Condor maturity: {ic.maturity}")

    print("\n" + "=" * 50)
    print("Instruments module smoke test passed")
    print("=" * 50)
