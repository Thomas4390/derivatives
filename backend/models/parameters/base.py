"""
Base Parameter Classes
======================

Immutable parameter containers for financial models using frozen dataclasses.

Design Principles:
    - Immutability: All params are frozen (thread-safe, hashable)
    - Validation: Constraints checked at creation time
    - Single Source: One definition for each parameter set
    - Composition: Complex models compose simpler param objects

Author: Derivatives Pricing Project
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, TypeVar, Type

T = TypeVar('T', bound='BaseParams')


@dataclass(frozen=True)
class BaseParams(ABC):
    """
    Abstract base class for immutable model parameter containers.

    All parameter classes should inherit from this and implement
    the _validate() method to check parameter constraints.

    Features:
        - frozen=True ensures immutability (hashable, thread-safe)
        - Validation happens automatically via __post_init__
        - to_dict() / from_dict() for serialization

    Example
    -------
    @dataclass(frozen=True)
    class MyParams(BaseParams):
        alpha: float
        beta: float

        def _validate(self) -> None:
            if self.alpha < 0:
                raise ValueError("alpha must be non-negative")
    """

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate()

    @abstractmethod
    def _validate(self) -> None:
        """
        Validate parameter constraints.

        Raises
        ------
        ValueError
            If any parameter constraint is violated
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter names and values
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create parameter object from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing parameter values

        Returns
        -------
        BaseParams
            New parameter object
        """
        # Get field names for this class
        field_names = {f.name for f in fields(cls)}
        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    def __repr__(self) -> str:
        """Compact string representation."""
        params = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({params})"
