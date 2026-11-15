"""Domain exceptions hierarchy for Open Logistics.

All domain exceptions inherit from the base `LogisticsError`.
"""

from typing import Any, Dict, Optional


class LogisticsError(Exception):
    """Base exception for all Open Logistics domain errors.

    Attributes:
        message: Human-readable error description.
        context: Optional dictionary containing additional diagnostic data.
    """

    def __init__(self, message: str = "", context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: Dict[str, Any] = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} (context: {self.context})"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception to a structured dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
        }


class RoutingError(LogisticsError):
    """Exception raised when routing, fleet dispatch, or schedule optimization fails."""


class InventoryError(LogisticsError):
    """Exception raised during inventory modeling, stock calculation, or replenishment."""


class StockoutError(InventoryError):
    """Exception raised when an inventory allocation or withdrawal causes an unauthorized stockout."""

    def __init__(
        self,
        sku: str,
        requested: int,
        available: int,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx.update({"sku": sku, "requested": requested, "available": available})
        msg = message or f"Stockout for SKU '{sku}': requested {requested}, but only {available} available."
        super().__init__(message=msg, context=ctx)
        self.sku = sku
        self.requested = requested
        self.available = available


class WarehouseError(LogisticsError):
    """Exception raised during warehouse slotting, wave batching, or task execution."""


class FSMStateError(WarehouseError):
    """Exception raised when an illegal order lifecycle state transition or invalid rollback is attempted."""

    def __init__(
        self,
        current_state: str,
        attempted_state: str,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx.update({"current_state": current_state, "attempted_state": attempted_state})
        msg = message or f"Illegal state transition from '{current_state}' to '{attempted_state}'."
        super().__init__(message=msg, context=ctx)
        self.current_state = current_state
        self.attempted_state = attempted_state


class InvalidTransitionError(FSMStateError):
    """Raised when an illegal order lifecycle state transition or invalid rollback is attempted."""


class CapacityExceededError(WarehouseError):
    """Raised when bin weight, volume, or layout capacity is exceeded."""


class ConfigurationError(LogisticsError):
    """Exception raised when configuration parameters are invalid, inconsistent, or missing."""


class ValidationError(LogisticsError):
    """Exception raised when domain entities or parameters fail schema validation."""
