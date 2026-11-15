"""Open Logistics: Lightweight, zero-heavy-dependency Python logistics framework.

Provides modular enterprise-ready algorithms and workflows for fleet routing,
scientific inventory optimization, warehouse automation, and event-driven dispatch.
"""

from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import Event, EventBus, Subscriber, topic_matches
from open_logistics.core.exceptions import (
    ConfigurationError,
    FSMStateError,
    InventoryError,
    LogisticsError,
    RoutingError,
    StockoutError,
    ValidationError,
    WarehouseError,
)
from open_logistics.core.models import (
    Depot,
    Dimensions,
    Location,
    Order,
    OrderItem,
    PurchaseOrder,
    Route,
    StockItem,
    Stop,
    Vehicle,
)
from open_logistics.api import LogisticsEngine, PickResult

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LogisticsEngine",
    "PickResult",
    "LogisticsConfig",
    "Event",
    "EventBus",
    "Subscriber",
    "topic_matches",
    "LogisticsError",
    "RoutingError",
    "InventoryError",
    "StockoutError",
    "WarehouseError",
    "FSMStateError",
    "ConfigurationError",
    "ValidationError",
    "Location",
    "Dimensions",
    "OrderItem",
    "Order",
    "Vehicle",
    "Depot",
    "Stop",
    "Route",
    "StockItem",
    "PurchaseOrder",
]
