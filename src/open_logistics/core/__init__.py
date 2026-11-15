"""Core package containing domain models, events, exceptions, and configuration."""

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

__all__ = [
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
