"""Unified ``LogisticsEngine`` facade.

One object that wires the routing, inventory, and warehouse sub-engines onto a
single shared :class:`~open_logistics.core.events.EventBus`, so integrators only
need to learn one entry point::

    from open_logistics.api import LogisticsEngine

    engine = LogisticsEngine()
    engine.event_bus.subscribe("inventory.replenishment_triggered", print)

    result = engine.dispatch_orders(orders, fleet, depot)
    statuses = engine.evaluate_inventory(stock_items)
    picks = engine.allocate_and_pick(orders)

The ``*_from_payload`` helpers accept plain JSON-compatible dicts and return
JSON-compatible dicts. They back both the REST server and the CLI, which keeps
the two transports behaviourally identical.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, TypeVar

from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import Event, EventBus
from open_logistics.core.exceptions import CapacityExceededError, InventoryError, ValidationError
from open_logistics.core.models import Depot, Order, PurchaseOrder, StockItem, Vehicle
from open_logistics.inventory.replenishment_engine import (
    ReplenishmentEventEngine,
    ReplenishmentStatus,
)
from open_logistics.routing.dispatch_engine import AutoDispatchEngine, DispatchResult
from open_logistics.warehouse.layout import WarehouseLayout
from open_logistics.warehouse.task_dispatcher import PickTask, WarehouseTaskDispatcher
from open_logistics.warehouse.wave_batcher import PickWave

__all__ = ["LogisticsEngine", "PickResult"]

T = TypeVar("T")


@dataclass
class PickResult:
    """Outcome of :meth:`LogisticsEngine.allocate_and_pick`.

    Attributes:
        waves: Pick waves created from the input orders.
        tasks: Route-sequenced pick tasks, keyed by wave ID.
    """

    waves: List[PickWave] = field(default_factory=list)
    tasks: Dict[str, List[PickTask]] = field(default_factory=dict)

    @property
    def wave(self) -> Optional[PickWave]:
        """The first wave, convenient when only one wave was produced."""
        return self.waves[0] if self.waves else None

    @property
    def all_tasks(self) -> List[PickTask]:
        """Every pick task across all waves, in wave order."""
        return [t for w in self.waves for t in self.tasks.get(w.id, [])]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "wave_count": len(self.waves),
            "task_count": len(self.all_tasks),
            "waves": [
                {
                    "id": w.id,
                    "status": w.status,
                    "order_ids": w.order_ids,
                    "total_weight": w.total_weight,
                    "total_volume": w.total_volume,
                    "carrier_cutoff": w.carrier_cutoff,
                    "tasks": [_task_to_dict(t) for t in self.tasks.get(w.id, [])],
                }
                for w in self.waves
            ],
        }


def _task_to_dict(task: PickTask) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "wave_id": task.wave_id,
        "order_id": task.order_id,
        "sku": task.sku,
        "quantity": task.quantity,
        "bin_id": task.bin_id,
        "location": list(task.location),
        "sequence": task.sequence,
        "status": task.status,
    }


def _require_instances(values: Any, cls: Type[T], name: str) -> List[T]:
    """Validate that ``values`` is a sequence whose elements are all ``cls``."""
    if values is None or isinstance(values, (str, bytes, dict)) or not isinstance(values, Iterable):
        raise ValidationError(f"'{name}' must be a list of {cls.__name__} objects")
    items = list(values)
    for idx, item in enumerate(items):
        if not isinstance(item, cls):
            raise ValidationError(
                f"'{name}[{idx}]' must be a {cls.__name__}, got {type(item).__name__}",
                context={"field": name, "index": idx},
            )
    return items


def _parse_list(payload: Dict[str, Any], key: str, factory: Callable[[Dict[str, Any]], T]) -> List[T]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValidationError(f"Payload field '{key}' must be a non-empty list", context={"field": key})
    out: List[T] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValidationError(f"'{key}[{idx}]' must be an object", context={"field": key, "index": idx})
        try:
            out.append(factory(entry))
        except ValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                f"Invalid '{key}[{idx}]': {exc!r}", context={"field": key, "index": idx}
            ) from exc
    return out


def _require_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("Request payload must be a non-empty JSON object")
    return payload


class LogisticsEngine:
    """Single entry point coordinating routing, inventory, and warehouse automation.

    Args:
        config: Shared configuration. Defaults to :class:`LogisticsConfig()`.
        event_bus: Bus to publish on. A new one is created if omitted; pass your
            own to bridge events into Kafka, webhooks, logs, etc.
        layout: Optional warehouse layout used for slot lookup during picking.

    Attributes:
        dispatcher: The :class:`AutoDispatchEngine` used for fleet routing.
        replenishment: The :class:`ReplenishmentEventEngine` tracking stock.
        warehouse: The :class:`WarehouseTaskDispatcher` producing pick waves.
    """

    def __init__(
        self,
        config: Optional[LogisticsConfig] = None,
        event_bus: Optional[EventBus] = None,
        layout: Optional[WarehouseLayout] = None,
    ) -> None:
        if config is not None and not isinstance(config, LogisticsConfig):
            raise ValidationError("config must be a LogisticsConfig instance")
        self.config: LogisticsConfig = config or LogisticsConfig()
        self.event_bus = event_bus if event_bus is not None else EventBus(
            isolated_errors=self.config.event_bus_isolated_errors
        )
        self.dispatcher = AutoDispatchEngine(event_bus=self.event_bus, config=self.config)
        self.replenishment = ReplenishmentEventEngine(
            event_bus=self.event_bus,
            default_service_level=self.config.default_service_level,
            default_order_cost=self.config.default_order_cost,
        )
        self.warehouse = WarehouseTaskDispatcher(layout=layout, event_bus=self.event_bus)
        # Serialises access from concurrent callers (e.g. the threaded HTTP server).
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ events
    def subscribe(self, topic_pattern: str, handler: Callable[[Event], Any]) -> str:
        """Shortcut for ``engine.event_bus.subscribe``. Supports ``*`` and ``#`` wildcards."""
        return self.event_bus.subscribe(topic_pattern, handler)

    # ----------------------------------------------------------------- routing
    def dispatch_orders(
        self,
        orders: Sequence[Order],
        fleet: Sequence[Vehicle],
        depot: Depot,
    ) -> DispatchResult:
        """Assign orders to vehicles and build optimised delivery routes.

        Raises:
            ValidationError: If inputs are not the expected model types.
        """
        order_list = _require_instances(orders, Order, "orders")
        fleet_list = _require_instances(fleet, Vehicle, "fleet")
        if not isinstance(depot, Depot):
            raise ValidationError(f"'depot' must be a Depot, got {type(depot).__name__}")
        with self._lock:
            return self.dispatcher.dispatch(orders=order_list, fleet=fleet_list, depot=depot)

    # Alias matching the underlying AutoDispatchEngine API.
    dispatch = dispatch_orders

    # --------------------------------------------------------------- inventory
    def evaluate_inventory(
        self,
        items: Sequence[StockItem],
        service_level: Optional[float] = None,
    ) -> List[ReplenishmentStatus]:
        """Register items for monitoring and compute ROP / safety stock / EOQ for each."""
        item_list = _require_instances(items, StockItem, "items")
        with self._lock:
            statuses: List[ReplenishmentStatus] = []
            for item in item_list:
                statuses.append(self.replenishment.evaluate_sku(item, service_level=service_level))
            return statuses

    def consume_stock(self, sku: str, quantity: int) -> Optional[PurchaseOrder]:
        """Record stock consumption; returns a PurchaseOrder if the reorder point is breached.

        The SKU must have been registered first (via :meth:`evaluate_inventory`).
        Publishes ``inventory.replenishment_triggered`` and
        ``inventory.reorder_triggered`` when a PO is raised.
        """
        with self._lock:
            if self.replenishment.get_item(sku) is None:
                raise InventoryError(f"SKU '{sku}' is not registered with this LogisticsEngine")
            return self.replenishment.process_stock_decrement(sku=sku, quantity=quantity)

    # --------------------------------------------------------------- warehouse
    def allocate_and_pick(
        self,
        orders: Sequence[Order | str],
        max_orders_per_wave: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> PickResult:
        """Batch orders into pick waves and generate route-sequenced pick tasks.

        ``orders`` may be :class:`Order` objects or plain order-ID strings.
        """
        if isinstance(orders, (str, bytes, dict)) or not isinstance(orders, Iterable):
            raise ValidationError("'orders' must be a list of Order objects or order IDs")
        order_list = list(orders)
        max_orders = self.config.max_wave_orders if max_orders_per_wave is None else max_orders_per_wave
        if isinstance(max_orders, bool) or not isinstance(max_orders, int) or max_orders <= 0:
            raise ValidationError("max_orders_per_wave must be a positive integer")
        routing = self.config.default_picking_strategy if strategy is None else strategy
        self.warehouse.validate_picking_strategy(routing)
        max_weight = self.config.max_wave_weight
        max_volume = self.config.max_wave_volume
        for idx, order in enumerate(order_list):
            if isinstance(order, Order):
                if order.weight > max_weight + 1e-9 or order.volume > max_volume + 1e-9:
                    raise CapacityExceededError(
                        f"Order '{order.id}' exceeds the configured pick wave capacity",
                        context={"order_id": order.id, "max_weight": max_weight, "max_volume": max_volume},
                    )
            elif not isinstance(order, str) or not order.strip():
                raise ValidationError(f"'orders[{idx}]' must be an Order or a non-empty order ID")

        with self._lock:
            waves = self.warehouse.create_waves(
                order_list,
                max_orders_per_wave=max_orders,
                max_wave_weight=max_weight,
                max_wave_volume=max_volume,
            )
            tasks = {
                w.id: self.warehouse.generate_picking_manifest(w, routing_strategy=routing)
                for w in waves
            }
        return PickResult(waves=waves, tasks=tasks)

    # ----------------------------------------------------- JSON payload helpers
    def dispatch_from_payload(self, payload: Any) -> Dict[str, Any]:
        """``{"orders": [...], "fleet": [...], "depot": {...}}`` -> dispatch result dict."""
        data = _require_payload(payload)
        orders = _parse_list(data, "orders", Order.from_dict)
        fleet = _parse_list(data, "fleet", Vehicle.from_dict)
        depot_raw = data.get("depot")
        if not isinstance(depot_raw, dict):
            raise ValidationError("Payload field 'depot' must be an object", context={"field": "depot"})
        try:
            depot = Depot.from_dict(depot_raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid 'depot': {exc!r}", context={"field": "depot"}) from exc
        return self.dispatch_orders(orders, fleet, depot).to_dict()

    def inventory_from_payload(self, payload: Any) -> Dict[str, Any]:
        """``{"items": [...], "service_level"?: float}`` -> replenishment statuses dict."""
        data = _require_payload(payload)
        items = _parse_list(data, "items", StockItem.from_dict)
        service_level = data.get("service_level")
        statuses = self.evaluate_inventory(items, service_level=service_level)
        return {
            "count": len(statuses),
            "reorder_count": sum(1 for s in statuses if s.needs_reorder),
            "items": [s.to_dict() for s in statuses],
        }

    def wave_from_payload(self, payload: Any) -> Dict[str, Any]:
        """``{"orders": [...]} or {"order_ids": [...]}`` (+ ``max_orders``, ``strategy``) -> waves dict."""
        data = _require_payload(payload)
        orders: Sequence[Order | str]
        if "orders" in data:
            orders = _parse_list(data, "orders", Order.from_dict)
        else:
            ids = data.get("order_ids")
            if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
                raise ValidationError(
                    "Payload must include 'orders' or a non-empty list of string 'order_ids'"
                )
            orders = list(ids)
        max_orders = data.get("max_orders")
        strategy = data.get("strategy")
        return self.allocate_and_pick(orders, max_orders_per_wave=max_orders, strategy=strategy).to_dict()
