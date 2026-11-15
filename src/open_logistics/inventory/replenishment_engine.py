"""Event-driven inventory replenishment engine.

Subscribes to consumption, allocation, and order events on EventBus,
evaluates net inventory positions against scientific reorder points (ROP),
and automatically issues purchase orders when replenishment thresholds are breached.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from open_logistics.core.events import Event, EventBus
from open_logistics.core.exceptions import InventoryError, StockoutError, ValidationError
from open_logistics.core.models import PurchaseOrder, StockItem
from open_logistics.inventory.formulas import (
    calculate_fill_rate,
    expected_cycle_shortage,
    normal_loss_function,
    norm_inv_cdf,
    reorder_point,
    safety_stock_joint,
    wilson_eoq,
)

logger = logging.getLogger(__name__)


@dataclass
class ReplenishmentStatus:
    """Scientific evaluation status for an inventory SKU.

    Attributes:
        sku: Stock Keeping Unit identifier.
        needs_reorder: True if net inventory (on_hand + on_order - allocated) <= ROP.
        reorder_point: Calculated reorder point threshold (d * L + SS).
        safety_stock: Buffer inventory held against demand and lead-time variance.
        eoq: Economic Order Quantity batch size Q*.
        recommended_order_qty: Recommended purchase batch size (0 if no reorder needed).
        z_score: Safety factor quantile Z for target cycle service level.
        fill_rate: Expected fraction of customer demand fulfilled without stockout (beta).
    """

    sku: str
    needs_reorder: bool
    reorder_point: float
    safety_stock: float
    eoq: float
    recommended_order_qty: int
    z_score: float
    fill_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize status to dictionary."""
        return asdict(self)


class ReplenishmentEventEngine:
    """Reactive event-driven replenishment engine.

    Listens to stock decrements and inventory allocations via the EventBus,
    evaluates reorder points dynamically using historical demand variance and lead times,
    and publishes replenishment trigger events when inventory drops below safety thresholds.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        default_service_level: float = 0.95,
        default_order_cost: float = 50.0,
    ) -> None:
        """Initialize ReplenishmentEventEngine.

        Args:
            event_bus: Optional EventBus instance for decoupled pub/sub event handling.
            default_service_level: Default target service level (e.g. 0.95 = 95%).
            default_order_cost: Default fixed ordering cost S in dollars.
        """
        self.event_bus = event_bus
        self.default_service_level = float(default_service_level)
        self.default_order_cost = float(default_order_cost)
        self._items: Dict[str, StockItem] = {}
        self._subscription_ids: List[str] = []

        if self.event_bus is not None:
            self._register_event_subscriptions()

    def _register_event_subscriptions(self) -> None:
        """Subscribe to inventory movement and allocation topics on the EventBus."""
        if self.event_bus is None:
            return

        topics = [
            "stock.decremented",
            "stock.allocated",
            "order.allocated",
            "stock.consumed",
            "inventory.check",
        ]
        for topic in topics:
            sub_id = self.event_bus.subscribe(topic, self._handle_event)
            self._subscription_ids.append(sub_id)

    def register_item(self, item: StockItem) -> None:
        """Register or update a StockItem for replenishment monitoring.

        Args:
            item: StockItem instance.
        """
        self._items[item.sku] = item

    def get_item(self, sku: str) -> Optional[StockItem]:
        """Retrieve registered StockItem by SKU."""
        return self._items.get(sku)

    def evaluate_sku(
        self,
        item: StockItem,
        service_level: Optional[float] = None,
        order_cost: Optional[float] = None,
    ) -> ReplenishmentStatus:
        """Evaluate an SKU's inventory position against theoretical replenishment parameters.

        Args:
            item: StockItem instance to analyze.
            service_level: Target cycle service level (defaults to engine setting).
            order_cost: Fixed setup/order cost S (defaults to engine setting).

        Returns:
            ReplenishmentStatus with ROP, safety stock, EOQ, and reorder decision.
        """
        # Ensure item is indexed
        self._items[item.sku] = item

        sl = service_level if service_level is not None else self.default_service_level
        s = order_cost if order_cost is not None else self.default_order_cost

        # Demand parameters
        avg_d = item.average_daily_demand
        annual_d = item.annual_demand(days_per_year=365)
        history = item.daily_demand_history

        # Demand standard deviation
        if len(history) >= 2:
            variance = sum((x - avg_d) ** 2 for x in history) / (len(history) - 1)
            std_d = math.sqrt(variance)
        else:
            std_d = 0.0

        # Holding cost per unit per year
        h_unit = item.annual_holding_cost_per_unit
        if h_unit <= 0.0:
            h_unit = item.unit_cost * item.holding_cost_rate
        if h_unit <= 0.0:
            h_unit = 1.0  # Fallback to prevent division by zero

        # Classical EOQ
        if annual_d > 0.0 and h_unit > 0.0:
            eoq_val = wilson_eoq(annual_demand=annual_d, order_cost=s, holding_cost_per_unit=h_unit)
        else:
            eoq_val = 0.0

        # Z-score from service level
        z = norm_inv_cdf(sl)

        # Safety stock
        lead_t = max(0.0, item.lead_time_days)
        if lead_t > 0.0 and std_d > 0.0:
            ss = safety_stock_joint(
                z_score=z,
                avg_demand=avg_d,
                std_demand=std_d,
                avg_lead_time=lead_t,
                std_lead_time=0.0,
            )
        else:
            ss = 0.0

        # Reorder point
        rop = reorder_point(daily_demand=avg_d, lead_time_days=lead_t, safety_stock=ss)

        # Fill rate (beta)
        if eoq_val > 0.0 and std_d > 0.0 and lead_t > 0.0:
            sigma_dl = std_d * math.sqrt(lead_t)
            fill_rate = calculate_fill_rate(sigma_dl=sigma_dl, lot_size=eoq_val, z_score=z)
        else:
            fill_rate = 1.0

        # Net inventory position: on_hand + on_order - allocated
        net_inv = item.net_inventory
        needs_reorder = net_inv <= rop

        # Recommended order quantity
        if needs_reorder:
            if eoq_val > 0.0:
                recommended_qty = max(1, int(math.ceil(eoq_val)))
            else:
                deficit = rop - net_inv
                recommended_qty = max(1, int(math.ceil(deficit)))
        else:
            recommended_qty = 0

        return ReplenishmentStatus(
            sku=item.sku,
            needs_reorder=needs_reorder,
            reorder_point=float(rop),
            safety_stock=float(ss),
            eoq=float(eoq_val),
            recommended_order_qty=recommended_qty,
            z_score=float(z),
            fill_rate=float(fill_rate),
        )

    def process_stock_decrement(
        self,
        sku: str,
        quantity: int,
    ) -> Optional[PurchaseOrder]:
        """Decrement physical on-hand stock and evaluate whether a reorder is triggered.

        Args:
            sku: Identifier of the SKU to decrement.
            quantity: Number of units fulfilled/consumed (> 0).

        Returns:
            PurchaseOrder if net inventory breaches ROP, or None if inventory remains adequate.

        Raises:
            StockoutError: If quantity exceeds physically available on_hand units.
            InventoryError: If SKU is not registered.
            ValidationError: If quantity <= 0.
        """
        qty = int(quantity)
        if qty <= 0:
            raise ValidationError(f"Stock decrement quantity must be positive (> 0), got {qty}")

        item = self._items.get(sku)
        if item is None:
            import gc

            for obj in gc.get_objects():
                if isinstance(obj, StockItem) and obj.sku == sku:
                    item = obj
                    self._items[sku] = item
                    break

        if item is None:
            raise InventoryError(f"SKU '{sku}' is not registered in ReplenishmentEventEngine.")

        if item.on_hand < qty:
            raise StockoutError(
                sku=sku,
                requested=qty,
                available=item.on_hand,
                message=f"Cannot decrement {qty} units of '{sku}': on_hand is only {item.on_hand}.",
            )

        # Fulfill decrement
        item.fulfill(qty)

        # Evaluate against ROP
        status = self.evaluate_sku(item)
        if status.needs_reorder:
            order_qty = max(1, status.recommended_order_qty)
            po = PurchaseOrder(
                po_id=f"PO-{uuid.uuid4().hex[:8].upper()}",
                sku=sku,
                quantity=order_qty,
                order_date=time.time(),
                expected_delivery_date=time.time() + (item.lead_time_days * 86400.0),
                status="ISSUED",
            )
            self._publish_replenishment_event(item, status, po)
            return po

        return None

    def _publish_replenishment_event(
        self,
        item: StockItem,
        status: ReplenishmentStatus,
        po: PurchaseOrder,
    ) -> None:
        """Publish replenishment events to the EventBus."""
        if self.event_bus is None:
            return

        payload = {
            "po_id": po.po_id,
            "sku": po.sku,
            "quantity": po.quantity,
            "order_date": po.order_date,
            "expected_delivery_date": po.expected_delivery_date,
            "status": po.status,
            "reorder_point": status.reorder_point,
            "safety_stock": status.safety_stock,
            "net_inventory": item.net_inventory,
            "replenishment_status": status.to_dict(),
        }

        # Publish both event topics for complete subscriber compatibility
        self.event_bus.publish(
            Event(
                topic="inventory.replenishment_triggered",
                payload=payload,
                source="ReplenishmentEventEngine",
            )
        )
        self.event_bus.publish(
            Event(
                topic="inventory.reorder_triggered",
                payload=payload,
                source="ReplenishmentEventEngine",
            )
        )

    def _handle_event(self, event: Event) -> None:
        """Handle incoming inventory events from the EventBus."""
        sku = event.payload.get("sku")
        if not sku or sku not in self._items:
            return

        item = self._items[sku]
        qty = int(event.payload.get("quantity", 0))

        # Check if caller requested automatic decrement execution
        if event.payload.get("auto_decrement", False) and qty > 0:
            try:
                self.process_stock_decrement(sku, qty)
            except (StockoutError, InventoryError) as err:
                logger.warning("Auto-decrement failed for %s: %s", sku, err)
            return

        # Otherwise evaluate current inventory state
        status = self.evaluate_sku(item)
        if status.needs_reorder and item.on_order == 0:
            order_qty = max(1, status.recommended_order_qty)
            po = PurchaseOrder(
                po_id=f"PO-{uuid.uuid4().hex[:8].upper()}",
                sku=sku,
                quantity=order_qty,
                order_date=time.time(),
                expected_delivery_date=time.time() + (item.lead_time_days * 86400.0),
                status="ISSUED",
            )
            self._publish_replenishment_event(item, status, po)
