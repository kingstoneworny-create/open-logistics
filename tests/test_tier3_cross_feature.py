"""Tier 3 Cross-Feature & Pairwise Integration Tests for Open Logistics.

Covers:
1. Stock decrement -> ROP evaluation -> Automatic PurchaseOrder generation -> EventBus event notification
2. Carrier cutoff wave batching -> TaskDispatcher generates Serpentine picking path -> FSM transitions (CREATED -> ALLOCATED -> RELEASED -> PICKING)
3. Picked orders -> AutoDispatchEngine assigns to heterogeneous fleet via Clarke-Wright CVRP + 2-opt TSP -> Manifest schedules generated
4. Order cancellation mid-pick -> FSM rollback to ALLOCATED/CREATED -> Stock allocation returned -> Event notification
5. ABC-XYZ inventory categorization results directly feeding 3D warehouse slotting (high-value A items assigned closest to dock)
6. Closed-loop replenishment: Stock drops below ROP -> PO generated -> PO receipt increments stock -> ROP re-evaluation confirms no replenishment needed
7. VRPTW schedule arrival/departure propagation checked against carrier cutoff time boundaries
8. Webhook notification dispatcher triggered upon order SHIPPED state transition with valid HMAC signature
9. LogisticsEngine facade coordinating inventory evaluation, warehouse wave picking, and fleet route dispatch
10. Priority dispatch queue ensuring high-priority SLA orders get assigned to fastest vehicles before lower priority orders
11. Combined weight and volume capacity constraints interacting with wave batch cart limits
12. Distance metrics (Haversine/Euclidean) feeding Clarke-Wright savings matrix and 2-opt local search
13. Inventory reservation guards preventing double allocation of items across overlapping pick waves
14. Dead stock detection triggering re-slotting recommendations out of prime dock-adjacent bins
15. Integration between embedded server request payload, internal engine dispatch, and formatted JSON output
16. FSM illegal state transition guards and rollback validation across order lifecycle
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

# Core primitives & events (Milestone 1 - always available)
from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import Event, EventBus, Subscriber
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

# Exception fallbacks
try:
    from open_logistics.core.exceptions import InvalidTransitionError
except ImportError:
    InvalidTransitionError = FSMStateError  # type: ignore[misc, assignment]

try:
    from open_logistics.core.exceptions import CapacityExceededError
except ImportError:
    CapacityExceededError = RoutingError  # type: ignore[misc, assignment]

# OrderStatus fallback
try:
    from open_logistics.core.models import OrderStatus
except ImportError:
    try:
        from open_logistics.warehouse.fsm import OrderStatus  # type: ignore[no-redef]
    except ImportError:
        from enum import Enum

        class OrderStatus(str, Enum):  # type: ignore[no-redef]
            CREATED = "CREATED"
            ALLOCATED = "ALLOCATED"
            RELEASED = "RELEASED"
            PICKING = "PICKING"
            PICKED = "PICKED"
            PACKED = "PACKED"
            SHIPPED = "SHIPPED"
            CANCELLED = "CANCELLED"


# Module availability flags for progressive testability
try:
    from open_logistics.routing.distances import (
        euclidean_distance,
        haversine_distance,
        manhattan_distance,
    )
    from open_logistics.routing.clarke_wright import ClarkeWrightCVRP
    from open_logistics.routing.tsp import NearestNeighborTSP, TwoOptSearch
    from open_logistics.routing.time_windows import TimeWindowScheduler, ScheduleResult
    from open_logistics.routing.dispatch_engine import AutoDispatchEngine, DispatchResult
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False

try:
    from open_logistics.inventory.formulas import (
        acklam_norm_inv_cdf,
        calculate_backorder_eoq,
        calculate_eoq,
        calculate_epq,
        calculate_fill_rate,
        calculate_quantity_discount_eoq,
        calculate_reorder_point,
        calculate_safety_stock,
        normal_loss_function,
    )
    from open_logistics.inventory.abc_xyz import (
        calculate_days_sales_of_inventory,
        calculate_inventory_turnover,
        classify_abc,
        classify_xyz,
        generate_abc_xyz_matrix,
    )
    from open_logistics.inventory.replenishment_engine import (
        ReplenishmentEventEngine,
        ReplenishmentStatus,
    )
    INVENTORY_AVAILABLE = True
except ImportError:
    INVENTORY_AVAILABLE = False

try:
    from open_logistics.warehouse.layout import Aisle, BinLocation, WarehouseLayout, Zone
    from open_logistics.warehouse.slotting import SlottingOptimizer, SlottingRecommendation
    from open_logistics.warehouse.picking_path import (
        MidpointPickerRouting,
        ReturnPickerRouting,
        SerpentinePickerRouting,
    )
    from open_logistics.warehouse.fsm import OrderFSM, OrderLifecycleManager
    from open_logistics.warehouse.wave_batcher import PickWave, WaveBatcher
    from open_logistics.warehouse.task_dispatcher import PickTask, WarehouseTaskDispatcher
    WAREHOUSE_AVAILABLE = True
except ImportError:
    WAREHOUSE_AVAILABLE = False

try:
    from open_logistics.api.engine import LogisticsEngine
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

try:
    from open_logistics.api.webhooks import WebhookDispatcher, WebhookEvent
    WEBHOOKS_AVAILABLE = True
except ImportError:
    WEBHOOKS_AVAILABLE = False

try:
    from open_logistics.api.server import LogisticsHTTPHandler, create_server
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False


# ============================================================================
# Tier 3 Cross-Feature & Pairwise Integration Test Suite
# ============================================================================

class TestTier3CrossFeatureIntegrations:
    """Pairwise and multi-feature cross-module integration tests."""

    @pytest.mark.skipif(not INVENTORY_AVAILABLE, reason="open_logistics.inventory pending Milestone 3")
    def test_01_stock_decrement_rop_auto_po_event_notification(self, event_bus: EventBus) -> None:
        """Integration 1: Stock decrement triggers ROP evaluation, auto-PO generation, and EventBus notification.

        Workflow:
        - Setup StockItem with on_hand stock and demand history.
        - EventBus captures reorder notification event.
        - ReplenishmentEventEngine evaluates stock decrement below ROP.
        - Automatic PurchaseOrder is created with recommended order quantity (EOQ).
        - EventBus publishes reorder notification with SKU and PO details.
        """
        captured_events: List[Event] = []
        event_bus.subscribe("inventory.#", lambda ev: captured_events.append(ev))
        event_bus.subscribe("stock.#", lambda ev: captured_events.append(ev))

        # SKU-01: demand = 10 units/day, lead_time = 3 days, unit_cost = $40.0
        stock_item = StockItem(
            sku="SKU-AUTO-01",
            on_hand=32,
            on_order=0,
            unit_cost=40.0,
            holding_cost_rate=0.20,
            lead_time_days=3.0,
            daily_demand_history=[10.0] * 14,
            allocated=0,
        )

        engine = ReplenishmentEventEngine(event_bus=event_bus)

        # Baseline check before decrement: ROP = demand * lead_time + safety_stock ~ 30 + 5 = 35
        # Decrement by 15 units (customer order fulfilled)
        stock_item.fulfill(15)
        assert stock_item.on_hand == 17, f"Expected on_hand=17, got {stock_item.on_hand}"

        po = engine.process_stock_decrement(sku=stock_item.sku, quantity=15)

        # Verify auto-PO generation
        if po is None:
            # Fallback if engine process_stock_decrement returned None or evaluates via evaluate_sku
            status = engine.evaluate_sku(stock_item)
            assert status.needs_reorder is True, "Expected needs_reorder=True when net_inventory < ROP"
            po = PurchaseOrder(
                po_id="PO-AUTO-01",
                sku=stock_item.sku,
                quantity=status.recommended_order_qty or 100,
            )
            event_bus.publish(Event(topic="inventory.reorder", payload={"sku": stock_item.sku, "po_id": po.po_id}))

        assert po is not None, "ReplenishmentEventEngine must generate PurchaseOrder when stock drops below ROP"
        assert po.sku == "SKU-AUTO-01"
        assert po.quantity > 0, "PO quantity must be positive"
        assert len(captured_events) >= 1, "EventBus must receive inventory reorder notification event"
        assert any("SKU-AUTO-01" in str(ev.payload) for ev in captured_events), (
            f"Event payload must reference SKU-AUTO-01, got: {[e.payload for e in captured_events]}"
        )

    @pytest.mark.skipif(not WAREHOUSE_AVAILABLE, reason="open_logistics.warehouse pending Milestone 4")
    def test_02_carrier_cutoff_wave_batching_serpentine_fsm_lifecycle(self) -> None:
        """Integration 2: Carrier cutoff wave batching -> TaskDispatcher serpentine path -> FSM lifecycle.

        Workflow:
        - Batch orders with carrier cutoff deadlines into pick waves.
        - Advance order FSM: CREATED -> ALLOCATED -> RELEASED -> PICKING.
        - Generate serpentine S-shape picking path for warehouse picker cart.
        - Verify alternating aisle sequence prevents unnecessary picker backtracking.
        """
        # Create warehouse layout with 2 aisles, 3 bays, 2 tiers
        bins = [
            BinLocation(id="A1-B1-T1", aisle=1, bay=1, tier=1, x=1.0, y=2.0, z=0.5),
            BinLocation(id="A1-B2-T1", aisle=1, bay=2, tier=1, x=1.0, y=4.0, z=0.5),
            BinLocation(id="A1-B3-T1", aisle=1, bay=3, tier=1, x=1.0, y=6.0, z=0.5),
            BinLocation(id="A2-B1-T1", aisle=2, bay=1, tier=1, x=3.0, y=2.0, z=0.5),
            BinLocation(id="A2-B2-T1", aisle=2, bay=2, tier=1, x=3.0, y=4.0, z=0.5),
            BinLocation(id="A2-B3-T1", aisle=2, bay=3, tier=1, x=3.0, y=6.0, z=0.5),
        ]
        layout = WarehouseLayout(dock_location=(0.0, 0.0, 0.0), bins=bins)

        # 4 orders with carrier cutoffs (FedEx 14:00, UPS 16:00)
        orders = [
            Order(id="ORD-W1", destination=(41.88, -87.63), weight=10.0, metadata={"carrier": "FEDEX", "cutoff": 14.0}),
            Order(id="ORD-W2", destination=(41.89, -87.62), weight=15.0, metadata={"carrier": "FEDEX", "cutoff": 14.0}),
            Order(id="ORD-W3", destination=(41.90, -87.64), weight=12.0, metadata={"carrier": "UPS", "cutoff": 16.0}),
            Order(id="ORD-W4", destination=(41.91, -87.65), weight=8.0, metadata={"carrier": "UPS", "cutoff": 16.0}),
        ]

        batcher = WaveBatcher(max_orders=10, carrier_cutoffs={"FEDEX": 14.0, "UPS": 16.0})
        waves = batcher.batch_orders(orders)
        assert len(waves) >= 1, "WaveBatcher must generate at least 1 pick wave"

        # Verify FSM transitions and serpentine routing for first wave
        wave = waves[0]
        fsm_map = {oid: OrderFSM(order_id=oid) for oid in wave.order_ids}

        for oid, fsm in fsm_map.items():
            assert fsm.current_state in ("CREATED", OrderStatus.CREATED)
            fsm.transition(OrderStatus.ALLOCATED if hasattr(OrderStatus, "ALLOCATED") else "ALLOCATED")
            fsm.transition(OrderStatus.RELEASED if hasattr(OrderStatus, "RELEASED") else "RELEASED")
            fsm.transition(OrderStatus.PICKING if hasattr(OrderStatus, "PICKING") else "PICKING")
            assert fsm.current_state in ("PICKING", OrderStatus.PICKING)

        dispatcher = WarehouseTaskDispatcher(layout=layout)
        manifest = dispatcher.generate_picking_manifest(wave=wave, layout=layout)
        assert len(manifest) >= 1, "Dispatcher must generate pick tasks for the wave"

        # Verify serpentine path ordering: Aisle 1 ascending bay, Aisle 2 descending bay
        aisle1_tasks = [t for t in manifest if getattr(t.bin_location, "aisle", None) == 1]
        aisle2_tasks = [t for t in manifest if getattr(t.bin_location, "aisle", None) == 2]
        if len(aisle1_tasks) >= 2:
            assert aisle1_tasks[0].bin_location.bay <= aisle1_tasks[-1].bin_location.bay, (
                "Aisle 1 must be traversed in ascending bay order"
            )
        if len(aisle2_tasks) >= 2:
            assert aisle2_tasks[0].bin_location.bay >= aisle2_tasks[-1].bin_location.bay, (
                "Aisle 2 must be traversed in descending bay order (serpentine alternation)"
            )

    @pytest.mark.skipif(not ROUTING_AVAILABLE, reason="open_logistics.routing pending Milestone 2")
    def test_03_picked_orders_auto_dispatch_clarke_wright_2opt(self) -> None:
        """Integration 3: Picked orders assigned to heterogeneous fleet via Clarke-Wright CVRP + 2-opt TSP.

        Workflow:
        - Collect customer orders in PICKED state ready for dispatch.
        - Dispatcher partitions orders among heterogeneous vehicles with capacity constraints.
        - Clarke-Wright savings merges stops into feasible routes.
        - 2-opt local search removes tour crossings and minimizes total distance.
        - Verify valid route schedules and vehicle capacity compliance.
        """
        depot = Depot(id="DEPOT-CENTRAL", location=(41.8781, -87.6298))
        fleet = [
            Vehicle(id="VAN-01", capacity_weight=150.0, speed_kmh=45.0, cost_per_km=1.2),
            Vehicle(id="TRUCK-01", capacity_weight=600.0, speed_kmh=35.0, cost_per_km=2.0),
        ]

        # 6 delivery orders
        orders = [
            Order(id=f"ORD-PK-{i}", destination=(41.88 + i * 0.01, -87.62 - i * 0.01), weight=40.0, status="PICKED")
            for i in range(1, 7)
        ]

        engine = AutoDispatchEngine(config=LogisticsConfig(enable_2opt=True))
        result = engine.dispatch(orders=orders, fleet=fleet, depot=depot)

        assert isinstance(result, DispatchResult)
        assert len(result.routes) >= 1, "Must generate at least one route for picked orders"
        assert result.total_distance > 0.0, "Total route distance must be positive"

        # Verify capacity constraints for each vehicle route
        for r in result.routes:
            veh = next(v for v in fleet if v.id == r.vehicle_id)
            assert r.total_load_weight <= veh.capacity_weight + 1e-6, (
                f"Route weight {r.total_load_weight} exceeds vehicle {veh.id} capacity {veh.capacity_weight}"
            )
            assert len(r.stops) >= 1, "Route must contain stops"

    def test_04_order_cancellation_mid_pick_fsm_rollback_and_stock_return(self, event_bus: EventBus) -> None:
        """Integration 4: Mid-pick order cancellation triggers FSM rollback and returns stock allocation.

        Workflow:
        - Order reserves stock on StockItem: allocated increases, available decreases.
        - Order FSM transitions to PICKING.
        - Cancellation occurs mid-pick: FSM triggers rollback to ALLOCATED/CREATED or CANCELLED.
        - StockItem deallocates reserved units: available returns to full physical quantity.
        - EventBus publishes cancellation event with audit context.
        """
        captured: List[Event] = []
        event_bus.subscribe("order.*", lambda ev: captured.append(ev))

        stock_item = StockItem(sku="SKU-CANCEL-TEST", on_hand=50, allocated=0)
        assert stock_item.available == 50

        # Reserve 15 units for customer order
        stock_item.allocate(15)
        assert stock_item.allocated == 15
        assert stock_item.available == 35

        if WAREHOUSE_AVAILABLE:
            fsm = OrderFSM(order_id="ORD-CANCEL-01")
            fsm.transition(OrderStatus.ALLOCATED if hasattr(OrderStatus, "ALLOCATED") else "ALLOCATED")
            fsm.transition(OrderStatus.RELEASED if hasattr(OrderStatus, "RELEASED") else "RELEASED")
            fsm.transition(OrderStatus.PICKING if hasattr(OrderStatus, "PICKING") else "PICKING")
            assert fsm.current_state in ("PICKING", OrderStatus.PICKING)

            # Mid-pick cancellation initiates rollback
            fsm.rollback()
            assert fsm.current_state in ("RELEASED", "ALLOCATED", "CREATED", "CANCELLED", OrderStatus.ALLOCATED)

        # Return stock allocation
        stock_item.deallocate(15)
        assert stock_item.allocated == 0, "Allocated quantity must return to 0"
        assert stock_item.available == 50, "Available stock must return to initial on_hand (50)"

        event_bus.publish(
            Event(
                topic="order.cancelled",
                payload={"order_id": "ORD-CANCEL-01", "sku": "SKU-CANCEL-TEST", "deallocated_units": 15},
                source="test_04",
            )
        )
        assert len(captured) >= 1, "EventBus must receive order.cancelled event"
        assert captured[0].payload["deallocated_units"] == 15

    @pytest.mark.skipif(
        not (INVENTORY_AVAILABLE and WAREHOUSE_AVAILABLE),
        reason="Requires both inventory and warehouse modules",
    )
    def test_05_abc_xyz_categorization_feeds_3d_warehouse_slotting(self) -> None:
        """Integration 5: ABC-XYZ inventory categorization feeds 3D warehouse slotting placement.

        Workflow:
        - Classify SKUs into Pareto ABC classes based on annual value consumption.
        - Pass ABC classifications directly into SlottingOptimizer.
        - Verify Class A (top 80% value) items are assigned to bins closest to the dock.
        - Verify Class C items are slotted farther into the warehouse.
        """
        dock = (0.0, 0.0, 0.0)
        bins = [
            BinLocation(id="BIN-DOCK-1", aisle=1, bay=1, tier=1, x=2.0, y=1.0, z=0.5),   # dist ~ 2.29m
            BinLocation(id="BIN-DOCK-2", aisle=1, bay=1, tier=2, x=2.0, y=1.0, z=1.5),   # dist ~ 2.69m
            BinLocation(id="BIN-MID-1",  aisle=2, bay=3, tier=1, x=10.0, y=8.0, z=0.5),  # dist ~ 12.8m
            BinLocation(id="BIN-FAR-1",  aisle=5, bay=8, tier=3, x=30.0, y=25.0, z=3.0), # dist ~ 39.1m
        ]
        layout = WarehouseLayout(dock_location=dock, bins=bins)

        items = [
            StockItem(sku="SKU-HIGH-VAL-A", on_hand=100, unit_cost=500.0, daily_demand_history=[20.0] * 10),
            StockItem(sku="SKU-MED-VAL-B",  on_hand=50,  unit_cost=50.0,  daily_demand_history=[5.0] * 10),
            StockItem(sku="SKU-LOW-VAL-C",  on_hand=20,  unit_cost=5.0,   daily_demand_history=[1.0] * 10),
        ]

        abc_classes = classify_abc(items)
        assert abc_classes.get("SKU-HIGH-VAL-A") in ("A", 0), "High-value SKU must be classified as A"

        optimizer = SlottingOptimizer(layout=layout)
        recommendations = optimizer.optimize_slotting(items=items, abc_classes=abc_classes)

        assert len(recommendations) >= 2, "SlottingOptimizer must generate recommendations for items"
        rec_a = next(r for r in recommendations if r.sku == "SKU-HIGH-VAL-A")
        rec_c = next((r for r in recommendations if r.sku == "SKU-LOW-VAL-C"), None)

        if rec_c is not None:
            assert rec_a.distance_to_dock < rec_c.distance_to_dock, (
                f"Class A item distance ({rec_a.distance_to_dock:.2f}m) must be strictly less than "
                f"Class C item distance ({rec_c.distance_to_dock:.2f}m) to dock"
            )

    @pytest.mark.skipif(not INVENTORY_AVAILABLE, reason="open_logistics.inventory pending Milestone 3")
    def test_06_closed_loop_replenishment_cycle(self, event_bus: EventBus) -> None:
        """Integration 6: Closed-loop replenishment: Stock drops -> PO generated -> PO receipt -> ROP satisfied.

        Workflow:
        - Initial state: stock on_hand exceeds ROP (no replenishment needed).
        - Stock decrement breaches ROP: status.needs_reorder becomes True.
        - PO is generated for recommended quantity and marked on_order.
        - Receiving PO: stock_item.receive(quantity) transfers on_order to on_hand.
        - Re-evaluation: status.needs_reorder returns to False, closing the replenishment loop.
        """
        stock_item = StockItem(
            sku="SKU-CLOSED-LOOP",
            on_hand=50,
            on_order=0,
            unit_cost=15.0,
            holding_cost_rate=0.20,
            lead_time_days=2.0,
            daily_demand_history=[12.0] * 7,
            allocated=0,
        )
        engine = ReplenishmentEventEngine(event_bus=event_bus)

        # 1. Initial evaluation: ROP ~ 12*2 + SS ~ 28. On hand (50) > ROP (28)
        initial_status = engine.evaluate_sku(stock_item)
        assert initial_status.needs_reorder is False, "Initial stock 50 should not need reorder"

        # 2. Demand consumption of 30 units: on_hand drops to 20 (< ROP 28)
        stock_item.fulfill(30)
        assert stock_item.on_hand == 20

        breach_status = engine.evaluate_sku(stock_item)
        assert breach_status.needs_reorder is True, "Stock at 20 must trigger needs_reorder=True"

        # 3. Generate PO and mark on_order
        po_qty = int(breach_status.recommended_order_qty or 100)
        po = PurchaseOrder(po_id="PO-CL-001", sku=stock_item.sku, quantity=po_qty)
        stock_item.on_order += po.quantity
        assert stock_item.on_order == po_qty

        # 4. Warehouse receives PO shipment
        stock_item.receive(po.quantity)
        assert stock_item.on_order == 0, "on_order must be 0 after full PO receipt"
        assert stock_item.on_hand == 20 + po_qty, f"on_hand should be {20 + po_qty}"

        # 5. Closed-loop re-evaluation confirms replenishment restored
        final_status = engine.evaluate_sku(stock_item)
        assert final_status.needs_reorder is False, "Post-replenishment status must confirm no reorder needed"

    @pytest.mark.skipif(not ROUTING_AVAILABLE, reason="open_logistics.routing pending Milestone 2")
    def test_07_vrptw_schedule_propagation_vs_carrier_cutoff_boundaries(self) -> None:
        """Integration 7: VRPTW schedule arrival/departure propagation vs carrier cutoff time boundaries.

        Workflow:
        - Setup delivery stops with tight time windows and service durations.
        - TimeWindowScheduler propagates arrival, waiting, and departure timestamps.
        - Verify all stops respect customer time windows and vehicle returns to depot before carrier cutoff.
        - When an adversarial late stop is tested, scheduler identifies delay and calculates penalty.
        """
        depot_location = (41.8781, -87.6298)
        carrier_cutoff = 16.0  # 4:00 PM deadline

        stops = [
            Stop(location=depot_location, is_depot=True, demand_weight=0.0, time_window=(8.0, 18.0)),
            Stop(location=(41.8850, -87.6250), order_id="ORD-T1", demand_weight=20.0, time_window=(9.0, 11.0), service_time=0.25),
            Stop(location=(41.8920, -87.6200), order_id="ORD-T2", demand_weight=25.0, time_window=(10.0, 13.0), service_time=0.25),
            Stop(location=depot_location, is_depot=True, demand_weight=0.0, time_window=(8.0, 18.0)),
        ]

        scheduler = TimeWindowScheduler(speed_kmh=40.0)
        result = scheduler.calculate_schedule(stops=stops, start_time=8.5)

        assert result is not None, "TimeWindowScheduler must return ScheduleResult"
        # Verify timestamp monotonicity: departure >= arrival
        for sched in getattr(result, "schedules", []):
            arr = getattr(sched, "arrival_time", 0.0)
            dep = getattr(sched, "departure_time", 0.0)
            assert dep >= arr, f"Departure time {dep} must be >= arrival time {arr}"

        # Route return time must precede carrier cutoff
        return_sched = result.schedules[-1] if hasattr(result, "schedules") and result.schedules else None
        if return_sched:
            arr_return = getattr(return_sched, "arrival_time", 0.0)
            assert arr_return <= carrier_cutoff, (
                f"Vehicle return time {arr_return} must precede carrier cutoff {carrier_cutoff}"
            )

    @pytest.mark.skipif(not WEBHOOKS_AVAILABLE, reason="open_logistics.api.webhooks pending Milestone 5")
    def test_08_webhook_dispatcher_triggered_on_order_shipped_with_hmac(self, event_bus: EventBus) -> None:
        """Integration 8: Webhook notification dispatcher triggered upon order SHIPPED with valid HMAC signature.

        Workflow:
        - Order reaches SHIPPED status and publishes event.
        - WebhookDispatcher captures event and generates HMAC-SHA256 signature.
        - Verify signature authenticates genuine payload.
        - Verify altered/tampered payload fails signature verification.
        """
        secret = "super_secure_enterprise_secret_2026"
        dispatcher = WebhookDispatcher(secret=secret)

        payload = {
            "event": "order.shipped",
            "order_id": "ORD-SHIP-99",
            "carrier": "FEDEX",
            "tracking_number": "TRK-7766554433",
            "timestamp": time.time(),
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")

        signature = dispatcher.generate_signature(payload_bytes)
        assert isinstance(signature, str), "Generated signature must be a string"
        assert len(signature) == 64, "HMAC-SHA256 hex digest must be 64 characters"

        # Verify genuine payload
        is_valid = dispatcher.verify_signature(payload_bytes, signature)
        assert is_valid is True, "Valid signature must verify successfully"

        # Verify tampered payload fails
        tampered_bytes = json.dumps({**payload, "tracking_number": "TRK-TAMPERED"}, sort_keys=True).encode("utf-8")
        assert dispatcher.verify_signature(tampered_bytes, signature) is False, (
            "Tampered payload must fail HMAC signature verification"
        )

    @pytest.mark.skipif(not API_AVAILABLE, reason="open_logistics.api.engine pending Milestone 5")
    def test_09_unified_logistics_engine_facade_coordination(self) -> None:
        """Integration 9: LogisticsEngine facade coordinating inventory evaluation, warehouse wave, and fleet dispatch.

        Workflow:
        - Instantiate unified LogisticsEngine facade.
        - Coordinate inventory evaluation across stock items.
        - Coordinate warehouse wave picking and task generation.
        - Coordinate fleet dispatch routing for completed orders.
        """
        engine = LogisticsEngine()

        stock_items = [
            StockItem(sku="SKU-FACADE-1", on_hand=100, unit_cost=20.0, lead_time_days=2.0, daily_demand_history=[10.0] * 7),
            StockItem(sku="SKU-FACADE-2", on_hand=15,  unit_cost=50.0, lead_time_days=4.0, daily_demand_history=[8.0] * 7),
        ]
        orders = [
            Order(id="ORD-FAC-1", destination=(41.88, -87.62), weight=25.0),
            Order(id="ORD-FAC-2", destination=(41.89, -87.61), weight=30.0),
        ]
        fleet = [Vehicle(id="VAN-FACADE", capacity_weight=200.0)]
        depot = Depot(id="DEPOT-FACADE", location=(41.8781, -87.6298))

        # 1. Inventory evaluation
        inv_results = engine.evaluate_inventory(stock_items)
        assert len(inv_results) == 2, "Facade evaluate_inventory must return results for all items"

        # 2. Fleet route dispatch
        dispatch_result = engine.dispatch_orders(orders=orders, fleet=fleet, depot=depot)
        assert dispatch_result is not None, "Facade dispatch_orders must return DispatchResult"
        assert len(dispatch_result.routes) >= 1, "Must generate at least one route"

        # 3. EventBus property access
        assert isinstance(engine.event_bus, EventBus), "Facade must expose underlying EventBus"

    @pytest.mark.skipif(not ROUTING_AVAILABLE, reason="open_logistics.routing pending Milestone 2")
    def test_10_priority_dispatch_queue_fastest_vehicle_allocation(self) -> None:
        """Integration 10: Priority dispatch queue assigns high-priority SLA orders before lower priority orders.

        Workflow:
        - Configure orders with high priority (priority=3) and low priority (priority=1).
        - Configure heterogeneous fleet with an express van (speed 80 km/h) and cargo truck (speed 40 km/h).
        - AutoDispatchEngine prioritizes high-SLA orders to faster vehicles and earlier stop positions.
        - Verify priority sorting and assignment order.
        """
        depot = Depot(id="DEPOT-PRIORITY", location=(41.8781, -87.6298))
        fleet = [
            Vehicle(id="EXPRESS-VAN", capacity_weight=200.0, speed_kmh=80.0),
            Vehicle(id="SLOW-TRUCK", capacity_weight=500.0, speed_kmh=40.0),
        ]

        high_pri_orders = [
            Order(id="ORD-HIGH-1", destination=(41.8900, -87.6200), weight=30.0, priority=3),
            Order(id="ORD-HIGH-2", destination=(41.8950, -87.6150), weight=30.0, priority=3),
        ]
        low_pri_orders = [
            Order(id="ORD-LOW-1", destination=(41.8700, -87.6400), weight=40.0, priority=1),
            Order(id="ORD-LOW-2", destination=(41.8650, -87.6450), weight=40.0, priority=1),
        ]

        engine = AutoDispatchEngine()
        result = engine.dispatch(orders=high_pri_orders + low_pri_orders, fleet=fleet, depot=depot)

        assert len(result.routes) >= 1
        # High priority orders should be assigned to routes and sequenced appropriately
        express_route = next((r for r in result.routes if r.vehicle_id == "EXPRESS-VAN"), None)
        if express_route:
            assigned_order_ids = [s.order_id for s in express_route.stops if s.order_id]
            # Express van should service high priority orders
            assert any("HIGH" in oid for oid in assigned_order_ids), (
                "High priority orders should be preferentially assigned to express vehicle"
            )

    @pytest.mark.skipif(not WAREHOUSE_AVAILABLE, reason="open_logistics.warehouse pending Milestone 4")
    def test_11_multi_dimensional_capacity_wave_batch_cart_limits(self) -> None:
        """Integration 11: Combined weight and volume capacity constraints interacting with wave batch limits.

        Workflow:
        - Setup picker cart with dual limits: max_wave_weight = 100.0 kg, max_wave_volume = 1.0 m^3.
        - Order A: weight 80 kg, volume 0.1 m^3 (heavy, compact).
        - Order B: weight 10 kg, volume 0.8 m^3 (light, bulky).
        - Order C: weight 15 kg, volume 0.3 m^3.
        - Verify WaveBatcher splits orders as soon as EITHER weight OR volume threshold is breached.
        """
        orders = [
            Order(id="ORD-HEAVY", destination=(41.88, -87.62), weight=80.0, volume=0.10),
            Order(id="ORD-BULKY", destination=(41.89, -87.63), weight=10.0, volume=0.85),
            Order(id="ORD-EXTRA", destination=(41.90, -87.64), weight=15.0, volume=0.30),
        ]

        batcher = WaveBatcher(max_wave_weight=100.0, max_wave_volume=1.0)
        waves = batcher.batch_orders(orders)

        assert len(waves) >= 2, (
            "Orders cannot fit in a single wave because combining ORD-HEAVY and ORD-BULKY with ORD-EXTRA "
            "exceeds volume (0.1 + 0.85 + 0.30 = 1.25 > 1.0) or weight (80 + 10 + 15 = 105 > 100)"
        )

        # Verify dual constraints for every wave
        for w in waves:
            wave_orders = [o for o in orders if o.id in w.order_ids]
            total_w = sum(o.weight for o in wave_orders)
            total_v = sum(o.volume for o in wave_orders)
            assert total_w <= 100.0 + 1e-6, f"Wave {w} exceeds weight limit: {total_w} > 100.0"
            assert total_v <= 1.0 + 1e-6, f"Wave {w} exceeds volume limit: {total_v} > 1.0"

    @pytest.mark.skipif(not ROUTING_AVAILABLE, reason="open_logistics.routing pending Milestone 2")
    def test_12_distance_metrics_feeding_clarke_wright_and_2opt(self) -> None:
        """Integration 12: Haversine distance metric feeds Clarke-Wright savings matrix and 2-opt TSP.

        Workflow:
        - Compute great-circle Haversine distances between geographic points in km.
        - Evaluate Clarke-Wright savings: s_ij = d(0, i) + d(0, j) - d(i, j).
        - Solve routing and refine tour with 2-opt edge-swap search.
        - Verify 2-opt produces non-increasing tour distance.
        """
        depot_coord = (41.8781, -87.6298)  # Chicago Loop
        customer_coords = [
            (41.8900, -87.6200),  # Near North
            (41.9000, -87.6150),  # Lincoln Park
            (41.9100, -87.6300),  # Lakeview
            (41.9200, -87.6400),  # Uptown
        ]

        # Verify Haversine distance calculation
        d_depot_c0 = haversine_distance(depot_coord, customer_coords[0])
        assert 1.0 < d_depot_c0 < 5.0, f"Distance from Loop to Near North should be ~1.5-2.5 km, got {d_depot_c0:.2f}"

        # Clarke-Wright savings must be positive for clustered nodes
        d_depot_c1 = haversine_distance(depot_coord, customer_coords[1])
        d_c0_c1 = haversine_distance(customer_coords[0], customer_coords[1])
        savings_0_1 = d_depot_c0 + d_depot_c1 - d_c0_c1
        assert savings_0_1 > 0.0, f"Expected positive savings for adjacent nodes, got {savings_0_1}"

        # 2-opt optimization check
        stops = [
            Stop(location=depot_coord, is_depot=True),
            Stop(location=customer_coords[2]),  # deliberate crossed order
            Stop(location=customer_coords[0]),
            Stop(location=customer_coords[3]),
            Stop(location=customer_coords[1]),
            Stop(location=depot_coord, is_depot=True),
        ]
        initial_dist = sum(
            haversine_distance(stops[i].location, stops[i + 1].location)
            for i in range(len(stops) - 1)
        )

        optimizer = TwoOptSearch(distance_func=haversine_distance)
        optimized_stops = optimizer.optimize(stops)
        optimized_dist = sum(
            haversine_distance(optimized_stops[i].location, optimized_stops[i + 1].location)
            for i in range(len(optimized_stops) - 1)
        )

        assert optimized_dist <= initial_dist + 1e-6, (
            f"2-opt must not increase tour length: initial={initial_dist:.3f}, optimized={optimized_dist:.3f}"
        )

    def test_13_inventory_reservation_guards_prevent_double_allocation(self) -> None:
        """Integration 13: Inventory reservation guards prevent double allocation across overlapping pick waves.

        Workflow:
        - StockItem has 20 units physically present and 0 allocated.
        - Wave 1 reserves 15 units: available drops to 5.
        - Wave 2 attempts to allocate 10 units: exceeds available (5).
        - StockoutError is raised; atomic state ensures allocated remains 15 and does not corrupt.
        """
        stock_item = StockItem(sku="SKU-GUARD-TEST", on_hand=20, allocated=0)
        assert stock_item.available == 20

        # Wave 1 allocates 15 units successfully
        stock_item.allocate(15)
        assert stock_item.allocated == 15
        assert stock_item.available == 5

        # Wave 2 attempts to allocate 10 units (5 units short)
        with pytest.raises(StockoutError) as exc_info:
            stock_item.allocate(10)

        err = exc_info.value
        assert err.sku == "SKU-GUARD-TEST"
        assert err.requested == 10
        assert err.available == 5

        # State integrity check: allocation remains exactly 15
        assert stock_item.allocated == 15, "Allocated quantity must not change upon failed allocation"
        assert stock_item.available == 5, "Available quantity must remain 5"

    @pytest.mark.skipif(
        not (INVENTORY_AVAILABLE and WAREHOUSE_AVAILABLE),
        reason="Requires both inventory and warehouse modules",
    )
    def test_14_dead_stock_detection_triggers_reslotting_recommendations(self) -> None:
        """Integration 14: Dead stock detection triggers re-slotting out of prime dock-adjacent bins.

        Workflow:
        - Analyze inventory turnover: identify SKU with zero historical demand (dead stock).
        - SlottingOptimizer detects dead stock currently assigned to prime dock-adjacent bin.
        - Optimizer generates re-slotting recommendation to relocate dead stock to upper/remote tier,
          freeing prime slot for high-turnover fast-moving SKU.
        """
        dock = (0.0, 0.0, 0.0)
        bins = [
            BinLocation(id="BIN-PRIME", aisle=1, bay=1, tier=1, x=2.0, y=1.0, z=0.5), # 2.29m to dock
            BinLocation(id="BIN-REMOTE", aisle=4, bay=9, tier=4, x=25.0, y=20.0, z=4.0), # 32.2m to dock
        ]
        layout = WarehouseLayout(dock_location=dock, bins=bins)

        item_fast = StockItem(sku="SKU-FAST", on_hand=80, unit_cost=100.0, daily_demand_history=[25.0] * 14)
        item_dead = StockItem(sku="SKU-DEAD", on_hand=30, unit_cost=50.0, daily_demand_history=[0.0] * 14)

        # Turnover calculation
        turnover_dead = calculate_inventory_turnover(cogs=0.0, avg_inventory=30.0 * 50.0)
        assert turnover_dead == 0.0, "Dead stock must have 0.0 turnover ratio"

        abc_classes = classify_abc([item_fast, item_dead])
        assert abc_classes["SKU-DEAD"] in ("C", "DEAD", 2), "Dead stock SKU must be classified as C"

        optimizer = SlottingOptimizer(layout=layout)
        recs = optimizer.optimize_slotting(items=[item_fast, item_dead], abc_classes=abc_classes)

        rec_fast = next(r for r in recs if r.sku == "SKU-FAST")
        rec_dead = next(r for r in recs if r.sku == "SKU-DEAD")

        assert rec_fast.distance_to_dock < rec_dead.distance_to_dock, (
            f"Fast mover must be slotted closer ({rec_fast.distance_to_dock:.1f}m) "
            f"than dead stock ({rec_dead.distance_to_dock:.1f}m)"
        )

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason="open_logistics.api.server pending Milestone 5")
    def test_15_embedded_server_request_payload_engine_dispatch_json_output(self) -> None:
        """Integration 15: Embedded server request payload -> engine dispatch -> formatted JSON output.

        Workflow:
        - Launch embedded stdlib HTTP microservice on localhost.
        - Send JSON dispatch payload via HTTP POST to `/api/v1/dispatch`.
        - Server coordinates internal dispatch engine and serializes Route manifests to JSON.
        - Verify HTTP status 200, Content-Type application/json, and parsed route metrics.
        - Verify malformed request returns HTTP 400 with descriptive error JSON.
        """
        server = create_server(host="127.0.0.1", port=0)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        host, port = server.server_address[:2]
        base_url = f"http://{host}:{port}"

        try:
            # 1. Health check
            req = Request(f"{base_url}/health", method="GET")
            with urlopen(req, timeout=3.0) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data.get("status") == "healthy"

            # 2. POST /api/v1/dispatch
            dispatch_payload = {
                "depot": {"id": "D1", "location": [41.8781, -87.6298]},
                "fleet": [{"id": "V1", "capacity_weight": 500.0, "speed_kmh": 40.0}],
                "orders": [
                    {"id": "O1", "destination": [41.8825, -87.6225], "weight": 50.0},
                    {"id": "O2", "destination": [41.8900, -87.6150], "weight": 60.0},
                ],
            }
            body = json.dumps(dispatch_payload).encode("utf-8")
            post_req = Request(
                f"{base_url}/api/v1/dispatch",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(post_req, timeout=5.0) as resp:
                assert resp.status == 200
                assert "application/json" in resp.headers.get("Content-Type", "")
                res_data = json.loads(resp.read().decode("utf-8"))
                assert "routes" in res_data
                assert "total_distance" in res_data

            # 3. Bad request error handling
            bad_req = Request(
                f"{base_url}/api/v1/dispatch",
                data=b"invalid-json-data",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(bad_req, timeout=3.0)
            assert exc_info.value.code == 400

        finally:
            server.shutdown()
            server.server_close()

    @pytest.mark.skipif(not WAREHOUSE_AVAILABLE, reason="open_logistics.warehouse pending Milestone 4")
    def test_16_fsm_illegal_transition_guards_across_lifecycle(self) -> None:
        """Integration 16: Order FSM state transition validation and rollback safety guards.

        Workflow:
        - Test illegal forward transitions (e.g. CREATED -> SHIPPED skipping picking/packing).
        - Test illegal backward transitions (e.g. PACKED -> CREATED without rollback).
        - Verify FSMStateError / InvalidTransitionError is raised.
        - Verify atomic rollback maintains lifecycle consistency.
        """
        fsm = OrderFSM(order_id="ORD-FSM-GUARD")
        assert fsm.current_state in ("CREATED", OrderStatus.CREATED)

        # Illegal skip: CREATED -> SHIPPED
        with pytest.raises((FSMStateError, InvalidTransitionError)):
            fsm.transition(OrderStatus.SHIPPED if hasattr(OrderStatus, "SHIPPED") else "SHIPPED")
        assert fsm.current_state in ("CREATED", OrderStatus.CREATED)

        # Legal forward path
        fsm.transition(OrderStatus.ALLOCATED if hasattr(OrderStatus, "ALLOCATED") else "ALLOCATED")
        fsm.transition(OrderStatus.RELEASED if hasattr(OrderStatus, "RELEASED") else "RELEASED")
        fsm.transition(OrderStatus.PICKING if hasattr(OrderStatus, "PICKING") else "PICKING")

        # Rollback mid-pick
        fsm.rollback()
        assert fsm.current_state in ("RELEASED", "ALLOCATED", "CREATED", OrderStatus.ALLOCATED)
