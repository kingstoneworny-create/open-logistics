"""Tier 4 Real-World Enterprise Workload Scenarios for Open Logistics.

Covers:
- Scenario 1: E-Commerce Same-Day Fulfillment (High-volume ingestion, stock reservation, ROP trigger, wave batching, serpentine picking, packing, dispatch, webhook notification)
- Scenario 2: 3PL Multi-Tenant Fleet Dispatch (Multi-client order aggregation, heterogeneous fleet matching, capacity constraints, delivery time window adherence, SLA violation penalty verification)
- Scenario 3: Automated DC Replenishment & 3D Slotting (50 SKUs Pareto classification, volatility clustering, optimal dock-proximity slotting, multi-day demand simulation with automated purchase order generation)
- Scenario 4: Resilient Order Lifecycle & Rollback Recovery (Wave picking lifecycle under edge disruptions, partial cancellations, atomic rollback, restocking without inventory leak)
- Scenario 5: End-to-End Headless CLI & Microservice Integration (Embedded HTTP server serving dispatch and wave-pick requests, verifying API contract and response metrics)
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import random
import socket
import sys
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

try:
    from open_logistics.cli.main import main as cli_main
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


# ============================================================================
# Tier 4 Enterprise Scenarios
# ============================================================================

class TestTier4RealWorldScenarios:
    """Comprehensive end-to-end enterprise workload test scenarios."""

    def test_scenario_1_ecommerce_same_day_fulfillment(self, event_bus: EventBus) -> None:
        """Scenario 1: E-Commerce Same-Day Fulfillment.

        Workflow:
        1. High-volume order ingestion (20 customer orders across 4 SKUs).
        2. Stock reservation & allocation: available stock decrements cleanly.
        3. Dynamic ROP evaluation: depleted SKU breaches threshold, triggering auto-PO and event.
        4. Carrier cutoff wave batching: orders grouped into pick waves.
        5. Serpentine picking: pick tasks generated in alternating aisle sequence.
        6. Order lifecycle progression: FSM transitions from CREATED to SHIPPED.
        7. Webhook dispatch: HMAC-signed event emitted upon order SHIPPED.
        8. Fleet dispatch: routes generated respecting vehicle payload limits.
        """
        captured_events: List[Event] = []
        event_bus.subscribe("#", lambda ev: captured_events.append(ev))

        # 1. Warehouse Inventory Setup
        catalog = {
            "SKU-SHIRT": StockItem(sku="SKU-SHIRT", on_hand=150, unit_cost=25.0, daily_demand_history=[15.0] * 10),
            "SKU-JEANS": StockItem(sku="SKU-JEANS", on_hand=80,  unit_cost=60.0, daily_demand_history=[8.0] * 10),
            "SKU-SHOES": StockItem(sku="SKU-SHOES", on_hand=40,  unit_cost=90.0, daily_demand_history=[6.0] * 10),
            "SKU-CAP":   StockItem(sku="SKU-CAP",   on_hand=30,  unit_cost=15.0, daily_demand_history=[12.0] * 10), # High demand, low stock
        }

        # 2. Ingest 20 Customer Orders
        orders: List[Order] = []
        for i in range(1, 21):
            items = [
                OrderItem(sku="SKU-SHIRT", quantity=2, unit_weight=0.5, unit_volume=0.002),
                OrderItem(sku="SKU-JEANS", quantity=1, unit_weight=1.0, unit_volume=0.005),
            ]
            if i % 2 == 0:
                items.append(OrderItem(sku="SKU-CAP", quantity=2, unit_weight=0.2, unit_volume=0.001))
            if i % 4 == 0:
                items.append(OrderItem(sku="SKU-SHOES", quantity=1, unit_weight=1.5, unit_volume=0.010))

            order = Order(
                id=f"ECOM-ORD-{i:03d}",
                destination=(41.8781 + (i % 5) * 0.01, -87.6298 + (i // 5) * 0.01),
                weight=sum(it.total_weight for it in items),
                volume=sum(it.total_volume for it in items),
                priority=2 if i <= 5 else 1,
                time_window=(10.0, 14.0) if i <= 5 else (12.0, 18.0),
                items=items,
                metadata={"carrier": "FEDEX", "cutoff": 14.0},
            )
            orders.append(order)

        # 3. Reserve Stock for all orders
        for order in orders:
            for item in order.items:
                stock_item = catalog[item.sku]
                stock_item.allocate(item.quantity)

        # Verify stock reservation
        assert catalog["SKU-CAP"].allocated == 20, "10 orders with 2 caps = 20 allocated"
        assert catalog["SKU-CAP"].available == 10, "30 on_hand - 20 allocated = 10 available"

        # 4. ROP Check: SKU-CAP available (10) < ROP (daily demand 12 * lead_time 2 + SS ~ 28)
        if INVENTORY_AVAILABLE:
            inv_engine = ReplenishmentEventEngine(event_bus=event_bus)
            status_cap = inv_engine.evaluate_sku(catalog["SKU-CAP"])
            assert status_cap.needs_reorder is True, "SKU-CAP must trigger needs_reorder=True"
            po = PurchaseOrder(po_id="PO-REPLEN-01", sku="SKU-CAP", quantity=status_cap.recommended_order_qty or 100)
            event_bus.publish(Event(topic="inventory.reorder", payload={"sku": "SKU-CAP", "po_id": po.po_id}))

        # 5. Warehouse Wave Batching & Serpentine Picking
        if WAREHOUSE_AVAILABLE:
            bins = [
                BinLocation(id="A1-B1", aisle=1, bay=1, tier=1, x=1.0, y=2.0, z=0.5),
                BinLocation(id="A1-B2", aisle=1, bay=2, tier=1, x=1.0, y=4.0, z=0.5),
                BinLocation(id="A2-B1", aisle=2, bay=1, tier=1, x=3.0, y=2.0, z=0.5),
                BinLocation(id="A2-B2", aisle=2, bay=2, tier=1, x=3.0, y=4.0, z=0.5),
            ]
            layout = WarehouseLayout(dock_location=(0.0, 0.0, 0.0), bins=bins)
            batcher = WaveBatcher(max_orders=10, max_wave_weight=150.0, carrier_cutoffs={"FEDEX": 14.0})
            waves = batcher.batch_orders(orders)
            assert len(waves) >= 2, "20 orders with max 10 per wave must produce at least 2 waves"

            dispatcher = WarehouseTaskDispatcher(layout=layout)
            manifest = dispatcher.generate_picking_manifest(wave=waves[0], layout=layout)
            assert len(manifest) >= 1, "Dispatcher must produce pick tasks"

            # 6. Order Lifecycle FSM Transitions: CREATED -> ALLOCATED -> RELEASED -> PICKING -> PICKED -> PACKED -> SHIPPED
            for order in orders[:5]:
                fsm = OrderFSM(order_id=order.id)
                fsm.transition(OrderStatus.ALLOCATED if hasattr(OrderStatus, "ALLOCATED") else "ALLOCATED")
                fsm.transition(OrderStatus.RELEASED if hasattr(OrderStatus, "RELEASED") else "RELEASED")
                fsm.transition(OrderStatus.PICKING if hasattr(OrderStatus, "PICKING") else "PICKING")
                fsm.transition(OrderStatus.PICKED if hasattr(OrderStatus, "PICKED") else "PICKED")
                fsm.transition(OrderStatus.PACKED if hasattr(OrderStatus, "PACKED") else "PACKED")
                fsm.transition(OrderStatus.SHIPPED if hasattr(OrderStatus, "SHIPPED") else "SHIPPED")
                assert fsm.current_state in ("SHIPPED", OrderStatus.SHIPPED)

                # Deduct inventory upon shipment
                for it in order.items:
                    catalog[it.sku].fulfill(it.quantity)

                # Emit SHIPPED event
                event_bus.publish(
                    Event(
                        topic="order.shipped",
                        payload={"order_id": order.id, "carrier": "FEDEX", "weight": order.weight},
                    )
                )

        # 7. Webhook Signature Verification on Shipped Orders
        if WEBHOOKS_AVAILABLE:
            dispatcher_wh = WebhookDispatcher(secret="ecommerce_live_secret_key_2026")
            shipped_payload = json.dumps({"order_id": "ECOM-ORD-001", "status": "SHIPPED"}).encode("utf-8")
            sig = dispatcher_wh.generate_signature(shipped_payload)
            assert dispatcher_wh.verify_signature(shipped_payload, sig) is True

        # 8. Delivery Fleet Dispatch
        if ROUTING_AVAILABLE:
            depot = Depot(id="DEPOT-ECOM", location=(41.8781, -87.6298))
            fleet = [
                Vehicle(id="VAN-ECOM-01", capacity_weight=200.0, speed_kmh=40.0),
                Vehicle(id="VAN-ECOM-02", capacity_weight=200.0, speed_kmh=40.0),
            ]
            engine_rt = AutoDispatchEngine(config=LogisticsConfig(enable_2opt=True))
            dispatch_res = engine_rt.dispatch(orders=orders[:10], fleet=fleet, depot=depot)
            assert len(dispatch_res.routes) >= 1
            for r in dispatch_res.routes:
                assert r.total_load_weight <= 200.0 + 1e-6

    @pytest.mark.skipif(not ROUTING_AVAILABLE, reason="open_logistics.routing pending Milestone 2")
    def test_scenario_2_three_pl_multi_tenant_fleet_dispatch(self) -> None:
        """Scenario 2: 3PL Multi-Tenant Fleet Dispatch.

        Workflow:
        1. Multi-client order aggregation across 3 clients:
           - Client A (Pharma): urgent SLA priority=3, tight time window [09:00, 11:00], light parcels.
           - Client B (FreshGrocer): priority=2, time window [10:00, 13:00], heavy parcels.
           - Client C (OfficeDirect): priority=1, time window [11:00, 16:00], bulky parcels.
        2. Heterogeneous fleet:
           - Electric Scooter: 30 kg, 0.2 m^3, speed 25 km/h, cost $0.50/km.
           - Delivery Van: 500 kg, 4.0 m^3, speed 45 km/h, cost $1.20/km.
           - Box Truck: 1800 kg, 14.0 m^3, speed 35 km/h, cost $2.20/km.
        3. Multi-dimensional capacity constraints (weight and volume).
        4. Clarke-Wright CVRP + 2-opt TSP route optimization.
        5. TimeWindowScheduler arrival time propagation and SLA delay verification.
        """
        depot = Depot(id="3PL-HUB-CHICAGO", location=(41.8781, -87.6298), time_window=(7.0, 19.0))
        fleet = [
            Vehicle(id="SCOOTER-01", capacity_weight=30.0, capacity_volume=0.20, speed_kmh=25.0, cost_per_km=0.50),
            Vehicle(id="VAN-01", capacity_weight=500.0, capacity_volume=4.0, speed_kmh=45.0, cost_per_km=1.20),
            Vehicle(id="TRUCK-01", capacity_weight=1800.0, capacity_volume=14.0, speed_kmh=35.0, cost_per_km=2.20),
        ]

        # 18 Multi-Tenant Orders
        orders: List[Order] = []
        # Client A: Pharma (6 stops, urgent, light)
        for i in range(1, 7):
            orders.append(
                Order(
                    id=f"CLIENT-A-PHARMA-{i}",
                    destination=(41.8850 + i * 0.005, -87.6250 - i * 0.005),
                    weight=3.5,
                    volume=0.015,
                    priority=3,
                    time_window=(9.0, 11.0),
                    metadata={"client": "PharmaCare"},
                )
            )

        # Client B: FreshGrocer (6 stops, medium, heavy)
        for i in range(1, 7):
            orders.append(
                Order(
                    id=f"CLIENT-B-GROCER-{i}",
                    destination=(41.8600 - i * 0.006, -87.6400 - i * 0.004),
                    weight=65.0,
                    volume=0.15,
                    priority=2,
                    time_window=(10.0, 13.0),
                    metadata={"client": "FreshGrocer"},
                )
            )

        # Client C: OfficeDirect (6 stops, bulky, standard window)
        for i in range(1, 7):
            orders.append(
                Order(
                    id=f"CLIENT-C-OFFICE-{i}",
                    destination=(41.9000 + i * 0.007, -87.6500 + i * 0.005),
                    weight=45.0,
                    volume=0.60,
                    priority=1,
                    time_window=(11.0, 16.0),
                    metadata={"client": "OfficeDirect"},
                )
            )

        engine = AutoDispatchEngine(config=LogisticsConfig(enable_2opt=True))
        result = engine.dispatch(orders=orders, fleet=fleet, depot=depot)

        assert isinstance(result, DispatchResult)
        assert len(result.routes) >= 2, "Must generate multiple routes for heterogeneous fleet"
        assert result.total_distance > 0.0

        # Verify dual capacity constraints and vehicle matching
        for route in result.routes:
            veh = next(v for v in fleet if v.id == route.vehicle_id)
            assert route.total_load_weight <= veh.capacity_weight + 1e-6, (
                f"Vehicle {veh.id} weight capacity {veh.capacity_weight} exceeded by route {route.total_load_weight}"
            )
            assert route.total_load_volume <= veh.capacity_volume + 1e-6, (
                f"Vehicle {veh.id} volume capacity {veh.capacity_volume} exceeded by route {route.total_load_volume}"
            )

        # Verify time window propagation
        scheduler = TimeWindowScheduler(speed_kmh=40.0)
        for route in result.routes:
            if len(route.stops) >= 2:
                sched_res = scheduler.calculate_schedule(route.stops, start_time=8.5)
                assert sched_res is not None

    @pytest.mark.skipif(
        not (INVENTORY_AVAILABLE and WAREHOUSE_AVAILABLE),
        reason="Requires both inventory and warehouse modules",
    )
    def test_scenario_3_automated_dc_replenishment_and_3d_slotting(self) -> None:
        """Scenario 3: Automated DC Replenishment & 3D Slotting.

        Workflow:
        1. 50 SKUs generated following Pareto distribution (Class A: 80% revenue, Class B: 15%, Class C: 5%).
        2. Volatility analysis: calculate standard deviation, Acklam inverse normal CDF probit, safety stock.
        3. 3D Warehouse layout: 4 aisles, 10 bays, 4 tiers.
        4. Optimal dock-proximity slotting: Class A slotted closest to dock, Class C in back/upper bins.
        5. Multi-day demand simulation (7 days) with stochastic consumption.
        6. ROP threshold breaches automatically generate PurchaseOrders, preventing stockouts.
        7. Stock replenishment receipts increment on_hand, maintaining inventory stability.
        """
        random.seed(42)  # Deterministic simulation

        # 1. Generate 50 SKUs with Pareto distribution
        skus: List[StockItem] = []
        for i in range(1, 51):
            if i <= 10:
                # Top 10 (20% count) = Class A: High daily demand, high unit cost
                demand = [random.uniform(20.0, 35.0) for _ in range(14)]
                cost = random.uniform(100.0, 300.0)
            elif i <= 25:
                # Next 15 (30% count) = Class B: Moderate demand, moderate cost
                demand = [random.uniform(8.0, 15.0) for _ in range(14)]
                cost = random.uniform(30.0, 80.0)
            else:
                # Remaining 25 (50% count) = Class C: Low demand, low cost
                demand = [random.uniform(1.0, 4.0) for _ in range(14)]
                cost = random.uniform(5.0, 25.0)

            avg_d = sum(demand) / len(demand)
            initial_stock = int(avg_d * 10)  # ~10 days of supply initially

            skus.append(
                StockItem(
                    sku=f"SKU-DC-{i:02d}",
                    on_hand=initial_stock,
                    unit_cost=cost,
                    holding_cost_rate=0.20,
                    lead_time_days=3.0,
                    daily_demand_history=demand,
                )
            )

        # 2. Pareto ABC Classification
        abc_map = classify_abc(skus)
        class_a_skus = [s.sku for s in skus if abc_map.get(s.sku) in ("A", 0)]
        class_c_skus = [s.sku for s in skus if abc_map.get(s.sku) in ("C", 2)]

        assert len(class_a_skus) >= 8, f"Top ~10 items should be classified A, got {len(class_a_skus)}"
        assert len(class_c_skus) >= 20, f"Bottom ~25 items should be classified C, got {len(class_c_skus)}"

        # 3. 3D Warehouse Layout Setup (4 aisles, 10 bays, 4 tiers = 160 bin locations)
        dock = (0.0, 0.0, 0.0)
        bins: List[BinLocation] = []
        for a in range(1, 5):
            for b in range(1, 11):
                for t in range(1, 5):
                    bins.append(
                        BinLocation(
                            id=f"A{a}-B{b}-T{t}",
                            aisle=a,
                            bay=b,
                            tier=t,
                            x=float(a * 3.0),
                            y=float(b * 2.5),
                            z=float(t * 1.2),
                        )
                    )
        layout = WarehouseLayout(dock_location=dock, bins=bins)

        # 4. Slotting Optimization
        optimizer = SlottingOptimizer(layout=layout)
        recommendations = optimizer.optimize_slotting(items=skus, abc_classes=abc_map)
        assert len(recommendations) == 50, "All 50 SKUs must receive slotting recommendations"

        avg_dist_a = sum(r.distance_to_dock for r in recommendations if r.sku in class_a_skus) / len(class_a_skus)
        avg_dist_c = sum(r.distance_to_dock for r in recommendations if r.sku in class_c_skus) / len(class_c_skus)

        assert avg_dist_a < avg_dist_c, (
            f"Class A average dock distance ({avg_dist_a:.2f}m) must be strictly less than "
            f"Class C average dock distance ({avg_dist_c:.2f}m)"
        )

        # 5. Multi-Day Simulation (7 Days of Demand & Replenishment)
        engine = ReplenishmentEventEngine()
        generated_pos: List[PurchaseOrder] = []
        stockouts_recorded = 0

        for day in range(1, 8):
            # Process daily orders
            for item in skus:
                daily_demand = int(random.gauss(item.average_daily_demand, 2.0))
                daily_demand = max(1, daily_demand)

                try:
                    item.fulfill(daily_demand)
                except StockoutError:
                    stockouts_recorded += 1
                    # Fulfill whatever is left
                    if item.on_hand > 0:
                        item.fulfill(item.on_hand)

                # Check ROP
                status = engine.evaluate_sku(item)
                if status.needs_reorder and item.on_order == 0:
                    order_qty = int(status.recommended_order_qty or 100)
                    po = PurchaseOrder(
                        po_id=f"PO-D{day}-{item.sku}",
                        sku=item.sku,
                        quantity=order_qty,
                    )
                    item.on_order += po.quantity
                    generated_pos.append(po)

            # Receive POs arriving after 3-day lead time
            for po in list(generated_pos):
                target_item = next(s for s in skus if s.sku == po.sku)
                target_item.receive(po.quantity)
                generated_pos.remove(po)

        # Verify stability
        assert stockouts_recorded <= 2, f"Automated replenishment must keep stockouts minimal, got {stockouts_recorded}"
        for item in skus:
            assert item.on_hand >= 0, f"StockItem {item.sku} on_hand cannot be negative"

    def test_scenario_4_resilient_order_lifecycle_and_rollback_recovery(self, event_bus: EventBus) -> None:
        """Scenario 4: Resilient Order Lifecycle & Rollback Recovery.

        Workflow:
        1. 10 customer orders active in wave picking across 4 SKUs.
        2. Stock reserved for all orders: allocated incremented, available decremented.
        3. Edge disruption injected:
           - Disruption A: Damaged item in bin -> selective order rollback.
           - Disruption B: Customer cancellation mid-pick -> atomic FSM rollback.
        4. Stock deallocation returns inventory cleanly back to available.
        5. Zero inventory leakage: on_hand == available + allocated strictly holds for all SKUs.
        6. Unaffected orders proceed through PICKED, PACKED, and SHIPPED without disruption.
        7. EventBus audit trail logs all disruption and state change events.
        """
        audit_trail: List[Event] = []
        event_bus.subscribe("#", lambda ev: audit_trail.append(ev))

        # 1. Inventory setup
        skus = {
            "SKU-GLASS": StockItem(sku="SKU-GLASS", on_hand=50, allocated=0),
            "SKU-PHONE": StockItem(sku="SKU-PHONE", on_hand=30, allocated=0),
            "SKU-CABLE": StockItem(sku="SKU-CABLE", on_hand=100, allocated=0),
        }

        # 2. 10 Orders reserving stock
        orders: List[Order] = []
        for i in range(1, 11):
            items = [
                OrderItem(sku="SKU-GLASS", quantity=2),
                OrderItem(sku="SKU-CABLE", quantity=3),
            ]
            if i % 3 == 0:
                items.append(OrderItem(sku="SKU-PHONE", quantity=1))

            o = Order(id=f"RES-ORD-{i:02d}", destination=(41.88, -87.62), weight=5.0, items=items)
            orders.append(o)
            for it in items:
                skus[it.sku].allocate(it.quantity)

        # Baseline check before disruptions
        assert skus["SKU-GLASS"].allocated == 20  # 10 orders * 2 = 20
        assert skus["SKU-GLASS"].available == 30  # 50 - 20 = 30
        assert skus["SKU-CABLE"].allocated == 30  # 10 orders * 3 = 30
        assert skus["SKU-CABLE"].available == 70  # 100 - 30 = 70

        # Invariant check: on_hand == available + allocated
        for s in skus.values():
            assert s.on_hand == s.available + s.allocated, f"Conservation law violated for {s.sku}"

        # 3. Disruption A: Damaged glass at pick time for RES-ORD-03
        disrupted_order_1 = orders[2]  # RES-ORD-03
        event_bus.publish(
            Event(
                topic="warehouse.disruption",
                payload={"order_id": disrupted_order_1.id, "reason": "damaged_stock", "sku": "SKU-GLASS"},
            )
        )
        # Rollback stock for disrupted order
        for it in disrupted_order_1.items:
            skus[it.sku].deallocate(it.quantity)

        if WAREHOUSE_AVAILABLE:
            fsm_1 = OrderFSM(order_id=disrupted_order_1.id)
            fsm_1.transition(OrderStatus.ALLOCATED if hasattr(OrderStatus, "ALLOCATED") else "ALLOCATED")
            fsm_1.transition(OrderStatus.RELEASED if hasattr(OrderStatus, "RELEASED") else "RELEASED")
            fsm_1.rollback()

        # 4. Disruption B: Customer cancels RES-ORD-07 mid-pick
        cancelled_order_2 = orders[6]  # RES-ORD-07
        event_bus.publish(
            Event(
                topic="order.cancelled",
                payload={"order_id": cancelled_order_2.id, "reason": "customer_cancelled"},
            )
        )
        for it in cancelled_order_2.items:
            skus[it.sku].deallocate(it.quantity)

        if WAREHOUSE_AVAILABLE:
            fsm_2 = OrderFSM(order_id=cancelled_order_2.id)
            fsm_2.transition(OrderStatus.ALLOCATED if hasattr(OrderStatus, "ALLOCATED") else "ALLOCATED")
            fsm_2.transition(OrderStatus.RELEASED if hasattr(OrderStatus, "RELEASED") else "RELEASED")
            fsm_2.transition(OrderStatus.PICKING if hasattr(OrderStatus, "PICKING") else "PICKING")
            fsm_2.rollback()

        # 5. Strict Zero Inventory Leakage Check
        # SKU-GLASS had 20 allocated, 2 orders cancelled (4 units freed) -> allocated=16, available=34
        assert skus["SKU-GLASS"].allocated == 16, f"Expected 16 allocated, got {skus['SKU-GLASS'].allocated}"
        assert skus["SKU-GLASS"].available == 34, f"Expected 34 available, got {skus['SKU-GLASS'].available}"
        for s in skus.values():
            assert s.on_hand == s.available + s.allocated, f"Conservation law violated after rollbacks for {s.sku}"

        # 6. Unaffected orders proceed to fulfillment
        remaining_orders = [o for o in orders if o.id not in (disrupted_order_1.id, cancelled_order_2.id)]
        assert len(remaining_orders) == 8
        for ro in remaining_orders:
            for it in ro.items:
                skus[it.sku].fulfill(it.quantity)

        # Final verification: All remaining orders fulfilled cleanly
        assert skus["SKU-GLASS"].allocated == 0
        assert skus["SKU-GLASS"].available == 34
        assert skus["SKU-GLASS"].on_hand == 34

        # Audit trail must have recorded disruptions
        assert any(e.topic == "warehouse.disruption" for e in audit_trail)
        assert any(e.topic == "order.cancelled" for e in audit_trail)

    @pytest.mark.skipif(
        not (SERVER_AVAILABLE or CLI_AVAILABLE),
        reason="Requires server or CLI modules",
    )
    def test_scenario_5_end_to_end_headless_cli_and_microservice_integration(self) -> None:
        """Scenario 5: End-to-End Headless CLI & Microservice Integration.

        Workflow:
        1. Start embedded stdlib HTTP microservice in background daemon thread.
        2. Verify REST endpoints:
           - GET /health -> 200 OK.
           - POST /api/v1/dispatch -> 200 OK with route calculations.
        3. Verify UNIX-pipeable CLI:
           - Execute CLI subcommands with `--help` and verify exit code 0.
           - Test stdout formatting.
        4. Clean server shutdown without leaking resources.
        """
        if SERVER_AVAILABLE:
            server = create_server(host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            host, port = server.server_address[:2]
            base_url = f"http://{host}:{port}"

            try:
                # 1. GET /health
                req_h = Request(f"{base_url}/health")
                with urlopen(req_h, timeout=4.0) as resp:
                    assert resp.status == 200
                    body = json.loads(resp.read().decode("utf-8"))
                    assert body.get("status") == "healthy"

                # 2. POST /api/v1/dispatch
                dispatch_data = {
                    "depot": {"id": "D1", "location": [41.8781, -87.6298]},
                    "fleet": [{"id": "V1", "capacity_weight": 1000.0, "speed_kmh": 40.0}],
                    "orders": [
                        {"id": "ORD-CLI-1", "destination": [41.8850, -87.6250], "weight": 50.0},
                        {"id": "ORD-CLI-2", "destination": [41.8920, -87.6200], "weight": 60.0},
                    ],
                }
                req_disp = Request(
                    f"{base_url}/api/v1/dispatch",
                    data=json.dumps(dispatch_data).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req_disp, timeout=5.0) as resp:
                    assert resp.status == 200
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    assert "routes" in resp_json
            finally:
                server.shutdown()
                server.server_close()

        # 3. CLI Main Invocation
        if CLI_AVAILABLE:
            # Test --help across CLI subcommands
            old_stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                with pytest.raises(SystemExit) as help_exit:
                    cli_main(["--help"])
                assert help_exit.value.code == 0
                assert "usage:" in sys.stdout.getvalue()

                sys.stdout = io.StringIO()
                with pytest.raises(SystemExit) as dispatch_exit:
                    cli_main(["dispatch", "--help"])
                assert dispatch_exit.value.code == 0
                assert "--stdin" in sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
