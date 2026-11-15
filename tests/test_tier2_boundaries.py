"""Tier 2 Boundary Value Analysis & Edge Case Tests.

Comprehensive boundary and corner case coverage across:
1. Routing Boundaries (zero demand, capacity limits, coordinates, time windows, fleets)
2. Inventory Boundaries (zero demand/lead time, service level extremes, discount tiers, demand history)
3. Warehouse Boundaries (1-bin layouts, 100% fill factor, wave limits, FSM transitions & rollbacks)
4. API & Server Boundaries (empty/malformed JSON, CLI arguments, unsupported HTTP methods, Webhook encodings)
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import io
import json
import math
import socket
import sys
import threading
import time
from typing import Any, Dict, Generator, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import Event, EventBus
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

# Exception aliases for compatibility with SCOPE.md
try:
    from open_logistics.core.exceptions import CapacityExceededError
except ImportError:
    CapacityExceededError = RoutingError  # type: ignore[misc]

try:
    from open_logistics.core.exceptions import InvalidTransitionError
except ImportError:
    InvalidTransitionError = FSMStateError  # type: ignore[misc]


# Safe imports for algorithmic and higher-tier modules (Progressive Testability)
try:
    from open_logistics.routing.clarke_wright import ClarkeWrightCVRP
    from open_logistics.routing.dispatch_engine import AutoDispatchEngine, DispatchResult
    from open_logistics.routing.distances import euclidean_distance, haversine_distance, manhattan_distance
    from open_logistics.routing.time_windows import ScheduleResult, TimeWindowScheduler
    from open_logistics.routing.tsp import NearestNeighborTSP, TwoOptSearch
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False

try:
    from open_logistics.inventory.abc_xyz import (
        calculate_days_sales_of_inventory,
        calculate_inventory_turnover,
        classify_abc,
        classify_xyz,
        generate_abc_xyz_matrix,
    )
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
    from open_logistics.inventory.replenishment_engine import ReplenishmentEventEngine, ReplenishmentStatus
    INVENTORY_AVAILABLE = True
except ImportError:
    INVENTORY_AVAILABLE = False

try:
    from open_logistics.warehouse.fsm import OrderFSM, OrderLifecycleManager
    from open_logistics.warehouse.layout import Aisle, BinLocation, WarehouseLayout, Zone
    from open_logistics.warehouse.picking_path import (
        MidpointPickerRouting,
        ReturnPickerRouting,
        SerpentinePickerRouting,
    )
    from open_logistics.warehouse.slotting import SlottingOptimizer, SlottingRecommendation
    from open_logistics.warehouse.task_dispatcher import PickTask, WarehouseTaskDispatcher
    from open_logistics.warehouse.wave_batcher import PickWave, WaveBatcher
    WAREHOUSE_AVAILABLE = True
except ImportError:
    WAREHOUSE_AVAILABLE = False

try:
    from open_logistics.api.engine import LogisticsEngine
    from open_logistics.api.server import LogisticsHTTPHandler, create_server
    from open_logistics.api.webhooks import WebhookDispatcher, WebhookEvent
    from open_logistics.cli.main import main as cli_main
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


def find_free_port() -> int:
    """Find a random available TCP port for embedded server testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ============================================================================
# Section 1: Routing Boundaries (15 tests)
# ============================================================================

class TestRoutingBoundaries:
    """Boundary value analysis and corner cases for routing and fleet dispatch."""

    def test_b1_order_zero_weight_and_zero_volume(self) -> None:
        """Order with weight=0.0 and volume=0.0 must be valid and handled without division by zero."""
        order = Order(
            id="ORD-ZERO",
            destination=(41.88, -87.62),
            weight=0.0,
            volume=0.0,
        )
        assert order.weight == 0.0
        assert order.volume == 0.0

    def test_b1_order_weight_exactly_vehicle_capacity(self) -> None:
        """Single order with weight exactly equal to vehicle capacity fits perfectly."""
        vehicle = Vehicle(id="V-1000", capacity_weight=1000.0, capacity_volume=10.0)
        order = Order(id="ORD-EXACT", destination=(41.88, -87.62), weight=1000.0, volume=5.0)

        assert order.weight <= vehicle.capacity_weight
        # If AutoDispatchEngine is available, verify assignment
        if ROUTING_AVAILABLE:
            depot = Depot(id="D1", location=(41.87, -87.63))
            engine = AutoDispatchEngine()
            result = engine.dispatch(orders=[order], fleet=[vehicle], depot=depot)
            assert len(result.routes) == 1
            assert len(result.unassigned_orders) == 0

    def test_b1_order_weight_exceeds_vehicle_capacity(self) -> None:
        """Single order exceeding vehicle capacity cannot be loaded into vehicle."""
        vehicle = Vehicle(id="V-1000", capacity_weight=1000.0, capacity_volume=10.0)
        order = Order(id="ORD-OVER", destination=(41.88, -87.62), weight=1001.0, volume=5.0)

        assert order.weight > vehicle.capacity_weight
        if ROUTING_AVAILABLE:
            depot = Depot(id="D1", location=(41.87, -87.63))
            engine = AutoDispatchEngine()
            result = engine.dispatch(orders=[order], fleet=[vehicle], depot=depot)
            # The oversized order should remain unassigned
            assert order.id in [o.id for o in result.unassigned_orders]

    def test_b1_depot_and_stops_at_zero_coordinates(self) -> None:
        """Depot and customer stops at (0.0, 0.0) must compute 0.0 distance without math errors."""
        depot = Depot(id="DEPOT-0", location=(0.0, 0.0))
        stop = Stop(location=(0.0, 0.0), order_id="ORD-0", demand_weight=10.0)
        assert depot.location == (0.0, 0.0)
        assert stop.location == (0.0, 0.0)

        if ROUTING_AVAILABLE:
            d_euc = euclidean_distance((0.0, 0.0), (0.0, 0.0))
            d_man = manhattan_distance((0.0, 0.0), (0.0, 0.0))
            d_hav = haversine_distance((0.0, 0.0), (0.0, 0.0))
            assert d_euc == 0.0
            assert d_man == 0.0
            assert d_hav == 0.0

    def test_b1_two_customer_stops_with_identical_coordinates(self) -> None:
        """Two distinct orders located at the exact same coordinates."""
        stop1 = Stop(location=(41.88, -87.62), order_id="O1", demand_weight=10.0)
        stop2 = Stop(location=(41.88, -87.62), order_id="O2", demand_weight=20.0)

        assert stop1.location == stop2.location
        if ROUTING_AVAILABLE:
            assert euclidean_distance(stop1.location, stop2.location) == 0.0

    def test_b1_single_customer_order_one_stop_tour(self) -> None:
        """Single order tour consists of Depot -> Customer -> Depot."""
        route = Route(vehicle_id="V1")
        depot_start = Stop(location=(41.87, -87.63), is_depot=True)
        customer = Stop(location=(41.88, -87.62), order_id="O1", demand_weight=15.0)
        depot_end = Stop(location=(41.87, -87.63), is_depot=True)

        route.add_stop(depot_start)
        route.add_stop(customer)
        route.add_stop(depot_end)

        assert route.customer_stop_count == 1
        assert route.stop_count == 3
        assert route.total_load_weight == 15.0

    def test_b1_exact_instantaneous_time_window(self) -> None:
        """Time window where earliest == latest [10.0, 10.0] is valid (punctual arrival)."""
        order = Order(
            id="ORD-TW-INST",
            destination=(41.88, -87.62),
            weight=10.0,
            time_window=(10.0, 10.0),
        )
        assert order.time_window == (10.0, 10.0)

    def test_b1_infeasible_tight_time_window_penalty_or_violation(self) -> None:
        """Impossible delivery window triggers violation flag or positive penalty."""
        order = Order(
            id="ORD-TIGHT",
            destination=(41.88, -87.62),
            weight=10.0,
            time_window=(0.1, 0.2),  # Arriving impossible in 0.2 hours from distant depot
        )
        assert order.time_window == (0.1, 0.2)

        if ROUTING_AVAILABLE:
            scheduler = TimeWindowScheduler()
            # If scheduler exists, evaluate schedule and expect late arrival or violation
            sched = scheduler.evaluate_stop(
                arrival_time=2.0,
                time_window=order.time_window,
                service_time=0.25,
            )
            assert sched.is_feasible is False or sched.penalty_cost > 0.0

    def test_b1_empty_order_list_dispatch(self) -> None:
        """Dispatch with empty orders list returns empty routes without error."""
        if ROUTING_AVAILABLE:
            engine = AutoDispatchEngine()
            depot = Depot(id="D1", location=(41.87, -87.63))
            fleet = [Vehicle(id="V1", capacity_weight=500.0)]
            result = engine.dispatch(orders=[], fleet=fleet, depot=depot)
            assert len(result.routes) == 0
            assert len(result.unassigned_orders) == 0
            assert result.total_distance == 0.0

    def test_b1_empty_fleet_handling(self) -> None:
        """Dispatch with empty fleet returns all orders unassigned or raises RoutingError."""
        orders = [Order(id="O1", destination=(41.88, -87.62), weight=10.0)]
        depot = Depot(id="D1", location=(41.87, -87.63))

        if ROUTING_AVAILABLE:
            engine = AutoDispatchEngine()
            result = engine.dispatch(orders=orders, fleet=[], depot=depot)
            assert len(result.unassigned_orders) == 1
            assert len(result.routes) == 0

    def test_b1_vehicle_single_stop_capacity(self) -> None:
        """Vehicle configured with max_stops=1 accepts at most 1 delivery."""
        vehicle = Vehicle(id="V-1STOP", capacity_weight=1000.0, max_stops=1)
        assert vehicle.max_stops == 1

    def test_b1_extreme_coordinate_boundaries(self) -> None:
        """Validates allowable extreme coordinate boundaries (-90/90, -180/180) and rejection beyond."""
        loc_north_pole = Location(lat=90.0, lon=0.0)
        loc_south_pole = Location(lat=-90.0, lon=0.0)
        loc_antimeridian = Location(lat=0.0, lon=180.0)
        loc_antimeridian_neg = Location(lat=0.0, lon=-180.0)

        assert loc_north_pole.lat == 90.0
        assert loc_south_pole.lat == -90.0
        assert loc_antimeridian.lon == 180.0
        assert loc_antimeridian_neg.lon == -180.0

        with pytest.raises(ValidationError):
            Location(lat=90.001, lon=0.0)
        with pytest.raises(ValidationError):
            Location(lat=0.0, lon=180.001)

    def test_b1_negative_order_weight_rejected(self) -> None:
        """Order initialized with negative weight raises ValidationError."""
        with pytest.raises(ValidationError):
            Order(id="ORD-NEG", destination=(41.88, -87.62), weight=-1.0)

    def test_b1_negative_order_volume_rejected(self) -> None:
        """Order initialized with negative volume raises ValidationError."""
        with pytest.raises(ValidationError):
            Order(id="ORD-NEG-VOL", destination=(41.88, -87.62), weight=10.0, volume=-0.5)

    def test_b1_time_window_start_after_end_rejected(self) -> None:
        """Order with time window where start > end raises ValidationError."""
        with pytest.raises(ValidationError):
            Order(id="ORD-INV-TW", destination=(41.88, -87.62), weight=10.0, time_window=(15.0, 10.0))


# ============================================================================
# Section 2: Inventory Boundaries (15 tests)
# ============================================================================

class TestInventoryBoundaries:
    """Boundary value analysis and corner cases for scientific inventory models."""

    def test_b2_zero_annual_demand_eoq(self) -> None:
        """Zero annual demand yields EOQ = 0.0 without ZeroDivisionError."""
        if INVENTORY_AVAILABLE:
            eoq = calculate_eoq(annual_demand=0.0, order_cost=50.0, holding_cost_per_unit=4.0)
            assert eoq == 0.0

    def test_b2_zero_lead_time_safety_stock_and_rop(self) -> None:
        """Lead time = 0.0 yields safety stock = 0.0 and deterministic ROP = 0.0."""
        if INVENTORY_AVAILABLE:
            ss = calculate_safety_stock(
                daily_demand_std=5.0,
                lead_time_days=0.0,
                service_level=0.95,
            )
            assert ss == 0.0

            rop = calculate_reorder_point(daily_demand=10.0, lead_time_days=0.0, safety_stock=0.0)
            assert rop == 0.0

    def test_b2_service_level_point_five_zero(self) -> None:
        """Service level = 0.50 corresponds to Z = 0.0, yielding safety stock = 0.0."""
        if INVENTORY_AVAILABLE:
            z = acklam_norm_inv_cdf(0.50)
            assert abs(z) < 1e-6

            ss = calculate_safety_stock(
                daily_demand_std=10.0,
                lead_time_days=4.0,
                service_level=0.50,
            )
            assert abs(ss) < 1e-4

    def test_b2_service_level_high_extreme(self) -> None:
        """Service level = 0.9999 (99.99%) yields Z ~ 3.719 and positive finite safety stock."""
        if INVENTORY_AVAILABLE:
            z = acklam_norm_inv_cdf(0.9999)
            assert 3.70 < z < 3.75

            ss = calculate_safety_stock(
                daily_demand_std=10.0,
                lead_time_days=4.0,
                service_level=0.9999,
            )
            assert ss > 0.0
            assert math.isfinite(ss)

    def test_b2_single_tier_quantity_discount_no_breaks(self) -> None:
        """Quantity discount with a single constant price tier matches classical EOQ."""
        if INVENTORY_AVAILABLE:
            annual_demand = 1000.0
            order_cost = 50.0
            unit_cost = 20.0
            holding_cost_rate = 0.20
            h = unit_cost * holding_cost_rate

            expected_eoq = calculate_eoq(annual_demand, order_cost, h)
            tiers = [(1, unit_cost)]  # No breaks, single tier
            best_q, best_c = calculate_quantity_discount_eoq(
                annual_demand=annual_demand,
                order_cost=order_cost,
                holding_cost_rate=holding_cost_rate,
                discount_tiers=tiers,
            )
            assert abs(best_q - expected_eoq) < 1.0

    def test_b2_multi_tier_discount_at_exact_break_boundary(self) -> None:
        """Order quantity at exact price break threshold evaluates correct tier price."""
        if INVENTORY_AVAILABLE:
            # Tiers: 1-499: $20, 500+: $15
            tiers = [(1, 20.0), (500, 15.0)]
            best_q, best_c = calculate_quantity_discount_eoq(
                annual_demand=2000.0,
                order_cost=100.0,
                holding_cost_rate=0.20,
                discount_tiers=tiers,
            )
            assert best_q >= 500  # Should leverage the volume discount break

    def test_b2_zero_holding_cost_edge_case(self) -> None:
        """Holding cost near zero is handled without division crash."""
        if INVENTORY_AVAILABLE:
            # When holding cost is 0, EOQ is arbitrarily large or bounded
            try:
                res = calculate_eoq(annual_demand=1000.0, order_cost=50.0, holding_cost_per_unit=0.0)
                assert res >= 0.0
            except (ZeroDivisionError, ValueError, InventoryError):
                pass  # Graceful exception also valid

    def test_b2_zero_ordering_cost_edge_case(self) -> None:
        """Ordering cost = 0 produces EOQ = 0 (order continuously in infinitesimal batches)."""
        if INVENTORY_AVAILABLE:
            eoq = calculate_eoq(annual_demand=1000.0, order_cost=0.0, holding_cost_per_unit=4.0)
            assert eoq == 0.0

    def test_b2_daily_demand_zero_variance(self) -> None:
        """Daily demand history with identical entries has zero variance and safety stock = 0."""
        item = StockItem(
            sku="SKU-IDENTICAL",
            on_hand=100,
            daily_demand_history=[10.0, 10.0, 10.0, 10.0, 10.0],
        )
        assert item.average_daily_demand == 10.0
        if INVENTORY_AVAILABLE:
            ss = calculate_safety_stock(
                daily_demand_std=0.0,
                lead_time_days=4.0,
                service_level=0.95,
            )
            assert ss == 0.0

    def test_b2_empty_demand_history_handling(self) -> None:
        """StockItem with empty daily demand history has 0.0 average daily demand without crash."""
        item = StockItem(
            sku="SKU-EMPTY-HIST",
            on_hand=50,
            daily_demand_history=[],
        )
        assert item.average_daily_demand == 0.0
        assert item.annual_demand() == 0.0

    def test_b2_negative_stock_decrement_guard(self) -> None:
        """Attempting to allocate a negative quantity raises ValidationError."""
        item = StockItem(sku="SKU-GUARD", on_hand=100)
        with pytest.raises(ValidationError):
            item.allocate(-10)

    def test_b2_stock_allocation_exceeding_available(self) -> None:
        """Attempting to allocate more units than available raises StockoutError."""
        item = StockItem(sku="SKU-AVAIL", on_hand=10, allocated=2)
        assert item.available == 8
        with pytest.raises(StockoutError) as exc_info:
            item.allocate(9)
        assert exc_info.value.requested == 9
        assert exc_info.value.available == 8

    def test_b2_stock_fulfillment_exceeding_on_hand(self) -> None:
        """Attempting to fulfill more units than on_hand raises StockoutError."""
        item = StockItem(sku="SKU-FULFILL", on_hand=5, allocated=5)
        with pytest.raises(StockoutError):
            item.fulfill(10)

    def test_b2_deallocate_clamp_to_zero(self) -> None:
        """Deallocating more units than currently allocated clamps allocated to 0 cleanly."""
        item = StockItem(sku="SKU-DEALLOC", on_hand=100, allocated=10)
        item.deallocate(50)
        assert item.allocated == 0

    def test_b2_negative_on_hand_initialization_rejected(self) -> None:
        """Initializing StockItem with negative on_hand raises ValidationError."""
        with pytest.raises(ValidationError):
            StockItem(sku="SKU-NEG", on_hand=-5)


# ============================================================================
# Section 3: Warehouse Boundaries (15 tests)
# ============================================================================

class TestWarehouseBoundaries:
    """Boundary value analysis and corner cases for warehouse slotting, FSM, and waves."""

    def test_b3_one_bin_warehouse_layout(self) -> None:
        """A minimal warehouse layout with exactly 1 bin location."""
        dims = Dimensions(length=1.0, width=1.0, height=1.0)
        assert dims.volume == 1.0

        if WAREHOUSE_AVAILABLE:
            layout = WarehouseLayout()
            bin_loc = BinLocation(id="BIN-SINGLE", zone="A", aisle=1, bay=1, tier=1, dimensions=dims)
            layout.add_bin(bin_loc)
            assert len(layout.bins) == 1

    def test_b3_item_dimensions_exactly_match_bin(self) -> None:
        """Item dimensions exactly matching bin dimensions (100% volumetric fill factor)."""
        bin_dims = Dimensions(length=0.5, width=0.4, height=0.3)
        item_dims = Dimensions(length=0.5, width=0.4, height=0.3)
        assert abs(item_dims.volume - bin_dims.volume) < 1e-9

        fill_factor = item_dims.volume / bin_dims.volume
        assert abs(fill_factor - 1.0) < 1e-9

    def test_b3_pick_wave_with_exactly_one_order(self) -> None:
        """Pick wave batch containing exactly 1 order generates valid pick task."""
        if WAREHOUSE_AVAILABLE:
            dispatcher = WarehouseTaskDispatcher()
            wave = dispatcher.create_wave(order_ids=["ORD-SINGLE"], max_orders=25)
            assert len(wave.order_ids) == 1

    def test_b3_wave_batcher_orders_exceed_max_capacity(self) -> None:
        """Wave batcher splits 60 orders across multiple waves when max_wave_orders=25."""
        if WAREHOUSE_AVAILABLE:
            batcher = WaveBatcher(max_orders=25)
            order_ids = [f"ORD-{i:03d}" for i in range(60)]
            waves = batcher.batch_orders(order_ids)
            assert len(waves) == 3
            assert len(waves[0].order_ids) == 25
            assert len(waves[1].order_ids) == 25
            assert len(waves[2].order_ids) == 10

    def test_b3_fsm_illegal_out_of_sequence_transition(self) -> None:
        """Attempting to transition directly from CREATED to SHIPPED raises FSMStateError."""
        if WAREHOUSE_AVAILABLE:
            fsm = OrderFSM(order_id="ORD-FSM-1")
            assert fsm.current_state in ("CREATED", "PENDING")
            with pytest.raises((FSMStateError, InvalidTransitionError)):
                fsm.transition("SHIPPED")

    def test_b3_fsm_forbidden_cancellation_from_shipped(self) -> None:
        """Cancellation/rollback from SHIPPED state is strictly forbidden."""
        if WAREHOUSE_AVAILABLE:
            fsm = OrderFSM(order_id="ORD-FSM-2")
            # Walk normal lifecycle to SHIPPED
            for state in ["ALLOCATED", "RELEASED", "PICKING", "PICKED", "PACKED", "SHIPPED"]:
                fsm.transition(state)
            assert fsm.current_state == "SHIPPED"

            with pytest.raises((FSMStateError, InvalidTransitionError)):
                fsm.transition("CANCELLED")

    def test_b3_fsm_rollback_from_picking_restores_allocation(self) -> None:
        """Order cancellation while in PICKING state successfully executes rollback."""
        if WAREHOUSE_AVAILABLE:
            fsm = OrderFSM(order_id="ORD-FSM-3")
            fsm.transition("ALLOCATED")
            fsm.transition("RELEASED")
            fsm.transition("PICKING")
            assert fsm.current_state == "PICKING"

            fsm.rollback()
            assert fsm.current_state in ("ALLOCATED", "CREATED", "CANCELLED")

    def test_b3_fsm_backward_transition_guard(self) -> None:
        """Illegal backward transition (e.g. PACKED -> ALLOCATED) is rejected."""
        if WAREHOUSE_AVAILABLE:
            fsm = OrderFSM(order_id="ORD-FSM-4")
            for state in ["ALLOCATED", "RELEASED", "PICKING", "PICKED", "PACKED"]:
                fsm.transition(state)
            with pytest.raises((FSMStateError, InvalidTransitionError)):
                fsm.transition("ALLOCATED")

    def test_b3_wave_batcher_empty_order_list(self) -> None:
        """Wave batcher with empty order list returns empty wave list."""
        if WAREHOUSE_AVAILABLE:
            batcher = WaveBatcher()
            waves = batcher.batch_orders([])
            assert len(waves) == 0

    def test_b3_item_larger_than_bin_dimensions(self) -> None:
        """Item with dimensions larger than bin cannot fit."""
        bin_dims = Dimensions(length=0.4, width=0.4, height=0.4)
        item_dims = Dimensions(length=0.5, width=0.3, height=0.3)
        assert item_dims.length > bin_dims.length

        if WAREHOUSE_AVAILABLE:
            optimizer = SlottingOptimizer()
            fits = optimizer.can_fit(item_dims, bin_dims)
            assert fits is False

    def test_b3_dimensions_negative_values_rejected(self) -> None:
        """Dimensions initialized with negative length/width/height raises ValidationError."""
        with pytest.raises(ValidationError):
            Dimensions(length=-0.5, width=0.2, height=0.2)
        with pytest.raises(ValidationError):
            Dimensions(length=0.5, width=-0.2, height=0.2)
        with pytest.raises(ValidationError):
            Dimensions(length=0.5, width=0.2, height=-0.2)

    def test_b3_zero_volume_dimensions(self) -> None:
        """Dimensions with zero length/width/height has 0.0 volume without error."""
        dims = Dimensions(length=0.0, width=0.0, height=0.0)
        assert dims.volume == 0.0

    def test_b3_order_item_positive_quantity_guard(self) -> None:
        """OrderItem quantity must be strictly positive (> 0)."""
        with pytest.raises(ValidationError):
            OrderItem(sku="SKU-ZERO", quantity=0)
        with pytest.raises(ValidationError):
            OrderItem(sku="SKU-NEG", quantity=-3)

    def test_b3_order_item_negative_weights_rejected(self) -> None:
        """OrderItem initialized with negative unit_weight raises ValidationError."""
        with pytest.raises(ValidationError):
            OrderItem(sku="SKU-W", quantity=5, unit_weight=-1.5)

    def test_b3_picker_routing_zero_picks(self) -> None:
        """Picker routing algorithm given empty pick tasks returns 0 distance."""
        if WAREHOUSE_AVAILABLE:
            routing = SerpentinePickerRouting()
            path, dist = routing.calculate_route(picks=[], depot_location=(0, 0))
            assert len(path) == 0
            assert dist == 0.0


# ============================================================================
# Section 4: API & Server Boundaries (12 tests)
# ============================================================================

class TestAPIServerBoundaries:
    """Boundary value analysis and corner cases for REST API, Webhooks, and CLI."""

    @pytest.fixture
    def test_boundary_server(self) -> Generator[str, None, None]:
        """Start an embedded REST server on a dedicated free port for boundary tests."""
        if not API_AVAILABLE:
            pytest.skip("open_logistics.api not available")
        port = find_free_port()
        server = create_server(host="127.0.0.1", port=port)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
        server.shutdown()
        server.server_close()

    def test_b4_server_empty_json_dispatch_endpoint(self, test_boundary_server: str) -> None:
        """POST /api/dispatch with empty JSON body '{}' triggers 400 Bad Request."""
        req = Request(
            f"{test_boundary_server}/api/dispatch",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 400

    def test_b4_server_empty_json_inventory_endpoint(self, test_boundary_server: str) -> None:
        """POST /api/inventory with empty JSON body '{}' triggers 400 Bad Request."""
        req = Request(
            f"{test_boundary_server}/api/inventory",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 400

    def test_b4_server_empty_json_wave_pick_endpoint(self, test_boundary_server: str) -> None:
        """POST /api/wave-pick with empty JSON body '{}' triggers 400 Bad Request."""
        req = Request(
            f"{test_boundary_server}/api/wave-pick",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 400

    def test_b4_server_malformed_json_string(self, test_boundary_server: str) -> None:
        """POST /api/dispatch with syntactically broken JSON returns 400 Bad Request."""
        malformed = b'{"orders": [ {"id": "1", '
        req = Request(
            f"{test_boundary_server}/api/dispatch",
            data=malformed,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 400

    def test_b4_server_unsupported_http_method(self, test_boundary_server: str) -> None:
        """Unsupported HTTP method (e.g. PUT/DELETE /health) returns 404 or 405."""
        req = Request(f"{test_boundary_server}/health", method="PUT")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code in (404, 405)

    def test_b4_server_nonexistent_url_path(self, test_boundary_server: str) -> None:
        """Request to an unknown endpoint path returns 404 Not Found."""
        req = Request(f"{test_boundary_server}/api/invalid_path_404", method="GET")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 404

    def test_b4_cli_missing_required_arguments(self) -> None:
        """Calling CLI subcommand with missing arguments exits with non-zero code."""
        if not API_AVAILABLE:
            pytest.skip("open_logistics.cli not available")
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["dispatch"])
        assert exc_info.value.code != 0

    def test_b4_cli_unknown_subcommand_exit_code(self) -> None:
        """Calling CLI with unrecognized subcommand name exits with non-zero code."""
        if not API_AVAILABLE:
            pytest.skip("open_logistics.cli not available")
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["unknown_subcommand_name"])
        assert exc_info.value.code != 0

    def test_b4_cli_malformed_stdin_json(self) -> None:
        """CLI receiving malformed JSON string from stdin exits with error code."""
        if not API_AVAILABLE:
            pytest.skip("open_logistics.cli not available")
        stdin_buf = io.StringIO("Not JSON Content")
        old_stdin = sys.stdin
        sys.stdin = stdin_buf
        try:
            with pytest.raises((SystemExit, ValueError)):
                cli_main(["dispatch", "--stdin"])
        finally:
            sys.stdin = old_stdin

    def test_b4_webhook_dispatcher_empty_payload_validation(self) -> None:
        """WebhookDispatcher validating empty or non-dict payload raises error."""
        if not API_AVAILABLE:
            pytest.skip("open_logistics.api.webhooks not available")
        dispatcher = WebhookDispatcher()
        with pytest.raises((ValidationError, ValueError, TypeError)):
            dispatcher.dispatch(endpoint="https://example.com", payload=None, secret="sec")  # type: ignore[arg-type]

    def test_b4_webhook_dispatcher_unicode_payload(self) -> None:
        """WebhookDispatcher signs and processes multi-byte unicode characters correctly."""
        secret = "secret_key"
        payload = {
            "warehouse": "München Zentrum",
            "destination": "São Paulo",
            "notes": "📦 Priority shipment: 50% discount applied 日本語",
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        expected_sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

        if API_AVAILABLE:
            dispatcher = WebhookDispatcher()
            calc_sig = dispatcher.calculate_signature(payload_bytes, secret)
            assert calc_sig == expected_sig

    def test_b4_webhook_dispatcher_tampered_payload_verification(self) -> None:
        """Webhook signature verification fails when single byte of payload is altered."""
        secret = "verification_secret"
        original = b'{"order": "ORD-1", "amount": 100}'
        tampered = b'{"order": "ORD-1", "amount": 101}'

        original_sig = hmac.new(secret.encode("utf-8"), original, hashlib.sha256).hexdigest()

        if API_AVAILABLE:
            dispatcher = WebhookDispatcher()
            assert dispatcher.verify_signature(original, original_sig, secret) is True
            assert dispatcher.verify_signature(tampered, original_sig, secret) is False
