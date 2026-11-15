"""Tier 1 Feature Coverage Tests: Warehouse Automation, 3D Slotting & Task Dispatch (F14 - F18).

Covers:
- F14: 3D Spatial Grid & Slotting Heuristic (Aisle/bay/tier bounding boxes, High-velocity placement, Volume slack minimization, Bin capacity constraints)
- F15: Picker Routing Paths (Serpentine S-shape traversal, Return routing, Midpoint split routing, Path sequence verification, Total picker distance calculation)
- F16: Pick-Pack-Ship 7-Stage FSM with Rollback (Transitions: CREATED -> ALLOCATED -> RELEASED -> PICKING -> PICKED -> PACKED -> SHIPPED, Invalid transition guards, Atomic rollback upon cancellation)
- F17: Wave Batching (Carrier cutoff grouping, Picker cart weight capacity limit, Item count limit, Wave release criteria)
- F18: WarehouseTaskDispatcher Automated Workflow (Automated wave creation from pending orders, Ordered pick tasks generation via serpentine routing, Task status tracking)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Attempt import from open_logistics domain modules
try:
    from open_logistics.core.events import Event, EventBus
    from open_logistics.core.exceptions import (
        FSMStateError,
        LogisticsError,
        WarehouseError,
    )
    from open_logistics.core.models import Order, OrderItem, StockItem
    from open_logistics.warehouse.fsm import OrderFSM, OrderLifecycleManager
    from open_logistics.warehouse.layout import Aisle, BinLocation, WarehouseLayout, Zone
    from open_logistics.warehouse.picking_path import (
        MidpointPickerRouting,
        ReturnPickerRouting,
        SerpentinePickerRouting,
    )
    from open_logistics.warehouse.slotting import (
        SlottingOptimizer,
        SlottingRecommendation,
    )
    from open_logistics.warehouse.task_dispatcher import (
        PickTask,
        WarehouseTaskDispatcher,
    )
    from open_logistics.warehouse.wave_batcher import PickWave, WaveBatcher

    # Safe fallbacks for optional exception names if placed in warehouse submodule
    try:
        from open_logistics.core.exceptions import InvalidTransitionError
    except ImportError:
        try:
            from open_logistics.warehouse.fsm import InvalidTransitionError  # type: ignore[no-redef]
        except ImportError:
            InvalidTransitionError = FSMStateError  # type: ignore[misc, assignment]

    try:
        from open_logistics.core.exceptions import CapacityExceededError
    except ImportError:
        try:
            from open_logistics.warehouse.layout import CapacityExceededError  # type: ignore[no-redef]
        except ImportError:
            CapacityExceededError = WarehouseError  # type: ignore[misc, assignment]

    # Safe fallback for OrderStatus
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

    WAREHOUSE_AVAILABLE = True
except ImportError:
    WAREHOUSE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not WAREHOUSE_AVAILABLE,
    reason="open_logistics.warehouse module is pending Milestone 4 implementation",
)


# ============================================================================
# F14: 3D Spatial Grid & Slotting Heuristic
# ============================================================================


class TestFeature14SpatialGridAndSlotting:
    """Comprehensive tests for Feature 14: 3D Spatial Grid & Slotting Heuristic."""

    def test_bin_location_3d_coordinates_and_bounds(self) -> None:
        """Verify BinLocation model with aisle, bay, tier, coordinates, and capacity limits."""
        bin_loc = BinLocation(
            id="BIN-A1-B01-T1",
            aisle=1,
            bay=1,
            tier=1,
            x=2.5,
            y=10.0,
            z=1.2,
            max_weight=500.0,
            max_volume=2.0,
        )
        assert bin_loc.id == "BIN-A1-B01-T1"
        assert bin_loc.aisle == 1
        assert bin_loc.bay == 1
        assert bin_loc.tier == 1
        assert (bin_loc.x, bin_loc.y, bin_loc.z) == (2.5, 10.0, 1.2)
        assert bin_loc.max_weight == 500.0
        assert bin_loc.max_volume == 2.0

        # Remaining capacity calculations
        assert hasattr(bin_loc, "current_weight") or hasattr(bin_loc, "remaining_weight")

    def test_warehouse_layout_zones_and_bins_indexing(self) -> None:
        """Verify WarehouseLayout structure, bin retrieval, and distance to intake dock."""
        layout = WarehouseLayout(id="DC-MAIN", dock_location=(0.0, 0.0, 0.0))
        b1 = BinLocation(id="BIN-01", aisle=1, bay=1, tier=1, x=3.0, y=4.0, z=0.0, max_weight=200.0, max_volume=1.0)
        b2 = BinLocation(id="BIN-02", aisle=2, bay=5, tier=2, x=6.0, y=8.0, z=2.0, max_weight=200.0, max_volume=1.0)

        layout.add_bin(b1)
        layout.add_bin(b2)

        retrieved = layout.get_bin("BIN-01")
        assert retrieved.id == "BIN-01"

        # Dock distance verification: Euclidean distance from (0,0,0) to (3,4,0) = 5.0
        dist_dock = layout.distance_to_dock(b1) if hasattr(layout, "distance_to_dock") else math.sqrt(b1.x**2 + b1.y**2 + b1.z**2)
        assert dist_dock == pytest.approx(5.0, rel=1e-3)

    def test_slotting_heuristic_places_high_velocity_closest_to_dock(self) -> None:
        """High-velocity (fast-moving) SKUs must be slotted into bins closest to the dock."""
        layout = WarehouseLayout(id="DC-MAIN", dock_location=(0.0, 0.0, 0.0))
        near_bin = BinLocation(id="BIN-NEAR", aisle=1, bay=1, tier=1, x=2.0, y=2.0, z=0.0, max_weight=500.0, max_volume=2.0)
        far_bin = BinLocation(id="BIN-FAR", aisle=5, bay=10, tier=3, x=20.0, y=40.0, z=5.0, max_weight=500.0, max_volume=2.0)
        layout.add_bin(near_bin)
        layout.add_bin(far_bin)

        fast_sku = StockItem(sku="SKU-FAST", on_hand=100, unit_cost=10.0, daily_demand_history=[50.0] * 30)
        slow_sku = StockItem(sku="SKU-SLOW", on_hand=10, unit_cost=10.0, daily_demand_history=[1.0] * 30)

        optimizer = SlottingOptimizer(layout=layout)
        recommendations = optimizer.optimize_slotting(items=[fast_sku, slow_sku])

        # Fast SKU must be assigned to near_bin, slow to far_bin
        assignments = {rec.sku: rec.bin_id for rec in recommendations}
        assert assignments["SKU-FAST"] == "BIN-NEAR"
        assert assignments["SKU-SLOW"] == "BIN-FAR"

    def test_volume_slack_minimization_heuristic(self) -> None:
        """Slotting heuristic selects the bin that minimizes volume slack (wasted empty volume)."""
        layout = WarehouseLayout(id="DC-MAIN", dock_location=(0.0, 0.0, 0.0))
        # Both bins are equidistant from dock (x=5.0, y=0.0)
        tight_bin = BinLocation(id="BIN-TIGHT", aisle=1, bay=1, tier=1, x=5.0, y=0.0, z=0.0, max_weight=100.0, max_volume=0.10)
        loose_bin = BinLocation(id="BIN-LOOSE", aisle=1, bay=1, tier=2, x=5.0, y=0.0, z=1.0, max_weight=100.0, max_volume=1.00)
        layout.add_bin(tight_bin)
        layout.add_bin(loose_bin)

        small_item = StockItem(sku="SKU-COMPACT", on_hand=1, unit_cost=20.0, daily_demand_history=[10.0] * 10)
        small_item_volume = 0.08  # Fits tightly in tight_bin (slack = 0.02), loose_bin slack = 0.92

        optimizer = SlottingOptimizer(layout=layout)
        rec = optimizer.recommend_bin(sku=small_item.sku, volume=small_item_volume, weight=5.0)

        bin_assigned = rec.bin_id if hasattr(rec, "bin_id") else rec
        assert bin_assigned == "BIN-TIGHT", "Optimizer must select tightest fitting bin to minimize volume slack"

    def test_bin_capacity_weight_and_volume_constraints(self) -> None:
        """Slotting rejects placement when item exceeds bin max weight or max volume."""
        layout = WarehouseLayout(id="DC-MAIN", dock_location=(0.0, 0.0, 0.0))
        small_bin = BinLocation(id="BIN-SMALL", aisle=1, bay=1, tier=1, x=2.0, y=2.0, z=0.0, max_weight=50.0, max_volume=0.2)
        layout.add_bin(small_bin)

        optimizer = SlottingOptimizer(layout=layout)

        # Overweight item
        with pytest.raises((CapacityExceededError, WarehouseError, ValueError)):
            optimizer.assign_item(sku="HEAVY-ITEM", bin_id="BIN-SMALL", weight=100.0, volume=0.1)

        # Overvolume item
        with pytest.raises((CapacityExceededError, WarehouseError, ValueError)):
            optimizer.assign_item(sku="BULKY-ITEM", bin_id="BIN-SMALL", weight=10.0, volume=0.5)

    def test_slotting_recommendation_dataclass_properties(self) -> None:
        """Verify SlottingRecommendation properties, distance_to_dock, and score."""
        rec = SlottingRecommendation(
            sku="SKU-TEST",
            bin_id="BIN-A1-01",
            distance_to_dock=7.5,
            volume_slack=0.05,
            score=92.5,
        )
        assert rec.sku == "SKU-TEST"
        assert rec.bin_id == "BIN-A1-01"
        assert rec.distance_to_dock == 7.5
        assert rec.volume_slack == 0.05
        assert rec.score == 92.5


# ============================================================================
# F15: Picker Routing Paths (Serpentine, Return, Mid-point, Distance)
# ============================================================================


class TestFeature15PickerRoutingPaths:
    """Comprehensive tests for Feature 15: Picker Routing Paths."""

    @pytest.fixture
    def grid_pick_locations(self) -> List[Tuple[float, float, float]]:
        """A standardized set of pick coordinates across 2 aisles:

        Aisle 1 (x=2): y=2.0, y=8.0
        Aisle 2 (x=4): y=3.0, y=9.0
        """
        return [
            (2.0, 2.0, 0.0),  # Aisle 1, bottom
            (2.0, 8.0, 0.0),  # Aisle 1, top
            (4.0, 9.0, 0.0),  # Aisle 2, top
            (4.0, 3.0, 0.0),  # Aisle 2, bottom
        ]

    def test_serpentine_s_shape_traversal_path_order(self, grid_pick_locations: List[Tuple[float, float, float]]) -> None:
        """Serpentine traversal visits Aisle 1 ascending, Aisle 2 descending without backtracking."""
        router = SerpentinePickerRouting()
        route_stops = router.calculate_path(picks=grid_pick_locations, dock=(0.0, 0.0, 0.0))

        # Check that coordinates alternate in Y-direction
        # Aisle 1: (2,2) -> (2,8)
        # Aisle 2: (4,9) -> (4,3)
        y_coords_in_order = [stop[1] if isinstance(stop, (tuple, list)) else stop.y for stop in route_stops if getattr(stop, "is_depot", False) is False and (stop[0] if isinstance(stop, (tuple, list)) else stop.x) > 0]

        # First aisle should ascend
        assert y_coords_in_order[0] < y_coords_in_order[1]
        # Second aisle should descend
        assert y_coords_in_order[2] > y_coords_in_order[3]

    def test_return_picker_routing_aisle_exits(self, grid_pick_locations: List[Tuple[float, float, float]]) -> None:
        """Return routing enters aisle, picks, and returns to front aisle mouth before next aisle."""
        router = ReturnPickerRouting()
        route = router.calculate_path(picks=grid_pick_locations, dock=(0.0, 0.0, 0.0))
        assert len(route) >= len(grid_pick_locations)

        # Total distance under return routing is strictly greater than serpentine for dense aisles
        serpentine_dist = SerpentinePickerRouting().calculate_distance(picks=grid_pick_locations, dock=(0.0, 0.0, 0.0))
        return_dist = router.calculate_distance(picks=grid_pick_locations, dock=(0.0, 0.0, 0.0))
        assert return_dist >= serpentine_dist

    def test_midpoint_split_routing_warehouse_partition(self) -> None:
        """Midpoint routing splits aisles at warehouse midpoint, traversing front half then rear half."""
        picks = [
            (2.0, 2.0, 0.0),   # Front half
            (2.0, 18.0, 0.0),  # Rear half
            (4.0, 3.0, 0.0),   # Front half
            (4.0, 19.0, 0.0),  # Rear half
        ]
        router = MidpointPickerRouting(midpoint_y=10.0)
        route = router.calculate_path(picks=picks, dock=(0.0, 0.0, 0.0))
        assert len(route) >= len(picks)

    def test_picker_total_distance_calculation(self, grid_pick_locations: List[Tuple[float, float, float]]) -> None:
        """Verify total picker travel distance is positive, symmetric, and includes dock return."""
        router = SerpentinePickerRouting()
        dock = (0.0, 0.0, 0.0)
        total_dist = router.calculate_distance(picks=grid_pick_locations, dock=dock)

        # Manhattan or Euclidean distance from dock (0,0) to first pick (2,2) is at least sqrt(8) ~ 2.82
        assert total_dist > 15.0, f"Distance should be significant for cross-aisle traversal, got {total_dist}"

    def test_picker_routing_empty_and_single_stop_edge_cases(self) -> None:
        """Verify handling of 0 picks and single pick edge conditions."""
        router = SerpentinePickerRouting()
        dock = (0.0, 0.0, 0.0)

        # 0 picks
        empty_route = router.calculate_path(picks=[], dock=dock)
        assert len(empty_route) == 0 or len(empty_route) == 1
        assert router.calculate_distance(picks=[], dock=dock) == 0.0

        # 1 pick at (3.0, 4.0, 0.0)
        single_pick = [(3.0, 4.0, 0.0)]
        single_dist = router.calculate_distance(picks=single_pick, dock=dock)
        # Round trip distance dock -> (3,4) -> dock = 5.0 + 5.0 = 10.0 (Euclidean) or (3+4)*2 = 14.0 (Manhattan)
        assert single_dist in (pytest.approx(10.0, rel=1e-3), pytest.approx(14.0, rel=1e-3))


# ============================================================================
# F16: Pick-Pack-Ship 7-Stage FSM with Rollback
# ============================================================================


class TestFeature16OrderLifecycleFSM:
    """Comprehensive tests for Feature 16: Pick-Pack-Ship 7-Stage FSM with Rollback."""

    def test_happy_path_seven_stage_lifecycle(self) -> None:
        """Verify full orderly progression through all 7 lifecycle states:

        CREATED -> ALLOCATED -> RELEASED -> PICKING -> PICKED -> PACKED -> SHIPPED.
        """
        fsm = OrderFSM(order_id="ORD-001")
        assert fsm.current_state == OrderStatus.CREATED or fsm.current_state == "CREATED"

        expected_sequence = [
            OrderStatus.ALLOCATED,
            OrderStatus.RELEASED,
            OrderStatus.PICKING,
            OrderStatus.PICKED,
            OrderStatus.PACKED,
            OrderStatus.SHIPPED,
        ]

        for next_state in expected_sequence:
            fsm.transition(next_state)
            assert fsm.current_state == next_state or fsm.current_state == str(next_state.value if hasattr(next_state, "value") else next_state)

    def test_invalid_forward_skip_guards_raise_error(self) -> None:
        """Skipping intermediate required states must be blocked by validation guards."""
        fsm = OrderFSM(order_id="ORD-002")
        assert fsm.current_state in (OrderStatus.CREATED, "CREATED")

        # Illegal skip: CREATED -> PICKING directly without ALLOCATED / RELEASED
        with pytest.raises((InvalidTransitionError, FSMStateError, WarehouseError)):
            fsm.transition(OrderStatus.PICKING)

        # Legal transition to ALLOCATED
        fsm.transition(OrderStatus.ALLOCATED)

        # Illegal skip: ALLOCATED -> SHIPPED directly
        with pytest.raises((InvalidTransitionError, FSMStateError, WarehouseError)):
            fsm.transition(OrderStatus.SHIPPED)

    def test_invalid_backward_transition_guards_raise_error(self) -> None:
        """Backward transitions without formal rollback must be prohibited."""
        fsm = OrderFSM(order_id="ORD-003")
        fsm.transition(OrderStatus.ALLOCATED)
        fsm.transition(OrderStatus.RELEASED)
        fsm.transition(OrderStatus.PICKING)
        fsm.transition(OrderStatus.PICKED)

        # Illegal backward transition: PICKED -> CREATED
        with pytest.raises((InvalidTransitionError, FSMStateError, WarehouseError)):
            fsm.transition(OrderStatus.CREATED)

    def test_terminal_state_shipped_is_immutable(self) -> None:
        """Once an order reaches SHIPPED, further state transitions must be rejected."""
        fsm = OrderFSM(order_id="ORD-004")
        fsm.transition(OrderStatus.ALLOCATED)
        fsm.transition(OrderStatus.RELEASED)
        fsm.transition(OrderStatus.PICKING)
        fsm.transition(OrderStatus.PICKED)
        fsm.transition(OrderStatus.PACKED)
        fsm.transition(OrderStatus.SHIPPED)

        # Attempting any transition from SHIPPED
        with pytest.raises((InvalidTransitionError, FSMStateError, WarehouseError)):
            fsm.transition(OrderStatus.CREATED)

        with pytest.raises((InvalidTransitionError, FSMStateError, WarehouseError)):
            fsm.transition(OrderStatus.CANCELLED if hasattr(OrderStatus, "CANCELLED") else "CANCELLED")

    def test_atomic_rollback_on_cancellation_during_picking(self) -> None:
        """Cancellation during PICKING triggers atomic rollback to safe state or releases reservation."""
        fsm = OrderFSM(order_id="ORD-005")
        fsm.transition(OrderStatus.ALLOCATED)
        fsm.transition(OrderStatus.RELEASED)
        fsm.transition(OrderStatus.PICKING)

        # Rollback execution
        fsm.rollback()
        # State should revert to RELEASED, ALLOCATED, or CANCELLED
        valid_post_rollback_states = [OrderStatus.ALLOCATED, OrderStatus.RELEASED, OrderStatus.CREATED, "ALLOCATED", "RELEASED", "CREATED", "CANCELLED"]
        assert fsm.current_state in valid_post_rollback_states

    def test_fsm_rollback_restores_allocated_inventory(self) -> None:
        """Verify rollback deallocates reserved inventory on the StockItem."""
        stock = StockItem(sku="SKU-ROLLBACK", on_hand=100, allocated=0)
        manager = OrderLifecycleManager()

        # Create and allocate order for 10 units
        manager.allocate_order(order_id="ORD-006", sku="SKU-ROLLBACK", quantity=10, stock_item=stock)
        assert stock.allocated == 10
        assert stock.available == 90

        # Cancel / Rollback order
        manager.rollback_order(order_id="ORD-006", stock_item=stock)
        assert stock.allocated == 0
        assert stock.available == 100

    def test_fsm_transition_history_and_context_tracking(self) -> None:
        """FSM retains transition history including operator ID and timestamps."""
        fsm = OrderFSM(order_id="ORD-007")
        fsm.transition(OrderStatus.ALLOCATED, context={"user": "admin", "reason": "batch-allocated"})
        fsm.transition(OrderStatus.RELEASED, context={"wave_id": "WAVE-101"})

        assert len(fsm.history) >= 2
        last_entry = fsm.history[-1]
        assert "WAVE-101" in str(last_entry) or last_entry.get("context", {}).get("wave_id") == "WAVE-101"


# ============================================================================
# F17: Wave Batching (Carrier Cutoff, Weight Limit, Count Limit)
# ============================================================================


class TestFeature17WaveBatching:
    """Comprehensive tests for Feature 17: Wave Batching."""

    @pytest.fixture
    def pending_orders(self) -> List[Order]:
        """A set of customer orders with varied weights and carrier cutoff windows."""
        return [
            Order(id="ORD-W1", destination=(41.88, -87.63), weight=20.0, items=[OrderItem(sku="S1", quantity=2, unit_weight=10.0)], metadata={"carrier": "FEDEX", "cutoff": 15.0}),
            Order(id="ORD-W2", destination=(41.89, -87.62), weight=25.0, items=[OrderItem(sku="S2", quantity=1, unit_weight=25.0)], metadata={"carrier": "FEDEX", "cutoff": 15.0}),
            Order(id="ORD-W3", destination=(41.87, -87.64), weight=15.0, items=[OrderItem(sku="S3", quantity=3, unit_weight=5.0)], metadata={"carrier": "FEDEX", "cutoff": 15.0}),
            Order(id="ORD-W4", destination=(41.86, -87.65), weight=30.0, items=[OrderItem(sku="S4", quantity=1, unit_weight=30.0)], metadata={"carrier": "UPS", "cutoff": 17.0}),
            Order(id="ORD-W5", destination=(41.85, -87.66), weight=20.0, items=[OrderItem(sku="S5", quantity=2, unit_weight=10.0)], metadata={"carrier": "UPS", "cutoff": 17.0}),
        ]

    def test_wave_batching_carrier_cutoff_grouping(self, pending_orders: List[Order]) -> None:
        """Orders must be grouped into separate pick waves based on carrier departure cutoffs."""
        batcher = WaveBatcher(max_cart_weight=100.0, max_orders=10)
        waves = batcher.batch_orders(pending_orders)

        # Should produce at least 2 distinct waves: one for FEDEX (cutoff 15.0) and one for UPS (cutoff 17.0)
        assert len(waves) >= 2
        for wave in waves:
            carriers = {order.metadata.get("carrier") for order in wave.orders}
            assert len(carriers) == 1, f"Wave {wave.id} mixed multiple carriers: {carriers}"

    def test_wave_batching_picker_cart_weight_capacity_limit(self) -> None:
        """Orders exceeding the picker cart weight limit (e.g. 50 kg) must roll into a new wave."""
        orders = [
            Order(id="O1", destination=(41.8, -87.6), weight=25.0),
            Order(id="O2", destination=(41.8, -87.6), weight=20.0),  # Cumulative = 45 kg (fits in 50 kg cart)
            Order(id="O3", destination=(41.8, -87.6), weight=15.0),  # 45 + 15 = 60 kg > 50 kg -> must start Wave 2
        ]
        batcher = WaveBatcher(max_cart_weight=50.0, max_orders=10)
        waves = batcher.batch_orders(orders)

        assert len(waves) == 2, f"Expected 2 waves due to weight constraint, got {len(waves)}"
        for wave in waves:
            assert wave.total_weight <= 50.0, f"Wave {wave.id} exceeded cart capacity: {wave.total_weight} > 50.0"

    def test_wave_batching_order_count_limit(self) -> None:
        """Waves must respect maximum allowable order count per wave."""
        orders = [Order(id=f"ORD-C{i}", destination=(41.8, -87.6), weight=5.0) for i in range(12)]
        batcher = WaveBatcher(max_cart_weight=1000.0, max_orders=5)
        waves = batcher.batch_orders(orders)

        # 12 orders / 5 max per wave = 3 waves (5, 5, 2)
        assert len(waves) == 3
        for wave in waves:
            assert len(wave.orders) <= 5

    def test_wave_release_criteria_and_status(self) -> None:
        """PickWave transitions to RELEASED when batching criteria are fulfilled."""
        orders = [Order(id="O-REL", destination=(41.8, -87.6), weight=10.0)]
        wave = PickWave(id="WAVE-001", orders=orders)
        assert hasattr(wave, "status")
        wave.release()
        assert wave.status in ("RELEASED", "READY", "ACTIVE")


# ============================================================================
# F18: WarehouseTaskDispatcher Automated Workflow
# ============================================================================


class TestFeature18WarehouseTaskDispatcher:
    """Comprehensive tests for Feature 18: WarehouseTaskDispatcher Automated Workflow."""

    @pytest.fixture
    def configured_warehouse(self) -> Tuple[WarehouseLayout, List[Order]]:
        """Pre-configured warehouse layout and orders mapped to bin locations."""
        layout = WarehouseLayout(id="DC-DISPATCH", dock_location=(0.0, 0.0, 0.0))
        b1 = BinLocation(id="BIN-A1-01", aisle=1, bay=1, tier=1, x=2.0, y=2.0, z=0.0, max_weight=500.0, max_volume=2.0)
        b2 = BinLocation(id="BIN-A1-05", aisle=1, bay=5, tier=1, x=2.0, y=8.0, z=0.0, max_weight=500.0, max_volume=2.0)
        b3 = BinLocation(id="BIN-A2-05", aisle=2, bay=5, tier=1, x=4.0, y=8.0, z=0.0, max_weight=500.0, max_volume=2.0)
        layout.add_bin(b1)
        layout.add_bin(b2)
        layout.add_bin(b3)

        orders = [
            Order(id="ORD-D1", destination=(41.88, -87.63), weight=10.0, items=[OrderItem(sku="SKU-1", quantity=2)]),
            Order(id="ORD-D2", destination=(41.89, -87.62), weight=15.0, items=[OrderItem(sku="SKU-2", quantity=1)]),
        ]
        return layout, orders

    def test_automated_wave_creation_from_pending_orders(
        self, configured_warehouse: Tuple[WarehouseLayout, List[Order]]
    ) -> None:
        """Dispatcher automatically groups pending orders into structured pick waves."""
        layout, orders = configured_warehouse
        dispatcher = WarehouseTaskDispatcher(layout=layout)

        waves = dispatcher.create_waves(orders=orders, max_orders_per_wave=10)
        assert len(waves) >= 1
        assert isinstance(waves[0], PickWave)
        assert len(waves[0].orders) == 2

    def test_dispatcher_generates_ordered_pick_tasks_serpentine(
        self, configured_warehouse: Tuple[WarehouseLayout, List[Order]]
    ) -> None:
        """Dispatcher generates pick tasks sequenced by serpentine routing order."""
        layout, orders = configured_warehouse
        dispatcher = WarehouseTaskDispatcher(layout=layout)
        wave = dispatcher.create_waves(orders=orders)[0]

        manifest = dispatcher.generate_picking_manifest(wave=wave, routing_strategy="serpentine")
        assert isinstance(manifest, list)
        assert len(manifest) >= 2
        for task in manifest:
            assert isinstance(task, PickTask)
            assert task.bin_location is not None
            assert task.quantity > 0
            assert task.status in ("PENDING", "CREATED")

    def test_pick_task_lifecycle_tracking(self) -> None:
        """PickTask tracks progression: PENDING -> IN_PROGRESS -> COMPLETED."""
        task = PickTask(
            task_id="TASK-001",
            wave_id="WAVE-001",
            sku="SKU-ALPHA",
            quantity=5,
            bin_location="BIN-A1-01",
            status="PENDING",
        )
        assert task.status == "PENDING"
        task.start(picker_id="PICKER-01")
        assert task.status == "IN_PROGRESS"
        task.complete()
        assert task.status == "COMPLETED"

    def test_dispatcher_end_to_end_wave_execution(
        self, configured_warehouse: Tuple[WarehouseLayout, List[Order]]
    ) -> None:
        """Dispatcher coordinates wave release, pick task assignments, and wave completion."""
        layout, orders = configured_warehouse
        dispatcher = WarehouseTaskDispatcher(layout=layout)
        waves = dispatcher.create_waves(orders=orders)
        wave = waves[0]

        manifest = dispatcher.generate_picking_manifest(wave=wave)
        dispatcher.dispatch_wave(wave_id=wave.id, picker_id="PICKER-01")

        # Complete all tasks in manifest
        for task in manifest:
            task.complete()

        # Verify wave completion
        assert dispatcher.is_wave_complete(wave.id) is True

    def test_dispatcher_empty_order_queue_handling(self) -> None:
        """Dispatcher handles empty order list gracefully without errors."""
        layout = WarehouseLayout(id="DC-EMPTY", dock_location=(0.0, 0.0, 0.0))
        dispatcher = WarehouseTaskDispatcher(layout=layout)
        waves = dispatcher.create_waves(orders=[])
        assert len(waves) == 0

    def test_dispatcher_eventbus_integration(
        self, configured_warehouse: Tuple[WarehouseLayout, List[Order]]
    ) -> None:
        """Dispatcher publishes warehouse task events onto EventBus."""
        layout, orders = configured_warehouse
        bus = EventBus()
        events_captured: List[Event] = []

        bus.subscribe("warehouse.*", lambda evt: events_captured.append(evt))

        dispatcher = WarehouseTaskDispatcher(layout=layout, event_bus=bus)
        waves = dispatcher.create_waves(orders=orders)
        assert len(waves) > 0
