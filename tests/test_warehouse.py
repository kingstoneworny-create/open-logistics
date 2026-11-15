"""Comprehensive Unit Tests for Milestone M4: Warehouse Automation, 3D Slotting & Task Dispatch.

Covers all 9 warehouse features:
1. Feature 29: 3D Spatial Warehouse Grid (BoundingBox3D, Bin, Aisle, Zone, Layout)
2. Feature 30: 3D Cell Placement Heuristic (Volume slack, dock proximity, 6 orthogonal rotations, capacity)
3. Feature 31: Serpentine S-Shape Picking Path (Alternating aisle traversal)
4. Feature 32: Return & Midpoint Picking Paths (Return routing, midpoint partitioning, travel distance)
5. Feature 33: Pick-Pack-Ship 7-Stage FSM (Sequential lifecycle CREATED -> ... -> SHIPPED)
6. Feature 34: State Transition Guards & Rollback (Validation guards, rollback from PICKING, terminal SHIPPED)
7. Feature 35: Carrier Cutoff Wave Batching (Carrier grouping, cutoff scheduling)
8. Feature 36: Capacity-Constrained Wave Batching (Cart weight, item count, cube limits)
9. Feature 37: WarehouseTaskDispatcher (Wave orchestration, serpentine manifests, task lifecycle, EventBus)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import pytest

from open_logistics.core.events import Event, EventBus
from open_logistics.core.exceptions import FSMStateError, WarehouseError
from open_logistics.core.models import Order, OrderItem, StockItem
from open_logistics.warehouse import (
    Aisle,
    Bin,
    BinLocation,
    BoundingBox3D,
    CapacityExceededError,
    InvalidTransitionError,
    MidpointPickerRouting,
    OrderFSM,
    OrderLifecycleManager,
    OrderStatus,
    PickTask,
    PickWave,
    ReturnPickerRouting,
    SerpentinePickerRouting,
    SlottingOptimizer,
    SlottingRecommendation,
    WarehouseLayout,
    Zone,
    WarehouseTaskDispatcher,
    WaveBatcher,
    batch_waves,
    calculate_path_distance,
    can_fit_dimensions,
    find_optimal_slot,
    midpoint_path,
    return_path,
    serpentine_path,
    slot_inventory,
)


# ============================================================================
# 1. Feature 29: 3D Spatial Warehouse Grid
# ============================================================================

class TestSpatialWarehouseGrid:
    """Tests for 3D BoundingBox3D, Bin, Aisle, Zone, and WarehouseLayout."""

    def test_bounding_box_geometry_and_volume(self) -> None:
        bbox = BoundingBox3D(x=1.0, y=2.0, z=0.5, width=2.0, length=3.0, height=1.5)
        assert bbox.volume == pytest.approx(9.0, rel=1e-6)
        assert bbox.center == (2.0, 3.5, 1.25)

    def test_bounding_box_contains_point(self) -> None:
        bbox = BoundingBox3D(x=0.0, y=0.0, z=0.0, width=5.0, length=5.0, height=5.0)
        assert bbox.contains_point(2.5, 2.5, 2.5) is True
        assert bbox.contains_point(0.0, 0.0, 0.0) is True
        assert bbox.contains_point(5.0, 5.0, 5.0) is True
        assert bbox.contains_point(5.1, 2.5, 2.5) is False
        assert bbox.contains_point(-0.1, 2.5, 2.5) is False

    def test_bounding_box_overlaps(self) -> None:
        b1 = BoundingBox3D(x=0.0, y=0.0, z=0.0, width=2.0, length=2.0, height=2.0)
        b2 = BoundingBox3D(x=1.0, y=1.0, z=1.0, width=2.0, length=2.0, height=2.0)
        b3 = BoundingBox3D(x=3.0, y=3.0, z=3.0, width=2.0, length=2.0, height=2.0)
        assert b1.overlaps(b2) is True
        assert b2.overlaps(b1) is True
        assert b1.overlaps(b3) is False

    def test_bounding_box_negative_dimensions_rejected(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox3D(x=0.0, y=0.0, z=0.0, width=-1.0, length=2.0, height=2.0)

    def test_bin_properties_and_capacities(self) -> None:
        bbox = BoundingBox3D(x=2.0, y=4.0, z=1.0, width=1.0, length=1.2, height=0.8)
        b = Bin(
            id="BIN-A1-B02-T1",
            zone="ZONE-A",
            aisle=1,
            bay=2,
            tier=1,
            bbox=bbox,
            max_weight=250.0,
            current_weight=50.0,
            current_volume=0.2,
        )
        assert b.id == "BIN-A1-B02-T1"
        assert b.zone == "ZONE-A"
        assert b.aisle == 1
        assert b.bay == 2
        assert b.tier == 1
        assert b.volume == pytest.approx(0.96, rel=1e-4)
        assert b.remaining_weight == pytest.approx(200.0, rel=1e-4)
        assert b.remaining_volume == pytest.approx(0.76, rel=1e-4)
        assert b.center == (2.5, 4.6, 1.4)

    def test_warehouse_layout_queries_and_dock_distance(self) -> None:
        layout = WarehouseLayout(id="DC-TEST", dock_location=(0.0, 0.0, 0.0))
        zone1 = Zone(id="Z1", name="Bulk", aisles=[Aisle(id=1, zone="Z1", bays=10)])
        layout.add_zone(zone1)

        b1 = Bin(id="B1", zone="Z1", aisle=1, bay=1, tier=1, x=3.0, y=4.0, z=0.0, sku="SKU-APPLE")
        b2 = Bin(id="B2", zone="Z1", aisle=1, bay=2, tier=1, x=6.0, y=8.0, z=0.0, sku="SKU-ORANGE")
        layout.add_bin(b1)
        layout.add_bin(b2)

        assert layout.get_bin("B1") == b1
        assert layout.get_bins_by_sku("SKU-APPLE") == [b1]
        assert len(layout.get_bins_by_zone("Z1")) == 2
        assert len(layout.get_bins_by_aisle(1)) == 2
        assert layout.distance_to_dock(b1) == pytest.approx(5.0, rel=1e-4)
        assert layout.distance_to_dock(b2) == pytest.approx(10.0, rel=1e-4)


# ============================================================================
# 2. Feature 30: 3D Cell Placement Heuristic
# ============================================================================

class TestCellPlacementHeuristic:
    """Tests for 3D orthogonal rotations, volume slack, dock proximity, and capacity."""

    def test_orthogonal_rotations_all_six_permutations(self) -> None:
        bin_dims = (1.0, 2.0, 3.0)
        # Permutations of (0.8, 1.8, 2.8) all fit
        for perm in [(0.8, 1.8, 2.8), (0.8, 2.8, 1.8), (1.8, 0.8, 2.8),
                     (1.8, 2.8, 0.8), (2.8, 0.8, 1.8), (2.8, 1.8, 0.8)]:
            assert can_fit_dimensions(perm, bin_dims) is True

        # Item with one dimension too large cannot fit regardless of rotation
        assert can_fit_dimensions((3.1, 1.0, 0.5), bin_dims) is False

    def test_find_optimal_slot_minimizes_slack_and_dock_distance(self) -> None:
        b_near_loose = Bin(id="B-NEAR-LOOSE", x=2.0, y=2.0, z=0.0, width=2.0, length=2.0, height=2.0)  # vol 8.0, dist ~ 2.82
        b_near_tight = Bin(id="B-NEAR-TIGHT", x=2.0, y=2.0, z=0.0, width=1.0, length=1.0, height=1.0)  # vol 1.0, dist ~ 2.82
        b_far_tight = Bin(id="B-FAR-TIGHT", x=10.0, y=10.0, z=0.0, width=1.0, length=1.0, height=1.0)  # vol 1.0, dist ~ 14.14

        pkg = (0.8, 0.8, 0.8)  # vol ~ 0.512
        available = [b_near_loose, b_near_tight, b_far_tight]

        # b_near_tight has minimal slack and minimal dock distance
        best = find_optimal_slot(package_dims=pkg, weight=5.0, available_bins=available)
        assert best is not None
        assert best.id == "B-NEAR-TIGHT"

    def test_capacity_exceeded_error_enforced(self) -> None:
        layout = WarehouseLayout()
        b = Bin(id="B-CAP", x=1.0, y=1.0, z=0.0, max_weight=50.0, max_volume=0.5)
        layout.add_bin(b)
        optimizer = SlottingOptimizer(layout=layout)

        with pytest.raises(CapacityExceededError):
            optimizer.assign_item("HEAVY", "B-CAP", weight=60.0, volume=0.1)

        with pytest.raises(CapacityExceededError):
            optimizer.assign_item("BULKY", "B-CAP", weight=10.0, volume=0.8)

    def test_slot_inventory_batch_assignment(self) -> None:
        layout = WarehouseLayout(dock_location=(0.0, 0.0, 0.0))
        b1 = Bin(id="B1", x=2.0, y=2.0, z=0.0)
        b2 = Bin(id="B2", x=5.0, y=5.0, z=0.0)
        layout.add_bin(b1)
        layout.add_bin(b2)

        items = [
            StockItem(sku="SKU-A", on_hand=10, daily_demand_history=[20.0] * 5),
            StockItem(sku="SKU-B", on_hand=10, daily_demand_history=[1.0] * 5),
        ]
        assigned = slot_inventory(items=items, layout=layout)
        assert assigned["SKU-A"] == "B1"
        assert assigned["SKU-B"] == "B2"


# ============================================================================
# 3. Feature 31 & 32: Picker Routing Paths
# ============================================================================

class TestPickerRoutingPaths:
    """Tests for Serpentine, Return, Midpoint routing and travel distance calculations."""

    @pytest.fixture
    def multi_aisle_bins(self) -> List[Bin]:
        return [
            Bin(id="A1-B1", aisle=1, bay=1, x=2.0, y=2.0, z=0.0),
            Bin(id="A1-B5", aisle=1, bay=5, x=2.0, y=10.0, z=0.0),
            Bin(id="A2-B2", aisle=2, bay=2, x=4.0, y=4.0, z=0.0),
            Bin(id="A2-B5", aisle=2, bay=5, x=4.0, y=10.0, z=0.0),
            Bin(id="A3-B1", aisle=3, bay=1, x=6.0, y=2.0, z=0.0),
            Bin(id="A3-B4", aisle=3, bay=4, x=6.0, y=8.0, z=0.0),
        ]

    def test_serpentine_s_shape_path_ordering(self, multi_aisle_bins: List[Bin]) -> None:
        path = serpentine_path(multi_aisle_bins)
        assert len(path) == len(multi_aisle_bins)

        # Aisle 1 must ascend
        a1 = [b for b in path if b.aisle == 1]
        assert a1[0].bay < a1[1].bay

        # Aisle 2 must descend (S-shape alternation)
        a2 = [b for b in path if b.aisle == 2]
        assert a2[0].bay > a2[1].bay

        # Aisle 3 must ascend again
        a3 = [b for b in path if b.aisle == 3]
        assert a3[0].bay < a3[1].bay

    def test_return_path_and_distance(self, multi_aisle_bins: List[Bin]) -> None:
        router_ret = ReturnPickerRouting()
        router_serp = SerpentinePickerRouting()

        dist_ret = router_ret.calculate_distance(multi_aisle_bins, dock=(0.0, 0.0, 0.0))
        dist_serp = router_serp.calculate_distance(multi_aisle_bins, dock=(0.0, 0.0, 0.0))

        # Return path visits mouth of aisle every time, hence travel distance >= serpentine
        assert dist_ret >= dist_serp

    def test_midpoint_path_partition(self) -> None:
        picks = [
            (2.0, 3.0, 0.0),   # Front (<= 10.0)
            (2.0, 15.0, 0.0),  # Rear (> 10.0)
            (4.0, 4.0, 0.0),   # Front (<= 10.0)
            (4.0, 16.0, 0.0),  # Rear (> 10.0)
        ]
        path = midpoint_path(picks, midpoint_y=10.0)
        assert len(path) == 4
        # First 2 stops must be front half
        assert path[0][1] <= 10.0
        assert path[1][1] <= 10.0
        # Next 2 stops must be rear half
        assert path[2][1] > 10.0
        assert path[3][1] > 10.0

    def test_path_distance_zero_and_single_stop(self) -> None:
        dock = (0.0, 0.0, 0.0)
        assert calculate_path_distance([], dock_location=dock) == 0.0

        single = [(3.0, 4.0, 0.0)]
        dist = calculate_path_distance(single, dock_location=dock)
        # 5.0 to pick + 5.0 return = 10.0
        assert dist == pytest.approx(10.0, rel=1e-4)


# ============================================================================
# 4. Feature 33 & 34: Pick-Pack-Ship 7-Stage FSM with Rollback
# ============================================================================

class TestOrderLifecycleFSM:
    """Tests for 7-stage sequential progression, transition guards, and atomic rollback."""

    def test_full_seven_stage_progression(self) -> None:
        fsm = OrderFSM(order_id="ORD-LIFECYCLE-01")
        assert fsm.current_state == OrderStatus.CREATED

        stages = [
            OrderStatus.ALLOCATED,
            OrderStatus.RELEASED,
            OrderStatus.PICKING,
            OrderStatus.PICKED,
            OrderStatus.PACKED,
            OrderStatus.SHIPPED,
        ]
        for st in stages:
            fsm.transition(st)
            assert fsm.current_state == st

    def test_invalid_forward_skip_raises_error(self) -> None:
        fsm = OrderFSM(order_id="ORD-SKIP")
        with pytest.raises(InvalidTransitionError):
            fsm.transition(OrderStatus.PICKING)

        fsm.transition(OrderStatus.ALLOCATED)
        with pytest.raises(InvalidTransitionError):
            fsm.transition(OrderStatus.SHIPPED)

    def test_invalid_backward_transition_raises_error(self) -> None:
        fsm = OrderFSM(order_id="ORD-BACK")
        fsm.transition(OrderStatus.ALLOCATED)
        fsm.transition(OrderStatus.RELEASED)
        fsm.transition(OrderStatus.PICKING)

        with pytest.raises(InvalidTransitionError):
            fsm.transition(OrderStatus.CREATED)

    def test_shipped_state_is_terminal_and_immutable(self) -> None:
        fsm = OrderFSM(order_id="ORD-TERM")
        for s in [OrderStatus.ALLOCATED, OrderStatus.RELEASED, OrderStatus.PICKING,
                  OrderStatus.PICKED, OrderStatus.PACKED, OrderStatus.SHIPPED]:
            fsm.transition(s)

        with pytest.raises(InvalidTransitionError):
            fsm.transition(OrderStatus.CANCELLED)

        with pytest.raises(InvalidTransitionError):
            fsm.rollback()

    def test_atomic_rollback_from_picking_restores_allocated(self) -> None:
        fsm = OrderFSM(order_id="ORD-RB")
        fsm.transition(OrderStatus.ALLOCATED)
        fsm.transition(OrderStatus.RELEASED)
        fsm.transition(OrderStatus.PICKING)
        assert fsm.current_state == OrderStatus.PICKING

        fsm.rollback()
        assert fsm.current_state == OrderStatus.ALLOCATED

    def test_order_lifecycle_manager_allocation_and_rollback(self) -> None:
        stock = StockItem(sku="SKU-TEST-MGR", on_hand=100, allocated=0)
        mgr = OrderLifecycleManager()

        mgr.allocate_order("ORD-MGR-01", "SKU-TEST-MGR", 25, stock)
        assert stock.allocated == 25
        assert stock.available == 75

        mgr.rollback_order("ORD-MGR-01", stock)
        assert stock.allocated == 0
        assert stock.available == 100

    def test_fsm_history_tracking_with_context(self) -> None:
        fsm = OrderFSM(order_id="ORD-HIST")
        fsm.transition(OrderStatus.ALLOCATED, context={"user": "picker_1"})
        fsm.transition(OrderStatus.RELEASED, context={"wave_id": "WAVE-001"})

        assert len(fsm.history) == 3  # init + 2 transitions
        assert fsm.history[-1]["context"]["wave_id"] == "WAVE-001"


# ============================================================================
# 5. Feature 35 & 36: Carrier Cutoff & Capacity-Constrained Wave Batching
# ============================================================================

class TestPickWaveBatching:
    """Tests for cutoff scheduling, picker cart weight, and item count limits."""

    def test_carrier_cutoff_grouping(self) -> None:
        orders = [
            Order(id="O1", destination=(41.8, -87.6), weight=10.0, metadata={"carrier": "DHL", "cutoff": 14.0}),
            Order(id="O2", destination=(41.8, -87.6), weight=10.0, metadata={"carrier": "DHL", "cutoff": 14.0}),
            Order(id="O3", destination=(41.8, -87.6), weight=10.0, metadata={"carrier": "UPS", "cutoff": 16.0}),
        ]
        waves = batch_waves(orders, max_orders=10)
        assert len(waves) == 2
        # Verify DHL wave has cutoff 14.0 and UPS has 16.0
        assert waves[0].carrier_cutoff == 14.0
        assert waves[1].carrier_cutoff == 16.0

    def test_picker_cart_weight_limit_splitting(self) -> None:
        orders = [
            Order(id="O1", destination=(41.8, -87.6), weight=40.0),
            Order(id="O2", destination=(41.8, -87.6), weight=40.0),  # Cumul = 80
            Order(id="O3", destination=(41.8, -87.6), weight=30.0),  # 80 + 30 = 110 > 100 -> splits
        ]
        waves = batch_waves(orders, max_orders=10, max_weight=100.0)
        assert len(waves) == 2
        for w in waves:
            assert w.total_weight <= 100.0

    def test_order_count_limit_splitting(self) -> None:
        orders = [Order(id=f"O-{i}", destination=(41.8, -87.6), weight=2.0) for i in range(15)]
        waves = batch_waves(orders, max_orders=5, max_weight=1000.0)
        assert len(waves) == 3
        for w in waves:
            assert len(w.orders) <= 5

    def test_empty_orders_returns_empty_waves(self) -> None:
        assert batch_waves([]) == []


# ============================================================================
# 6. Feature 37: WarehouseTaskDispatcher
# ============================================================================

class TestWarehouseTaskDispatcherWorkflow:
    """Tests for automated wave creation, manifest generation, task lifecycle, and events."""

    @pytest.fixture
    def setup_dispatcher(self) -> Tuple[WarehouseTaskDispatcher, List[Order], EventBus]:
        layout = WarehouseLayout(dock_location=(0.0, 0.0, 0.0))
        b1 = Bin(id="BIN-1", aisle=1, bay=1, tier=1, x=2.0, y=2.0, z=0.0, sku="SKU-1")
        b2 = Bin(id="BIN-2", aisle=1, bay=5, tier=1, x=2.0, y=10.0, z=0.0, sku="SKU-2")
        layout.add_bin(b1)
        layout.add_bin(b2)

        bus = EventBus()
        dispatcher = WarehouseTaskDispatcher(layout=layout, event_bus=bus)

        orders = [
            Order(id="ORD-T1", destination=(41.8, -87.6), weight=5.0, items=[OrderItem(sku="SKU-1", quantity=2)]),
            Order(id="ORD-T2", destination=(41.8, -87.6), weight=8.0, items=[OrderItem(sku="SKU-2", quantity=1)]),
        ]
        return dispatcher, orders, bus

    def test_dispatcher_end_to_end_lifecycle(
        self,
        setup_dispatcher: Tuple[WarehouseTaskDispatcher, List[Order], EventBus],
    ) -> None:
        dispatcher, orders, bus = setup_dispatcher
        captured: List[Event] = []
        bus.subscribe("warehouse.*", lambda e: captured.append(e))

        # 1. Create wave
        wave = dispatcher.create_wave(orders=orders)
        assert wave.status == "CREATED"
        assert len(wave.orders) == 2

        # 2. Generate manifest
        manifest = dispatcher.generate_picking_manifest(wave=wave, routing_strategy="serpentine")
        assert len(manifest) == 2
        assert all(t.status == "PENDING" for t in manifest)

        # 3. Dispatch wave
        dispatcher.dispatch_wave(wave_id=wave.id, picker_id="PICKER-42")
        assert wave.status == "PICKING"
        assert all(t.status == "IN_PROGRESS" for t in manifest)
        assert all(t.picker_id == "PICKER-42" for t in manifest)

        # 4. Complete tasks and verify wave completion
        for t in manifest:
            t.complete()

        assert dispatcher.is_wave_complete(wave.id) is True
        assert wave.status == "COMPLETED"

        # Verify event topics published
        topics = [e.topic for e in captured]
        assert "warehouse.wave_created" in topics
        assert "warehouse.manifest_generated" in topics
        assert "warehouse.wave_dispatched" in topics
        assert "warehouse.wave_completed" in topics
