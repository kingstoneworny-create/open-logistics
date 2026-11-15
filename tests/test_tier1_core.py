"""Tier 1 Core Tests: Features 1-4.

F1: In-Memory Pub/Sub EventBus
F2: Core Domain Models (Order, Vehicle, Depot, Route, Stop, StockItem, PurchaseOrder)
F3: Domain Exceptions Hierarchy
F4: Distance Metrics (Euclidean, Manhattan, Haversine)
"""

import math
import pytest
from typing import Any, Dict, List

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

# Distance metrics are implemented under open_logistics.routing.distances in M2
try:
    from open_logistics.routing.distances import (
        euclidean_distance,
        haversine_distance,
        manhattan_distance,
    )
    HAS_DISTANCES = True
except ImportError:
    HAS_DISTANCES = False
    euclidean_distance = None  # type: ignore
    haversine_distance = None  # type: ignore
    manhattan_distance = None  # type: ignore


# ==============================================================================
# Feature 1: In-Memory Pub/Sub EventBus (F1)
# ==============================================================================


class TestFeature1EventBus:
    """Comprehensive test suite for Feature 1: In-Memory Pub/Sub EventBus."""

    def test_f1_basic_pub_sub(self, event_bus: EventBus) -> None:
        """Verify basic publish and subscribe delivers exact event to callback."""
        received_events: List[Event] = []

        def on_order_created(event: Event) -> None:
            received_events.append(event)

        sub_id = event_bus.subscribe("order.created", on_order_created)
        assert isinstance(sub_id, str) and len(sub_id) > 0

        event = Event(
            topic="order.created",
            payload={"order_id": "ORD-101", "weight": 25.0},
            source="test_runner",
        )
        notified = event_bus.publish(event)

        assert notified == 1
        assert len(received_events) == 1
        delivered = received_events[0]
        assert delivered.topic == "order.created"
        assert delivered.payload == {"order_id": "ORD-101", "weight": 25.0}
        assert delivered.source == "test_runner"
        assert delivered.timestamp > 0.0
        assert delivered.event_id == event.event_id

    def test_f1_multiple_subscribers(self, event_bus: EventBus) -> None:
        """Verify multiple independent subscribers to same topic receive event in order."""
        call_log: List[str] = []

        event_bus.subscribe("stock.depleted", lambda e: call_log.append("subscriber_alpha"))
        event_bus.subscribe("stock.depleted", lambda e: call_log.append("subscriber_beta"))
        event_bus.subscribe("stock.depleted", lambda e: call_log.append("subscriber_gamma"))

        event = Event(topic="stock.depleted", payload={"sku": "SKU-999", "on_hand": 0})
        notified = event_bus.publish(event)

        assert notified == 3
        assert call_log == ["subscriber_alpha", "subscriber_beta", "subscriber_gamma"]

    def test_f1_single_level_wildcard_topic_matching(self, event_bus: EventBus) -> None:
        """Verify '*' matches exactly one dot-delimited segment."""
        matched: List[str] = []

        event_bus.subscribe("order.*", lambda e: matched.append(e.topic))

        event_bus.publish(Event(topic="order.created", payload={}))
        event_bus.publish(Event(topic="order.cancelled", payload={}))
        event_bus.publish(Event(topic="order", payload={}))  # No segment after dot
        event_bus.publish(Event(topic="order.item.added", payload={}))  # Two segments after dot
        event_bus.publish(Event(topic="inventory.created", payload={}))  # Different root

        assert matched == ["order.created", "order.cancelled"]

    def test_f1_multi_level_wildcard_topic_matching(self, event_bus: EventBus) -> None:
        """Verify '#' and '**' match zero or more dot-delimited segments."""
        matched_warehouse: List[str] = []
        matched_alerts: List[str] = []

        event_bus.subscribe("warehouse.#", lambda e: matched_warehouse.append(e.topic))
        event_bus.subscribe("#.alert", lambda e: matched_alerts.append(e.topic))

        # Warehouse events
        event_bus.publish(Event(topic="warehouse", payload={}))
        event_bus.publish(Event(topic="warehouse.zone_a", payload={}))
        event_bus.publish(Event(topic="warehouse.zone_a.aisle_3.bin_12", payload={}))
        event_bus.publish(Event(topic="routing.vehicle.dispatched", payload={}))

        assert matched_warehouse == [
            "warehouse",
            "warehouse.zone_a",
            "warehouse.zone_a.aisle_3.bin_12",
        ]

        # Alert events
        event_bus.publish(Event(topic="alert", payload={}))
        event_bus.publish(Event(topic="stock.alert", payload={}))
        event_bus.publish(Event(topic="system.sensor.temperature.alert", payload={}))
        event_bus.publish(Event(topic="system.normal", payload={}))

        assert matched_alerts == [
            "alert",
            "stock.alert",
            "system.sensor.temperature.alert",
        ]

    def test_f1_unsubscription_and_token(self, event_bus: EventBus) -> None:
        """Verify unsubscription stops delivery and returns correct boolean status."""
        calls: List[str] = []

        def callback(event: Event) -> None:
            calls.append(event.topic)

        sub_id = event_bus.subscribe("dispatch.route", callback)
        assert len(event_bus) == 1

        event_bus.publish(Event(topic="dispatch.route", payload={"route_id": "R1"}))
        assert len(calls) == 1

        # Unsubscribe with valid token
        removed = event_bus.unsubscribe(sub_id)
        assert removed is True
        assert len(event_bus) == 0

        # Subsequent publish should not invoke callback
        event_bus.publish(Event(topic="dispatch.route", payload={"route_id": "R2"}))
        assert len(calls) == 1

        # Unsubscribe with non-existent token
        removed_again = event_bus.unsubscribe("non-existent-token")
        assert removed_again is False

    def test_f1_event_payload_integrity_and_serialization(self) -> None:
        """Verify Event serialization to and from dict preserves data fidelity and types."""
        complex_payload = {
            "order_id": "ORD-555",
            "weights": [12.5, 34.2, 5.0],
            "metadata": {"customer_tier": "VIP", "express": True, "notes": None},
        }
        event = Event(
            topic="orders.processed",
            payload=complex_payload,
            source="order_service",
            timestamp=1700000000.0,
            event_id="custom-uuid-1234",
        )

        serialized = event.to_dict()
        assert serialized["topic"] == "orders.processed"
        assert serialized["payload"] == complex_payload
        assert serialized["source"] == "order_service"
        assert serialized["timestamp"] == 1700000000.0
        assert serialized["event_id"] == "custom-uuid-1234"

        reconstructed = Event.from_dict(serialized)
        assert reconstructed.topic == event.topic
        assert reconstructed.payload == event.payload
        assert reconstructed.source == event.source
        assert reconstructed.timestamp == event.timestamp
        assert reconstructed.event_id == event.event_id

    def test_f1_subscriber_error_isolation(self, event_bus: EventBus) -> None:
        """Verify exceptions in one subscriber do not block other subscribers."""
        executed: List[str] = []

        def failing_handler(event: Event) -> None:
            raise RuntimeError("Database connection timed out during event processing")

        def successful_handler(event: Event) -> None:
            executed.append("successful_handler_executed")

        event_bus.subscribe("payment.received", failing_handler)
        event_bus.subscribe("payment.received", successful_handler)

        event = Event(topic="payment.received", payload={"amount": 99.99})
        notified = event_bus.publish(event)

        assert notified == 2
        assert executed == ["successful_handler_executed"]

        failed = event_bus.get_failed_dispatches()
        assert len(failed) == 1
        assert "Database connection timed out" in failed[0]["error"]
        assert failed[0]["topic"] == "payment.received"


# ==============================================================================
# Feature 2: Core Domain Models (F2)
# ==============================================================================


class TestFeature2CoreModels:
    """Comprehensive test suite for Feature 2: Core Domain Models."""

    def test_f2_order_creation_validation_and_serialization(self) -> None:
        """Validate Order dataclass constraints, line items aggregation, and serialization."""
        items = [
            OrderItem(sku="SKU-A", quantity=2, unit_weight=5.0, unit_volume=0.01),
            OrderItem(sku="SKU-B", quantity=3, unit_weight=10.0, unit_volume=0.02),
        ]
        order = Order(
            id="ORD-200",
            destination=(41.8825, -87.6225),
            weight=40.0,
            volume=0.08,
            priority=3,
            time_window=(9.0, 13.0),
            items=items,
            status="PENDING",
            metadata={"tag": "fragile"},
        )

        assert order.id == "ORD-200"
        assert order.weight == 40.0
        assert order.volume == 0.08
        assert order.priority == 3
        assert order.time_window == (9.0, 13.0)
        assert len(order.items) == 2

        # Test sync_metrics_from_items
        order.sync_metrics_from_items()
        assert order.weight == 40.0  # 2*5.0 + 3*10.0 = 40.0
        assert order.volume == 0.08  # 2*0.01 + 3*0.02 = 0.08

        # Serialization roundtrip
        order_dict = order.to_dict()
        reconstructed = Order.from_dict(order_dict)
        assert reconstructed.id == order.id
        assert reconstructed.destination == order.destination
        assert reconstructed.weight == order.weight
        assert reconstructed.volume == order.volume
        assert len(reconstructed.items) == 2

    def test_f2_order_invalid_inputs_raise_validation_error(self) -> None:
        """Verify invalid parameters for Order raise ValidationError."""
        # Empty ID
        with pytest.raises(ValidationError):
            Order(id="", destination=(41.88, -87.62), weight=10.0)

        # Inverted latitude
        with pytest.raises(ValidationError):
            Order(id="ORD-BAD", destination=(95.0, -87.62), weight=10.0)

        # Inverted longitude
        with pytest.raises(ValidationError):
            Order(id="ORD-BAD", destination=(41.88, -190.0), weight=10.0)

        # Negative weight
        with pytest.raises(ValidationError):
            Order(id="ORD-BAD", destination=(41.88, -87.62), weight=-5.0)

        # Invalid priority (< 1)
        with pytest.raises(ValidationError):
            Order(id="ORD-BAD", destination=(41.88, -87.62), weight=10.0, priority=0)

        # Inverted time window (start > end)
        with pytest.raises(ValidationError):
            Order(id="ORD-BAD", destination=(41.88, -87.62), weight=10.0, time_window=(15.0, 10.0))

    def test_f2_vehicle_model_validation_and_serialization(self) -> None:
        """Validate Vehicle dataclass properties, constraints, and serialization."""
        vehicle = Vehicle(
            id="TRUCK-01",
            capacity_weight=5000.0,
            capacity_volume=25.0,
            max_stops=40,
            speed_kmh=60.0,
            fixed_cost=150.0,
            cost_per_km=1.8,
            available_time_window=(6.0, 21.0),
        )

        assert vehicle.id == "TRUCK-01"
        assert vehicle.capacity_weight == 5000.0
        assert vehicle.capacity_volume == 25.0
        assert vehicle.speed_kmh == 60.0
        assert vehicle.fixed_cost == 150.0

        # Serialization roundtrip
        v_dict = vehicle.to_dict()
        reconstructed = Vehicle.from_dict(v_dict)
        assert reconstructed.id == vehicle.id
        assert reconstructed.capacity_weight == vehicle.capacity_weight
        assert reconstructed.capacity_volume == vehicle.capacity_volume
        assert reconstructed.available_time_window == (6.0, 21.0)

        # Negative capacity
        with pytest.raises(ValidationError):
            Vehicle(id="BAD-V", capacity_weight=-100.0)

        # Negative cost
        with pytest.raises(ValidationError):
            Vehicle(id="BAD-V", capacity_weight=1000.0, cost_per_km=-1.0)

    def test_f2_depot_and_stop_models(self) -> None:
        """Validate Depot and Stop models, location handling, and serialization."""
        depot = Depot(
            id="DEPOT-NORTH",
            location=(42.0, -87.7),
            name="North Chicago Hub",
            time_window=(5.0, 23.0),
        )
        assert depot.id == "DEPOT-NORTH"
        assert depot.location == (42.0, -87.7)
        depot_dict = depot.to_dict()
        assert Depot.from_dict(depot_dict).id == "DEPOT-NORTH"

        stop = Stop(
            location=(42.05, -87.68),
            order_id="ORD-301",
            demand_weight=22.0,
            demand_volume=0.05,
            time_window=(9.0, 11.0),
            service_time=10.0,
            is_depot=False,
        )
        assert stop.order_id == "ORD-301"
        assert stop.demand_weight == 22.0
        assert stop.is_depot is False

        stop_dict = stop.to_dict()
        reconstructed_stop = Stop.from_dict(stop_dict)
        assert reconstructed_stop.location == stop.location
        assert reconstructed_stop.demand_weight == stop.demand_weight

    def test_f2_route_stop_management_and_metrics(self, sample_stop: Stop) -> None:
        """Validate Route creation, stop addition, metrics recalculation, and serialization."""
        depot_stop = Stop(
            location=(41.8781, -87.6298),
            order_id=None,
            demand_weight=0.0,
            demand_volume=0.0,
            is_depot=True,
        )
        route = Route(
            vehicle_id="VAN-01",
            stops=[depot_stop, sample_stop],
            total_load_weight=12.5,
            total_load_volume=0.025,
            total_distance=5.2,
        )

        assert len(route.stops) == 2
        assert route.vehicle_id == "VAN-01"

        # Add second stop
        second_stop = Stop(
            location=(41.8900, -87.6150),
            order_id="ORD-002",
            demand_weight=20.0,
            demand_volume=0.04,
            is_depot=False,
        )
        route.add_stop(second_stop)
        assert len(route.stops) == 3

        route.recalculate_totals()
        assert route.total_load_weight == 32.5  # 12.5 + 20.0
        assert route.total_load_volume == 0.065  # 0.025 + 0.04

        # Serialization
        r_dict = route.to_dict()
        reconstructed = Route.from_dict(r_dict)
        assert reconstructed.vehicle_id == route.vehicle_id
        assert len(reconstructed.stops) == 3

    def test_f2_stock_item_inventory_lifecycle(self) -> None:
        """Validate StockItem state transitions: allocation, fulfillment, receipt, stockout guard."""
        item = StockItem(
            sku="SKU-PHONE-12",
            on_hand=50,
            on_order=20,
            unit_cost=500.0,
            holding_cost_rate=0.20,
            lead_time_days=3.0,
            daily_demand_history=[5.0, 7.0, 4.0, 6.0, 8.0],
            allocated=10,
        )

        assert item.available == 40  # 50 on_hand - 10 allocated
        assert item.net_inventory == 60  # 50 on_hand + 20 on_order - 10 allocated
        assert item.average_daily_demand == 6.0  # (5 + 7 + 4 + 6 + 8) / 5 = 6.0
        assert item.annual_holding_cost_per_unit == 100.0  # 500.0 * 0.20

        # Successful allocation
        item.allocate(15)
        assert item.allocated == 25
        assert item.available == 25

        # Stockout error when allocating more than available
        with pytest.raises(StockoutError) as exc_info:
            item.allocate(30)
        assert exc_info.value.sku == "SKU-PHONE-12"
        assert exc_info.value.requested == 30
        assert exc_info.value.available == 25

        # Fulfillment decrements both on_hand and allocated
        item.fulfill(20)
        assert item.on_hand == 30  # 50 - 20
        assert item.allocated == 5  # 25 - 20

        # Goods receipt increments on_hand and decrements on_order
        item.receive(20)
        assert item.on_hand == 50  # 30 + 20
        assert item.on_order == 0  # 20 - 20

        # Serialization
        item_dict = item.to_dict()
        reconstructed = StockItem.from_dict(item_dict)
        assert reconstructed.sku == item.sku
        assert reconstructed.on_hand == item.on_hand

    def test_f2_purchase_order_creation_and_validation(self) -> None:
        """Validate PurchaseOrder fields, positive quantity guard, and serialization."""
        po = PurchaseOrder(
            po_id="PO-77001",
            sku="SKU-LAPTOP-X1",
            quantity=100,
            order_date=1700000000.0,
            expected_delivery_date=1700432000.0,
            status="ISSUED",
        )

        assert po.po_id == "PO-77001"
        assert po.quantity == 100
        assert po.status == "ISSUED"

        po_dict = po.to_dict()
        reconstructed = PurchaseOrder.from_dict(po_dict)
        assert reconstructed.po_id == po.po_id
        assert reconstructed.sku == po.sku
        assert reconstructed.quantity == po.quantity

        # Non-positive quantity guard
        with pytest.raises(ValidationError):
            PurchaseOrder(po_id="PO-BAD", sku="SKU-BAD", quantity=0)

    def test_f2_order_item_and_dimensions_validation(self) -> None:
        """Validate OrderItem and Dimensions calculations and boundary conditions."""
        dim = Dimensions(length=1.5, width=0.8, height=1.2)
        assert math.isclose(dim.volume, 1.44, rel_tol=1e-5)
        assert dim.to_dict()["volume"] == dim.volume

        with pytest.raises(ValidationError):
            Dimensions(length=-1.0, width=1.0, height=1.0)

        item = OrderItem(sku="SKU-WIDGET", quantity=10, unit_weight=2.5, unit_volume=0.004)
        assert item.total_weight == 25.0
        assert item.total_volume == 0.04

        with pytest.raises(ValidationError):
            OrderItem(sku="", quantity=5)

        with pytest.raises(ValidationError):
            OrderItem(sku="SKU-OK", quantity=-2)


# ==============================================================================
# Feature 3: Domain Exceptions Hierarchy (F3)
# ==============================================================================


class TestFeature3DomainExceptions:
    """Comprehensive test suite for Feature 3: Domain Exceptions Hierarchy."""

    def test_f3_logistics_error_base_and_serialization(self) -> None:
        """Verify LogisticsError base exception, string formatting, and dict serialization."""
        err = LogisticsError("General logistics failure", context={"operation": "dispatch", "job_id": 42})
        assert isinstance(err, Exception)
        assert err.message == "General logistics failure"
        assert err.context == {"operation": "dispatch", "job_id": 42}
        assert "General logistics failure" in str(err)
        assert "job_id" in str(err)

        d = err.to_dict()
        assert d["error_type"] == "LogisticsError"
        assert d["message"] == "General logistics failure"
        assert d["context"]["operation"] == "dispatch"

    def test_f3_routing_error_inheritance(self) -> None:
        """Verify RoutingError inherits from LogisticsError and behaves correctly in try/except."""
        err = RoutingError("Vehicle capacity exceeded", context={"vehicle_id": "V1", "overload_kg": 250.0})
        assert isinstance(err, LogisticsError)

        caught = False
        try:
            raise err
        except LogisticsError as caught_err:
            caught = True
            assert caught_err.context["overload_kg"] == 250.0
        assert caught is True

        assert err.to_dict()["error_type"] == "RoutingError"

    def test_f3_stockout_error_attributes_and_context(self) -> None:
        """Verify StockoutError provides typed attributes and structured diagnostics."""
        err = StockoutError(sku="SKU-BOLT-M8", requested=100, available=45)
        assert isinstance(err, InventoryError)
        assert isinstance(err, LogisticsError)
        assert err.sku == "SKU-BOLT-M8"
        assert err.requested == 100
        assert err.available == 45
        assert err.context["sku"] == "SKU-BOLT-M8"
        assert err.context["requested"] == 100
        assert err.context["available"] == 45
        assert "Stockout for SKU 'SKU-BOLT-M8'" in str(err)

    def test_f3_fsm_state_error_attributes_and_context(self) -> None:
        """Verify FSMStateError captures illegal state transitions."""
        err = FSMStateError(current_state="CREATED", attempted_state="SHIPPED")
        assert isinstance(err, WarehouseError)
        assert isinstance(err, LogisticsError)
        assert err.current_state == "CREATED"
        assert err.attempted_state == "SHIPPED"
        assert err.context["current_state"] == "CREATED"
        assert err.context["attempted_state"] == "SHIPPED"
        assert "Illegal state transition from 'CREATED' to 'SHIPPED'" in str(err)

    def test_f3_configuration_and_validation_errors(self) -> None:
        """Verify ConfigurationError and ValidationError inherit from LogisticsError."""
        cfg_err = ConfigurationError("Invalid solver timeout", context={"timeout_sec": -10})
        assert isinstance(cfg_err, LogisticsError)
        assert cfg_err.context["timeout_sec"] == -10

        val_err = ValidationError("Field 'id' cannot be blank")
        assert isinstance(val_err, LogisticsError)

    def test_f3_capacity_exceeded_error_subclass(self) -> None:
        """Verify CapacityExceededError is defined and inherits from RoutingError or LogisticsError."""
        from open_logistics.core import exceptions

        if not hasattr(exceptions, "CapacityExceededError"):
            pytest.xfail("CapacityExceededError is not yet implemented in open_logistics.core.exceptions (escalate to implementer)")
        cls = getattr(exceptions, "CapacityExceededError")
        assert issubclass(cls, LogisticsError)
        assert issubclass(cls, RoutingError) or issubclass(cls, LogisticsError)

    def test_f3_invalid_transition_error_subclass(self) -> None:
        """Verify InvalidTransitionError is defined and inherits from FSMStateError or LogisticsError."""
        from open_logistics.core import exceptions

        if not hasattr(exceptions, "InvalidTransitionError"):
            pytest.xfail("InvalidTransitionError is not yet implemented in open_logistics.core.exceptions (escalate to implementer)")
        cls = getattr(exceptions, "InvalidTransitionError")
        assert issubclass(cls, LogisticsError)


# ==============================================================================
# Feature 4: Distance Metrics (F4)
# ==============================================================================


@pytest.mark.skipif(not HAS_DISTANCES, reason="Feature 4 distances implemented in Milestone 2")
class TestFeature4DistanceMetrics:
    """Comprehensive test suite for Feature 4: Distance Metrics."""

    def test_f4_euclidean_distance_cartesian(self) -> None:
        """Verify pure-Python Euclidean distance matches exact Pythagorean theorem."""
        # Standard 3-4-5 right triangle
        d1 = euclidean_distance((0.0, 0.0), (3.0, 4.0))
        assert math.isclose(d1, 5.0, abs_tol=1e-7)

        # Negative coordinate displacement
        d2 = euclidean_distance((-1.0, -2.0), (2.0, 2.0))
        # sqrt((2 - (-1))^2 + (2 - (-2))^2) = sqrt(9 + 16) = 5.0
        assert math.isclose(d2, 5.0, abs_tol=1e-7)

        # Coincident coordinates
        assert euclidean_distance((10.5, 20.2), (10.5, 20.2)) == 0.0

    def test_f4_manhattan_distance_grid(self) -> None:
        """Verify Manhattan (L1 norm / taxicab) distance matches |dx| + |dy|."""
        d1 = manhattan_distance((0.0, 0.0), (3.0, 4.0))
        assert math.isclose(d1, 7.0, abs_tol=1e-7)

        d2 = manhattan_distance((-5.0, 10.0), (5.0, -10.0))
        # |5 - (-5)| + |-10 - 10| = 10 + 20 = 30.0
        assert math.isclose(d2, 30.0, abs_tol=1e-7)

    def test_f4_haversine_distance_known_real_world_cities(self) -> None:
        """Verify Haversine spherical distance against authoritative great-circle values."""
        # New York (40.7128, -74.0060) to London (51.5074, -0.1278)
        # Authoritative great-circle distance: ~5570 km
        d_nyc_lon = haversine_distance((40.7128, -74.0060), (51.5074, -0.1278))
        assert 5550.0 <= d_nyc_lon <= 5590.0, f"Expected ~5570 km, got {d_nyc_lon} km"

        # Paris (48.8566, 2.3522) to Berlin (52.5200, 13.4050)
        # Authoritative great-circle distance: ~878 km
        d_par_ber = haversine_distance((48.8566, 2.3522), (52.5200, 13.4050))
        assert 870.0 <= d_par_ber <= 885.0, f"Expected ~878 km, got {d_par_ber} km"

        # Tokyo (35.6762, 139.6503) to Sydney (-33.8688, 151.2093)
        # Authoritative great-circle distance: ~7825 km
        d_tyo_syd = haversine_distance((35.6762, 139.6503), (-33.8688, 151.2093))
        assert 7800.0 <= d_tyo_syd <= 7850.0, f"Expected ~7825 km, got {d_tyo_syd} km"

    def test_f4_distance_symmetry(self) -> None:
        """Verify distance symmetry: dist(A, B) == dist(B, A) across all distance engines."""
        p1 = (41.8781, -87.6298)  # Chicago
        p2 = (34.0522, -118.2437)  # Los Angeles

        assert math.isclose(euclidean_distance(p1, p2), euclidean_distance(p2, p1), rel_tol=1e-9)
        assert math.isclose(manhattan_distance(p1, p2), manhattan_distance(p2, p1), rel_tol=1e-9)
        assert math.isclose(haversine_distance(p1, p2), haversine_distance(p2, p1), rel_tol=1e-9)

    def test_f4_distance_zero_identity(self) -> None:
        """Verify distance between identical points is exactly zero."""
        p = (52.52, 13.405)
        assert euclidean_distance(p, p) == 0.0
        assert manhattan_distance(p, p) == 0.0
        assert haversine_distance(p, p) == 0.0

    def test_f4_distance_triangle_inequality(self) -> None:
        """Verify triangle inequality: dist(A, C) <= dist(A, B) + dist(B, C)."""
        p_a = (40.7128, -74.0060)  # NYC
        p_b = (41.8781, -87.6298)  # Chicago
        p_c = (34.0522, -118.2437)  # LA

        # Euclidean
        assert euclidean_distance(p_a, p_c) <= euclidean_distance(p_a, p_b) + euclidean_distance(p_b, p_c) + 1e-9

        # Manhattan
        assert manhattan_distance(p_a, p_c) <= manhattan_distance(p_a, p_b) + manhattan_distance(p_b, p_c) + 1e-9

        # Haversine
        assert haversine_distance(p_a, p_c) <= haversine_distance(p_a, p_b) + haversine_distance(p_b, p_c) + 1e-9
