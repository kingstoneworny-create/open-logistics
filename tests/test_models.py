"""Unit tests for Open Logistics domain models, config, and exceptions."""

import pytest

from open_logistics.core.config import LogisticsConfig
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


class TestLocation:
    """Tests for Location entity."""

    def test_valid_location_creation(self) -> None:
        loc = Location(lat=40.7128, lon=-74.0060, name="NYC Port")
        assert loc.lat == 40.7128
        assert loc.lon == -74.0060
        assert loc.name == "NYC Port"
        assert loc.as_tuple() == (40.7128, -74.0060)

    def test_location_from_tuple(self) -> None:
        loc = Location.from_tuple((34.0522, -118.2437), name="LA Facility")
        assert loc.lat == 34.0522
        assert loc.lon == -118.2437
        assert loc.name == "LA Facility"

    def test_invalid_latitude_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="latitude"):
            Location(lat=95.0, lon=10.0)
        with pytest.raises(ValidationError, match="latitude"):
            Location(lat=-91.0, lon=10.0)

    def test_invalid_longitude_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="longitude"):
            Location(lat=10.0, lon=181.0)
        with pytest.raises(ValidationError, match="longitude"):
            Location(lat=10.0, lon=-185.0)

    def test_non_numeric_coordinates(self) -> None:
        with pytest.raises(ValidationError):
            Location.from_tuple((1.0, 2.0, 3.0))  # type: ignore[arg-type]

    def test_location_serialization_roundtrip(self) -> None:
        loc = Location(lat=51.5074, lon=-0.1278, name="London DC")
        d = loc.to_dict()
        assert d == {"lat": 51.5074, "lon": -0.1278, "name": "London DC"}
        restored = Location.from_dict(d)
        assert restored.lat == loc.lat
        assert restored.lon == loc.lon
        assert restored.name == loc.name


class TestDimensions:
    """Tests for Dimensions entity."""

    def test_volume_calculation(self) -> None:
        dims = Dimensions(length=2.0, width=1.5, height=0.5)
        assert dims.volume == pytest.approx(1.5)

    def test_negative_dimensions_raise_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="negative"):
            Dimensions(length=-1.0, width=1.0, height=1.0)
        with pytest.raises(ValidationError, match="negative"):
            Dimensions(length=1.0, width=-0.5, height=1.0)
        with pytest.raises(ValidationError, match="negative"):
            Dimensions(length=1.0, width=1.0, height=-2.0)

    def test_zero_dimensions_allowed(self) -> None:
        dims = Dimensions(length=0.0, width=0.0, height=0.0)
        assert dims.volume == 0.0

    def test_dimensions_serialization(self) -> None:
        dims = Dimensions(length=1.2, width=0.8, height=1.0)
        d = dims.to_dict()
        assert d["volume"] == pytest.approx(0.96)
        restored = Dimensions.from_dict(d)
        assert restored.length == dims.length
        assert restored.volume == pytest.approx(dims.volume)


class TestOrderItem:
    """Tests for OrderItem entity."""

    def test_valid_order_item(self) -> None:
        item = OrderItem(sku="SKU-A", quantity=10, unit_weight=1.5, unit_volume=0.01)
        assert item.total_weight == 15.0
        assert item.total_volume == pytest.approx(0.10)

    def test_empty_sku_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="SKU"):
            OrderItem(sku="", quantity=5)

    def test_non_positive_quantity_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="quantity"):
            OrderItem(sku="SKU-A", quantity=0)
        with pytest.raises(ValidationError, match="quantity"):
            OrderItem(sku="SKU-A", quantity=-3)

    def test_negative_weight_or_volume_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="unit_weight"):
            OrderItem(sku="SKU-A", quantity=1, unit_weight=-0.5)
        with pytest.raises(ValidationError, match="unit_volume"):
            OrderItem(sku="SKU-A", quantity=1, unit_volume=-0.01)

    def test_order_item_serialization(self) -> None:
        item = OrderItem(sku="SKU-B", quantity=4, unit_weight=3.0, unit_volume=0.02)
        d = item.to_dict()
        restored = OrderItem.from_dict(d)
        assert restored.sku == "SKU-B"
        assert restored.quantity == 4
        assert restored.total_weight == 12.0


class TestOrder:
    """Tests for Order entity."""

    def test_order_creation_and_defaults(self, sample_order_item: OrderItem) -> None:
        order = Order(
            id="ORD-100",
            destination=(41.8781, -87.6298),
            weight=10.0,
            volume=0.05,
            items=[sample_order_item],
        )
        assert order.id == "ORD-100"
        assert order.priority == 1
        assert order.status == "PENDING"
        assert order.time_window is None
        assert len(order.items) == 1

    def test_order_location_normalization(self) -> None:
        loc = Location(lat=41.8781, lon=-87.6298)
        order = Order(id="ORD-101", destination=loc, weight=5.0)  # type: ignore[arg-type]
        assert order.destination == (41.8781, -87.6298)

    def test_order_invalid_inputs(self) -> None:
        with pytest.raises(ValidationError, match="Order ID"):
            Order(id="", destination=(40.0, -74.0), weight=10.0)

        with pytest.raises(ValidationError, match="destination"):
            Order(id="ORD-1", destination=(100.0, 0.0), weight=10.0)

        with pytest.raises(ValidationError, match="weight"):
            Order(id="ORD-1", destination=(40.0, -74.0), weight=-5.0)

        with pytest.raises(ValidationError, match="volume"):
            Order(id="ORD-1", destination=(40.0, -74.0), weight=5.0, volume=-1.0)

        with pytest.raises(ValidationError, match="priority"):
            Order(id="ORD-1", destination=(40.0, -74.0), weight=5.0, priority=0)

    def test_order_time_window_validation(self) -> None:
        with pytest.raises(ValidationError, match="time_window"):
            Order(id="ORD-1", destination=(40.0, -74.0), weight=1.0, time_window=(10.0, 8.0))
        with pytest.raises(ValidationError, match="negative"):
            Order(id="ORD-1", destination=(40.0, -74.0), weight=1.0, time_window=(-1.0, 5.0))

    def test_sync_metrics_from_items(self) -> None:
        item1 = OrderItem(sku="SKU-1", quantity=2, unit_weight=3.0, unit_volume=0.01)
        item2 = OrderItem(sku="SKU-2", quantity=3, unit_weight=2.0, unit_volume=0.02)
        order = Order(
            id="ORD-SYNC",
            destination=(41.0, -87.0),
            weight=0.0,
            volume=0.0,
            items=[item1, item2],
        )
        assert order.calculate_weight_from_items() == 12.0
        assert order.calculate_volume_from_items() == pytest.approx(0.08)
        order.sync_metrics_from_items()
        assert order.weight == 12.0
        assert order.volume == pytest.approx(0.08)

    def test_order_serialization(self, sample_order: Order) -> None:
        d = sample_order.to_dict()
        restored = Order.from_dict(d)
        assert restored.id == sample_order.id
        assert restored.destination == sample_order.destination
        assert restored.weight == sample_order.weight
        assert restored.time_window == sample_order.time_window
        assert len(restored.items) == len(sample_order.items)
        assert restored.items[0].sku == sample_order.items[0].sku


class TestVehicle:
    """Tests for Vehicle entity."""

    def test_vehicle_creation_and_defaults(self) -> None:
        v = Vehicle(id="V1", capacity_weight=500.0)
        assert v.capacity_weight == 500.0
        assert v.capacity_volume == 1e9
        assert v.max_stops == 50
        assert v.speed_kmh == 40.0
        assert v.fixed_cost == 0.0
        assert v.cost_per_km == 1.0

    def test_vehicle_invalid_inputs(self) -> None:
        with pytest.raises(ValidationError, match="Vehicle ID"):
            Vehicle(id="", capacity_weight=100.0)
        with pytest.raises(ValidationError, match="capacity_weight"):
            Vehicle(id="V1", capacity_weight=0.0)
        with pytest.raises(ValidationError, match="capacity_volume"):
            Vehicle(id="V1", capacity_weight=100.0, capacity_volume=-5.0)
        with pytest.raises(ValidationError, match="max_stops"):
            Vehicle(id="V1", capacity_weight=100.0, max_stops=0)
        with pytest.raises(ValidationError, match="speed_kmh"):
            Vehicle(id="V1", capacity_weight=100.0, speed_kmh=-10.0)
        with pytest.raises(ValidationError, match="fixed_cost"):
            Vehicle(id="V1", capacity_weight=100.0, fixed_cost=-1.0)
        with pytest.raises(ValidationError, match="cost_per_km"):
            Vehicle(id="V1", capacity_weight=100.0, cost_per_km=-0.5)

    def test_vehicle_serialization(self, sample_vehicle: Vehicle) -> None:
        d = sample_vehicle.to_dict()
        restored = Vehicle.from_dict(d)
        assert restored.id == sample_vehicle.id
        assert restored.capacity_weight == sample_vehicle.capacity_weight
        assert restored.speed_kmh == sample_vehicle.speed_kmh


class TestDepotAndStop:
    """Tests for Depot and Stop entities."""

    def test_depot_creation(self) -> None:
        depot = Depot(id="DEPOT-A", location=(41.8781, -87.6298), name="West Facility")
        assert depot.id == "DEPOT-A"
        assert depot.location == (41.8781, -87.6298)

    def test_depot_invalid_inputs(self) -> None:
        with pytest.raises(ValidationError, match="Depot ID"):
            Depot(id="", location=(40.0, -74.0))
        with pytest.raises(ValidationError, match="location"):
            Depot(id="D1", location=(95.0, 0.0))

    def test_stop_creation(self) -> None:
        stop = Stop(
            location=(42.0, -88.0),
            order_id="ORD-1",
            demand_weight=20.0,
            demand_volume=0.1,
            service_time=10.0,
        )
        assert stop.order_id == "ORD-1"
        assert stop.demand_weight == 20.0
        assert not stop.is_depot

    def test_stop_negative_values(self) -> None:
        with pytest.raises(ValidationError, match="demand_weight"):
            Stop(location=(42.0, -88.0), demand_weight=-5.0)
        with pytest.raises(ValidationError, match="demand_volume"):
            Stop(location=(42.0, -88.0), demand_volume=-0.1)
        with pytest.raises(ValidationError, match="service_time"):
            Stop(location=(42.0, -88.0), service_time=-2.0)

    def test_depot_and_stop_serialization(self, sample_depot: Depot, sample_stop: Stop) -> None:
        depot_dict = sample_depot.to_dict()
        restored_depot = Depot.from_dict(depot_dict)
        assert restored_depot.id == sample_depot.id

        stop_dict = sample_stop.to_dict()
        restored_stop = Stop.from_dict(stop_dict)
        assert restored_stop.order_id == sample_stop.order_id
        assert restored_stop.demand_weight == sample_stop.demand_weight


class TestRoute:
    """Tests for Route entity."""

    def test_route_creation_and_add_stop(self) -> None:
        route = Route(vehicle_id="VAN-1")
        assert route.stop_count == 0
        assert route.customer_stop_count == 0

        depot = Stop(location=(41.8, -87.6), is_depot=True)
        cust1 = Stop(location=(41.9, -87.7), order_id="O1", demand_weight=10.0, demand_volume=0.05)
        cust2 = Stop(location=(41.95, -87.75), order_id="O2", demand_weight=15.0, demand_volume=0.08)

        route.add_stop(depot)
        route.add_stop(cust1)
        route.add_stop(cust2)

        assert route.stop_count == 3
        assert route.customer_stop_count == 2
        assert route.total_load_weight == 25.0
        assert route.total_load_volume == pytest.approx(0.13)

    def test_route_recalculate_totals(self) -> None:
        route = Route(
            vehicle_id="VAN-2",
            stops=[
                Stop(location=(41.8, -87.6), demand_weight=5.0, demand_volume=0.02),
                Stop(location=(41.9, -87.7), demand_weight=15.0, demand_volume=0.08),
            ],
            total_load_weight=0.0,
            total_load_volume=0.0,
        )
        route.recalculate_totals()
        assert route.total_load_weight == 20.0
        assert route.total_load_volume == pytest.approx(0.10)

    def test_route_serialization(self, sample_route: Route) -> None:
        d = sample_route.to_dict()
        restored = Route.from_dict(d)
        assert restored.vehicle_id == sample_route.vehicle_id
        assert restored.stop_count == sample_route.stop_count
        assert restored.total_load_weight == sample_route.total_load_weight


class TestStockItem:
    """Tests for StockItem inventory entity."""

    def test_stock_item_properties(self, sample_stock_item: StockItem) -> None:
        assert sample_stock_item.available == 80  # 100 on_hand - 20 allocated
        assert sample_stock_item.net_inventory == 130  # 100 on_hand + 50 on_order - 20 allocated
        assert sample_stock_item.average_daily_demand == pytest.approx(11.42857, rel=1e-3)
        assert sample_stock_item.annual_demand(365) == pytest.approx(4171.428, rel=1e-3)
        assert sample_stock_item.annual_holding_cost_per_unit == pytest.approx(4.0)

    def test_stock_allocation_success(self, sample_stock_item: StockItem) -> None:
        initial_allocated = sample_stock_item.allocated
        sample_stock_item.allocate(30)
        assert sample_stock_item.allocated == initial_allocated + 30
        assert sample_stock_item.available == 50

    def test_stock_allocation_stockout_raises_error(self, sample_stock_item: StockItem) -> None:
        with pytest.raises(StockoutError) as exc_info:
            sample_stock_item.allocate(90)  # only 80 available
        assert exc_info.value.sku == "SKU-1001"
        assert exc_info.value.requested == 90
        assert exc_info.value.available == 80

    def test_stock_deallocation_and_fulfillment(self, sample_stock_item: StockItem) -> None:
        sample_stock_item.deallocate(10)
        assert sample_stock_item.allocated == 10
        assert sample_stock_item.available == 90

        sample_stock_item.fulfill(10)
        assert sample_stock_item.on_hand == 90
        assert sample_stock_item.allocated == 0

    def test_stock_receipt(self, sample_stock_item: StockItem) -> None:
        sample_stock_item.receive(50)
        assert sample_stock_item.on_hand == 150
        assert sample_stock_item.on_order == 0

    def test_stock_item_validation(self) -> None:
        with pytest.raises(ValidationError, match="SKU"):
            StockItem(sku="", on_hand=10)
        with pytest.raises(ValidationError, match="on_hand"):
            StockItem(sku="SKU-1", on_hand=-5)
        with pytest.raises(ValidationError, match="on_order"):
            StockItem(sku="SKU-1", on_hand=5, on_order=-1)
        with pytest.raises(ValidationError, match="unit_cost"):
            StockItem(sku="SKU-1", on_hand=5, unit_cost=-2.0)
        with pytest.raises(ValidationError, match="holding_cost_rate"):
            StockItem(sku="SKU-1", on_hand=5, holding_cost_rate=-0.1)

    def test_stock_item_serialization(self, sample_stock_item: StockItem) -> None:
        d = sample_stock_item.to_dict()
        restored = StockItem.from_dict(d)
        assert restored.sku == sample_stock_item.sku
        assert restored.on_hand == sample_stock_item.on_hand
        assert restored.available == sample_stock_item.available


class TestPurchaseOrder:
    """Tests for PurchaseOrder entity."""

    def test_purchase_order_creation_and_defaults(self) -> None:
        po = PurchaseOrder(po_id="PO-1", sku="SKU-A", quantity=100)
        assert po.po_id == "PO-1"
        assert po.sku == "SKU-A"
        assert po.quantity == 100
        assert po.status == "ISSUED"
        assert po.order_date > 0.0

    def test_purchase_order_validation(self) -> None:
        with pytest.raises(ValidationError, match="po_id"):
            PurchaseOrder(po_id="", sku="SKU-A", quantity=10)
        with pytest.raises(ValidationError, match="sku"):
            PurchaseOrder(po_id="PO-1", sku="", quantity=10)
        with pytest.raises(ValidationError, match="quantity"):
            PurchaseOrder(po_id="PO-1", sku="SKU-A", quantity=0)

    def test_purchase_order_serialization(self, sample_purchase_order: PurchaseOrder) -> None:
        d = sample_purchase_order.to_dict()
        restored = PurchaseOrder.from_dict(d)
        assert restored.po_id == sample_purchase_order.po_id
        assert restored.quantity == sample_purchase_order.quantity


class TestLogisticsConfig:
    """Tests for LogisticsConfig settings."""

    def test_default_config_valid(self, default_config: LogisticsConfig) -> None:
        assert default_config.default_vehicle_speed_kmh == 40.0
        assert default_config.default_service_level == 0.95
        assert default_config.cvrp_algorithm == "parallel"
        assert default_config.distance_metric == "haversine"

    def test_custom_config_validation(self) -> None:
        with pytest.raises(ConfigurationError, match="distance_metric"):
            LogisticsConfig(distance_metric="invalid_metric")

        with pytest.raises(ConfigurationError, match="default_service_level"):
            LogisticsConfig(default_service_level=1.5)

        with pytest.raises(ConfigurationError, match="abc_cutoffs"):
            LogisticsConfig(abc_cutoffs=(0.95, 0.80))

    def test_config_serialization(self, default_config: LogisticsConfig) -> None:
        d = default_config.to_dict()
        restored = LogisticsConfig.from_dict(d)
        assert restored.default_vehicle_speed_kmh == default_config.default_vehicle_speed_kmh
        assert restored.distance_metric == default_config.distance_metric


class TestExceptionsHierarchy:
    """Tests verifying domain exception inheritance and context."""

    def test_hierarchy(self) -> None:
        assert issubclass(RoutingError, LogisticsError)
        assert issubclass(InventoryError, LogisticsError)
        assert issubclass(StockoutError, InventoryError)
        assert issubclass(WarehouseError, LogisticsError)
        assert issubclass(FSMStateError, WarehouseError)
        assert issubclass(ConfigurationError, LogisticsError)
        assert issubclass(ValidationError, LogisticsError)

    def test_exception_context_and_serialization(self) -> None:
        err = StockoutError(sku="SKU-XYZ", requested=10, available=2)
        assert err.sku == "SKU-XYZ"
        assert err.requested == 10
        assert err.available == 2
        d = err.to_dict()
        assert d["error_type"] == "StockoutError"
        assert d["context"]["sku"] == "SKU-XYZ"
        assert "SKU 'SKU-XYZ'" in str(err)
