"""Shared fixtures and test utilities for Open Logistics unit and integration tests."""

import pytest
from typing import Any, Dict, Generator, List

from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import EventBus
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


@pytest.fixture
def sample_location() -> Location:
    """Standard warehouse depot location in Chicago, IL."""
    return Location(lat=41.8781, lon=-87.6298, name="Central Hub")


@pytest.fixture
def sample_dimensions() -> Dimensions:
    """Standard cardboard shipping box dimensions (0.4m x 0.3m x 0.2m = 0.024 m^3)."""
    return Dimensions(length=0.4, width=0.3, height=0.2)


@pytest.fixture
def sample_order_item() -> OrderItem:
    """Sample line item: 5 units of SKU-1001."""
    return OrderItem(
        sku="SKU-1001",
        quantity=5,
        unit_weight=2.5,
        unit_volume=0.005,
    )


@pytest.fixture
def sample_order(sample_order_item: OrderItem) -> Order:
    """Sample single customer order with destination and line items."""
    return Order(
        id="ORD-001",
        destination=(41.8825, -87.6225),
        weight=12.5,
        volume=0.025,
        priority=2,
        time_window=(8.0, 12.0),
        items=[sample_order_item],
        status="PENDING",
    )


@pytest.fixture
def sample_orders_batch() -> List[Order]:
    """Batch of customer orders with varied weights, volumes, destinations, priorities, and time windows."""
    return [
        Order(
            id="ORD-BATCH-01",
            destination=(41.8825, -87.6225),
            weight=15.0,
            volume=0.03,
            priority=2,
            time_window=(8.0, 11.0),
            items=[OrderItem(sku="SKU-1001", quantity=3, unit_weight=5.0, unit_volume=0.01)],
            status="PENDING",
        ),
        Order(
            id="ORD-BATCH-02",
            destination=(41.8900, -87.6150),
            weight=45.0,
            volume=0.08,
            priority=3,
            time_window=(9.0, 12.0),
            items=[OrderItem(sku="SKU-1002", quantity=9, unit_weight=5.0, unit_volume=0.0088)],
            status="PENDING",
        ),
        Order(
            id="ORD-BATCH-03",
            destination=(41.8700, -87.6400),
            weight=25.0,
            volume=0.05,
            priority=1,
            time_window=(10.0, 14.0),
            items=[OrderItem(sku="SKU-1003", quantity=5, unit_weight=5.0, unit_volume=0.01)],
            status="PENDING",
        ),
        Order(
            id="ORD-BATCH-04",
            destination=(41.8650, -87.6100),
            weight=80.0,
            volume=0.15,
            priority=2,
            time_window=(8.5, 13.0),
            items=[OrderItem(sku="SKU-1001", quantity=16, unit_weight=5.0, unit_volume=0.0093)],
            status="PENDING",
        ),
        Order(
            id="ORD-BATCH-05",
            destination=(41.8950, -87.6350),
            weight=10.0,
            volume=0.02,
            priority=1,
            time_window=(13.0, 17.0),
            items=[OrderItem(sku="SKU-1004", quantity=2, unit_weight=5.0, unit_volume=0.01)],
            status="PENDING",
        ),
        Order(
            id="ORD-BATCH-06",
            destination=(41.8550, -87.6500),
            weight=120.0,
            volume=0.25,
            priority=3,
            time_window=(9.0, 15.0),
            items=[OrderItem(sku="SKU-1002", quantity=24, unit_weight=5.0, unit_volume=0.0104)],
            status="PENDING",
        ),
    ]


@pytest.fixture
def sample_vehicle_van() -> Vehicle:
    """Light commercial delivery van."""
    return Vehicle(
        id="VAN-01",
        capacity_weight=1000.0,
        capacity_volume=8.0,
        max_stops=30,
        speed_kmh=45.0,
        fixed_cost=50.0,
        cost_per_km=1.2,
        available_time_window=(7.0, 19.0),
    )


@pytest.fixture
def sample_vehicle_box_truck() -> Vehicle:
    """Medium freight box truck."""
    return Vehicle(
        id="BOX-01",
        capacity_weight=3500.0,
        capacity_volume=20.0,
        max_stops=50,
        speed_kmh=40.0,
        fixed_cost=100.0,
        cost_per_km=2.0,
        available_time_window=(6.0, 20.0),
    )


@pytest.fixture
def sample_vehicle_semi() -> Vehicle:
    """Heavy semi-trailer commercial transporter."""
    return Vehicle(
        id="SEMI-01",
        capacity_weight=15000.0,
        capacity_volume=60.0,
        max_stops=60,
        speed_kmh=35.0,
        fixed_cost=250.0,
        cost_per_km=3.5,
        available_time_window=(5.0, 22.0),
    )


@pytest.fixture
def sample_vehicle(sample_vehicle_van: Vehicle) -> Vehicle:
    """Default vehicle fixture (standard delivery van)."""
    return sample_vehicle_van


@pytest.fixture
def sample_heterogeneous_fleet(
    sample_vehicle_van: Vehicle,
    sample_vehicle_box_truck: Vehicle,
    sample_vehicle_semi: Vehicle,
) -> List[Vehicle]:
    """Heterogeneous vehicle fleet containing van, box truck, and semi."""
    return [sample_vehicle_van, sample_vehicle_box_truck, sample_vehicle_semi]


@pytest.fixture
def sample_depot() -> Depot:
    """Primary distribution center depot."""
    return Depot(
        id="DEPOT-01",
        location=(41.8781, -87.6298),
        name="Main Depot",
        time_window=(6.0, 22.0),
    )


@pytest.fixture
def sample_depot_cartesian() -> Depot:
    """Depot at cartesian origin (0.0, 0.0)."""
    return Depot(
        id="DEPOT-CART-01",
        location=(0.0, 0.0),
        name="Cartesian Origin Depot",
        time_window=(0.0, 24.0),
    )


@pytest.fixture
def sample_depot_geo() -> Depot:
    """Depot at real-world geographical coordinates in Chicago, IL."""
    return Depot(
        id="DEPOT-GEO-01",
        location=(41.8781, -87.6298),
        name="Chicago Distribution Center",
        time_window=(6.0, 22.0),
    )


@pytest.fixture
def sample_stop() -> Stop:
    """Customer delivery stop."""
    return Stop(
        location=(41.8825, -87.6225),
        order_id="ORD-001",
        demand_weight=12.5,
        demand_volume=0.025,
        time_window=(8.0, 12.0),
        service_time=15.0,
        is_depot=False,
    )


@pytest.fixture
def sample_route(sample_stop: Stop) -> Route:
    """Assigned delivery route with depot start and one stop."""
    depot_stop = Stop(
        location=(41.8781, -87.6298),
        order_id=None,
        demand_weight=0.0,
        demand_volume=0.0,
        is_depot=True,
    )
    return Route(
        vehicle_id="VAN-01",
        stops=[depot_stop, sample_stop],
        total_load_weight=12.5,
        total_load_volume=0.025,
        total_distance=5.2,
    )


@pytest.fixture
def sample_stock_item() -> StockItem:
    """Inventory item with historical daily demand."""
    return StockItem(
        sku="SKU-1001",
        on_hand=100,
        on_order=50,
        unit_cost=20.0,
        holding_cost_rate=0.20,
        lead_time_days=4.0,
        daily_demand_history=[10.0, 12.0, 15.0, 8.0, 11.0, 14.0, 10.0],
        allocated=20,
    )


@pytest.fixture
def sample_stock_items_batch() -> List[StockItem]:
    """Batch of 5 stock items spanning various demand volumes, costs, lead times, and volatility."""
    return [
        StockItem(
            sku="SKU-A1-FAST",
            on_hand=500,
            on_order=100,
            unit_cost=150.0,
            holding_cost_rate=0.20,
            lead_time_days=3.0,
            daily_demand_history=[50.0, 52.0, 48.0, 55.0, 49.0, 51.0, 50.0],
            allocated=50,
        ),
        StockItem(
            sku="SKU-A2-VOLATILE",
            on_hand=150,
            on_order=50,
            unit_cost=300.0,
            holding_cost_rate=0.25,
            lead_time_days=7.0,
            daily_demand_history=[10.0, 25.0, 5.0, 30.0, 15.0, 20.0, 8.0],
            allocated=20,
        ),
        StockItem(
            sku="SKU-B1-STEADY",
            on_hand=200,
            on_order=40,
            unit_cost=40.0,
            holding_cost_rate=0.18,
            lead_time_days=5.0,
            daily_demand_history=[20.0, 22.0, 18.0, 25.0, 19.0, 21.0, 20.0],
            allocated=15,
        ),
        StockItem(
            sku="SKU-C1-ERRATIC",
            on_hand=80,
            on_order=0,
            unit_cost=8.0,
            holding_cost_rate=0.15,
            lead_time_days=10.0,
            daily_demand_history=[2.0, 0.0, 5.0, 0.0, 1.0, 3.0, 0.0],
            allocated=5,
        ),
        StockItem(
            sku="SKU-C2-DEAD",
            on_hand=30,
            on_order=0,
            unit_cost=5.0,
            holding_cost_rate=0.15,
            lead_time_days=14.0,
            daily_demand_history=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            allocated=0,
        ),
    ]


@pytest.fixture
def sample_warehouse_layout() -> Any:
    """Sample 3D warehouse layout with zones, aisles, bays, and tiers."""
    try:
        from open_logistics.warehouse.layout import WarehouseLayout
        return WarehouseLayout.create_grid(zones=["ZONE-FAST", "ZONE-SLOW"], aisles=4, bays=10, tiers=4)
    except (ImportError, AttributeError, TypeError):
        return {
            "zones": ["RECEIVING", "ZONE-FAST", "ZONE-SLOW", "SHIPPING"],
            "aisles": 4,
            "bays_per_aisle": 10,
            "tiers_per_bay": 4,
            "dock_location": (0.0, 0.0, 0.0),
            "bin_dimensions": (1.0, 1.2, 1.5),
            "total_bins": 160,
        }


@pytest.fixture
def sample_purchase_order() -> PurchaseOrder:
    """Issued supplier replenishment purchase order."""
    return PurchaseOrder(
        po_id="PO-9001",
        sku="SKU-1001",
        quantity=200,
        order_date=1700000000.0,
        expected_delivery_date=1700345600.0,
        status="ISSUED",
    )


@pytest.fixture
def event_bus() -> Generator[EventBus, None, None]:
    """Fresh in-memory EventBus instance."""
    bus = EventBus(isolated_errors=True)
    yield bus
    bus.clear()


@pytest.fixture
def default_config() -> LogisticsConfig:
    """Default system configuration."""
    return LogisticsConfig()
