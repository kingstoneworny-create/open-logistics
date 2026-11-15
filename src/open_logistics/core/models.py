"""Core domain models for Open Logistics.

Defines typed dataclasses for logistics operations including locations, dimensions,
orders, vehicles, depots, routes, inventory stock items, and purchase orders.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from open_logistics.core.exceptions import StockoutError, ValidationError


def _validate_coordinates(lat: float, lon: float, context_name: str = "Coordinates") -> None:
    """Validate latitude and longitude ranges."""
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise ValidationError(f"{context_name} latitude and longitude must be numbers, got ({lat!r}, {lon!r})")
    if not (-90.0 <= lat <= 90.0):
        raise ValidationError(f"{context_name} latitude must be between -90.0 and 90.0, got {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValidationError(f"{context_name} longitude must be between -180.0 and 180.0, got {lon}")


@dataclass
class Location:
    """Geographic or cartesian point representation.

    Attributes:
        lat: Latitude in decimal degrees (between -90.0 and 90.0).
        lon: Longitude in decimal degrees (between -180.0 and 180.0).
        name: Optional human-readable place identifier.
    """

    lat: float
    lon: float
    name: str = ""

    def __post_init__(self) -> None:
        self.lat = float(self.lat)
        self.lon = float(self.lon)
        _validate_coordinates(self.lat, self.lon, f"Location '{self.name or 'unnamed'}'")

    def as_tuple(self) -> Tuple[float, float]:
        """Return coordinates as (lat, lon) tuple."""
        return (self.lat, self.lon)

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float], name: str = "") -> Location:
        """Create a Location from a 2-element coordinate tuple."""
        if len(coords) != 2:
            raise ValidationError(f"Expected 2-tuple for coordinates, got {coords!r}")
        return cls(lat=coords[0], lon=coords[1], name=name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Location to dictionary."""
        return {"lat": self.lat, "lon": self.lon, "name": self.name}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Location:
        """Instantiate Location from dictionary."""
        return cls(
            lat=data["lat"],
            lon=data["lon"],
            name=data.get("name", ""),
        )


@dataclass
class Dimensions:
    """3D spatial bounding box dimensions in meters or standardized units.

    Attributes:
        length: Dimension length (>= 0.0).
        width: Dimension width (>= 0.0).
        height: Dimension height (>= 0.0).
    """

    length: float
    width: float
    height: float

    def __post_init__(self) -> None:
        self.length = float(self.length)
        self.width = float(self.width)
        self.height = float(self.height)
        if self.length < 0 or self.width < 0 or self.height < 0:
            raise ValidationError(
                f"Dimensions cannot be negative: length={self.length}, width={self.width}, height={self.height}"
            )

    @property
    def volume(self) -> float:
        """Calculated volume (length * width * height)."""
        return self.length * self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        """Convert Dimensions to dictionary."""
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dimensions:
        """Instantiate Dimensions from dictionary."""
        return cls(
            length=data["length"],
            width=data["width"],
            height=data["height"],
        )


@dataclass
class OrderItem:
    """Line item in a customer order.

    Attributes:
        sku: Stock Keeping Unit identifier.
        quantity: Units ordered (must be > 0).
        unit_weight: Weight per unit (kg or configured unit).
        unit_volume: Volume per unit (m^3 or configured unit).
    """

    sku: str
    quantity: int
    unit_weight: float = 0.0
    unit_volume: float = 0.0

    def __post_init__(self) -> None:
        if not self.sku or not isinstance(self.sku, str):
            raise ValidationError(f"OrderItem SKU must be a non-empty string, got {self.sku!r}")
        self.quantity = int(self.quantity)
        if self.quantity <= 0:
            raise ValidationError(f"OrderItem quantity must be positive (> 0), got {self.quantity}")
        self.unit_weight = float(self.unit_weight)
        self.unit_volume = float(self.unit_volume)
        if self.unit_weight < 0:
            raise ValidationError(f"OrderItem unit_weight cannot be negative, got {self.unit_weight}")
        if self.unit_volume < 0:
            raise ValidationError(f"OrderItem unit_volume cannot be negative, got {self.unit_volume}")

    @property
    def total_weight(self) -> float:
        """Total weight for this line item."""
        return self.quantity * self.unit_weight

    @property
    def total_volume(self) -> float:
        """Total volume for this line item."""
        return self.quantity * self.unit_volume

    def to_dict(self) -> Dict[str, Any]:
        """Convert OrderItem to dictionary."""
        return {
            "sku": self.sku,
            "quantity": self.quantity,
            "unit_weight": self.unit_weight,
            "unit_volume": self.unit_volume,
            "total_weight": self.total_weight,
            "total_volume": self.total_volume,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrderItem:
        """Instantiate OrderItem from dictionary."""
        return cls(
            sku=data["sku"],
            quantity=data["quantity"],
            unit_weight=data.get("unit_weight", 0.0),
            unit_volume=data.get("unit_volume", 0.0),
        )


@dataclass
class Order:
    """Customer order requiring delivery or fulfillment.

    Attributes:
        id: Unique order identifier.
        destination: Delivery coordinates (lat, lon).
        weight: Total payload weight.
        volume: Total payload volume.
        priority: Priority ranking (integer >= 1, higher value = higher priority).
        time_window: Optional allowable delivery time window (earliest, latest).
        items: List of line items associated with this order.
        status: Current order fulfillment status.
        metadata: Arbitrary additional data.
    """

    id: str
    destination: Tuple[float, float]
    weight: float
    volume: float = 0.0
    priority: int = 1
    time_window: Optional[Tuple[float, float]] = None
    items: List[OrderItem] = field(default_factory=list)
    status: str = "PENDING"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValidationError(f"Order ID must be a non-empty string, got {self.id!r}")

        # Normalize destination if passed as Location or list
        if isinstance(self.destination, Location):
            self.destination = self.destination.as_tuple()
        elif isinstance(self.destination, (list, tuple)) and len(self.destination) == 2:
            self.destination = (float(self.destination[0]), float(self.destination[1]))
        else:
            raise ValidationError(f"Order destination must be a 2-tuple (lat, lon), got {self.destination!r}")

        _validate_coordinates(self.destination[0], self.destination[1], f"Order '{self.id}' destination")

        self.weight = float(self.weight)
        self.volume = float(self.volume)
        if self.weight < 0:
            raise ValidationError(f"Order weight cannot be negative, got {self.weight}")
        if self.volume < 0:
            raise ValidationError(f"Order volume cannot be negative, got {self.volume}")

        self.priority = int(self.priority)
        if self.priority < 1:
            raise ValidationError(f"Order priority must be an integer >= 1, got {self.priority}")

        if self.time_window is not None:
            if not isinstance(self.time_window, (tuple, list)) or len(self.time_window) != 2:
                raise ValidationError(f"Order time_window must be a 2-tuple (start, end), got {self.time_window!r}")
            tw_start, tw_end = float(self.time_window[0]), float(self.time_window[1])
            if tw_start < 0 or tw_end < 0:
                raise ValidationError(f"Order time_window timestamps cannot be negative: {self.time_window}")
            if tw_start > tw_end:
                raise ValidationError(f"Order time_window start ({tw_start}) cannot exceed end ({tw_end})")
            self.time_window = (tw_start, tw_end)

        # Ensure items list contains OrderItem instances
        parsed_items: List[OrderItem] = []
        for it in self.items:
            if isinstance(it, dict):
                parsed_items.append(OrderItem.from_dict(it))
            elif isinstance(it, OrderItem):
                parsed_items.append(it)
            else:
                raise ValidationError(f"Invalid item entry in order: {it!r}")
        self.items = parsed_items

    def calculate_weight_from_items(self) -> float:
        """Calculate total weight aggregated from line items."""
        return sum(item.total_weight for item in self.items)

    def calculate_volume_from_items(self) -> float:
        """Calculate total volume aggregated from line items."""
        return sum(item.total_volume for item in self.items)

    def sync_metrics_from_items(self) -> None:
        """Synchronize weight and volume from line items if items are defined."""
        if self.items:
            self.weight = self.calculate_weight_from_items()
            self.volume = self.calculate_volume_from_items()

    def to_dict(self) -> Dict[str, Any]:
        """Convert Order to dictionary."""
        return {
            "id": self.id,
            "destination": self.destination,
            "weight": self.weight,
            "volume": self.volume,
            "priority": self.priority,
            "time_window": self.time_window,
            "items": [it.to_dict() for it in self.items],
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Order:
        """Instantiate Order from dictionary."""
        return cls(
            id=data["id"],
            destination=tuple(data["destination"]),  # type: ignore[arg-type]
            weight=data["weight"],
            volume=data.get("volume", 0.0),
            priority=data.get("priority", 1),
            time_window=tuple(data["time_window"]) if data.get("time_window") else None,  # type: ignore[arg-type]
            items=[OrderItem.from_dict(i) if isinstance(i, dict) else i for i in data.get("items", [])],
            status=data.get("status", "PENDING"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Vehicle:
    """Vehicle in a logistics delivery fleet.

    Attributes:
        id: Vehicle identifier.
        capacity_weight: Maximum allowable cargo weight.
        capacity_volume: Maximum allowable cargo volume.
        max_stops: Maximum number of delivery stops in a single route.
        speed_kmh: Average operational speed in km/h.
        fixed_cost: Fixed dispatch cost per route assignment.
        cost_per_km: Variable cost per kilometer traveled.
        available_time_window: Operational shift time window (earliest_start, latest_return).
    """

    id: str
    capacity_weight: float
    capacity_volume: float = 1e9
    max_stops: int = 50
    speed_kmh: float = 40.0
    fixed_cost: float = 0.0
    cost_per_km: float = 1.0
    available_time_window: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValidationError(f"Vehicle ID must be a non-empty string, got {self.id!r}")
        self.capacity_weight = float(self.capacity_weight)
        self.capacity_volume = float(self.capacity_volume)
        if self.capacity_weight <= 0:
            raise ValidationError(f"Vehicle capacity_weight must be positive (> 0), got {self.capacity_weight}")
        if self.capacity_volume <= 0:
            raise ValidationError(f"Vehicle capacity_volume must be positive (> 0), got {self.capacity_volume}")
        self.max_stops = int(self.max_stops)
        if self.max_stops <= 0:
            raise ValidationError(f"Vehicle max_stops must be positive (> 0), got {self.max_stops}")
        self.speed_kmh = float(self.speed_kmh)
        if self.speed_kmh <= 0:
            raise ValidationError(f"Vehicle speed_kmh must be positive (> 0), got {self.speed_kmh}")
        self.fixed_cost = float(self.fixed_cost)
        self.cost_per_km = float(self.cost_per_km)
        if self.fixed_cost < 0:
            raise ValidationError(f"Vehicle fixed_cost cannot be negative, got {self.fixed_cost}")
        if self.cost_per_km < 0:
            raise ValidationError(f"Vehicle cost_per_km cannot be negative, got {self.cost_per_km}")

        if self.available_time_window is not None:
            if not isinstance(self.available_time_window, (tuple, list)) or len(self.available_time_window) != 2:
                raise ValidationError("Vehicle available_time_window must be a 2-tuple (start, end)")
            s, e = float(self.available_time_window[0]), float(self.available_time_window[1])
            if s < 0 or e < 0 or s > e:
                raise ValidationError(f"Invalid available_time_window: ({s}, {e})")
            self.available_time_window = (s, e)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Vehicle to dictionary."""
        return {
            "id": self.id,
            "capacity_weight": self.capacity_weight,
            "capacity_volume": self.capacity_volume,
            "max_stops": self.max_stops,
            "speed_kmh": self.speed_kmh,
            "fixed_cost": self.fixed_cost,
            "cost_per_km": self.cost_per_km,
            "available_time_window": self.available_time_window,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Vehicle:
        """Instantiate Vehicle from dictionary."""
        return cls(
            id=data["id"],
            capacity_weight=data["capacity_weight"],
            capacity_volume=data.get("capacity_volume", 1e9),
            max_stops=data.get("max_stops", 50),
            speed_kmh=data.get("speed_kmh", 40.0),
            fixed_cost=data.get("fixed_cost", 0.0),
            cost_per_km=data.get("cost_per_km", 1.0),
            available_time_window=tuple(data["available_time_window"]) if data.get("available_time_window") else None,  # type: ignore[arg-type]
        )


@dataclass
class Depot:
    """Distribution center or base depot for fleet dispatch.

    Attributes:
        id: Unique depot identifier.
        location: Depot coordinates (lat, lon).
        name: Human-readable depot name.
        time_window: Operational hours window (opening_time, closing_time).
    """

    id: str
    location: Tuple[float, float]
    name: str = ""
    time_window: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValidationError(f"Depot ID must be a non-empty string, got {self.id!r}")

        if isinstance(self.location, Location):
            self.location = self.location.as_tuple()
        elif isinstance(self.location, (list, tuple)) and len(self.location) == 2:
            self.location = (float(self.location[0]), float(self.location[1]))
        else:
            raise ValidationError(f"Depot location must be a 2-tuple (lat, lon), got {self.location!r}")

        _validate_coordinates(self.location[0], self.location[1], f"Depot '{self.id}' location")

        if self.time_window is not None:
            if not isinstance(self.time_window, (tuple, list)) or len(self.time_window) != 2:
                raise ValidationError("Depot time_window must be a 2-tuple (start, end)")
            s, e = float(self.time_window[0]), float(self.time_window[1])
            if s < 0 or e < 0 or s > e:
                raise ValidationError(f"Invalid depot time_window: ({s}, {e})")
            self.time_window = (s, e)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Depot to dictionary."""
        return {
            "id": self.id,
            "location": self.location,
            "name": self.name,
            "time_window": self.time_window,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Depot:
        """Instantiate Depot from dictionary."""
        return cls(
            id=data["id"],
            location=tuple(data["location"]),  # type: ignore[arg-type]
            name=data.get("name", ""),
            time_window=tuple(data["time_window"]) if data.get("time_window") else None,  # type: ignore[arg-type]
        )


@dataclass
class Stop:
    """Way-point or delivery visit in a route.

    Attributes:
        location: Coordinates (lat, lon) of the stop.
        order_id: Associated customer order ID (or None if depot).
        demand_weight: Weight to be delivered/collected at this stop.
        demand_volume: Volume to be delivered/collected at this stop.
        time_window: Delivery time window (earliest, latest).
        service_time: Service duration (minutes or hours) required at stop.
        is_depot: Whether this stop represents the base depot.
    """

    location: Tuple[float, float]
    order_id: Optional[str] = None
    demand_weight: float = 0.0
    demand_volume: float = 0.0
    time_window: Optional[Tuple[float, float]] = None
    service_time: float = 0.0
    is_depot: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.location, Location):
            self.location = self.location.as_tuple()
        elif isinstance(self.location, (list, tuple)) and len(self.location) == 2:
            self.location = (float(self.location[0]), float(self.location[1]))
        else:
            raise ValidationError(f"Stop location must be a 2-tuple (lat, lon), got {self.location!r}")

        _validate_coordinates(self.location[0], self.location[1], f"Stop location")

        self.demand_weight = float(self.demand_weight)
        self.demand_volume = float(self.demand_volume)
        self.service_time = float(self.service_time)

        if self.demand_weight < 0:
            raise ValidationError(f"Stop demand_weight cannot be negative, got {self.demand_weight}")
        if self.demand_volume < 0:
            raise ValidationError(f"Stop demand_volume cannot be negative, got {self.demand_volume}")
        if self.service_time < 0:
            raise ValidationError(f"Stop service_time cannot be negative, got {self.service_time}")

        if self.time_window is not None:
            if not isinstance(self.time_window, (tuple, list)) or len(self.time_window) != 2:
                raise ValidationError("Stop time_window must be a 2-tuple (start, end)")
            s, e = float(self.time_window[0]), float(self.time_window[1])
            if s < 0 or e < 0 or s > e:
                raise ValidationError(f"Invalid stop time_window: ({s}, {e})")
            self.time_window = (s, e)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Stop to dictionary."""
        return {
            "location": self.location,
            "order_id": self.order_id,
            "demand_weight": self.demand_weight,
            "demand_volume": self.demand_volume,
            "time_window": self.time_window,
            "service_time": self.service_time,
            "is_depot": self.is_depot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stop:
        """Instantiate Stop from dictionary."""
        return cls(
            location=tuple(data["location"]),  # type: ignore[arg-type]
            order_id=data.get("order_id"),
            demand_weight=data.get("demand_weight", 0.0),
            demand_volume=data.get("demand_volume", 0.0),
            time_window=tuple(data["time_window"]) if data.get("time_window") else None,  # type: ignore[arg-type]
            service_time=data.get("service_time", 0.0),
            is_depot=data.get("is_depot", False),
        )


@dataclass
class Route:
    """Sequential route assigned to a vehicle.

    Attributes:
        vehicle_id: ID of the vehicle assigned to this route.
        stops: Ordered sequence of delivery/depot stops.
        total_load_weight: Aggregated demand weight served on the route.
        total_load_volume: Aggregated demand volume served on the route.
        total_distance: Total travel distance (km).
        schedule: Sequence of schedule details (arrival, wait, departure times).
    """

    vehicle_id: str
    stops: List[Stop] = field(default_factory=list)
    total_load_weight: float = 0.0
    total_load_volume: float = 0.0
    total_distance: float = 0.0
    schedule: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.vehicle_id or not isinstance(self.vehicle_id, str):
            raise ValidationError(f"Route vehicle_id must be a non-empty string, got {self.vehicle_id!r}")
        # Parse stops if dicts are passed
        parsed_stops: List[Stop] = []
        for s in self.stops:
            if isinstance(s, dict):
                parsed_stops.append(Stop.from_dict(s))
            elif isinstance(s, Stop):
                parsed_stops.append(s)
            else:
                raise ValidationError(f"Invalid stop in route: {s!r}")
        self.stops = parsed_stops
        self.total_load_weight = float(self.total_load_weight)
        self.total_load_volume = float(self.total_load_volume)
        self.total_distance = float(self.total_distance)

    def add_stop(self, stop: Stop) -> None:
        """Append a stop to the route and update load totals."""
        self.stops.append(stop)
        self.total_load_weight += stop.demand_weight
        self.total_load_volume += stop.demand_volume

    def recalculate_totals(self) -> None:
        """Recalculate total payload weight and volume from current stops."""
        self.total_load_weight = sum(s.demand_weight for s in self.stops)
        self.total_load_volume = sum(s.demand_volume for s in self.stops)

    @property
    def stop_count(self) -> int:
        """Total number of stops."""
        return len(self.stops)

    @property
    def customer_stop_count(self) -> int:
        """Number of customer delivery stops (excluding depots)."""
        return sum(1 for s in self.stops if not s.is_depot)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Route to dictionary."""
        return {
            "vehicle_id": self.vehicle_id,
            "stops": [s.to_dict() for s in self.stops],
            "total_load_weight": self.total_load_weight,
            "total_load_volume": self.total_load_volume,
            "total_distance": self.total_distance,
            "schedule": [entry.to_dict() if hasattr(entry, "to_dict") else entry for entry in self.schedule],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Route:
        """Instantiate Route from dictionary."""
        return cls(
            vehicle_id=data["vehicle_id"],
            stops=[Stop.from_dict(s) if isinstance(s, dict) else s for s in data.get("stops", [])],
            total_load_weight=data.get("total_load_weight", 0.0),
            total_load_volume=data.get("total_load_volume", 0.0),
            total_distance=data.get("total_distance", 0.0),
            schedule=data.get("schedule", []),
        )


@dataclass
class StockItem:
    """Inventory item tracked for scientific replenishment and warehouse slotting.

    Attributes:
        sku: Unique Stock Keeping Unit code.
        on_hand: Physically present inventory units.
        on_order: Units currently ordered on outstanding purchase orders.
        unit_cost: Acquisition cost per unit.
        holding_cost_rate: Annual holding cost fraction (e.g. 0.20 = 20%).
        lead_time_days: Replenishment supplier lead time in days.
        daily_demand_history: Historical sequence of daily sales/consumption.
        allocated: Units reserved for active or pending customer orders.
    """

    sku: str
    on_hand: int
    on_order: int = 0
    unit_cost: float = 0.0
    holding_cost_rate: float = 0.2
    lead_time_days: float = 1.0
    daily_demand_history: List[float] = field(default_factory=list)
    allocated: int = 0

    def __post_init__(self) -> None:
        if not self.sku or not isinstance(self.sku, str):
            raise ValidationError(f"StockItem SKU must be a non-empty string, got {self.sku!r}")
        self.on_hand = int(self.on_hand)
        self.on_order = int(self.on_order)
        self.allocated = int(self.allocated)
        if self.on_hand < 0:
            raise ValidationError(f"StockItem on_hand cannot be negative, got {self.on_hand}")
        if self.on_order < 0:
            raise ValidationError(f"StockItem on_order cannot be negative, got {self.on_order}")
        if self.allocated < 0:
            raise ValidationError(f"StockItem allocated cannot be negative, got {self.allocated}")
        self.unit_cost = float(self.unit_cost)
        self.holding_cost_rate = float(self.holding_cost_rate)
        self.lead_time_days = float(self.lead_time_days)
        if self.unit_cost < 0:
            raise ValidationError(f"StockItem unit_cost cannot be negative, got {self.unit_cost}")
        if self.holding_cost_rate < 0:
            raise ValidationError(f"StockItem holding_cost_rate cannot be negative, got {self.holding_cost_rate}")
        if self.lead_time_days < 0:
            raise ValidationError(f"StockItem lead_time_days cannot be negative, got {self.lead_time_days}")
        self.daily_demand_history = [float(x) for x in self.daily_demand_history]

    @property
    def available(self) -> int:
        """Available stock for new allocation (on_hand - allocated)."""
        return self.on_hand - self.allocated

    @property
    def net_inventory(self) -> int:
        """Net inventory position (on_hand + on_order - allocated)."""
        return self.on_hand + self.on_order - self.allocated

    @property
    def average_daily_demand(self) -> float:
        """Arithmetic mean daily demand based on historical entries."""
        if not self.daily_demand_history:
            return 0.0
        return sum(self.daily_demand_history) / len(self.daily_demand_history)

    def annual_demand(self, days_per_year: int = 365) -> float:
        """Projected annual demand."""
        return self.average_daily_demand * days_per_year

    @property
    def annual_holding_cost_per_unit(self) -> float:
        """Annual cost of holding one unit in inventory (unit_cost * holding_cost_rate)."""
        return self.unit_cost * self.holding_cost_rate

    def allocate(self, quantity: int) -> None:
        """Reserve inventory for an order. Raises StockoutError if insufficient stock."""
        quantity = int(quantity)
        if quantity <= 0:
            raise ValidationError(f"Allocation quantity must be positive (> 0), got {quantity}")
        if self.available < quantity:
            raise StockoutError(
                sku=self.sku,
                requested=quantity,
                available=self.available,
            )
        self.allocated += quantity

    def deallocate(self, quantity: int) -> None:
        """Release previously reserved inventory (e.g. order cancellation)."""
        quantity = int(quantity)
        if quantity <= 0:
            raise ValidationError(f"Deallocation quantity must be positive (> 0), got {quantity}")
        self.allocated = max(0, self.allocated - quantity)

    def fulfill(self, quantity: int) -> None:
        """Deduct shipped units from on_hand and allocated inventory."""
        quantity = int(quantity)
        if quantity <= 0:
            raise ValidationError(f"Fulfillment quantity must be positive (> 0), got {quantity}")
        if self.on_hand < quantity:
            raise StockoutError(
                sku=self.sku,
                requested=quantity,
                available=self.on_hand,
                message=f"Cannot fulfill {quantity} units of '{self.sku}': on_hand is only {self.on_hand}."
            )
        self.on_hand -= quantity
        self.allocated = max(0, self.allocated - quantity)

    def receive(self, quantity: int) -> None:
        """Receive replenishment units from an outstanding purchase order."""
        quantity = int(quantity)
        if quantity <= 0:
            raise ValidationError(f"Received quantity must be positive (> 0), got {quantity}")
        self.on_hand += quantity
        self.on_order = max(0, self.on_order - quantity)

    def to_dict(self) -> Dict[str, Any]:
        """Convert StockItem to dictionary."""
        return {
            "sku": self.sku,
            "on_hand": self.on_hand,
            "on_order": self.on_order,
            "unit_cost": self.unit_cost,
            "holding_cost_rate": self.holding_cost_rate,
            "lead_time_days": self.lead_time_days,
            "daily_demand_history": list(self.daily_demand_history),
            "allocated": self.allocated,
            "available": self.available,
            "net_inventory": self.net_inventory,
            "average_daily_demand": self.average_daily_demand,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StockItem:
        """Instantiate StockItem from dictionary."""
        return cls(
            sku=data["sku"],
            on_hand=data["on_hand"],
            on_order=data.get("on_order", 0),
            unit_cost=data.get("unit_cost", 0.0),
            holding_cost_rate=data.get("holding_cost_rate", 0.2),
            lead_time_days=data.get("lead_time_days", 1.0),
            daily_demand_history=data.get("daily_demand_history", []),
            allocated=data.get("allocated", 0),
        )


@dataclass
class PurchaseOrder:
    """Purchase order issued to a supplier for inventory replenishment.

    Attributes:
        po_id: Unique PO identifier.
        sku: SKU to replenish.
        quantity: Order quantity.
        order_date: Epoch timestamp when PO was issued.
        expected_delivery_date: Epoch timestamp of expected arrival.
        status: Status code ("ISSUED", "CONFIRMED", "RECEIVED", "CANCELLED").
    """

    po_id: str
    sku: str
    quantity: int
    order_date: float = 0.0
    expected_delivery_date: float = 0.0
    status: str = "ISSUED"

    def __post_init__(self) -> None:
        if not self.po_id or not isinstance(self.po_id, str):
            raise ValidationError(f"PurchaseOrder po_id must be a non-empty string, got {self.po_id!r}")
        if not self.sku or not isinstance(self.sku, str):
            raise ValidationError(f"PurchaseOrder sku must be a non-empty string, got {self.sku!r}")
        self.quantity = int(self.quantity)
        if self.quantity <= 0:
            raise ValidationError(f"PurchaseOrder quantity must be positive (> 0), got {self.quantity}")
        if self.order_date == 0.0:
            self.order_date = time.time()
        self.expected_delivery_date = float(self.expected_delivery_date)

    def to_dict(self) -> Dict[str, Any]:
        """Convert PurchaseOrder to dictionary."""
        return {
            "po_id": self.po_id,
            "sku": self.sku,
            "quantity": self.quantity,
            "order_date": self.order_date,
            "expected_delivery_date": self.expected_delivery_date,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PurchaseOrder:
        """Instantiate PurchaseOrder from dictionary."""
        return cls(
            po_id=data["po_id"],
            sku=data["sku"],
            quantity=data["quantity"],
            order_date=data.get("order_date", 0.0),
            expected_delivery_date=data.get("expected_delivery_date", 0.0),
            status=data.get("status", "ISSUED"),
        )
