"""3D Spatial Layout Modeling for Warehouses.

Provides classes and data structures for modeling warehouse layouts in 3D:
zones, aisles, bays, tiers, 3D bounding boxes, coordinate systems, and bins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from open_logistics.core.exceptions import CapacityExceededError, WarehouseError  # noqa: F401  (re-exported)


@dataclass
class BoundingBox3D:
    """Axis-Aligned Bounding Box (AABB) in 3D space.

    Attributes:
        x: X-coordinate of origin (width axis).
        y: Y-coordinate of origin (length / aisle depth axis).
        z: Z-coordinate of origin (height / vertical tier axis).
        width: Dimension along X axis (> 0).
        length: Dimension along Y axis (> 0).
        height: Dimension along Z axis (> 0).
    """

    x: float
    y: float
    z: float
    width: float
    length: float
    height: float

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
        self.width = float(self.width)
        self.length = float(self.length)
        self.height = float(self.height)
        if self.width < 0.0 or self.length < 0.0 or self.height < 0.0:
            raise ValueError("Bounding box dimensions cannot be negative")

    @property
    def volume(self) -> float:
        """Total volume of the 3D bounding box."""
        return self.width * self.length * self.height

    @property
    def center(self) -> Tuple[float, float, float]:
        """Centroid of the 3D bounding box."""
        return (
            self.x + self.width / 2.0,
            self.y + self.length / 2.0,
            self.z + self.height / 2.0,
        )

    def contains_point(self, x: float, y: float, z: float, tolerance: float = 1e-9) -> bool:
        """Check if the point (x, y, z) lies inside this bounding box."""
        return (
            (self.x - tolerance) <= x <= (self.x + self.width + tolerance)
            and (self.y - tolerance) <= y <= (self.y + self.length + tolerance)
            and (self.z - tolerance) <= z <= (self.z + self.height + tolerance)
        )

    def overlaps(self, other: BoundingBox3D, tolerance: float = 1e-9) -> bool:
        """Check if this bounding box overlaps another bounding box."""
        no_overlap = (
            (self.x + self.width) <= (other.x + tolerance)
            or (other.x + other.width) <= (self.x + tolerance)
            or (self.y + self.length) <= (other.y + tolerance)
            or (other.y + other.length) <= (self.y + tolerance)
            or (self.z + self.height) <= (other.z + tolerance)
            or (other.z + other.height) <= (self.z + tolerance)
        )
        return not no_overlap


class Bin:
    """Warehouse storage bin / cell location with 3D spatial properties and capacity constraints.

    Can be initialized with explicit coordinates and bounding box or legacy BinLocation parameters.
    """

    def __init__(
        self,
        id: str,
        aisle: int = 1,
        bay: int = 1,
        tier: int = 1,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        bbox: Optional[BoundingBox3D] = None,
        max_weight: float = 1000.0,
        max_volume: Optional[float] = None,
        current_weight: float = 0.0,
        current_volume: float = 0.0,
        occupied: bool = False,
        sku: Optional[str] = None,
        zone: str = "DEFAULT",
        dimensions: Optional[Any] = None,
        width: float = 1.0,
        length: float = 1.0,
        height: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.id = str(id)
        self.aisle = int(aisle)
        self.bay = int(bay)
        self.tier = int(tier)
        self.zone = str(zone)
        self.max_weight = float(max_weight)
        self.current_weight = float(current_weight)
        self.current_volume = float(current_volume)
        self.occupied = bool(occupied)
        self.sku = sku
        self.dimensions = dimensions

        if dimensions is not None:
            width = float(getattr(dimensions, "width", width))
            length = float(getattr(dimensions, "length", length))
            height = float(getattr(dimensions, "height", height))
            if max_volume is None:
                max_volume = float(getattr(dimensions, "volume", width * length * height))

        if bbox is not None:
            self.bbox = bbox
            self.x = bbox.x
            self.y = bbox.y
            self.z = bbox.z
            self.width = bbox.width
            self.length = bbox.length
            self.height = bbox.height
            self._volume = float(max_volume if max_volume is not None else bbox.volume)
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.width = float(width)
            self.length = float(length)
            self.height = float(height)
            self.bbox = BoundingBox3D(self.x, self.y, self.z, self.width, self.length, self.height)
            self._volume = float(max_volume if max_volume is not None else (self.width * self.length * self.height))

    @property
    def volume(self) -> float:
        """Total volumetric capacity of the bin."""
        return self._volume

    @property
    def max_volume(self) -> float:
        """Alias for volume capacity."""
        return self._volume

    @max_volume.setter
    def max_volume(self, val: float) -> None:
        self._volume = float(val)

    @property
    def remaining_weight(self) -> float:
        """Available payload weight capacity remaining in this bin."""
        return max(0.0, self.max_weight - self.current_weight)

    @property
    def remaining_volume(self) -> float:
        """Available volumetric space remaining in this bin."""
        return max(0.0, self._volume - self.current_volume)

    @property
    def center(self) -> Tuple[float, float, float]:
        """3D centroid coordinate of the bin."""
        return (
            self.x + self.width / 2.0,
            self.y + self.length / 2.0,
            self.z + self.height / 2.0,
        )

    def __repr__(self) -> str:
        return (
            f"Bin(id={self.id!r}, zone={self.zone!r}, aisle={self.aisle}, bay={self.bay}, "
            f"tier={self.tier}, coords=({self.x}, {self.y}, {self.z}), "
            f"max_weight={self.max_weight}, volume={self.volume})"
        )


# BinLocation is fully compatible with Bin
BinLocation = Bin


@dataclass
class Aisle:
    """Aisle grouping bays along a traversal corridor in a warehouse zone.

    Attributes:
        id: Numeric or string aisle identifier.
        zone: Zone identifier containing this aisle.
        length: Physical aisle length along traversal axis.
        width: Corridor width.
        bays: Number of bays per side.
    """

    id: int
    zone: str = "DEFAULT"
    length: float = 50.0
    width: float = 3.0
    bays: int = 10


@dataclass
class Zone:
    """Warehouse operational zone (e.g. ambient, chilled, mezzanine, high-bay).

    Attributes:
        id: Unique zone identifier (e.g. 'A', 'ZONE-BULK').
        name: Human-readable zone designation.
        aisles: List of aisles situated in this zone.
    """

    id: str
    name: str = ""
    aisles: List[Aisle] = field(default_factory=list)


class WarehouseLayout:
    """Comprehensive 3D warehouse layout indexing zones, aisles, and storage bins."""

    def __init__(
        self,
        id: str = "DEFAULT",
        zones: Optional[Union[Dict[str, Zone], List[Zone]]] = None,
        bins: Optional[Union[Dict[str, Bin], List[Bin]]] = None,
        dock_location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.id = str(id)
        self.dock_location = (float(dock_location[0]), float(dock_location[1]), float(dock_location[2]) if len(dock_location) > 2 else 0.0)

        self.zones: Dict[str, Zone] = {}
        if zones is not None:
            if isinstance(zones, dict):
                self.zones = dict(zones)
            else:
                for z in zones:
                    self.zones[z.id] = z

        self.bins: Dict[str, Bin] = {}
        if bins is not None:
            if isinstance(bins, dict):
                self.bins = dict(bins)
            else:
                for b in bins:
                    self.bins[b.id] = b

    def add_zone(self, zone: Zone) -> None:
        """Register an operational zone in the layout."""
        self.zones[zone.id] = zone

    def add_bin(self, bin_loc: Bin) -> None:
        """Register a storage bin location in the layout."""
        self.bins[bin_loc.id] = bin_loc

    def get_bin(self, bin_id: str) -> Optional[Bin]:
        """Retrieve a bin by unique identifier."""
        return self.bins.get(bin_id)

    def get_bins_by_sku(self, sku: str) -> List[Bin]:
        """Retrieve all bins currently storing the given SKU."""
        return [b for b in self.bins.values() if b.sku == sku]

    def get_bins_by_zone(self, zone_id: str) -> List[Bin]:
        """Retrieve all bins located in a specified zone."""
        return [b for b in self.bins.values() if b.zone == zone_id]

    def get_bins_by_aisle(self, aisle_id: int, zone_id: Optional[str] = None) -> List[Bin]:
        """Retrieve all bins located along a specified aisle."""
        return [
            b for b in self.bins.values()
            if b.aisle == aisle_id and (zone_id is None or b.zone == zone_id)
        ]

    def distance_to_dock(self, target: Union[Bin, Tuple[float, ...]]) -> float:
        """Calculate 3D Euclidean distance from dock location to a bin or 3D coordinate."""
        if isinstance(target, (tuple, list)):
            tx = float(target[0])
            ty = float(target[1])
            tz = float(target[2]) if len(target) > 2 else 0.0
        else:
            tx = float(target.x)
            ty = float(target.y)
            tz = float(target.z)

        dx, dy, dz = self.dock_location
        return math.sqrt((tx - dx) ** 2 + (ty - dy) ** 2 + (tz - dz) ** 2)
