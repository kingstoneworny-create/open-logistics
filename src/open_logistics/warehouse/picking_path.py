"""Warehouse Aisle Picker Routing Algorithms.

Implements routing algorithms to optimize warehouse picker travel and eliminate backtracking:
1. Serpentine (S-Shape) Picker Routing: Alternating traversal direction across adjacent aisles.
2. Return Picker Routing: Enters aisle, picks items, and returns out the same front cross-aisle mouth.
3. Midpoint Split Routing: Partitions warehouse at midpoint_y, traversing front half then rear half.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from open_logistics.warehouse.layout import Bin, WarehouseLayout


def _coord(pt: Any) -> Tuple[float, float, float]:
    """Extract (x, y, z) coordinate from tuple, list, Bin, or PickTask."""
    if isinstance(pt, (tuple, list)):
        x = float(pt[0])
        y = float(pt[1])
        z = float(pt[2]) if len(pt) > 2 else 0.0
        return (x, y, z)
    if hasattr(pt, "bin_location") and pt.bin_location is not None:
        return _coord(pt.bin_location)
    x = float(getattr(pt, "x", 0.0))
    y = float(getattr(pt, "y", 0.0))
    z = float(getattr(pt, "z", 0.0))
    return (x, y, z)


def _euclidean_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Compute 3D Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def calculate_path_distance(
    route: List[Any],
    dock_location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """Calculate total picker travel distance including dock departure and return.

    Args:
        route: Sequence of stops (coordinates, Bins, or tasks) in order of traversal.
        dock_location: Origin and return coordinates of the dispatch dock.

    Returns:
        Total round-trip travel distance.
    """
    if not route:
        return 0.0

    dock = (float(dock_location[0]), float(dock_location[1]), float(dock_location[2]) if len(dock_location) > 2 else 0.0)
    current = dock
    total_dist = 0.0

    for stop in route:
        nxt = _coord(stop)
        total_dist += _euclidean_distance(current, nxt)
        current = nxt

    # Return to dock
    total_dist += _euclidean_distance(current, dock)
    return total_dist


def _group_by_aisle(picks: List[Any]) -> List[Tuple[float, List[Any]]]:
    """Group pick locations by aisle (using aisle attribute or X-coordinate)."""
    aisle_dict: Dict[float, List[Any]] = {}
    for p in picks:
        if hasattr(p, "aisle"):
            k = float(p.aisle)
        elif hasattr(p, "bin_location") and hasattr(p.bin_location, "aisle"):
            k = float(p.bin_location.aisle)
        else:
            c = _coord(p)
            k = round(c[0], 2)
        aisle_dict.setdefault(k, []).append(p)

    sorted_aisles = sorted(aisle_dict.items(), key=lambda item: item[0])
    return sorted_aisles


def serpentine_path(bins: List[Any], layout: Optional[WarehouseLayout] = None) -> List[Any]:
    """Compute serpentine S-shape picking path across aisles."""
    router = SerpentinePickerRouting()
    dock = layout.dock_location if layout else (0.0, 0.0, 0.0)
    return router.calculate_path(bins, dock=dock)


def return_path(bins: List[Any], layout: Optional[WarehouseLayout] = None) -> List[Any]:
    """Compute return routing picking path."""
    router = ReturnPickerRouting()
    dock = layout.dock_location if layout else (0.0, 0.0, 0.0)
    return router.calculate_path(bins, dock=dock)


def midpoint_path(
    bins: List[Any],
    layout: Optional[WarehouseLayout] = None,
    midpoint_y: float = 10.0,
) -> List[Any]:
    """Compute midpoint split routing picking path."""
    router = MidpointPickerRouting(midpoint_y=midpoint_y)
    dock = layout.dock_location if layout else (0.0, 0.0, 0.0)
    return router.calculate_path(bins, dock=dock)


class SerpentinePickerRouting:
    """Serpentine S-shape picker routing alternating traversal direction across adjacent aisles."""

    def calculate_path(
        self,
        picks: List[Any],
        dock: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[Any]:
        """Sequence pick locations in serpentine order (aisle 1 ascending, aisle 2 descending, etc.)."""
        if not picks:
            return []

        aisle_groups = _group_by_aisle(picks)
        ordered: List[Any] = []

        for idx, (_, aisle_picks) in enumerate(aisle_groups):
            def pick_key(p: Any) -> float:
                if hasattr(p, "bay"):
                    return float(p.bay)
                if hasattr(p, "bin_location") and hasattr(p.bin_location, "bay"):
                    return float(p.bin_location.bay)
                return _coord(p)[1]

            reverse = (idx % 2 == 1)
            sorted_aisle = sorted(aisle_picks, key=pick_key, reverse=reverse)
            ordered.extend(sorted_aisle)

        return ordered

    def calculate_distance(
        self,
        picks: List[Any],
        dock: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> float:
        """Compute total serpentine picker round-trip travel distance."""
        if not picks:
            return 0.0
        path = self.calculate_path(picks, dock=dock)
        return calculate_path_distance(path, dock_location=dock)

    def calculate_route(
        self,
        picks: List[Any],
        depot_location: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[List[Any], float]:
        """Convenience method returning (path, distance)."""
        dock_3d = (float(depot_location[0]), float(depot_location[1]), 0.0)
        path = self.calculate_path(picks, dock=dock_3d)
        dist = calculate_path_distance(path, dock_location=dock_3d)
        return path, dist


class ReturnPickerRouting:
    """Return routing: picker enters aisle, picks items, and returns out the front mouth."""

    def calculate_path(
        self,
        picks: List[Any],
        dock: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[Any]:
        """Sequence pick locations in return order."""
        if not picks:
            return []

        aisle_groups = _group_by_aisle(picks)
        ordered: List[Any] = []

        for _, aisle_picks in aisle_groups:
            def pick_key(p: Any) -> float:
                if hasattr(p, "bay"):
                    return float(p.bay)
                if hasattr(p, "bin_location") and hasattr(p.bin_location, "bay"):
                    return float(p.bin_location.bay)
                return _coord(p)[1]

            sorted_aisle = sorted(aisle_picks, key=pick_key)
            ordered.extend(sorted_aisle)

        return ordered

    def calculate_distance(
        self,
        picks: List[Any],
        dock: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> float:
        """Calculate return routing distance including returning to front cross-aisle (y=0) after each aisle."""
        if not picks:
            return 0.0

        dock_3d = (float(dock[0]), float(dock[1]), float(dock[2]) if len(dock) > 2 else 0.0)
        aisle_groups = _group_by_aisle(picks)

        current = dock_3d
        total_dist = 0.0

        for _, aisle_picks in aisle_groups:
            if not aisle_picks:
                continue
            sorted_aisle = sorted(aisle_picks, key=lambda p: _coord(p)[1])
            aisle_x = _coord(sorted_aisle[0])[0]
            mouth = (aisle_x, 0.0, 0.0)

            total_dist += _euclidean_distance(current, mouth)
            curr_pos = mouth

            for p in sorted_aisle:
                nxt = _coord(p)
                total_dist += _euclidean_distance(curr_pos, nxt)
                curr_pos = nxt

            total_dist += _euclidean_distance(curr_pos, mouth)
            current = mouth

        total_dist += _euclidean_distance(current, dock_3d)
        return total_dist

    def calculate_route(
        self,
        picks: List[Any],
        depot_location: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[List[Any], float]:
        """Convenience method returning (path, distance)."""
        dock_3d = (float(depot_location[0]), float(depot_location[1]), 0.0)
        path = self.calculate_path(picks, dock=dock_3d)
        dist = self.calculate_distance(picks, dock=dock_3d)
        return path, dist


class MidpointPickerRouting:
    """Midpoint routing splitting aisles at warehouse midpoint, traversing front then rear half."""

    def __init__(self, midpoint_y: float = 10.0) -> None:
        self.midpoint_y = float(midpoint_y)

    def calculate_path(
        self,
        picks: List[Any],
        dock: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[Any]:
        """Sequence pick locations traversing front half (<= midpoint_y) then rear half (> midpoint_y)."""
        if not picks:
            return []

        front_picks = [p for p in picks if _coord(p)[1] <= self.midpoint_y]
        rear_picks = [p for p in picks if _coord(p)[1] > self.midpoint_y]

        router = SerpentinePickerRouting()
        front_path = router.calculate_path(front_picks, dock=dock)
        rear_path = router.calculate_path(rear_picks, dock=dock)

        return front_path + rear_path

    def calculate_distance(
        self,
        picks: List[Any],
        dock: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> float:
        """Calculate total midpoint picker travel distance."""
        if not picks:
            return 0.0
        path = self.calculate_path(picks, dock=dock)
        return calculate_path_distance(path, dock_location=dock)

    def calculate_route(
        self,
        picks: List[Any],
        depot_location: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[List[Any], float]:
        """Convenience method returning (path, distance)."""
        dock_3d = (float(depot_location[0]), float(depot_location[1]), 0.0)
        path = self.calculate_path(picks, dock=dock_3d)
        dist = self.calculate_distance(picks, dock=dock_3d)
        return path, dist
