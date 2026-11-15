"""3D Slotting Optimization and Cell Placement Heuristics.

Provides algorithms for optimal inventory slotting in 3D warehouse grids:
- Minimizing volume slack (wasted space)
- Minimizing travel distance to intake/shipping docks
- 3D orthogonal rotation checking across all 6 orientations
- Weight and volumetric capacity constraint enforcement
- ABC-XYZ velocity categorization-informed placement
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from open_logistics.warehouse.layout import Bin, CapacityExceededError, WarehouseLayout


@dataclass
class SlottingRecommendation:
    """Recommendation result for assigning a SKU to an optimal warehouse bin.

    Attributes:
        sku: SKU identifier.
        bin_id: Target bin location identifier.
        distance_to_dock: Travel distance from target bin to shipping/intake dock.
        volume_slack: Unused volumetric space in the bin (bin.volume - item.volume).
        score: Composite optimization fitness score (higher is better).
    """

    sku: str
    bin_id: str
    distance_to_dock: float
    volume_slack: float
    score: float


def _extract_dims(dims: Any) -> Tuple[float, float, float]:
    """Extract (width, length, height) triple from tuple, list, or Dimensions object."""
    if hasattr(dims, "width") and hasattr(dims, "length") and hasattr(dims, "height"):
        return (float(dims.width), float(dims.length), float(dims.height))
    if isinstance(dims, (tuple, list)):
        if len(dims) >= 3:
            return (float(dims[0]), float(dims[1]), float(dims[2]))
        if len(dims) == 2:
            return (float(dims[0]), float(dims[1]), 1.0)
    return (1.0, 1.0, 1.0)


def can_fit_dimensions(
    item_dims: Any,
    bin_dims: Any,
    tolerance: float = 1e-9,
) -> bool:
    """Test whether item fits into bin across all 6 orthogonal 3D rotations."""
    iw, il, ih = _extract_dims(item_dims)
    bw, bl, bh = _extract_dims(bin_dims)

    # Sort dimensions for orthogonal rotation check
    i_sorted = sorted([iw, il, ih])
    b_sorted = sorted([bw, bl, bh])

    return (
        i_sorted[0] <= b_sorted[0] + tolerance
        and i_sorted[1] <= b_sorted[1] + tolerance
        and i_sorted[2] <= b_sorted[2] + tolerance
    )


def find_optimal_slot(
    package_dims: Tuple[float, float, float],
    weight: float,
    available_bins: List[Bin],
    dock_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Optional[Bin]:
    """Find the optimal bin for a package by minimizing volume slack and dock travel.

    Args:
        package_dims: (width, length, height) of the package.
        weight: Package weight in kg.
        available_bins: Candidate bins to search.
        dock_origin: 3D coordinates of the shipping/receiving dock.
        alpha: Weight assigned to volume slack penalty (0.0 to 1.0).
        beta: Weight assigned to travel distance penalty (0.0 to 1.0).

    Returns:
        Best-fit Bin instance, or None if no available bin can accommodate the package.
    """
    pkg_w, pkg_l, pkg_h = package_dims
    pkg_vol = pkg_w * pkg_l * pkg_h

    dx, dy, dz = dock_origin[:3]
    best_bin: Optional[Bin] = None
    best_cost = float("inf")

    for b in available_bins:
        if b.occupied:
            continue
        # Weight capacity constraint
        if b.current_weight + weight > b.max_weight + 1e-9:
            continue
        # Volume capacity constraint
        if b.current_volume + pkg_vol > b.volume + 1e-9:
            continue
        # Geometric 3D orientation fit check
        bin_dims = (b.width, b.length, b.height)
        if not can_fit_dimensions(package_dims, bin_dims):
            continue

        slack = max(0.0, b.volume - pkg_vol)
        dist = math.sqrt((b.x - dx) ** 2 + (b.y - dy) ** 2 + (b.z - dz) ** 2)

        cost = alpha * slack + beta * dist
        if cost < best_cost:
            best_cost = cost
            best_bin = b

    return best_bin


def slot_inventory(
    items: List[Any],
    layout: WarehouseLayout,
    dock_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Dict[str, str]:
    """Slot a collection of inventory items into warehouse layout bins.

    Returns:
        Dictionary mapping SKU -> assigned bin_id.
    """
    optimizer = SlottingOptimizer(layout=layout)
    recs = optimizer.optimize_slotting(items=items)
    return {rec.sku: rec.bin_id for rec in recs}


class SlottingOptimizer:
    """High-level 3D slotting optimizer coordinating layout placement and velocity ranking."""

    def __init__(self, layout: Optional[WarehouseLayout] = None) -> None:
        self.layout = layout or WarehouseLayout()

    def can_fit(self, item_dims: Any, bin_dims: Any) -> bool:
        """Check if item fits in bin considering 3D rotations."""
        return can_fit_dimensions(item_dims, bin_dims)

    def assign_item(
        self,
        sku: str,
        bin_id: str,
        weight: float,
        volume: float,
    ) -> None:
        """Explicitly allocate an item to a bin, enforcing weight and volume limits."""
        target_bin = self.layout.get_bin(bin_id)
        if target_bin is None:
            raise CapacityExceededError(f"Target bin {bin_id!r} not found in layout")

        if target_bin.current_weight + weight > target_bin.max_weight + 1e-9:
            raise CapacityExceededError(
                f"Bin {bin_id} weight limit exceeded: {target_bin.current_weight + weight} > {target_bin.max_weight}"
            )
        if target_bin.current_volume + volume > target_bin.volume + 1e-9:
            raise CapacityExceededError(
                f"Bin {bin_id} volume limit exceeded: {target_bin.current_volume + volume} > {target_bin.volume}"
            )

        target_bin.current_weight += weight
        target_bin.current_volume += volume
        target_bin.sku = sku
        target_bin.occupied = True

    def recommend_bin(
        self,
        sku: str,
        volume: float = 0.0,
        weight: float = 0.0,
        dimensions: Optional[Tuple[float, float, float]] = None,
    ) -> SlottingRecommendation:
        """Recommend single best-fit bin for a given SKU."""
        available = [b for b in self.layout.bins.values() if not b.occupied]
        if not available:
            available = [b for b in self.layout.bins.values() if b.remaining_weight >= weight and b.remaining_volume >= volume]

        if not available:
            raise CapacityExceededError(f"No available bin in layout for SKU {sku}")

        best_bin: Optional[Bin] = None
        best_slack = float("inf")
        best_dist = float("inf")

        for b in available:
            if b.max_weight < weight or b.volume < volume - 1e-9:
                continue
            slack = b.volume - volume
            if slack < -1e-9:
                continue
            dist = self.layout.distance_to_dock(b)

            if slack < best_slack - 1e-9:
                best_slack = slack
                best_dist = dist
                best_bin = b
            elif abs(slack - best_slack) <= 1e-9 and dist < best_dist:
                best_dist = dist
                best_bin = b

        if best_bin is None:
            best_bin = min(available, key=lambda b: (abs(b.volume - volume), self.layout.distance_to_dock(b)))

        dist = self.layout.distance_to_dock(best_bin)
        slack = max(0.0, best_bin.volume - volume)
        score = max(0.0, 100.0 - (dist * 0.5 + slack * 10.0))

        return SlottingRecommendation(
            sku=sku,
            bin_id=best_bin.id,
            distance_to_dock=dist,
            volume_slack=slack,
            score=round(score, 2),
        )

    def optimize_slotting(
        self,
        items: List[Any],
        abc_classes: Optional[Dict[str, Any]] = None,
    ) -> List[SlottingRecommendation]:
        """Batch slotting optimizer placing high-velocity / Class A items closest to dock."""
        def item_priority(it: Any) -> Tuple[int, float]:
            sku = getattr(it, "sku", str(it))
            abc_rank = 1
            if abc_classes and sku in abc_classes:
                val = str(abc_classes[sku]).upper()
                if val in ("A", "0"):
                    abc_rank = 0
                elif val in ("B", "1"):
                    abc_rank = 1
                elif val in ("C", "2"):
                    abc_rank = 2
                elif val in ("DEAD", "3"):
                    abc_rank = 3

            demand = 0.0
            if hasattr(it, "average_daily_demand") and it.average_daily_demand:
                demand = float(it.average_daily_demand)
            elif hasattr(it, "daily_demand_history") and it.daily_demand_history:
                demand = float(sum(it.daily_demand_history))
            elif hasattr(it, "unit_cost") and hasattr(it, "on_hand"):
                demand = float(it.unit_cost * it.on_hand)

            return (abc_rank, -demand)

        sorted_items = sorted(items, key=item_priority)

        candidate_bins = sorted(
            [b for b in self.layout.bins.values() if not b.occupied],
            key=lambda b: self.layout.distance_to_dock(b),
        )
        if not candidate_bins:
            candidate_bins = sorted(
                list(self.layout.bins.values()),
                key=lambda b: self.layout.distance_to_dock(b),
            )

        recommendations: List[SlottingRecommendation] = []
        assigned_bin_ids: set[str] = set()

        for it in sorted_items:
            sku = getattr(it, "sku", str(it))
            it_vol = getattr(it, "volume", 0.05)
            it_weight = getattr(it, "weight", 1.0)

            chosen_bin: Optional[Bin] = None
            for b in candidate_bins:
                if b.id in assigned_bin_ids:
                    continue
                if b.max_weight >= it_weight and b.volume >= it_vol - 1e-9:
                    chosen_bin = b
                    break

            if chosen_bin is None:
                chosen_bin = min(candidate_bins, key=lambda b: self.layout.distance_to_dock(b))

            assigned_bin_ids.add(chosen_bin.id)
            dist = self.layout.distance_to_dock(chosen_bin)
            slack = max(0.0, chosen_bin.volume - it_vol)
            score = max(0.0, 100.0 - (dist * 0.5 + slack * 10.0))

            recommendations.append(
                SlottingRecommendation(
                    sku=sku,
                    bin_id=chosen_bin.id,
                    distance_to_dock=dist,
                    volume_slack=slack,
                    score=round(score, 2),
                )
            )

        return recommendations
