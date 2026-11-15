"""Carrier Cutoff and Capacity-Constrained Pick Wave Batching.

Groups customer orders into optimal pick waves subject to:
- Carrier departure cutoff deadlines
- Picker cart maximum payload weight capacity
- Picker cart maximum order count limit
- Volumetric cube capacity limits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class PickWave:
    """A batched pick wave grouping orders for consolidated warehouse picking.

    Attributes:
        id: Unique wave identifier.
        orders: List of customer orders or order IDs assigned to this wave.
        carrier_cutoff: Earliest carrier departure deadline (decimal hours or timestamp).
        total_weight: Aggregated payload weight of orders in the wave.
        total_volume: Aggregated cubic volume of orders in the wave.
        status: Wave execution state (CREATED, RELEASED, PICKING, COMPLETED).
    """

    id: str
    orders: List[Any] = field(default_factory=list)
    carrier_cutoff: Optional[float] = None
    total_weight: float = 0.0
    total_volume: float = 0.0
    status: str = "CREATED"

    @property
    def order_ids(self) -> List[str]:
        """Extract list of order ID strings in the wave."""
        return [o.id if hasattr(o, "id") else str(o) for o in self.orders]

    def release(self) -> None:
        """Transition wave state to RELEASED upon batch finalization."""
        self.status = "RELEASED"


class WaveBatcher:
    """Batch processor partitioning orders into waves respecting cutoff and capacity constraints."""

    def __init__(
        self,
        max_orders: int = 25,
        max_cart_weight: float = 200.0,
        max_volume: float = 1.5,
        carrier_cutoffs: Optional[Dict[str, float]] = None,
        max_weight: Optional[float] = None,
        max_wave_weight: Optional[float] = None,
        max_wave_volume: Optional[float] = None,
        max_wave_orders: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.max_orders = int(max_wave_orders if max_wave_orders is not None else max_orders)
        if max_wave_weight is not None:
            self.max_cart_weight = float(max_wave_weight)
        elif max_weight is not None:
            self.max_cart_weight = float(max_weight)
        else:
            self.max_cart_weight = float(max_cart_weight)

        if max_wave_volume is not None:
            self.max_volume = float(max_wave_volume)
        else:
            self.max_volume = float(max_volume)

        self.carrier_cutoffs = carrier_cutoffs or {}

    def batch_orders(
        self,
        orders: List[Any],
        carrier_cutoffs: Optional[Dict[str, float]] = None,
    ) -> List[PickWave]:
        """Batch a list of pending orders into feasible pick waves."""
        if not orders:
            return []

        active_cutoffs = dict(self.carrier_cutoffs)
        if carrier_cutoffs:
            active_cutoffs.update(carrier_cutoffs)

        def get_carrier_and_cutoff(order: Any) -> Tuple[Optional[str], Optional[float]]:
            carrier = None
            cutoff = None
            if hasattr(order, "metadata") and isinstance(order.metadata, dict):
                carrier = order.metadata.get("carrier")
                cutoff = order.metadata.get("cutoff")
            if carrier is None and hasattr(order, "carrier"):
                carrier = getattr(order, "carrier")
            if cutoff is None and hasattr(order, "cutoff"):
                cutoff = getattr(order, "cutoff")
            if cutoff is None and carrier and carrier in active_cutoffs:
                cutoff = active_cutoffs[carrier]
            return (carrier, cutoff)

        def get_order_weight(order: Any) -> float:
            if hasattr(order, "weight"):
                return float(order.weight)
            return 0.0

        def get_order_volume(order: Any) -> float:
            if hasattr(order, "volume"):
                return float(order.volume)
            return 0.0

        groups: Dict[Tuple[Optional[str], Optional[float]], List[Any]] = {}
        for o in orders:
            c, cut = get_carrier_and_cutoff(o)
            groups.setdefault((c, cut), []).append(o)

        sorted_groups = sorted(
            groups.items(),
            key=lambda item: (item[0][1] if item[0][1] is not None else float("inf"), str(item[0][0])),
        )

        waves: List[PickWave] = []
        wave_seq = 1

        for (carrier, cutoff), group_orders in sorted_groups:
            current_wave: Optional[PickWave] = None

            for o in group_orders:
                w = get_order_weight(o)
                v = get_order_volume(o)

                fits = (
                    current_wave is not None
                    and len(current_wave.orders) < self.max_orders
                    and (current_wave.total_weight + w <= self.max_cart_weight + 1e-9 or w == 0.0)
                    and (current_wave.total_volume + v <= self.max_volume + 1e-9 or v == 0.0)
                )

                if fits and current_wave is not None:
                    current_wave.orders.append(o)
                    current_wave.total_weight += w
                    current_wave.total_volume += v
                else:
                    current_wave = PickWave(
                        id=f"WAVE-{wave_seq:03d}",
                        orders=[o],
                        carrier_cutoff=cutoff,
                        total_weight=w,
                        total_volume=v,
                        status="CREATED",
                    )
                    waves.append(current_wave)
                    wave_seq += 1

        return waves

    def batch_waves(
        self,
        orders: List[Any],
        carrier_cutoffs: Optional[Dict[str, float]] = None,
    ) -> List[PickWave]:
        """Alias for batch_orders."""
        return self.batch_orders(orders, carrier_cutoffs=carrier_cutoffs)


def batch_waves(
    orders: List[Any],
    max_orders: int = 25,
    max_weight: float = 200.0,
    max_volume: float = 1.5,
    carrier_cutoffs: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> List[PickWave]:
    """Functional helper to batch orders into pick waves."""
    batcher = WaveBatcher(
        max_orders=max_orders,
        max_cart_weight=max_weight,
        max_volume=max_volume,
        carrier_cutoffs=carrier_cutoffs,
        **kwargs,
    )
    return batcher.batch_orders(orders, carrier_cutoffs=carrier_cutoffs)
