"""Warehouse Task Dispatcher and Picking Workflow Orchestration.

Orchestrates pick wave creation, picking manifest generation via picker routing,
picker task lifecycle progression (PENDING -> IN_PROGRESS -> COMPLETED), and event emission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Dict, List, Optional, Tuple, Union

from open_logistics.core.events import Event, EventBus
from open_logistics.core.exceptions import ValidationError
from open_logistics.warehouse.fsm import OrderFSM, OrderStatus
from open_logistics.warehouse.layout import Bin, WarehouseLayout
from open_logistics.warehouse.picking_path import (
    MidpointPickerRouting,
    ReturnPickerRouting,
    SerpentinePickerRouting,
)
from open_logistics.warehouse.wave_batcher import PickWave, WaveBatcher


@dataclass
class PickTask:
    """An individual picking action assigned to a warehouse operator or autonomous picker cart.

    Attributes:
        task_id: Unique task identifier.
        wave_id: Identifier of the parent pick wave.
        order_id: Customer order ID requiring this item.
        sku: SKU identifier to pick.
        quantity: Units to pick (> 0).
        bin_id: Target storage bin ID.
        bin_location: Bin instance, coordinates, or location string.
        location: 3D coordinates (x, y, z) of the pick.
        sequence: Execution sequence index within the wave manifest.
        status: Task progression (PENDING, IN_PROGRESS, COMPLETED).
        picker_id: Assigned operator ID.
    """

    task_id: str
    wave_id: str
    order_id: str = ""
    sku: str = ""
    quantity: int = 1
    bin_id: str = ""
    bin_location: Optional[Any] = None
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    sequence: int = 0
    status: str = "PENDING"
    picker_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.bin_location is not None and not self.bin_id:
            if hasattr(self.bin_location, "id"):
                self.bin_id = str(self.bin_location.id)
            elif isinstance(self.bin_location, str):
                self.bin_id = self.bin_location

        if self.bin_location is not None and self.location == (0.0, 0.0, 0.0):
            if hasattr(self.bin_location, "x") and hasattr(self.bin_location, "y") and hasattr(self.bin_location, "z"):
                self.location = (float(self.bin_location.x), float(self.bin_location.y), float(self.bin_location.z))
            elif isinstance(self.bin_location, (tuple, list)) and len(self.bin_location) >= 3:
                self.location = (float(self.bin_location[0]), float(self.bin_location[1]), float(self.bin_location[2]))

    def start(self, picker_id: str = "") -> None:
        """Begin task execution by assigning picker and setting IN_PROGRESS status."""
        self.status = "IN_PROGRESS"
        if picker_id:
            self.picker_id = picker_id

    def complete() -> None:
        pass  # Will be defined below


    def complete(self) -> None:
        """Mark task as successfully completed."""
        self.status = "COMPLETED"


class WarehouseTaskDispatcher:
    """Automated warehouse workflow dispatcher orchestrating waves, manifests, and order FSMs."""

    _PICKING_ROUTERS = {
        "serpentine": SerpentinePickerRouting,
        "return": ReturnPickerRouting,
        "midpoint": MidpointPickerRouting,
    }

    def __init__(
        self,
        layout: Optional[WarehouseLayout] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.layout = layout or WarehouseLayout()
        self.event_bus = event_bus
        self.waves: Dict[str, PickWave] = {}
        self.tasks: Dict[str, List[PickTask]] = {}
        self.order_fsms: Dict[str, OrderFSM] = {}
        self.batcher = WaveBatcher()
        self._wave_sequence = count(1)

    @classmethod
    def validate_picking_strategy(cls, strategy: str) -> str:
        """Return the normalized name of a supported picker routing strategy."""
        if isinstance(strategy, str) and strategy.lower() in cls._PICKING_ROUTERS:
            return strategy.lower()
        raise ValidationError("Picking strategy must be one of serpentine, return, or midpoint")

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.event_bus:
            self.event_bus.publish(
                Event(
                    topic=topic,
                    payload=payload,
                    source="WarehouseTaskDispatcher",
                    timestamp=time.time(),
                )
            )

    def _register_wave(self, wave: PickWave) -> None:
        wave.id = f"WAVE-{next(self._wave_sequence):03d}"
        self.waves[wave.id] = wave

        for order_id in wave.order_ids:
            if order_id not in self.order_fsms:
                self.order_fsms[order_id] = OrderFSM(order_id=order_id)

        self._publish("warehouse.wave_created", {"wave_id": wave.id, "order_count": len(wave.orders)})

    def create_wave(
        self,
        order_ids: Optional[List[str]] = None,
        orders: Optional[List[Any]] = None,
        max_orders: int = 25,
    ) -> PickWave:
        """Create a single consolidated pick wave from order IDs or order objects."""
        raw_orders: List[Any] = []
        if orders is not None:
            raw_orders = list(orders)
        elif order_ids is not None:
            raw_orders = list(order_ids)

        wave = PickWave(
            id="",
            orders=raw_orders,
            total_weight=sum(float(getattr(o, "weight", 0.0)) for o in raw_orders),
            total_volume=sum(float(getattr(o, "volume", 0.0)) for o in raw_orders),
            status="CREATED",
        )
        self._register_wave(wave)
        return wave

    def create_waves(
        self,
        orders: List[Any],
        max_orders_per_wave: int = 25,
        *,
        max_wave_weight: Optional[float] = None,
        max_wave_volume: Optional[float] = None,
    ) -> List[PickWave]:
        """Batch multiple orders into pick waves using WaveBatcher."""
        if not orders:
            return []

        batcher = WaveBatcher(
            max_orders=max_orders_per_wave,
            max_wave_weight=max_wave_weight,
            max_wave_volume=max_wave_volume,
        )
        waves = batcher.batch_orders(orders)

        for wave in waves:
            self._register_wave(wave)

        return waves

    def generate_picking_manifest(
        self,
        wave: PickWave,
        layout: Optional[WarehouseLayout] = None,
        routing_strategy: str = "serpentine",
        strategy: Optional[str] = None,
    ) -> List[PickTask]:
        """Generate ordered list of pick tasks for a pick wave, sequenced by routing algorithm."""
        active_layout = layout or self.layout
        selected = self.validate_picking_strategy(routing_strategy if strategy is None else strategy)
        router = self._PICKING_ROUTERS[selected]()

        tasks: List[PickTask] = []
        task_seq = 1
        bin_pool = list(active_layout.bins.values())

        for o in wave.orders:
            oid = o.id if hasattr(o, "id") else str(o)
            fsm = self.order_fsms.setdefault(oid, OrderFSM(order_id=oid))
            if fsm.current_state == OrderStatus.CREATED:
                fsm.transition(OrderStatus.ALLOCATED, context={"wave_id": wave.id})
                fsm.transition(OrderStatus.RELEASED, context={"wave_id": wave.id})
            elif fsm.current_state == OrderStatus.ALLOCATED:
                fsm.transition(OrderStatus.RELEASED, context={"wave_id": wave.id})

            items = getattr(o, "items", None)
            if items:
                for it in items:
                    sku = getattr(it, "sku", "SKU-GENERIC")
                    qty = int(getattr(it, "quantity", 1))

                    matching_bins = active_layout.get_bins_by_sku(sku)
                    if matching_bins:
                        target_bin = matching_bins[0]
                    elif bin_pool:
                        target_bin = bin_pool[(task_seq - 1) % len(bin_pool)]
                    else:
                        target_bin = Bin(id=f"BIN-{task_seq:02d}", aisle=1, bay=task_seq, tier=1, x=2.0, y=task_seq * 2.0, z=0.0)

                    task = PickTask(
                        task_id=f"TASK-{wave.id}-{task_seq:03d}",
                        wave_id=wave.id,
                        order_id=oid,
                        sku=sku,
                        quantity=qty,
                        bin_id=target_bin.id,
                        bin_location=target_bin,
                        location=(target_bin.x, target_bin.y, target_bin.z),
                        sequence=task_seq,
                        status="PENDING",
                    )
                    tasks.append(task)
                    task_seq += 1
            else:
                sku = getattr(o, "sku", f"SKU-{oid}")
                matching_bins = active_layout.get_bins_by_sku(sku)
                if matching_bins:
                    target_bin = matching_bins[0]
                elif bin_pool:
                    target_bin = bin_pool[(task_seq - 1) % len(bin_pool)]
                else:
                    target_bin = Bin(id=f"BIN-{task_seq:02d}", aisle=1, bay=task_seq, tier=1, x=2.0, y=task_seq * 2.0, z=0.0)

                task = PickTask(
                    task_id=f"TASK-{wave.id}-{task_seq:03d}",
                    wave_id=wave.id,
                    order_id=oid,
                    sku=sku,
                    quantity=1,
                    bin_id=target_bin.id,
                    bin_location=target_bin,
                    location=(target_bin.x, target_bin.y, target_bin.z),
                    sequence=task_seq,
                    status="PENDING",
                )
                tasks.append(task)
                task_seq += 1

        ordered_tasks = router.calculate_path(tasks, dock=active_layout.dock_location)

        for idx, t in enumerate(ordered_tasks):
            t.sequence = idx + 1

        self.tasks[wave.id] = ordered_tasks
        self._publish("warehouse.manifest_generated", {"wave_id": wave.id, "task_count": len(ordered_tasks)})
        return ordered_tasks

    def dispatch_wave(self, wave_id: str, picker_id: str = "PICKER-01") -> None:
        """Dispatch wave to warehouse floor, transitioning orders to PICKING."""
        wave = self.waves.get(wave_id)
        if wave:
            wave.status = "PICKING"
            for oid in wave.order_ids:
                if oid in self.order_fsms:
                    fsm = self.order_fsms[oid]
                    if fsm.current_state in (OrderStatus.RELEASED, OrderStatus.ALLOCATED):
                        if fsm.current_state == OrderStatus.ALLOCATED:
                            fsm.transition(OrderStatus.RELEASED)
                        fsm.transition(OrderStatus.PICKING, context={"picker_id": picker_id, "wave_id": wave_id})

        for t in self.tasks.get(wave_id, []):
            t.start(picker_id)

        self._publish("warehouse.wave_dispatched", {"wave_id": wave_id, "picker_id": picker_id})

    def is_wave_complete(self, wave_id: str) -> bool:
        """Check if all pick tasks in wave are completed and finalize wave state."""
        tasks = self.tasks.get(wave_id, [])
        if not tasks:
            return False

        if all(t.status == "COMPLETED" for t in tasks):
            wave = self.waves.get(wave_id)
            if wave and wave.status != "COMPLETED":
                wave.status = "COMPLETED"
                for oid in wave.order_ids:
                    if oid in self.order_fsms:
                        fsm = self.order_fsms[oid]
                        if fsm.current_state == OrderStatus.PICKING:
                            fsm.transition(OrderStatus.PICKED, context={"wave_id": wave_id})
                self._publish("warehouse.wave_completed", {"wave_id": wave_id})
            return True
        return False
