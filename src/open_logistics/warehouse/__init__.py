"""Warehouse Automation, 3D Slotting & Task Dispatch Module.

Provides comprehensive warehouse management capabilities:
- 3D spatial layout modeling (zones, aisles, bays, tiers, bounding boxes)
- 3D slotting optimization with 6 orthogonal package rotations
- Picking path optimization (Serpentine S-shape, Return, Midpoint)
- 7-Stage order lifecycle FSM with atomic rollback guards
- Carrier cutoff and capacity-constrained wave batching
- Automated warehouse task dispatching and manifest generation
"""

from open_logistics.core.exceptions import FSMStateError, WarehouseError
from open_logistics.warehouse.fsm import (
    InvalidTransitionError,
    OrderFSM,
    OrderLifecycleManager,
    OrderStatus,
)
from open_logistics.warehouse.layout import (
    Aisle,
    Bin,
    BinLocation,
    BoundingBox3D,
    CapacityExceededError,
    WarehouseLayout,
    Zone,
)
from open_logistics.warehouse.picking_path import (
    MidpointPickerRouting,
    ReturnPickerRouting,
    SerpentinePickerRouting,
    calculate_path_distance,
    midpoint_path,
    return_path,
    serpentine_path,
)
from open_logistics.warehouse.slotting import (
    SlottingOptimizer,
    SlottingRecommendation,
    can_fit_dimensions,
    find_optimal_slot,
    slot_inventory,
)
from open_logistics.warehouse.task_dispatcher import (
    PickTask,
    WarehouseTaskDispatcher,
)
from open_logistics.warehouse.wave_batcher import (
    PickWave,
    WaveBatcher,
    batch_waves,
)

__all__ = [
    # Exceptions
    "WarehouseError",
    "FSMStateError",
    "CapacityExceededError",
    "InvalidTransitionError",
    # 3D Spatial Layout
    "BoundingBox3D",
    "Bin",
    "BinLocation",
    "Aisle",
    "Zone",
    "WarehouseLayout",
    # 3D Slotting
    "SlottingOptimizer",
    "SlottingRecommendation",
    "find_optimal_slot",
    "slot_inventory",
    "can_fit_dimensions",
    # Picking Routing
    "SerpentinePickerRouting",
    "ReturnPickerRouting",
    "MidpointPickerRouting",
    "serpentine_path",
    "return_path",
    "midpoint_path",
    "calculate_path_distance",
    # Order Lifecycle FSM
    "OrderStatus",
    "OrderFSM",
    "OrderLifecycleManager",
    # Wave Batching
    "PickWave",
    "WaveBatcher",
    "batch_waves",
    # Task Dispatcher
    "PickTask",
    "WarehouseTaskDispatcher",
]
