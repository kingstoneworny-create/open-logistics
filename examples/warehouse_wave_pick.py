"""Build serpentine pick manifests for a warehouse and complete each wave."""

import json

from open_logistics import LogisticsConfig, LogisticsEngine, Order, OrderItem
from open_logistics.warehouse.layout import Bin, WarehouseLayout


def main() -> None:
    layout = WarehouseLayout(bins=[
        Bin("A1-B1", aisle=1, bay=1, x=1, y=2, sku="TEA"),
        Bin("A2-B1", aisle=2, bay=1, x=3, y=2, sku="COFFEE"),
        Bin("A2-B2", aisle=2, bay=2, x=3, y=5, sku="COCOA"),
    ])
    orders = [
        Order("WEB-1", (41.885, -87.625), weight=2, items=[OrderItem("COFFEE", 2)]),
        Order("WEB-2", (41.891, -87.617), weight=1, items=[OrderItem("TEA", 1)]),
        Order("WEB-3", (41.893, -87.615), weight=1, items=[OrderItem("COCOA", 1)]),
    ]
    engine = LogisticsEngine(LogisticsConfig(max_wave_orders=2), layout=layout)
    result = engine.allocate_and_pick(orders, strategy="serpentine")
    for wave in result.waves:
        engine.warehouse.dispatch_wave(wave.id, picker_id="PICKER-1")
        for task in result.tasks[wave.id]:
            task.complete()
        if not engine.warehouse.is_wave_complete(wave.id):
            raise RuntimeError(f"Wave {wave.id} is still open")

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
