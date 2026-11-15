"""Plan routes for a third-party logistics provider serving two customers."""

import json

from open_logistics import Depot, LogisticsEngine, Order, Vehicle


def main() -> None:
    orders = [
        Order(
            "CLIENT-A-EXPRESS", (41.885, -87.625), weight=30, volume=0.2, priority=2,
            time_window=(8.5, 10.0), metadata={"client": "A"},
        ),
        Order(
            "CLIENT-B-STD", (41.891, -87.617), weight=90, volume=0.7,
            time_window=(9.0, 11.0), metadata={"client": "B"},
        ),
        Order(
            "CLIENT-B-OVERSIZED", (41.893, -87.615), weight=170, volume=1.5,
            time_window=(9.0, 11.0), metadata={"client": "B"},
        ),
    ]
    fleet = [
        Vehicle("FAST-VAN", capacity_weight=60, capacity_volume=0.4, max_stops=2, speed_kmh=45,
                available_time_window=(8.0, 12.0)),
        Vehicle("BOX-TRUCK", capacity_weight=150, capacity_volume=1.2, max_stops=3, speed_kmh=35,
                available_time_window=(8.0, 12.0)),
    ]
    result = LogisticsEngine().dispatch_orders(
        orders, fleet, Depot("CHICAGO-DC", location=(41.878, -87.629))
    )
    # A dispatcher can reschedule or subcontract the unassigned orders.
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
