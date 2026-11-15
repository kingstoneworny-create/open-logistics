"""Register a SKU, consume stock, and record a replenishment order."""

import json

from open_logistics import LogisticsEngine, StockItem


def main() -> None:
    engine = LogisticsEngine()
    engine.subscribe(
        "inventory.replenishment_triggered",
        lambda event: print(f"Reorder event for {event.payload['sku']}"),
    )
    stock = StockItem(
        sku="FILTER-01",
        on_hand=40,
        unit_cost=12.0,
        lead_time_days=3,
        daily_demand_history=[8, 10, 9, 11, 10],
    )
    status, = engine.evaluate_inventory([stock])
    print(f"Starting stock {stock.on_hand}, reorder point {status.reorder_point:.1f}")

    purchase_order = engine.consume_stock(stock.sku, 15)
    if purchase_order is None:
        raise RuntimeError("The sample stock consumption should trigger replenishment")
    print(json.dumps(purchase_order.to_dict(), indent=2))

    # Record inbound units once the supplier system accepts the purchase order.
    stock.on_order += purchase_order.quantity
    print(f"On hand {stock.on_hand}, on order {stock.on_order}")


if __name__ == "__main__":
    main()
