"""Process a sample store order and deliver a signed shipment event offline."""

import json
from secrets import token_hex

from open_logistics import Depot, Event, LogisticsEngine, Order, Vehicle
from open_logistics.api.webhooks import SIGNATURE_HEADER, WebhookDispatcher
from open_logistics.warehouse.fsm import OrderStatus
from open_logistics.warehouse.layout import Bin, WarehouseLayout


class DemoShipmentReceiver(WebhookDispatcher):
    """Simulate a partner receiving the exact HTTP body that was signed."""

    def _http_post(self, url: str, payload_bytes: bytes, headers: dict[str, str]) -> int:
        if not self.verify_signature(payload_bytes, headers[SIGNATURE_HEADER]):
            return 401
        print("Partner accepted:", payload_bytes.decode("utf-8"))
        return 204


def main() -> None:
    # The store adapter has already authenticated the incoming request.
    order = Order.from_dict({
        "id": "SHOP-1042",
        "destination": [41.885, -87.625],
        "weight": 0.8,
        "volume": 0.004,
        "items": [{"sku": "MUG", "quantity": 2, "unit_weight": 0.4, "unit_volume": 0.002}],
    })
    layout = WarehouseLayout(bins=[Bin("A1-B1", x=1, y=3, sku="MUG")])
    engine = LogisticsEngine(layout=layout)
    webhooks = DemoShipmentReceiver(secret=token_hex(32), max_retries=1)
    webhooks.register("https://partner.example.invalid/shipments", events=["order.shipped"])
    webhooks.attach(engine.event_bus, "order.shipped")

    picks = engine.allocate_and_pick([order])
    for wave in picks.waves:
        engine.warehouse.dispatch_wave(wave.id, picker_id="PICKER-1")
        for task in picks.tasks[wave.id]:
            task.complete()
        if not engine.warehouse.is_wave_complete(wave.id):
            raise RuntimeError(f"Wave {wave.id} is still open")

    order_state = engine.warehouse.order_fsms[order.id]
    order_state.transition(OrderStatus.PACKED)
    dispatch = engine.dispatch_orders(
        [order],
        [Vehicle("VAN-1", capacity_weight=50)],
        Depot("SHOP-DEPOT", location=(41.878, -87.629)),
    )
    if dispatch.unassigned_orders:
        raise RuntimeError(f"No vehicle assigned to {order.id}")

    order_state.transition(OrderStatus.SHIPPED)
    engine.event_bus.publish(Event("order.shipped", {
        "order_id": order.id,
        "status": order_state.current_state.value,
        "vehicle_id": dispatch.routes[0].vehicle_id,
    }))
    if not webhooks.history or not webhooks.history[-1].success:
        raise RuntimeError("The shipment notification was not accepted")
    print(json.dumps(dispatch.to_dict(), indent=2))


if __name__ == "__main__":
    main()
