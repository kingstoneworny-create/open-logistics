# Open Logistics

Open Logistics is a Python SDK for fleet dispatch, inventory replenishment, and warehouse picking. Embed it in a store, ERP, or warehouse management system, or run the CLI and HTTP API against JSON payloads. It supports Python 3.10 and later and has zero runtime dependencies.

## Install from source

Run these commands from the repository root in a Python virtual environment.

```sh
python -m pip install .
open-logistics --version
```

You can also invoke the CLI with `python -m open_logistics`.

## Python quickstart

```python
from open_logistics import Depot, LogisticsEngine, Order, Vehicle

engine = LogisticsEngine()
engine.subscribe("dispatch.completed", lambda event: print(event.payload))

result = engine.dispatch_orders(
    orders=[Order("ORDER-1", destination=(41.885, -87.625), weight=5.0)],
    fleet=[Vehicle("VAN-1", capacity_weight=100.0)],
    depot=Depot("DEPOT-1", location=(41.878, -87.629)),
)

print(result.to_dict())
```

- `dispatch_orders` returns routes, stop schedules, and unassigned orders. Check the unassigned list and inspect schedule feasibility before releasing routes. By default, coordinates are latitude and longitude in degrees, distances are in kilometers, and delivery windows use decimal hours such as `[8.5, 10.0]` for 08:30–10:00.
- `evaluate_inventory` registers stock and calculates reorder points, safety stock, and economic order quantities. Call `consume_stock` to record a stock decrement and receive a purchase order when replenishment is triggered. Record accepted supplier orders in `on_order` so later evaluations include inbound units.
- `allocate_and_pick` creates waves, prepares pick tasks, and brings orders into the warehouse workflow. Pass `Order` objects with item lines and a `WarehouseLayout` with SKU bins for operational manifests. Order IDs alone produce placeholder tasks for quick trials. Use `engine.warehouse` to dispatch waves and confirm completion, and use your inventory system to reserve stock.

Each engine keeps stock registrations, waves, and task state in memory for the life of the process. Keep your source of truth in your existing systems and use a separate engine for each inventory or warehouse that needs isolated state.

## CLI and HTTP API

Use `dispatch`, `inventory`, or `wave` with `--input FILE` or `--stdin`. Successful commands write JSON to stdout. Invalid input writes a diagnostic to stderr and exits with code 2.

Save this payload as `dispatch.json`.

```json
{
  "orders": [{"id": "ORDER-1", "destination": [41.885, -87.625], "weight": 5}],
  "fleet": [{"id": "VAN-1", "capacity_weight": 100}],
  "depot": {"id": "DEPOT-1", "location": [41.878, -87.629]}
}
```

```sh
open-logistics dispatch --input dispatch.json
open-logistics dispatch --stdin < dispatch.json
```

In PowerShell, use `Get-Content -Raw dispatch.json | open-logistics dispatch --stdin` for a pipe.

The inventory command accepts `{"items":[{"sku":"FILTER-01","on_hand":28,"unit_cost":12,"lead_time_days":3,"daily_demand_history":[8,10,9]}],"service_level":0.95}`. The wave command accepts `{"order_ids":["ORDER-1","ORDER-2"],"max_orders":1,"strategy":"serpentine"}` or a full `orders` array using the dispatch order shape. The supported picking strategies are `serpentine`, `return`, and `midpoint`.

Run `open-logistics serve` to start the embedded HTTP server at `127.0.0.1:8080`. Use `--host` and `--port` to configure the address.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health |
| GET | `/api/v1/info` | Version, uptime, and active configuration |
| POST | `/api/v1/dispatch` | Dispatch orders |
| POST | `/api/v1/inventory` | Evaluate replenishment |
| POST | `/api/v1/wave-pick` | Generate pick waves and tasks |

The POST endpoints take the same JSON as their CLI counterparts. The embedded server uses plain HTTP and has no built-in authentication. Use it on loopback or behind a proxy that provides TLS and authentication.

## Events and company integrations

Subscribe with `engine.subscribe` or pass your own `EventBus` to the constructor. Topics include `dispatch.completed`, `inventory.replenishment_triggered`, and `warehouse.wave_created`. Use `WebhookDispatcher.register` and `attach` to forward selected events. The receiver verifies the `X-Signature-SHA256` header against the raw request body and shared secret.

Run these scripts from the repository root after installing the package.

| Example | Command |
| --- | --- |
| E-commerce order through picking, packing, routing, and an offline signed shipment notification | `python examples/ecommerce_fulfillment.py` |
| Third-party dispatch with delivery windows, mixed fleet capacity, and an unassigned order | `python examples/three_pl_dispatch.py` |
| Stock consumption and an automatic purchase order | `python examples/auto_replenishment.py` |
| Warehouse waves from slot lookup through task completion | `python examples/warehouse_wave_pick.py` |

## Test and build

```sh
python -m pip install -e ".[test]" build
python -m pytest
python -m build
```

The HTTP integration tests bind to local ports and need loopback access. The build writes a wheel and source archive to `dist/`.

## First GitHub push

Create an empty GitHub repository named `open-logistics`. Run the commands below from this folder and replace `YOUR-ACCOUNT` with your account or organization.

```sh
git init -b main
git add .
git status --short
git commit -m "Release Open Logistics 0.1.0"
git remote add origin https://github.com/YOUR-ACCOUNT/open-logistics.git
git push -u origin main
```

Open Logistics is available under the [MIT License](LICENSE).
