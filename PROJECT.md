# Project: Open Logistics (open-logistics)

## Architecture
An advanced, lightweight, zero-heavy-dependency Python open-source logistics framework and SDK designed for modular enterprise integration into existing operations (e-commerce, 3PLs, distribution centers).

```
open_logistics/
├── pyproject.toml              # Modern PEP 517/621 build configuration
├── README.md                   # Comprehensive documentation, quickstart, integration guides
├── LICENSE                     # MIT License
├── TEST_INFRA.md               # E2E Test Suite Specification & Architecture
├── TEST_READY.md               # E2E Test Suite Readiness Signal & Coverage Summary
├── src/
│   └── open_logistics/
│       ├── __init__.py         # Package entrypoint, version, public exports
│       ├── py.typed            # PEP 561 typing marker
│       ├── core/               # Shared domain models & event bus
│       │   ├── __init__.py
│       │   ├── models.py       # Order, Item, Vehicle, Depot, Route, StockItem, PurchaseOrder
│       │   ├── events.py       # In-memory pub/sub EventBus, Event, Subscriber
│       │   ├── exceptions.py   # Domain exceptions hierarchy
│       │   └── config.py       # LogisticsConfig dataclass
│       ├── routing/            # Routing & fleet dispatch algorithms
│       │   ├── __init__.py
│       │   ├── distances.py    # Euclidean, Manhattan, Haversine calculators
│       │   ├── clarke_wright.py# Parallel & sequential CVRP savings heuristic
│       │   ├── tsp.py          # Nearest neighbor & 2-opt local search optimizer
│       │   ├── time_windows.py # VRPTW schedule propagation & penalty models
│       │   └── dispatch_engine.py # AutoDispatchEngine & priority allocation
│       ├── inventory/          # Scientific inventory optimization & auto-replenishment
│       │   ├── __init__.py
│       │   ├── formulas.py     # EOQ, EPQ, Quantity discounts, Backorders, Acklam norm_inv_cdf, Safety Stock, Normal loss
│       │   ├── abc_xyz.py      # Pareto ABC, XYZ volatility matrix, turnover metrics
│       │   └── replenishment_engine.py # ReplenishmentEventEngine & auto-PO triggers
│       ├── warehouse/          # Warehouse automation & 3D slotting
│       │   ├── __init__.py
│       │   ├── layout.py       # 3D spatial bins, zones, aisles, coordinate bounding boxes
│       │   ├── slotting.py     # 3D cell placement heuristic (volume-slack + dock-distance)
│       │   ├── picking_path.py # Serpentine S-shape, return, mid-point picker routing
│       │   ├── fsm.py          # 7-stage order lifecycle state machine with rollback guards
│       │   ├── wave_batcher.py # Carrier cutoff & capacity-constrained pick wave batching
│       │   └── task_dispatcher.py # WarehouseTaskDispatcher
│       ├── api/                # Integration layer
│       │   ├── __init__.py
│       │   ├── engine.py       # Unified LogisticsEngine facade
│       │   ├── webhooks.py     # Outbound webhook dispatcher with HMAC signature
│       │   └── server.py       # Embedded pure-stdlib HTTP microservice server
│       └── cli/                # Command-line interface
│           ├── __init__.py
│           └── main.py         # UNIX pipeable open-logistics CLI
├── tests/                      # Automated test suite (Tiers 1-4)
│   ├── conftest.py             # Fixtures and deterministic test dataset
│   ├── test_tier1_core.py      # F1-F4 tests
│   ├── test_tier1_routing.py   # F5-F9 tests
│   ├── test_tier1_inventory.py # F10-F13 tests
│   ├── test_tier1_warehouse.py # F14-F18 tests
│   ├── test_tier1_api_cli.py   # F19-F22 tests
│   ├── test_tier2_boundaries.py# Tier 2 Boundary Value Analysis
│   ├── test_tier3_cross_feature.py # Tier 3 Pairwise & Cross-Feature Integration
│   └── test_tier4_scenarios.py # Tier 4 Real-World Application Workloads
└── examples/                   # Executable company integration examples
    ├── ecommerce_fulfillment.py# Webhook -> Pick -> Pack -> Dispatch
    ├── three_pl_dispatch.py    # Multi-client fleet routing with SLAs
    ├── auto_replenishment.py   # Stock consumption -> ROP -> Auto-PO
    └── warehouse_wave_pick.py  # Serpentine routing & state machine
```
