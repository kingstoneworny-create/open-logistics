# E2E Test Infra: open-logistics

## Test Philosophy
- Opaque-box, requirement-driven testing based on `ORIGINAL_REQUEST.md` and public APIs in `PROJECT.md § Interface Contracts`.
- No dependency on private or unexported implementation details.
- Comprehensive 4-tier methodology: Category-Partition (Tier 1), Boundary Value Analysis (Tier 2), Pairwise Combinatorial Testing (Tier 3), Real-World Workload Testing (Tier 4).
- Standard runner: `pytest` with zero external binary dependencies.

## Feature Inventory & Test Matrix
| # | Feature | Requirement Source | Tier 1 (Coverage ≥5) | Tier 2 (Boundaries ≥5) | Tier 3 (Cross-Feature) | Tier 4 (Workloads) |
|---|---------|-------------------|:--------------------:|:----------------------:|:----------------------:|:------------------:|
| F1 | In-Memory Pub/Sub EventBus | R3 / PROJECT #1 | 5 | 5 | ✓ | ✓ |
| F2 | Core Domain Models & Schemas | R1 / PROJECT #2 | 5 | 5 | ✓ | ✓ |
| F3 | Domain Exceptions & Error Handling | R1 / PROJECT #4 | 5 | 5 | ✓ | ✓ |
| F4 | Distance Metrics (Euclidean, Manhattan, Haversine) | R2 / PROJECT #5 | 5 | 5 | ✓ | ✓ |
| F5 | Clarke-Wright CVRP Routing Heuristic | R2 / PROJECT #6,7 | 5 | 5 | ✓ | ✓ |
| F6 | TSP Optimizers (Nearest Neighbor & 2-Opt) | R2 / PROJECT #8,9 | 5 | 5 | ✓ | ✓ |
| F7 | Time Windows & Schedule Propagation | R2 / PROJECT #10,11 | 5 | 5 | ✓ | ✓ |
| F8 | Heterogeneous Fleet & Multi-Dim Capacity Dispatch | R2 / PROJECT #12,13,14 | 5 | 5 | ✓ | ✓ |
| F9 | AutoDispatchEngine Automated Workflow | R3 / PROJECT #15 | 5 | 5 | ✓ | ✓ |
| F10 | Scientific Lot Sizing (EOQ, Discount, EPQ, Backorders) | R2 / PROJECT #16-19 | 5 | 5 | ✓ | ✓ |
| F11 | Safety Stock & Stochastic Fill Rate Models | R2 / PROJECT #20-23 | 5 | 5 | ✓ | ✓ |
| F12 | ABC-XYZ Categorization & Stock Analytics | R2 / PROJECT #24-27 | 5 | 5 | ✓ | ✓ |
| F13 | ReplenishmentEventEngine & Auto-PO Workflow | R3 / PROJECT #28 | 5 | 5 | ✓ | ✓ |
| F14 | 3D Spatial Warehouse Grid & Slotting Heuristic | R2 / PROJECT #29,30 | 5 | 5 | ✓ | ✓ |
| F15 | Picker Routing Paths (Serpentine, Return, Mid-point) | R2 / PROJECT #31,32 | 5 | 5 | ✓ | ✓ |
| F16 | Pick-Pack-Ship 7-Stage FSM with Rollbacks | R2, R3 / PROJECT #33,34 | 5 | 5 | ✓ | ✓ |
| F17 | Carrier Cutoff & Capacity Wave Batching | R2 / PROJECT #35,36 | 5 | 5 | ✓ | ✓ |
| F18 | WarehouseTaskDispatcher Automated Workflow | R3 / PROJECT #37 | 5 | 5 | ✓ | ✓ |
| F19 | Unified LogisticsEngine Facade | R1, R2 / PROJECT #38 | 5 | 5 | ✓ | ✓ |
| F20 | Outbound Webhook Dispatcher (HMAC Signed) | R3 / PROJECT #40 | 5 | 5 | ✓ | ✓ |
| F21 | Pure-Stdlib Embedded REST Server | R1, R2 / PROJECT #41 | 5 | 5 | ✓ | ✓ |
| F22 | UNIX-Pipeable CLI | R1, R2 / PROJECT #39 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Location**: `C:\Users\Kiroi\teamwork_projects\open_logistics\tests\`
- **Runner Invocation**: `pytest -v tests/` or `python -m pytest tests/`
- **Pass/Fail Semantics**: 100% exit code 0, 0 failures, 0 errors, 0 unhandled warnings.
- **Directory Layout**:
  ```
  tests/
  ├── conftest.py                   # Shared test fixtures, mock data generators
  ├── test_tier1_core.py            # F1-F4: Models, Events, Exceptions, Distances
  ├── test_tier1_routing.py         # F5-F9: CVRP, TSP, Time Windows, Fleet Dispatch
  ├── test_tier1_inventory.py       # F10-F13: EOQ, Safety Stock, ABC-XYZ, Auto-PO
  ├── test_tier1_warehouse.py       # F14-F18: 3D Grid, Slotting, Picker Routing, FSM, Waves
  ├── test_tier1_api_cli.py         # F19-F22: Facade, Webhooks, REST Server, CLI
  ├── test_tier2_boundaries.py      # Tier 2: Boundary Value Analysis & Corner Cases
  ├── test_tier3_cross_feature.py   # Tier 3: Pairwise & Cross-Feature Integration Workflows
  └── test_tier4_scenarios.py       # Tier 4: Real-World Enterprise Workload Scenarios
  ```

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity | Description |
|---|----------|--------------------|------------|-------------|
| 1 | E-Commerce Same-Day Fulfillment | F1, F2, F9, F13, F16, F17, F18, F19, F20 | High | Customer orders arrive via Webhook/API -> stock allocated -> ROP evaluated & PO triggered -> orders batched into pick wave -> Serpentine picking manifest -> FSM transitions (Allocated -> Picking -> Picked -> Packed -> Shipped) -> Webhook dispatched. |
| 2 | 3PL Multi-Tenant Fleet Dispatch | F4, F5, F6, F7, F8, F9, F19 | High | Heterogeneous fleet of 4 vehicles serving 20 customer stops with tight time windows, weight & volume capacity constraints -> Clarke-Wright savings + 2-opt tour improvement -> multi-route manifest schedule generation. |
| 3 | Automated DC Replenishment & 3D Slotting | F10, F11, F12, F13, F14, F15, F19 | High | 50 SKUs evaluated for Pareto ABC-XYZ -> stochastic lead time/demand safety stock calculated -> high-velocity A SKUs slotted nearest to dock in 3D grid -> stock consumption simulation triggering automated PO generation. |
| 4 | Resilient Order Lifecycle & Rollback Recovery | F1, F3, F16, F18, F19 | Medium | High-concurrency cancellation during wave picking triggering atomic FSM rollback, restocking allocation, and exception handling without system corruption. |
| 5 | End-to-End Headless CLI & Microservice Integration | F19, F21, F22 | Medium | Start embedded REST microservice -> submit dispatch and inventory payload via HTTP API -> invoke CLI pipe commands to generate pick manifests and inspect system status. |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: ≥5 tests per feature × 22 features = **≥110 tests**.
- **Tier 2 (Boundary & Corner Cases)**: ≥5 tests per feature domain = **≥50 boundary tests**.
- **Tier 3 (Cross-Feature Combinations)**: Major pairwise integrations = **≥15 cross-feature tests**.
- **Tier 4 (Real-World Application Scenarios)**: Realistic end-to-end multi-feature workflows = **≥5 comprehensive application tests**.
- **Total Suite Minimum**: **≥180 tests** executing with 100% pass rate.
