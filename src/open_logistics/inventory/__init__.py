"""Scientific inventory optimization and automated replenishment package.

Provides pure Python standard library implementations of:
- Classical Wilson EOQ, EPQ, Quantity Discount optimization, Planned Backorders
- Peter John Acklam rational Chebyshev probit approximation for exact arbitrary Z-scores
- Standard normal PDF, CDF, normal loss function G(z), and Item Fill Rate beta
- Dynamic safety stock with joint stochastic demand and lead-time variance
- Deterministic and dynamic Reorder Points (ROP)
- Pareto ABC analysis, XYZ volatility classification, 9-box matrix
- Inventory turnover ratio and Days Sales of Inventory (DSI)
- Dead stock and stockout risk detection
- Reactive ReplenishmentEventEngine with EventBus pub/sub integration
"""

from __future__ import annotations

from open_logistics.inventory.abc_xyz import (
    abc_classification,
    abc_xyz_matrix,
    calculate_days_sales_of_inventory,
    calculate_inventory_turnover,
    classify_abc,
    classify_xyz,
    days_sales_of_inventory,
    detect_dead_stock,
    detect_stockout_risks,
    generate_abc_xyz_matrix,
    inventory_turnover,
    xyz_classification,
)
from open_logistics.inventory.formulas import (
    BackorderResult,
    QuantityDiscountResult,
    acklam_norm_inv_cdf,
    calculate_backorder_eoq,
    calculate_eoq,
    calculate_epq,
    calculate_fill_rate,
    calculate_quantity_discount_eoq,
    calculate_reorder_point,
    calculate_safety_stock,
    eoq_with_backorders,
    epq,
    expected_cycle_shortage,
    item_fill_rate,
    norm_cdf,
    norm_inv_cdf,
    norm_pdf,
    normal_loss_function,
    quantity_discount_eoq,
    reorder_point,
    safety_stock_joint,
    total_cost_eoq,
    wilson_eoq,
)
from open_logistics.inventory.replenishment_engine import (
    ReplenishmentEventEngine,
    ReplenishmentStatus,
)

__all__ = [
    # Formulas
    "BackorderResult",
    "QuantityDiscountResult",
    "acklam_norm_inv_cdf",
    "calculate_backorder_eoq",
    "calculate_eoq",
    "calculate_epq",
    "calculate_fill_rate",
    "calculate_quantity_discount_eoq",
    "calculate_reorder_point",
    "calculate_safety_stock",
    "eoq_with_backorders",
    "epq",
    "expected_cycle_shortage",
    "item_fill_rate",
    "norm_cdf",
    "norm_inv_cdf",
    "norm_pdf",
    "normal_loss_function",
    "quantity_discount_eoq",
    "reorder_point",
    "safety_stock_joint",
    "total_cost_eoq",
    "wilson_eoq",
    # ABC-XYZ & Metrics
    "abc_classification",
    "abc_xyz_matrix",
    "calculate_days_sales_of_inventory",
    "calculate_inventory_turnover",
    "classify_abc",
    "classify_xyz",
    "days_sales_of_inventory",
    "detect_dead_stock",
    "detect_stockout_risks",
    "generate_abc_xyz_matrix",
    "inventory_turnover",
    "xyz_classification",
    # Replenishment Engine
    "ReplenishmentEventEngine",
    "ReplenishmentStatus",
]
