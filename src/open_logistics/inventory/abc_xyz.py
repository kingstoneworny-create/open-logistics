"""ABC-XYZ inventory categorization, segmentation matrix, and portfolio analytics.

Pure Python standard library implementation of:
- Pareto ABC classification based on annual dollar usage value (80/15/5 rule)
- XYZ volatility classification based on daily demand coefficient of variation (CV)
- 9-box ABC-XYZ segmentation matrix
- Inventory turnover ratio and Days Sales of Inventory (DSI)
- Dead stock detection and stockout risk alerts
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from open_logistics.core.exceptions import ValidationError
from open_logistics.core.models import StockItem


def classify_abc(
    items: Sequence[StockItem],
    days_per_year: int = 365,
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
) -> Dict[str, str]:
    """Pareto ABC classification partitioning items by cumulative annual consumption value.

    Standard rule:
    - Class A: Top ~80% of annual value (high monetary importance)
    - Class B: Next ~15% of annual value (moderate monetary importance)
    - Class C: Remaining ~5% of annual value (low monetary importance)

    Args:
        items: Collection of StockItem instances to classify.
        days_per_year: Number of operational days used to project annual demand.
        a_threshold: Upper cumulative fraction for Class A (default 0.80).
        b_threshold: Upper cumulative fraction for Class B (default 0.95).

    Returns:
        Dictionary mapping SKU string to category ('A', 'B', or 'C').
    """
    if not items:
        return {}

    if a_threshold <= 0 or b_threshold <= a_threshold or b_threshold > 1.0:
        raise ValidationError(
            f"Invalid Pareto thresholds: a_threshold={a_threshold}, b_threshold={b_threshold}. "
            "Must satisfy 0 < a_threshold < b_threshold <= 1.0"
        )

    # Compute annual consumption value per item: D * C
    item_values: List[Tuple[StockItem, float]] = []
    total_value = 0.0

    for item in items:
        annual_demand = item.annual_demand(days_per_year=days_per_year)
        val = annual_demand * item.unit_cost
        item_values.append((item, val))
        total_value += val

    # Boundary case: single item portfolio represents 100% of value
    if len(items) == 1:
        return {items[0].sku: "A"}

    # Boundary case: zero total portfolio value
    if total_value <= 0.0:
        return {item.sku: "C" for item in items}

    # Sort descending by annual consumption value
    item_values.sort(key=lambda x: x[1], reverse=True)

    result: Dict[str, str] = {}
    cum_val = 0.0

    for item, val in item_values:
        prev_cum_fraction = cum_val / total_value
        if prev_cum_fraction < a_threshold:
            category = "A"
        elif prev_cum_fraction < b_threshold:
            category = "B"
        else:
            category = "C"

        result[item.sku] = category
        cum_val += val

    return result


def abc_classification(
    items: Sequence[StockItem],
    days_per_year: int = 365,
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
) -> Dict[str, str]:
    """Alias for classify_abc."""
    return classify_abc(
        items=items,
        days_per_year=days_per_year,
        a_threshold=a_threshold,
        b_threshold=b_threshold,
    )


def classify_xyz(
    items: Sequence[StockItem],
    x_threshold: float = 0.15,
    y_threshold: float = 0.40,
) -> Dict[str, str]:
    """XYZ volatility classification based on Coefficient of Variation (CV = sigma / mu).

    Segmentation criteria:
    - Class X: Stable, predictable demand (CV <= 0.15)
    - Class Y: Moderate demand fluctuations (0.15 < CV <= 0.40)
    - Class Z: High irregularity, sporadic, or zero demand (CV > 0.40)

    Args:
        items: Collection of StockItem instances to analyze.
        x_threshold: Upper CV threshold for Class X (default 0.15).
        y_threshold: Upper CV threshold for Class Y (default 0.40).

    Returns:
        Dictionary mapping SKU string to volatility class ('X', 'Y', or 'Z').
    """
    if not items:
        return {}

    result: Dict[str, str] = {}

    for item in items:
        history = item.daily_demand_history
        if not history or len(history) == 0:
            result[item.sku] = "Z"
            continue

        mean = sum(history) / len(history)
        if mean <= 0.0:
            result[item.sku] = "Z"
            continue

        if len(history) == 1:
            result[item.sku] = "X"
            continue

        # Sample standard deviation: s = sqrt(sum((x - mean)^2) / (N - 1))
        var = sum((x - mean) ** 2 for x in history) / (len(history) - 1)
        std_dev = math.sqrt(var)
        cv = std_dev / mean

        if cv <= x_threshold:
            result[item.sku] = "X"
        elif cv <= y_threshold:
            result[item.sku] = "Y"
        else:
            result[item.sku] = "Z"

    return result


def xyz_classification(
    items: Sequence[StockItem],
    x_threshold: float = 0.15,
    y_threshold: float = 0.40,
) -> Dict[str, str]:
    """Alias for classify_xyz."""
    return classify_xyz(
        items=items,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
    )


def generate_abc_xyz_matrix(
    items: Sequence[StockItem],
    days_per_year: int = 365,
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
    x_threshold: float = 0.15,
    y_threshold: float = 0.40,
) -> Dict[str, List[str]]:
    """Generate 9-box ABC-XYZ segmentation matrix.

    Maps items across monetary value (A, B, C) and volatility (X, Y, Z):
    - AX, AY, AZ
    - BX, BY, BZ
    - CX, CY, CZ

    Args:
        items: Collection of StockItem instances to classify.
        days_per_year: Number of operational days per year.
        a_threshold: ABC threshold for Class A.
        b_threshold: ABC threshold for Class B.
        x_threshold: XYZ threshold for Class X.
        y_threshold: XYZ threshold for Class Y.

    Returns:
        Dictionary mapping all 9 box codes to lists of SKU strings.
    """
    matrix: Dict[str, List[str]] = {
        "AX": [],
        "AY": [],
        "AZ": [],
        "BX": [],
        "BY": [],
        "BZ": [],
        "CX": [],
        "CY": [],
        "CZ": [],
    }

    if not items:
        return matrix

    abc = classify_abc(items, days_per_year=days_per_year, a_threshold=a_threshold, b_threshold=b_threshold)
    xyz = classify_xyz(items, x_threshold=x_threshold, y_threshold=y_threshold)

    for item in items:
        cat_abc = abc.get(item.sku, "C")
        cat_xyz = xyz.get(item.sku, "Z")
        cell = f"{cat_abc}{cat_xyz}"
        if cell in matrix:
            matrix[cell].append(item.sku)
        else:
            matrix[cell] = [item.sku]

    return matrix


def abc_xyz_matrix(
    items: Sequence[StockItem],
    days_per_year: int = 365,
) -> Dict[str, List[str]]:
    """Alias for generate_abc_xyz_matrix."""
    return generate_abc_xyz_matrix(items=items, days_per_year=days_per_year)


def calculate_inventory_turnover(
    cogs: float = 0.0,
    average_inventory: Optional[float] = None,
    average_inventory_value: Optional[float] = None,
    avg_inventory: Optional[float] = None,
    cost_of_goods_sold: Optional[float] = None,
) -> float:
    """Calculate Inventory Turnover Ratio.

    Inventory Turnover = Cost of Goods Sold (COGS) / Average Inventory Value

    Args:
        cogs: Annual Cost of Goods Sold.
        average_inventory: Average value of inventory held.
        average_inventory_value: Alternative alias for average_inventory.
        avg_inventory: Alternative alias for average_inventory.
        cost_of_goods_sold: Alternative alias for cogs.

    Returns:
        Turnover ratio (number of times inventory is sold and replaced per year).
    """
    actual_cogs = cost_of_goods_sold if cost_of_goods_sold is not None else cogs
    avg_inv = (
        average_inventory
        if average_inventory is not None
        else (
            average_inventory_value
            if average_inventory_value is not None
            else (avg_inventory if avg_inventory is not None else 0.0)
        )
    )

    if avg_inv <= 0.0 or actual_cogs <= 0.0:
        return 0.0

    return float(actual_cogs / avg_inv)


def inventory_turnover(
    cost_of_goods_sold: float,
    average_inventory_value: float,
) -> float:
    """Alias for calculate_inventory_turnover."""
    return calculate_inventory_turnover(
        cost_of_goods_sold=cost_of_goods_sold,
        average_inventory_value=average_inventory_value,
    )


def calculate_days_sales_of_inventory(
    turnover: Optional[float] = None,
    turnover_ratio: Optional[float] = None,
    days: int = 365,
    days_per_year: int = 365,
) -> float:
    """Calculate Days Sales of Inventory (DSI).

    DSI = Days in Period / Inventory Turnover Ratio

    Args:
        turnover: Annual inventory turnover ratio.
        turnover_ratio: Alternative alias for turnover.
        days: Days in evaluation period (default 365).
        days_per_year: Alternative alias for days.

    Returns:
        Average number of days required to turn inventory into sales.
    """
    t = turnover if turnover is not None else turnover_ratio
    if t is None or t <= 0.0:
        return 0.0

    period_days = days if days != 365 else days_per_year
    return float(period_days / t)


def days_sales_of_inventory(
    turnover_ratio: float,
    days: int = 365,
) -> float:
    """Alias for calculate_days_sales_of_inventory."""
    return calculate_days_sales_of_inventory(
        turnover_ratio=turnover_ratio,
        days=days,
    )


def detect_dead_stock(
    items: Sequence[StockItem],
    threshold_days_no_demand: int = 90,
) -> List[str]:
    """Detect dormant SKUs with zero consumption over an observation window.

    An item is flagged as dead stock if:
    - on_hand > 0 units are tied up in storage, AND
    - average daily demand is 0.0, or all daily demand observations are 0.0.

    Args:
        items: Collection of StockItem instances to inspect.
        threshold_days_no_demand: Inactivity window threshold in days.

    Returns:
        List of dormant SKU identifiers.
    """
    dead_skus: List[str] = []

    for item in items:
        if item.on_hand <= 0:
            continue

        history = item.daily_demand_history
        if not history:
            # No demand recorded at all
            dead_skus.append(item.sku)
            continue

        if item.average_daily_demand == 0.0:
            dead_skus.append(item.sku)
            continue

        # Check trailing observations up to threshold_days_no_demand
        trailing_window = history[-threshold_days_no_demand:]
        if all(d == 0.0 for d in trailing_window):
            dead_skus.append(item.sku)

    return dead_skus


def detect_stockout_risks(
    items: Sequence[StockItem],
    days_horizon: float = 7.0,
) -> List[Dict[str, Any]]:
    """Detect items at imminent risk of stockout within the planning horizon.

    Args:
        items: Collection of StockItem instances to evaluate.
        days_horizon: Planning window in days (default 7.0 days).

    Returns:
        List of diagnostic dictionaries for high-risk items.
    """
    alerts: List[Dict[str, Any]] = []

    for item in items:
        avg_d = item.average_daily_demand
        projected_demand = avg_d * float(days_horizon)
        available = item.available
        net_inv = item.net_inventory

        # Risk triggered if available stock is depleted or net inventory cannot cover horizon
        if available <= 0 or (avg_d > 0 and net_inv < projected_demand):
            days_of_supply = (available / avg_d) if avg_d > 0 else 0.0
            alerts.append(
                {
                    "sku": item.sku,
                    "on_hand": item.on_hand,
                    "available": available,
                    "net_inventory": net_inv,
                    "projected_demand": projected_demand,
                    "days_of_supply": round(days_of_supply, 2),
                    "risk_level": "CRITICAL" if available <= 0 else "HIGH",
                }
            )

    return alerts
