"""Scientific inventory optimization formulas and statistical models.

Pure Python standard library implementation of lot sizing algorithms,
stochastic safety stock models, Peter John Acklam's inverse normal CDF probit
approximation, standard normal loss functions, and fill rates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from open_logistics.core.exceptions import InventoryError, ValidationError


# ============================================================================
# Result Tuples with Attribute & Index Access
# ============================================================================


class QuantityDiscountResult(NamedTuple):
    """Result of all-units quantity discount optimization.

    Attributes:
        optimal_quantity: Optimal order batch size minimizing total annual cost.
        min_total_cost: Minimum total annual cost achievable across all tiers.
    """

    optimal_quantity: float
    min_total_cost: float

    @property
    def eoq(self) -> float:
        """Alias for optimal_quantity."""
        return self.optimal_quantity


class BackorderResult(NamedTuple):
    """Result of EOQ optimization with planned backorders.

    Attributes:
        eoq: Optimal order batch size Q*.
        max_backorder: Maximum backorder / shortage level b*.
    """

    eoq: float
    max_backorder: float

    @property
    def optimal_order_quantity(self) -> float:
        """Alias for eoq."""
        return self.eoq

    @property
    def max_backorder_quantity(self) -> float:
        """Alias for max_backorder."""
        return self.max_backorder


# ============================================================================
# Statistical Distribution & Probit Functions (Acklam Rational Approximation)
# ============================================================================

# Acklam coefficients for rational Chebyshev approximation of the normal inverse CDF
_A_COEFFS: Tuple[float, ...] = (
    -3.969683011353505e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)

_B_COEFFS: Tuple[float, ...] = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)

_C_COEFFS: Tuple[float, ...] = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)

_D_COEFFS: Tuple[float, ...] = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)

_P_LOW: float = 0.02425
_P_HIGH: float = 1.0 - _P_LOW


def norm_inv_cdf(p: float) -> float:
    """Peter John Acklam's rational Chebyshev approximation of the standard normal inverse CDF.

    Computes the probit function z = Phi^-1(p) for standard normal distribution N(0, 1)
    with relative error < 1.15e-9 across the entire domain p in (0, 1).

    Args:
        p: Cumulative probability value strictly between 0.0 and 1.0.

    Returns:
        Standard normal quantile (Z-score).

    Raises:
        ValidationError: If p is not in the open interval (0, 1).
    """
    if not isinstance(p, (int, float)):
        raise ValidationError(f"Probability p must be a float, got {type(p).__name__}")
    if p <= 0.0 or p >= 1.0:
        raise ValidationError(f"Probability p must be strictly in range (0, 1), got {p}")

    p_val = float(p)

    # Lower tail: p < 0.02425
    if p_val < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p_val))
        num = (
            ((((_C_COEFFS[0] * q + _C_COEFFS[1]) * q + _C_COEFFS[2]) * q + _C_COEFFS[3]) * q + _C_COEFFS[4]) * q
            + _C_COEFFS[5]
        )
        den = (((_D_COEFFS[0] * q + _D_COEFFS[1]) * q + _D_COEFFS[2]) * q + _D_COEFFS[3]) * q + 1.0
        return num / den

    # Central region: 0.02425 <= p <= 0.97575
    if p_val <= _P_HIGH:
        q = p_val - 0.5
        r = q * q
        num = (
            ((((_A_COEFFS[0] * r + _A_COEFFS[1]) * r + _A_COEFFS[2]) * r + _A_COEFFS[3]) * r + _A_COEFFS[4]) * r
            + _A_COEFFS[5]
        ) * q
        den = (
            ((((_B_COEFFS[0] * r + _B_COEFFS[1]) * r + _B_COEFFS[2]) * r + _B_COEFFS[3]) * r + _B_COEFFS[4]) * r
            + 1.0
        )
        return num / den

    # Upper tail: p > 0.97575
    q = math.sqrt(-2.0 * math.log(1.0 - p_val))
    num = (
        ((((_C_COEFFS[0] * q + _C_COEFFS[1]) * q + _C_COEFFS[2]) * q + _C_COEFFS[3]) * q + _C_COEFFS[4]) * q
        + _C_COEFFS[5]
    )
    den = (((_D_COEFFS[0] * q + _D_COEFFS[1]) * q + _D_COEFFS[2]) * q + _D_COEFFS[3]) * q + 1.0
    return -num / den


def acklam_norm_inv_cdf(p: float) -> float:
    """Alias for norm_inv_cdf."""
    return norm_inv_cdf(p)


def norm_pdf(z: float) -> float:
    """Standard normal probability density function phi(z).

    Args:
        z: Standard score (quantile).

    Returns:
        Probability density value phi(z) = (1 / sqrt(2 * pi)) * exp(-z^2 / 2).
    """
    z_val = float(z)
    return math.exp(-0.5 * z_val * z_val) / math.sqrt(2.0 * math.pi)


def norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Phi(z).

    Args:
        z: Standard score (quantile).

    Returns:
        Cumulative probability Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
    """
    z_val = float(z)
    return 0.5 * (1.0 + math.erf(z_val / math.sqrt(2.0)))


def normal_loss_function(z: float) -> float:
    """Standard normal unit loss function G_u(z).

    Computes the expected shortage in standard deviation units:
    G_u(z) = phi(z) - z * (1 - Phi(z))

    Args:
        z: Safety factor (standard normal quantile).

    Returns:
        Unit loss value G_u(z) >= 0.0.
    """
    z_val = float(z)
    if z_val > 10.0:
        return 0.0
    if z_val < -10.0:
        return -z_val
    return norm_pdf(z_val) - z_val * (1.0 - norm_cdf(z_val))


# ============================================================================
# Lot Sizing Formulas: Wilson EOQ, Total Cost, EPQ, Quantity Discount, Backorders
# ============================================================================


def wilson_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_per_unit: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> float:
    """Classical Wilson Economic Order Quantity (EOQ).

    Minimizes total annual ordering and inventory carrying costs:
    Q* = sqrt(2 * D * S / H)

    Args:
        annual_demand: Total annual units demanded (D >= 0).
        order_cost: Fixed administrative/setup cost per order (S >= 0).
        holding_cost_per_unit: Annual carrying cost per unit per year (H > 0).
        holding_cost: Alternative alias for holding_cost_per_unit.

    Returns:
        Optimal order quantity Q* in units.
    """
    h = holding_cost_per_unit if holding_cost_per_unit is not None else holding_cost
    if h is None:
        raise ValidationError("Holding cost parameter must be provided.")

    if annual_demand < 0:
        raise ValidationError(f"Annual demand cannot be negative, got {annual_demand}")
    if order_cost < 0:
        raise ValidationError(f"Order cost cannot be negative, got {order_cost}")
    if h < 0:
        raise ValidationError(f"Holding cost cannot be negative, got {h}")

    if annual_demand == 0.0 or order_cost == 0.0:
        return 0.0
    if h == 0.0:
        raise InventoryError("Holding cost per unit cannot be zero when annual demand is non-zero.")

    return math.sqrt(2.0 * float(annual_demand) * float(order_cost) / float(h))


def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_per_unit: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> float:
    """Alias for wilson_eoq."""
    return wilson_eoq(
        annual_demand=annual_demand,
        order_cost=order_cost,
        holding_cost_per_unit=holding_cost_per_unit,
        holding_cost=holding_cost,
    )


def total_cost_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_per_unit: float,
    order_quantity: float,
    unit_cost: float = 0.0,
) -> float:
    """Calculate total annual inventory cost for a given lot size Q.

    TC = (D / Q) * S + (Q / 2) * H + D * C

    Args:
        annual_demand: Annual demand D in units.
        order_cost: Fixed order setup cost S.
        holding_cost_per_unit: Holding cost per unit per year H.
        order_quantity: Order lot size Q (> 0).
        unit_cost: Acquisition cost per unit C (optional).

    Returns:
        Total annual inventory cost.
    """
    if order_quantity <= 0:
        raise ValidationError(f"Order quantity Q must be strictly positive, got {order_quantity}")
    if annual_demand < 0 or order_cost < 0 or holding_cost_per_unit < 0 or unit_cost < 0:
        raise ValidationError("Parameters cannot be negative.")

    q = float(order_quantity)
    d = float(annual_demand)
    s = float(order_cost)
    h = float(holding_cost_per_unit)
    c = float(unit_cost)

    ordering_cost = (d / q) * s
    holding_cost = (q / 2.0) * h
    purchase_cost = d * c
    return ordering_cost + holding_cost + purchase_cost


def quantity_discount_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_rate: float,
    price_breaks: Optional[Sequence[Any]] = None,
    discount_tiers: Optional[Sequence[Any]] = None,
    price_tiers: Optional[Sequence[Any]] = None,
) -> QuantityDiscountResult:
    """Optimize order batch size under all-units quantity discount price breaks.

    Evaluates feasible unconstrained EOQ per tier and price break threshold boundaries,
    finding the global minimum total annual cost including purchase acquisition costs.

    Supported tier schemas:
    - List of dicts: [{'min_qty': 0, 'max_qty': 499, 'price': 20.0}, ...]
    - List of tuples: [(min_qty, price), (min_qty_2, price_2), ...]

    Args:
        annual_demand: Annual demand D in units.
        order_cost: Fixed order cost S.
        holding_cost_rate: Annual carrying fraction h (e.g. 0.20 for 20%).
        price_breaks: Price break tier definitions.
        discount_tiers: Alternative alias for price_breaks.
        price_tiers: Alternative alias for price_breaks.

    Returns:
        QuantityDiscountResult namedtuple: (optimal_quantity, min_total_cost, unit_price).
    """
    raw_tiers = price_breaks or discount_tiers or price_tiers
    if not raw_tiers:
        raise ValidationError("At least one price tier or price break must be provided.")
    if annual_demand < 0 or order_cost < 0 or holding_cost_rate <= 0:
        raise ValidationError("Invalid input parameters for quantity discount optimization.")

    d = float(annual_demand)
    s = float(order_cost)
    h_rate = float(holding_cost_rate)

    # Standardize tiers to list of (min_qty, max_qty, price)
    parsed_tiers: List[Tuple[float, float, float]] = []
    first_item = raw_tiers[0]

    if isinstance(first_item, dict):
        for t in raw_tiers:
            min_q = float(t.get("min_qty", t.get("min_quantity", 0.0)))
            max_q = float(t.get("max_qty", t.get("max_quantity", float("inf"))))
            p = float(t.get("price", t.get("unit_price", 0.0)))
            parsed_tiers.append((min_q, max_q, p))
    elif isinstance(first_item, (tuple, list)):
        sorted_tuples = sorted(raw_tiers, key=lambda x: float(x[0]))
        for i, item in enumerate(sorted_tuples):
            min_q = float(item[0])
            p = float(item[1])
            if i + 1 < len(sorted_tuples):
                max_q = float(sorted_tuples[i + 1][0]) - 1.0
            else:
                max_q = float("inf")
            parsed_tiers.append((min_q, max_q, p))
    else:
        raise ValidationError(f"Unsupported price tier format: {type(first_item).__name__}")

    best_q: Optional[float] = None
    best_cost: float = float("inf")
    best_price: float = 0.0

    for min_q, max_q, price in parsed_tiers:
        if price <= 0:
            continue
        h_tier = price * h_rate
        # Calculate theoretical EOQ for this tier price
        q_calc = math.sqrt(2.0 * d * s / h_tier) if (d > 0 and s > 0 and h_tier > 0) else min_q

        candidates: List[float] = []
        if min_q <= q_calc <= max_q:
            candidates.append(q_calc)
        elif q_calc < min_q:
            # Must order at least min_q to qualify for this tier's discounted price
            candidates.append(min_q)
        else:
            # q_calc exceeds max_q
            if math.isfinite(max_q):
                candidates.append(max_q)

        # Always check the tier boundary threshold
        if min_q > 0:
            candidates.append(min_q)

        for candidate_q in candidates:
            if candidate_q <= 0:
                continue
            cost = total_cost_eoq(
                annual_demand=d,
                order_cost=s,
                holding_cost_per_unit=h_tier,
                order_quantity=candidate_q,
                unit_cost=price,
            )
            if cost < best_cost:
                best_cost = cost
                best_q = candidate_q
                best_price = price

    if best_q is None:
        best_q = 0.0
        best_cost = 0.0

    return QuantityDiscountResult(
        optimal_quantity=float(best_q),
        min_total_cost=float(best_cost),
    )


def calculate_quantity_discount_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_rate: float,
    price_breaks: Optional[Sequence[Any]] = None,
    discount_tiers: Optional[Sequence[Any]] = None,
    price_tiers: Optional[Sequence[Any]] = None,
) -> QuantityDiscountResult:
    """Alias for quantity_discount_eoq."""
    return quantity_discount_eoq(
        annual_demand=annual_demand,
        order_cost=order_cost,
        holding_cost_rate=holding_cost_rate,
        price_breaks=price_breaks,
        discount_tiers=discount_tiers,
        price_tiers=price_tiers,
    )


def epq(
    annual_demand: float,
    production_rate: float,
    order_cost: Optional[float] = None,
    setup_cost: Optional[float] = None,
    holding_cost_per_unit: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> float:
    """Economic Production Quantity (EPQ) for finite production rate P > D.

    Models gradual inventory replenishment during ongoing manufacturing runs:
    Q* = sqrt(2 * D * S / (H * (1 - D / P)))

    Args:
        annual_demand: Annual consumption / demand rate D.
        production_rate: Annual manufacturing / production rate P (must be > D).
        order_cost: Machine setup cost S per production run.
        setup_cost: Alternative alias for order_cost.
        holding_cost_per_unit: Annual carrying cost per unit H.
        holding_cost: Alternative alias for holding_cost_per_unit.

    Returns:
        Optimal production lot size in units.
    """
    s = order_cost if order_cost is not None else setup_cost
    h = holding_cost_per_unit if holding_cost_per_unit is not None else holding_cost
    if s is None or h is None:
        raise ValidationError("Setup/order cost and holding cost must be specified.")

    d = float(annual_demand)
    p = float(production_rate)
    s_val = float(s)
    h_val = float(h)

    if d < 0 or p <= 0 or s_val < 0 or h_val <= 0:
        raise ValidationError("Invalid input parameters for EPQ.")
    if p <= d:
        raise InventoryError(
            f"Production rate ({p}) must be strictly greater than annual demand rate ({d}) for finite batch EPQ."
        )

    if d == 0.0 or s_val == 0.0:
        return 0.0

    capacity_slack = 1.0 - (d / p)
    return math.sqrt(2.0 * d * s_val / (h_val * capacity_slack))


def calculate_epq(
    annual_demand: float,
    production_rate: float,
    order_cost: Optional[float] = None,
    setup_cost: Optional[float] = None,
    holding_cost_per_unit: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> float:
    """Alias for epq."""
    return epq(
        annual_demand=annual_demand,
        production_rate=production_rate,
        order_cost=order_cost,
        setup_cost=setup_cost,
        holding_cost_per_unit=holding_cost_per_unit,
        holding_cost=holding_cost,
    )


def eoq_with_backorders(
    annual_demand: float,
    order_cost: float,
    backorder_cost_per_unit: Optional[float] = None,
    backorder_cost: Optional[float] = None,
    holding_cost_per_unit: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> BackorderResult:
    """Economic Order Quantity with planned backorders and shortage penalty cost.

    Optimal order lot size and optimal maximum backorder level:
    Q* = sqrt((2 * D * S / H) * ((H + B) / B))
    b* = Q* * (H / (H + B))

    Args:
        annual_demand: Annual demand D in units.
        order_cost: Order setup cost S.
        backorder_cost_per_unit: Penalty cost B per backordered unit per year.
        backorder_cost: Alternative alias for backorder_cost_per_unit.
        holding_cost_per_unit: Holding cost H per unit per year.
        holding_cost: Alternative alias for holding_cost_per_unit.

    Returns:
        BackorderResult namedtuple: (eoq, max_backorder).
    """
    b = backorder_cost_per_unit if backorder_cost_per_unit is not None else backorder_cost
    h = holding_cost_per_unit if holding_cost_per_unit is not None else holding_cost
    if b is None or h is None:
        raise ValidationError("Holding cost and backorder penalty cost must both be specified.")

    d = float(annual_demand)
    s = float(order_cost)
    b_val = float(b)
    h_val = float(h)

    if d < 0 or s < 0 or h_val <= 0 or b_val <= 0:
        raise ValidationError("Invalid input parameters for planned backorders EOQ.")

    if d == 0.0 or s == 0.0:
        return BackorderResult(eoq=0.0, max_backorder=0.0)

    # Factor (H + B) / B
    factor = (h_val + b_val) / b_val
    q_star = math.sqrt((2.0 * d * s / h_val) * factor)
    max_bo = q_star * (h_val / (h_val + b_val))

    return BackorderResult(eoq=float(q_star), max_backorder=float(max_bo))


def calculate_backorder_eoq(
    annual_demand: float,
    order_cost: float,
    backorder_cost_per_unit: Optional[float] = None,
    backorder_cost: Optional[float] = None,
    holding_cost_per_unit: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> BackorderResult:
    """Alias for eoq_with_backorders."""
    return eoq_with_backorders(
        annual_demand=annual_demand,
        order_cost=order_cost,
        backorder_cost_per_unit=backorder_cost_per_unit,
        backorder_cost=backorder_cost,
        holding_cost_per_unit=holding_cost_per_unit,
        holding_cost=holding_cost,
    )


# ============================================================================
# Reorder Point & Stochastic Safety Stock Models
# ============================================================================


def reorder_point(
    daily_demand: float,
    lead_time_days: float,
    safety_stock: float = 0.0,
) -> float:
    """Calculate deterministic Reorder Point (ROP).

    ROP = (daily_demand * lead_time_days) + safety_stock

    Args:
        daily_demand: Average daily consumption rate (units/day).
        lead_time_days: Supplier lead time in days.
        safety_stock: Buffer stock held against uncertainty.

    Returns:
        Reorder point threshold in units.
    """
    if daily_demand < 0 or lead_time_days < 0 or safety_stock < 0:
        raise ValidationError("Demand, lead time, and safety stock cannot be negative.")
    return float(daily_demand * lead_time_days + safety_stock)


def calculate_reorder_point(
    daily_demand: float,
    lead_time_days: float,
    safety_stock: float = 0.0,
) -> float:
    """Alias for reorder_point."""
    return reorder_point(
        daily_demand=daily_demand,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock,
    )


def safety_stock_joint(
    z_score: float,
    avg_demand: float,
    std_demand: float,
    avg_lead_time: float,
    std_lead_time: float = 0.0,
) -> float:
    """Calculate safety stock with joint stochastic demand and lead-time variance.

    sigma_DL = sqrt(avg_lead_time * std_demand^2 + avg_demand^2 * std_lead_time^2)
    SS = z * sigma_DL

    Args:
        z_score: Standard normal quantile Z corresponding to desired service level.
        avg_demand: Average demand rate per period.
        std_demand: Standard deviation of demand per period.
        avg_lead_time: Average replenishment lead time.
        std_lead_time: Standard deviation of supplier lead time (0.0 for constant lead time).

    Returns:
        Safety stock in units.
    """
    if avg_demand < 0 or std_demand < 0 or avg_lead_time < 0 or std_lead_time < 0:
        raise ValidationError("Variance parameters cannot be negative.")

    var_demand_during_lt = avg_lead_time * (std_demand**2)
    var_lt_during_demand = (avg_demand**2) * (std_lead_time**2)
    sigma_dl = math.sqrt(var_demand_during_lt + var_lt_during_demand)

    if sigma_dl == 0.0 or z_score <= 0.0:
        return 0.0

    return float(z_score * sigma_dl)


def calculate_safety_stock(
    avg_demand: float = 0.0,
    std_demand: Optional[float] = None,
    avg_lead_time: Optional[float] = None,
    std_lead_time: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    daily_demand_std: Optional[float] = None,
    lead_time_days: Optional[float] = None,
) -> float:
    """Flexible helper calculating safety stock under constant or stochastic lead time.

    Supports both parameter nomenclature conventions:
    - (avg_demand, std_demand, avg_lead_time, std_lead_time, service_level)
    - (daily_demand_std, lead_time_days, service_level)

    Args:
        avg_demand: Mean daily demand.
        std_demand: Standard deviation of daily demand.
        avg_lead_time: Mean supplier lead time in days.
        std_lead_time: Standard deviation of supplier lead time in days.
        service_level: Target cycle service level (0.0 < service_level < 1.0).
        z_score: Direct Z-score override.
        daily_demand_std: Alternative alias for std_demand.
        lead_time_days: Alternative alias for avg_lead_time.

    Returns:
        Safety stock in units.
    """
    sigma_d = std_demand if std_demand is not None else (daily_demand_std if daily_demand_std is not None else 0.0)
    lead_t = avg_lead_time if avg_lead_time is not None else (lead_time_days if lead_time_days is not None else 0.0)

    if z_score is not None:
        z = float(z_score)
    elif service_level is not None:
        z = norm_inv_cdf(service_level)
    else:
        # Default 95% service level
        z = 1.6448536269514722

    return safety_stock_joint(
        z_score=z,
        avg_demand=avg_demand,
        std_demand=sigma_d,
        avg_lead_time=lead_t,
        std_lead_time=std_lead_time,
    )


# ============================================================================
# Expected Cycle Shortage & Item Fill Rate (Type II Service Level)
# ============================================================================


def expected_cycle_shortage(
    z: float,
    std_lead_time_demand: float,
) -> float:
    """Expected shortage per replenishment cycle ESC.

    ESC = sigma_DL * G_u(z)

    Args:
        z: Safety factor Z.
        std_lead_time_demand: Standard deviation of demand during lead time (sigma_DL).

    Returns:
        Expected units short per order cycle.
    """
    if std_lead_time_demand < 0:
        raise ValidationError("Demand standard deviation cannot be negative.")
    return float(std_lead_time_demand * normal_loss_function(z))


def item_fill_rate(
    order_quantity: float,
    expected_shortage: float,
) -> float:
    """Calculate Item Fill Rate beta (fraction of customer demand satisfied directly from stock).

    beta = 1 - (ESC / Q)

    Args:
        order_quantity: Replenishment batch size Q (> 0).
        expected_shortage: Expected units short per cycle ESC.

    Returns:
        Fill rate fraction between 0.0 and 1.0.
    """
    if order_quantity <= 0:
        raise ValidationError(f"Order quantity must be strictly positive, got {order_quantity}")
    if expected_shortage < 0:
        raise ValidationError("Expected shortage cannot be negative.")

    fill = 1.0 - (float(expected_shortage) / float(order_quantity))
    return float(max(0.0, min(1.0, fill)))


def calculate_fill_rate(
    sigma_dl: Optional[float] = None,
    lot_size: Optional[float] = None,
    z_score: Optional[float] = None,
    order_quantity: Optional[float] = None,
    expected_shortage: Optional[float] = None,
) -> float:
    """Unified fill rate calculator accepting lot sizing and distribution parameters.

    Args:
        sigma_dl: Standard deviation of lead-time demand.
        lot_size: Order quantity batch size Q.
        z_score: Safety factor Z.
        order_quantity: Alternative alias for lot_size.
        expected_shortage: Pre-calculated expected shortage ESC.

    Returns:
        Item fill rate beta in range [0.0, 1.0].
    """
    q = lot_size if lot_size is not None else order_quantity
    if q is None or q <= 0:
        raise ValidationError("Valid positive lot size must be provided.")

    if expected_shortage is not None:
        esc = float(expected_shortage)
    elif sigma_dl is not None and z_score is not None:
        esc = expected_cycle_shortage(z=z_score, std_lead_time_demand=sigma_dl)
    else:
        raise ValidationError("Either expected_shortage or both (sigma_dl, z_score) must be provided.")

    return item_fill_rate(order_quantity=q, expected_shortage=esc)
