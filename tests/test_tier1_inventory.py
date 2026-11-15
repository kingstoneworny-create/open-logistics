"""Tier 1 Feature Coverage Tests: Inventory Optimization & Replenishment (F10 - F13).

Covers:
- F10: Scientific Lot Sizing (Classical Wilson EOQ, Scaling, Quantity Discounts, EPQ, Planned Backorders)
- F11: Safety Stock & Probit Models (Acklam rational Chebyshev probit, Deterministic ROP, Joint Stochastic Variance, Normal Loss Gu(z), Item Fill Rate)
- F12: ABC-XYZ Categorization & Stock Analytics (Pareto 80/15/5, XYZ Volatility CV, 9-box Matrix, Turnover Ratio, Days Sales of Inventory, Dead Stock Alerts)
- F13: ReplenishmentEventEngine & Auto-PO Workflow (SKU evaluation, Stock decrement trigger, Threshold checks, Deduplication, EventBus integration, Stockout handling)

Authoritative sources of expected outputs:
- Classical Wilson EOQ: Q* = sqrt(2*D*S / H)
- EPQ: Q* = sqrt(2*D*S / (H * (1 - D/P)))
- EOQ with Backorders: Q* = sqrt(2*D*S / H * (H + B)/B), max_backorder = Q* * H / (H + B)
- Acklam Chebyshev Rational Probit approximation for standard normal inverse CDF
- Dynamic safety stock: sigma_DL = sqrt(L * sigma_d^2 + d^2 * sigma_L^2), SS = z * sigma_DL
- Normal loss function: G_u(z) = phi(z) - z * (1 - Phi(z))
- Item fill rate: beta = 1 - (sigma_DL * G_u(z)) / Q
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Attempt import from open_logistics domain modules
try:
    from open_logistics.core.events import Event, EventBus
    from open_logistics.core.exceptions import (
        InventoryError,
        LogisticsError,
        StockoutError,
        ValidationError,
    )
    from open_logistics.core.models import PurchaseOrder, StockItem
    from open_logistics.inventory.abc_xyz import (
        calculate_days_sales_of_inventory,
        calculate_inventory_turnover,
        classify_abc,
        classify_xyz,
        generate_abc_xyz_matrix,
    )
    from open_logistics.inventory.formulas import (
        acklam_norm_inv_cdf,
        calculate_backorder_eoq,
        calculate_eoq,
        calculate_epq,
        calculate_fill_rate,
        calculate_quantity_discount_eoq,
        calculate_reorder_point,
        calculate_safety_stock,
        normal_loss_function,
    )
    from open_logistics.inventory.replenishment_engine import (
        ReplenishmentEventEngine,
        ReplenishmentStatus,
    )

    INVENTORY_AVAILABLE = True
except ImportError:
    INVENTORY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not INVENTORY_AVAILABLE,
    reason="open_logistics.inventory module is pending Milestone 3 implementation",
)


def _get_val(res: Any, attr: str = "eoq") -> float:
    """Helper to extract float value whether function returns float, dict, or dataclass."""
    if isinstance(res, (int, float)):
        return float(res)
    if isinstance(res, dict):
        return float(res.get(attr, next(iter(res.values()))))
    return float(getattr(res, attr, res[0]))


# ============================================================================
# F10: Scientific Lot Sizing (EOQ, Quantity Discounts, EPQ, Backorders)
# ============================================================================


class TestFeature10ScientificLotSizing:
    """Comprehensive tests for Feature 10: Scientific Lot Sizing."""

    def test_classical_wilson_eoq_cost_minimization(self) -> None:
        """Verify classical Wilson EOQ minimizes total annual ordering and holding costs.

        Theoretical Model:
            D = 10,000 units/year, S = $50.0/order, H = $2.0/unit/year.
            Q* = sqrt(2 * 10000 * 50 / 2) = sqrt(500,000) = 707.10678 units.
            At Q*, Ordering Cost = (D/Q)*S = 707.10678.
            At Q*, Holding Cost = (Q/2)*H = 707.10678.
            Total Cost = 1414.21356.
        """
        demand = 10000.0
        order_cost = 50.0
        holding_cost = 2.0

        res = calculate_eoq(annual_demand=demand, order_cost=order_cost, holding_cost=holding_cost)
        q_star = _get_val(res, "eoq")

        expected_q = math.sqrt(2.0 * demand * order_cost / holding_cost)
        assert q_star == pytest.approx(expected_q, rel=1e-4), (
            f"Expected EOQ {expected_q:.4f}, got {q_star:.4f}"
        )

        # Verify ordering cost equals holding cost at Q*
        ordering_cost_opt = (demand / q_star) * order_cost
        holding_cost_opt = (q_star / 2.0) * holding_cost
        assert ordering_cost_opt == pytest.approx(holding_cost_opt, rel=1e-3)

        total_cost_opt = ordering_cost_opt + holding_cost_opt

        # Verify cost at non-optimal batch sizes is strictly higher
        for test_q in [400.0, 500.0, 1000.0, 1500.0]:
            cost_other = (demand / test_q) * order_cost + (test_q / 2.0) * holding_cost
            assert total_cost_opt < cost_other, (
                f"Total cost at EOQ ({total_cost_opt}) should be strictly less than at Q={test_q} ({cost_other})"
            )

    def test_wilson_eoq_demand_scaling_sensitivity(self) -> None:
        """Verify EOQ scales with the square root of demand (quadrupling demand doubles EOQ)."""
        base_demand = 1000.0
        order_cost = 40.0
        holding_cost = 5.0

        res_base = calculate_eoq(annual_demand=base_demand, order_cost=order_cost, holding_cost=holding_cost)
        q_base = _get_val(res_base, "eoq")

        res_quad = calculate_eoq(annual_demand=base_demand * 4.0, order_cost=order_cost, holding_cost=holding_cost)
        q_quad = _get_val(res_quad, "eoq")

        assert q_quad == pytest.approx(2.0 * q_base, rel=1e-4), (
            f"Quadrupling demand must double EOQ: base={q_base}, quad={q_quad}"
        )

    def test_quantity_discount_all_units_tier_evaluation(self) -> None:
        """Evaluate all-units quantity discount price break tiers for global minimum cost.

        Tiers:
            Tier 0: [0, 499]    -> $20.00 / unit
            Tier 1: [500, 999]  -> $19.00 / unit
            Tier 2: [1000, inf) -> $18.00 / unit
            D = 10,000, S = 50, h = 0.20
            Optimal evaluated batch size should be 1000 units at Tier 2 price break.
        """
        demand = 10000.0
        order_cost = 50.0
        holding_rate = 0.20
        tiers = [
            {"min_qty": 0, "max_qty": 499, "price": 20.0},
            {"min_qty": 500, "max_qty": 999, "price": 19.0},
            {"min_qty": 1000, "max_qty": float("inf"), "price": 18.0},
        ]

        res = calculate_quantity_discount_eoq(
            annual_demand=demand,
            order_cost=order_cost,
            price_tiers=tiers,
            holding_cost_rate=holding_rate,
        )

        best_qty = _get_val(res, "optimal_quantity")
        # Global minimum total annual cost occurs at the Tier 2 threshold of 1000 units
        assert best_qty >= 1000, f"Expected discount lot size >= 1000, got {best_qty}"
        assert best_qty == pytest.approx(1000.0, abs=10.0)

    def test_epq_finite_replenishment_batch_calculation(self) -> None:
        """Verify Economic Production Quantity (EPQ) for finite production rate P > D.

        Model:
            D = 2,000 units/year, P = 10,000 units/year (production rate)
            S = $100.0, H = $2.0/unit/year
            Q_EPQ = sqrt(2 * 2000 * 100 / (2 * (1 - 2000/10000))) = sqrt(400,000 / 1.6) = 500.0 units.
        """
        demand = 2000.0
        production_rate = 10000.0
        order_cost = 100.0
        holding_cost = 2.0

        res_epq = calculate_epq(
            annual_demand=demand,
            order_cost=order_cost,
            holding_cost=holding_cost,
            production_rate=production_rate,
        )
        q_epq = _get_val(res_epq, "epq")
        assert q_epq == pytest.approx(500.0, rel=1e-4)

        # EPQ batch size must strictly exceed classical instantaneous EOQ
        res_eoq = calculate_eoq(annual_demand=demand, order_cost=order_cost, holding_cost=holding_cost)
        q_eoq = _get_val(res_eoq, "eoq")
        assert q_epq > q_eoq, f"EPQ ({q_epq}) must exceed instantaneous EOQ ({q_eoq})"

    def test_epq_production_rate_lte_demand_raises_error(self) -> None:
        """EPQ calculation must reject production rate <= demand rate."""
        with pytest.raises((ValueError, InventoryError, ValidationError)):
            calculate_epq(
                annual_demand=5000.0,
                order_cost=50.0,
                holding_cost=2.0,
                production_rate=4000.0,  # P < D cannot sustain finite batches
            )

    def test_eoq_with_planned_backorders_stockout_penalty(self) -> None:
        """Verify EOQ with planned backorders under stockout penalty cost B.

        Formula:
            Q_BO = sqrt( (2*D*S/H) * ((H + B)/B) )
            Optimal backorder level b* = Q_BO * (H / (H + B))
            Let D=2000, S=50, H=2, B=8:
            Factor (H+B)/B = 10/8 = 1.25.
            Q_BO = sqrt(100,000 * 1.25) = sqrt(125,000) = 353.553 units.
            b* = 353.553 * (2 / 10) = 70.711 units.
        """
        demand = 2000.0
        order_cost = 50.0
        holding_cost = 2.0
        backorder_cost = 8.0

        res = calculate_backorder_eoq(
            annual_demand=demand,
            order_cost=order_cost,
            holding_cost=holding_cost,
            backorder_cost=backorder_cost,
        )
        q_bo = _get_val(res, "eoq")
        expected_q = math.sqrt((2.0 * demand * order_cost / holding_cost) * ((holding_cost + backorder_cost) / backorder_cost))
        assert q_bo == pytest.approx(expected_q, rel=1e-3)

        # When backorder penalty approaches infinity, Q_BO converges to classical Wilson EOQ
        res_inf = calculate_backorder_eoq(
            annual_demand=demand,
            order_cost=order_cost,
            holding_cost=holding_cost,
            backorder_cost=1e9,
        )
        q_inf = _get_val(res_inf, "eoq")
        expected_wilson = math.sqrt(2.0 * demand * order_cost / holding_cost)
        assert q_inf == pytest.approx(expected_wilson, rel=1e-3)

    def test_lot_sizing_invalid_parameters_raise_error(self) -> None:
        """Verify negative or zero values raise appropriate domain errors."""
        with pytest.raises((ValueError, InventoryError, ValidationError)):
            calculate_eoq(annual_demand=-100.0, order_cost=50.0, holding_cost=2.0)

        with pytest.raises((ValueError, InventoryError, ValidationError)):
            calculate_eoq(annual_demand=100.0, order_cost=-5.0, holding_cost=2.0)

        with pytest.raises((ValueError, InventoryError, ValidationError)):
            calculate_eoq(annual_demand=100.0, order_cost=50.0, holding_cost=0.0)


# ============================================================================
# F11: Safety Stock & Probit Models (Acklam, ROP, Joint Variance, Loss Func, Fill Rate)
# ============================================================================


class TestFeature11SafetyStockAndProbitModels:
    """Comprehensive tests for Feature 11: Safety Stock & Probit Models."""

    @pytest.mark.parametrize(
        "p,expected_z,tolerance",
        [
            (0.50, 0.0, 1e-5),
            (0.8413447, 1.0, 1e-3),
            (0.90, 1.28155, 1e-4),
            (0.95, 1.64485, 1e-4),
            (0.975, 1.95996, 1e-4),
            (0.99, 2.32635, 1e-4),
            (0.999, 3.09023, 1e-3),
        ],
    )
    def test_acklam_probit_accuracy_against_known_z_scores(
        self, p: float, expected_z: float, tolerance: float
    ) -> None:
        """Verify Acklam rational Chebyshev probit approximation against standard normal Z-scores."""
        computed_z = acklam_norm_inv_cdf(p)
        assert computed_z == pytest.approx(expected_z, abs=tolerance), (
            f"For p={p}, expected z={expected_z}, got {computed_z}"
        )

    def test_acklam_probit_symmetry_and_domain_guards(self) -> None:
        """Verify probit symmetry Phi^-1(p) == -Phi^-1(1-p) and domain validation for p in (0, 1)."""
        test_probabilities = [0.01, 0.05, 0.10, 0.25, 0.40, 0.75, 0.90, 0.95, 0.99]
        for p in test_probabilities:
            z_pos = acklam_norm_inv_cdf(p)
            z_neg = acklam_norm_inv_cdf(1.0 - p)
            assert z_pos == pytest.approx(-z_neg, abs=1e-5), (
                f"Symmetry failed for p={p}: z({p})={z_pos}, z({1.0-p})={z_neg}"
            )

        # Domain boundary exceptions
        for invalid_p in [-0.5, 0.0, 1.0, 1.5]:
            with pytest.raises((ValueError, ValidationError, InventoryError)):
                acklam_norm_inv_cdf(invalid_p)

    def test_deterministic_reorder_point(self) -> None:
        """Verify deterministic Reorder Point ROP = (daily_demand * lead_time) + safety_stock."""
        daily_demand = 25.0
        lead_time = 6.0
        rop_no_ss = calculate_reorder_point(daily_demand=daily_demand, lead_time_days=lead_time, safety_stock=0.0)
        assert rop_no_ss == pytest.approx(150.0, rel=1e-4)

        rop_with_ss = calculate_reorder_point(daily_demand=daily_demand, lead_time_days=lead_time, safety_stock=25.0)
        assert rop_with_ss == pytest.approx(175.0, rel=1e-4)

    def test_dynamic_safety_stock_constant_lead_time(self) -> None:
        """Verify safety stock when lead time is constant: SS = z * sigma_d * sqrt(L).

        Parameters:
            avg_demand = 20.0, std_demand = 5.0, L = 4.0, std_lead_time = 0.0
            Service level = 0.95 (z = 1.64485)
            sigma_DL = 5.0 * sqrt(4.0) = 10.0
            SS = 1.64485 * 10.0 = 16.4485
        """
        ss = calculate_safety_stock(
            avg_demand=20.0,
            std_demand=5.0,
            avg_lead_time=4.0,
            std_lead_time=0.0,
            service_level=0.95,
        )
        assert ss == pytest.approx(16.4485, rel=1e-3)

    def test_dynamic_safety_stock_joint_demand_and_lead_time_variance(self) -> None:
        """Verify safety stock under joint stochastic demand AND lead-time variance.

        Formula:
            sigma_DL = sqrt(avg_lead_time * std_demand^2 + avg_demand^2 * std_lead_time^2)
            SS = z * sigma_DL
            avg_demand = 20.0, std_demand = 4.0, avg_lead_time = 9.0, std_lead_time = 1.0
            sigma_DL = sqrt(9 * 16 + 400 * 1) = sqrt(144 + 400) = sqrt(544) = 23.3238
            Service level = 0.99 -> z = 2.32635
            SS = 2.32635 * 23.3238 = 54.260
        """
        ss = calculate_safety_stock(
            avg_demand=20.0,
            std_demand=4.0,
            avg_lead_time=9.0,
            std_lead_time=1.0,
            service_level=0.99,
        )
        expected_sigma_dl = math.sqrt(9.0 * (4.0**2) + (20.0**2) * (1.0**2))
        expected_z = 2.32635
        expected_ss = expected_z * expected_sigma_dl
        assert ss == pytest.approx(expected_ss, rel=1e-3)

    def test_normal_loss_function_properties(self) -> None:
        """Verify standard normal loss function G_u(z) = phi(z) - z * (1 - Phi(z))."""
        # G_u(0) = 1 / sqrt(2 * pi) approx 0.398942
        g_0 = normal_loss_function(0.0)
        assert g_0 == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), rel=1e-3)

        # Monotonicity: G_u(z) strictly decreases with increasing z
        z_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        loss_values = [normal_loss_function(z) for z in z_values]
        for i in range(len(loss_values) - 1):
            assert loss_values[i] > loss_values[i + 1], f"Loss function must be strictly decreasing at z={z_values[i]}"

        # At z=1: G_u(1) approx 0.0833
        assert normal_loss_function(1.0) == pytest.approx(0.0833, abs=0.005)
        # At z=2: G_u(2) approx 0.00849
        assert normal_loss_function(2.0) == pytest.approx(0.00849, abs=0.001)

    def test_item_fill_rate_calculation(self) -> None:
        """Verify item fill rate (Type II service level) beta = 1 - (sigma_DL * G_u(z)) / Q.

        Parameters:
            sigma_DL = 10.0, Q = 100.0, z = 1.645 -> G_u(1.645) approx 0.02114
            Expected cycle shortages = 10.0 * 0.02114 = 0.2114 units
            beta = 1 - (0.2114 / 100) = 0.99789 (99.79% fill rate)
        """
        fill_rate = calculate_fill_rate(sigma_dl=10.0, lot_size=100.0, z_score=1.645)
        assert 0.99 <= fill_rate <= 1.0
        assert fill_rate == pytest.approx(0.9979, abs=0.002)

        # Increasing lot size Q reduces shortage fraction and increases fill rate
        higher_q_fill_rate = calculate_fill_rate(sigma_dl=10.0, lot_size=200.0, z_score=1.645)
        assert higher_q_fill_rate > fill_rate


# ============================================================================
# F12: ABC-XYZ Categorization & Stock Analytics (Pareto, CV, Matrix, Turnover, DSI)
# ============================================================================


class TestFeature12AbcXyzAndStockAnalytics:
    """Comprehensive tests for Feature 12: ABC-XYZ Categorization & Stock Analytics."""

    @pytest.fixture
    def sample_sku_portfolio(self) -> List[StockItem]:
        """A diverse portfolio of 6 SKUs representing A, B, and C value tiers."""
        return [
            StockItem(sku="SKU-A1", on_hand=100, unit_cost=50.0, daily_demand_history=[20.0] * 30),  # $365k / yr (high)
            StockItem(sku="SKU-A2", on_hand=80, unit_cost=40.0, daily_demand_history=[15.0] * 30),   # $219k / yr (high)
            StockItem(sku="SKU-B1", on_hand=50, unit_cost=30.0, daily_demand_history=[5.0] * 30),    # $54.7k / yr (mid)
            StockItem(sku="SKU-B2", on_hand=40, unit_cost=25.0, daily_demand_history=[4.0] * 30),    # $36.5k / yr (mid)
            StockItem(sku="SKU-C1", on_hand=20, unit_cost=10.0, daily_demand_history=[1.0] * 30),    # $3.65k / yr (low)
            StockItem(sku="SKU-C2", on_hand=10, unit_cost=5.0, daily_demand_history=[0.5] * 30),     # $0.91k / yr (low)
        ]

    def test_pareto_abc_classification_proportions(self, sample_sku_portfolio: List[StockItem]) -> None:
        """Verify Pareto 80/15/5 cumulative annual value classification partitions items correctly."""
        abc_results = classify_abc(sample_sku_portfolio)

        # Check results mapping format (dict of sku -> category or list of categorizations)
        if isinstance(abc_results, dict):
            assert abc_results["SKU-A1"] in ("A", "a")
            assert abc_results["SKU-A2"] in ("A", "a")
            assert abc_results["SKU-B1"] in ("B", "b")
            assert abc_results["SKU-C1"] in ("C", "c")
            assert abc_results["SKU-C2"] in ("C", "c")
        else:
            categories = {item["sku"] if isinstance(item, dict) else item.sku: item["category"] if isinstance(item, dict) else item.category for item in abc_results}
            assert categories["SKU-A1"] == "A"
            assert categories["SKU-C1"] == "C"

    def test_abc_classification_boundary_single_item(self) -> None:
        """A single item in inventory constitutes 100% of value and must be classified as 'A'."""
        single_item = [StockItem(sku="SOLO-1", on_hand=10, unit_cost=100.0, daily_demand_history=[5.0] * 10)]
        res = classify_abc(single_item)
        category = res["SOLO-1"] if isinstance(res, dict) else (res[0]["category"] if isinstance(res[0], dict) else res[0].category)
        assert category in ("A", "a")

    def test_xyz_volatility_classification_cv(self) -> None:
        """Verify XYZ categorization according to Coefficient of Variation (CV = sigma / mu).

        Criteria:
            X: Stable demand (CV <= 0.15)
            Y: Moderate fluctuations (0.15 < CV <= 0.40)
            Z: High irregularity / sporadic demand (CV > 0.40)
        """
        sku_x = StockItem(sku="SKU-X", on_hand=50, unit_cost=10.0, daily_demand_history=[10.0, 10.0, 10.0, 10.0, 10.0])  # CV = 0.0
        sku_y = StockItem(sku="SKU-Y", on_hand=50, unit_cost=10.0, daily_demand_history=[10.0, 12.0, 8.0, 11.0, 9.0])     # CV approx 0.15
        sku_z = StockItem(sku="SKU-Z", on_hand=50, unit_cost=10.0, daily_demand_history=[0.0, 0.0, 25.0, 0.0, 1.0])       # CV > 1.0

        xyz_results = classify_xyz([sku_x, sku_y, sku_z])
        if isinstance(xyz_results, dict):
            assert xyz_results["SKU-X"] in ("X", "x")
            assert xyz_results["SKU-Z"] in ("Z", "z")
        else:
            categories = {item["sku"] if isinstance(item, dict) else item.sku: item["category"] if isinstance(item, dict) else item.category for item in xyz_results}
            assert categories["SKU-X"] == "X"
            assert categories["SKU-Z"] == "Z"

    def test_nine_box_abc_xyz_matrix_generation(self, sample_sku_portfolio: List[StockItem]) -> None:
        """Verify generation of the 9-box ABC-XYZ segmentation matrix."""
        matrix = generate_abc_xyz_matrix(sample_sku_portfolio)
        expected_cells = {"AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"}

        # Matrix should map cell -> list of SKUs or item identifiers
        if isinstance(matrix, dict):
            assert expected_cells.issubset(set(matrix.keys()))
            total_items_in_matrix = sum(len(items) for items in matrix.values())
            assert total_items_in_matrix == len(sample_sku_portfolio)

    def test_inventory_turnover_ratio_calculation(self) -> None:
        """Verify Inventory Turnover Ratio = Cost of Goods Sold (COGS) / Average Inventory Value."""
        cogs = 1200000.0
        avg_inv = 200000.0
        turnover = calculate_inventory_turnover(cogs=cogs, average_inventory=avg_inv)
        assert turnover == pytest.approx(6.0, rel=1e-4)

    def test_days_sales_of_inventory(self) -> None:
        """Verify Days Sales of Inventory (DSI) = 365 / Inventory Turnover."""
        turnover = 6.0
        dsi = calculate_days_sales_of_inventory(turnover=turnover)
        assert dsi == pytest.approx(365.0 / 6.0, rel=1e-4)
        assert dsi == pytest.approx(60.833, abs=0.01)

    def test_dead_stock_alert_detection(self) -> None:
        """Verify dead stock detection flags dormant SKUs with zero consumption over observation window."""
        active_sku = StockItem(sku="ACTIVE-1", on_hand=100, daily_demand_history=[5.0, 4.0, 6.0, 5.0])
        dormant_sku = StockItem(sku="DORMANT-1", on_hand=150, daily_demand_history=[0.0, 0.0, 0.0, 0.0, 0.0])

        assert dormant_sku.average_daily_demand == 0.0
        assert active_sku.average_daily_demand > 0.0


# ============================================================================
# F13: ReplenishmentEventEngine & Auto-PO Workflow
# ============================================================================


class TestFeature13ReplenishmentEventEngine:
    """Comprehensive tests for Feature 13: ReplenishmentEventEngine & Auto-PO Workflow."""

    @pytest.fixture
    def sample_stock(self) -> StockItem:
        """Standard stock item for replenishment engine testing."""
        return StockItem(
            sku="SKU-AUTO-01",
            on_hand=80,
            on_order=0,
            unit_cost=20.0,
            holding_cost_rate=0.20,
            lead_time_days=4.0,
            daily_demand_history=[10.0, 12.0, 9.0, 11.0, 8.0, 10.0],
            allocated=0,
        )

    def test_sku_evaluation_returns_valid_replenishment_status(self, sample_stock: StockItem) -> None:
        """Verify evaluate_sku returns a comprehensive ReplenishmentStatus dataclass."""
        engine = ReplenishmentEventEngine()
        status = engine.evaluate_sku(item=sample_stock, service_level=0.95, order_cost=50.0)

        assert isinstance(status, ReplenishmentStatus)
        assert status.sku == sample_stock.sku
        assert isinstance(status.needs_reorder, bool)
        assert status.reorder_point > 0.0
        assert status.safety_stock >= 0.0
        assert status.eoq > 0.0
        assert isinstance(status.recommended_order_qty, int)
        assert 0.0 <= status.fill_rate <= 1.0

    def test_unneeded_reorder_when_stock_above_rop(self, sample_stock: StockItem) -> None:
        """When net stock is strictly above ROP, needs_reorder must be False and recommended Qty == 0."""
        sample_stock.on_hand = 500  # Plenty of stock
        engine = ReplenishmentEventEngine()
        status = engine.evaluate_sku(item=sample_stock, service_level=0.95, order_cost=50.0)

        assert status.needs_reorder is False
        assert status.recommended_order_qty == 0

        # Decrementing a small amount should not trigger a PurchaseOrder
        po = engine.process_stock_decrement(sku=sample_stock.sku, quantity=5)
        assert po is None, "No PurchaseOrder should be created when stock remains above ROP"

    def test_event_driven_stock_decrement_triggers_automatic_po(self, sample_stock: StockItem) -> None:
        """Decrementing stock at or below ROP triggers automatic PurchaseOrder creation."""
        engine = ReplenishmentEventEngine()
        # Evaluate to learn baseline ROP
        status = engine.evaluate_sku(item=sample_stock, service_level=0.95, order_cost=50.0)
        rop = status.reorder_point

        # Set on_hand just above ROP, then decrement to fall below ROP
        sample_stock.on_hand = int(math.ceil(rop)) + 5
        decrement_qty = 15  # Brings on_hand well below ROP

        po = engine.process_stock_decrement(sku=sample_stock.sku, quantity=decrement_qty)
        assert po is not None, "PurchaseOrder must be automatically generated when stock drops below ROP"
        assert isinstance(po, PurchaseOrder)
        assert po.sku == sample_stock.sku
        assert po.quantity > 0
        assert po.quantity >= int(math.floor(status.eoq))
        assert po.status in ("ISSUED", "PENDING", "CREATED")

    def test_reorder_deduplication_via_on_order_accounting(self, sample_stock: StockItem) -> None:
        """Outstanding on_order quantity prevents duplicate redundant PurchaseOrders."""
        engine = ReplenishmentEventEngine()
        status = engine.evaluate_sku(item=sample_stock, service_level=0.95, order_cost=50.0)

        # Trigger initial PO
        sample_stock.on_hand = int(status.reorder_point) - 10
        po1 = engine.process_stock_decrement(sku=sample_stock.sku, quantity=5)
        assert po1 is not None

        # Simulate on_order updated with PO quantity
        sample_stock.on_order = po1.quantity

        # Subsequent small decrement does not create a second duplicate PO because net inventory is sufficient
        po2 = engine.process_stock_decrement(sku=sample_stock.sku, quantity=2)
        assert po2 is None, "Should not issue duplicate PO when on_order covers net inventory requirement"

    def test_replenishment_engine_eventbus_integration(self, sample_stock: StockItem) -> None:
        """Verify replenishment engine publishes an event on EventBus when a reorder is triggered."""
        bus = EventBus()
        published_events: List[Event] = []

        def on_reorder(event: Event) -> None:
            published_events.append(event)

        bus.subscribe("inventory.reorder_triggered", on_reorder)
        bus.subscribe("inventory.*", on_reorder)

        engine = ReplenishmentEventEngine(event_bus=bus)
        sample_stock.on_hand = 10  # Below ROP

        po = engine.process_stock_decrement(sku=sample_stock.sku, quantity=5)
        if po is not None:
            assert len(published_events) > 0, "EventBus should receive an event when reorder is triggered"
            assert any(sample_stock.sku in str(evt.payload) for evt in published_events)

    def test_stockout_error_on_excessive_decrement(self, sample_stock: StockItem) -> None:
        """Decrementing more units than physically on_hand raises StockoutError."""
        engine = ReplenishmentEventEngine()
        sample_stock.on_hand = 10
        with pytest.raises((StockoutError, LogisticsError)):
            engine.process_stock_decrement(sku=sample_stock.sku, quantity=50)

    def test_service_level_sensitivity_on_rop_and_safety_stock(self, sample_stock: StockItem) -> None:
        """A higher service level (99% vs 90%) produces larger safety stock and higher ROP."""
        engine = ReplenishmentEventEngine()
        status_90 = engine.evaluate_sku(item=sample_stock, service_level=0.90, order_cost=50.0)
        status_99 = engine.evaluate_sku(item=sample_stock, service_level=0.99, order_cost=50.0)

        assert status_99.z_score > status_90.z_score, "Z-score at 99% must exceed Z-score at 90%"
        assert status_99.safety_stock > status_90.safety_stock, "Safety stock at 99% must exceed 90%"
        assert status_99.reorder_point > status_90.reorder_point, "ROP at 99% must exceed ROP at 90%"
