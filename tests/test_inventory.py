"""Unit tests for Open Logistics scientific inventory optimization & replenishment.

Validates:
- Formulas (Wilson classical EOQ, total cost, quantity discounts, EPQ, planned backorders)
- Acklam's rational Chebyshev probit approximation for exact arbitrary Z-scores
- Standard normal PDF, CDF, normal loss function G(z), and Item Fill Rate beta
- Stochastic safety stock with constant and joint lead-time variance
- Deterministic and dynamic Reorder Points (ROP)
- Pareto ABC classification (80/15/5% cutoffs)
- XYZ volatility classification based on Coefficient of Variation (CV)
- 9-box ABC-XYZ segmentation matrix
- Financial metrics: Inventory turnover ratio and Days Sales of Inventory (DSI)
- Dead stock detection and stockout risk alerts
- Reactive ReplenishmentEventEngine with EventBus pub/sub integration
"""

from __future__ import annotations

import math
from typing import List

import pytest

from open_logistics.core.events import Event, EventBus
from open_logistics.core.exceptions import (
    InventoryError,
    LogisticsError,
    StockoutError,
    ValidationError,
)
from open_logistics.core.models import PurchaseOrder, StockItem
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


# ============================================================================
# 1. Classical Wilson EOQ & Cost Models
# ============================================================================


class TestWilsonEOQAndCost:
    """Test suite for classical Wilson EOQ and inventory cost functions."""

    def test_wilson_eoq_textbook_calculation(self) -> None:
        """Verify EOQ matches theoretical Q* = sqrt(2*D*S / H)."""
        d = 10000.0
        s = 50.0
        h = 2.0
        expected_q = math.sqrt(2.0 * d * s / h)  # sqrt(500,000) ~ 707.1068
        res = wilson_eoq(annual_demand=d, order_cost=s, holding_cost_per_unit=h)
        assert res == pytest.approx(expected_q, rel=1e-5)

    def test_calculate_eoq_alias(self) -> None:
        """Verify calculate_eoq function alias works with holding_cost parameter."""
        res = calculate_eoq(annual_demand=5000.0, order_cost=40.0, holding_cost=4.0)
        expected = math.sqrt(2.0 * 5000.0 * 40.0 / 4.0)
        assert res == pytest.approx(expected, rel=1e-5)

    def test_eoq_demand_quadrupling_doubles_batch_size(self) -> None:
        """Square root relationship: 4x demand yields exactly 2x EOQ."""
        q1 = wilson_eoq(annual_demand=1000.0, order_cost=50.0, holding_cost_per_unit=2.0)
        q2 = wilson_eoq(annual_demand=4000.0, order_cost=50.0, holding_cost_per_unit=2.0)
        assert q2 == pytest.approx(2.0 * q1, rel=1e-5)

    def test_total_cost_eoq_strict_convexity(self) -> None:
        """Total cost at EOQ is strictly lower than at any other order quantity."""
        d = 12000.0
        s = 60.0
        h = 2.5
        q_star = wilson_eoq(annual_demand=d, order_cost=s, holding_cost_per_unit=h)
        cost_optimal = total_cost_eoq(
            annual_demand=d, order_cost=s, holding_cost_per_unit=h, order_quantity=q_star, unit_cost=10.0
        )

        for perturbation in [0.5, 0.8, 1.2, 1.5, 2.0]:
            q_other = q_star * perturbation
            cost_other = total_cost_eoq(
                annual_demand=d, order_cost=s, holding_cost_per_unit=h, order_quantity=q_other, unit_cost=10.0
            )
            assert cost_optimal < cost_other, f"Cost at Q*={q_star} must be strictly less than at Q={q_other}"

    def test_eoq_zero_demand_and_ordering_cost(self) -> None:
        """Zero demand or zero ordering cost returns 0.0 batch size."""
        assert wilson_eoq(annual_demand=0.0, order_cost=50.0, holding_cost_per_unit=2.0) == 0.0
        assert wilson_eoq(annual_demand=1000.0, order_cost=0.0, holding_cost_per_unit=2.0) == 0.0

    def test_eoq_zero_holding_cost_raises_inventory_error(self) -> None:
        """Non-zero demand with zero holding cost raises InventoryError."""
        with pytest.raises(InventoryError):
            wilson_eoq(annual_demand=1000.0, order_cost=50.0, holding_cost_per_unit=0.0)

    def test_eoq_negative_parameters_raise_validation_error(self) -> None:
        """Negative demand, order cost, or holding cost raises ValidationError."""
        with pytest.raises(ValidationError):
            wilson_eoq(annual_demand=-500.0, order_cost=50.0, holding_cost_per_unit=2.0)
        with pytest.raises(ValidationError):
            wilson_eoq(annual_demand=500.0, order_cost=-10.0, holding_cost_per_unit=2.0)
        with pytest.raises(ValidationError):
            wilson_eoq(annual_demand=500.0, order_cost=50.0, holding_cost_per_unit=-1.0)


# ============================================================================
# 2. All-Units Quantity Discount EOQ
# ============================================================================


class TestQuantityDiscountEOQ:
    """Test suite for all-units quantity discount optimization."""

    def test_quantity_discount_multi_tier_dict_evaluation(self) -> None:
        """Evaluate multi-tier price breaks in dict format for global minimum."""
        demand = 10000.0
        order_cost = 50.0
        holding_rate = 0.20
        tiers = [
            {"min_qty": 0, "max_qty": 499, "price": 20.0},
            {"min_qty": 500, "max_qty": 999, "price": 19.0},
            {"min_qty": 1000, "max_qty": float("inf"), "price": 18.0},
        ]

        res = quantity_discount_eoq(
            annual_demand=demand,
            order_cost=order_cost,
            holding_cost_rate=holding_rate,
            price_breaks=tiers,
        )
        assert isinstance(res, QuantityDiscountResult)
        assert res.optimal_quantity >= 1000.0
        assert res.optimal_quantity == pytest.approx(1000.0, abs=10.0)
        assert res.min_total_cost > 0.0

    def test_quantity_discount_tuple_tier_format(self) -> None:
        """Evaluate multi-tier price breaks in (min_quantity, price) tuple format."""
        demand = 2000.0
        order_cost = 100.0
        holding_rate = 0.20
        tiers = [(1, 20.0), (500, 15.0)]

        q_opt, cost_opt = calculate_quantity_discount_eoq(
            annual_demand=demand,
            order_cost=order_cost,
            holding_cost_rate=holding_rate,
            discount_tiers=tiers,
        )
        assert q_opt >= 500.0
        assert cost_opt > 0.0

    def test_quantity_discount_single_tier_matches_classical_eoq(self) -> None:
        """When only a single constant price tier exists, result matches classical Wilson EOQ."""
        demand = 1000.0
        order_cost = 50.0
        unit_price = 20.0
        holding_rate = 0.20
        h = unit_price * holding_rate

        expected_eoq = wilson_eoq(annual_demand=demand, order_cost=order_cost, holding_cost_per_unit=h)
        res = quantity_discount_eoq(
            annual_demand=demand,
            order_cost=order_cost,
            holding_cost_rate=holding_rate,
            price_breaks=[(1, unit_price)],
        )
        assert res.optimal_quantity == pytest.approx(expected_eoq, rel=1e-4)


# ============================================================================
# 3. Economic Production Quantity (EPQ)
# ============================================================================


class TestEPQ:
    """Test suite for finite production rate EPQ."""

    def test_epq_calculation_with_finite_production_rate(self) -> None:
        """Verify EPQ formula Q* = sqrt(2*D*S / (H * (1 - D/P)))."""
        d = 2000.0
        p = 10000.0
        s = 100.0
        h = 2.0
        expected_epq = math.sqrt(2.0 * d * s / (h * (1.0 - d / p)))  # 500.0
        res = epq(annual_demand=d, production_rate=p, order_cost=s, holding_cost_per_unit=h)
        assert res == pytest.approx(expected_epq, rel=1e-5)
        assert res == pytest.approx(500.0, rel=1e-5)

    def test_epq_strictly_exceeds_instantaneous_eoq(self) -> None:
        """Finite production rate batch size must strictly exceed instantaneous EOQ."""
        d = 3000.0
        p = 6000.0
        s = 80.0
        h = 3.0
        q_eoq = wilson_eoq(annual_demand=d, order_cost=s, holding_cost_per_unit=h)
        q_epq = epq(annual_demand=d, production_rate=p, setup_cost=s, holding_cost=h)
        assert q_epq > q_eoq

    def test_epq_production_rate_lte_demand_raises_inventory_error(self) -> None:
        """Production rate <= demand rate cannot sustain cyclic manufacturing and must raise InventoryError."""
        with pytest.raises(InventoryError):
            calculate_epq(annual_demand=5000.0, production_rate=5000.0, order_cost=50.0, holding_cost=2.0)
        with pytest.raises(InventoryError):
            calculate_epq(annual_demand=5000.0, production_rate=4000.0, order_cost=50.0, holding_cost=2.0)


# ============================================================================
# 4. Planned Backorders EOQ
# ============================================================================


class TestPlannedBackordersEOQ:
    """Test suite for EOQ with planned backorders."""

    def test_eoq_with_planned_backorders_calculation(self) -> None:
        """Verify backorder batch size Q* and max shortage level b*."""
        d = 2000.0
        s = 50.0
        h = 2.0
        b = 8.0

        factor = (h + b) / b  # 1.25
        expected_q = math.sqrt((2.0 * d * s / h) * factor)  # ~353.553
        expected_b = expected_q * (h / (h + b))  # ~70.711

        res = eoq_with_backorders(
            annual_demand=d,
            order_cost=s,
            holding_cost_per_unit=h,
            backorder_cost_per_unit=b,
        )
        assert isinstance(res, BackorderResult)
        assert res.eoq == pytest.approx(expected_q, rel=1e-4)
        assert res.max_backorder == pytest.approx(expected_b, rel=1e-4)
        assert res.optimal_order_quantity == res.eoq
        assert res.max_backorder_quantity == res.max_backorder

    def test_backorders_converges_to_wilson_when_penalty_approaches_infinity(self) -> None:
        """As backorder penalty B -> inf, Q* converges to classical Wilson EOQ and b* -> 0."""
        d = 4000.0
        s = 60.0
        h = 3.0
        q_wilson = wilson_eoq(annual_demand=d, order_cost=s, holding_cost_per_unit=h)

        res = calculate_backorder_eoq(
            annual_demand=d,
            order_cost=s,
            holding_cost=h,
            backorder_cost=1e9,
        )
        assert res.eoq == pytest.approx(q_wilson, rel=1e-4)
        assert res.max_backorder == pytest.approx(0.0, abs=1e-3)


# ============================================================================
# 5. Deterministic Reorder Point (ROP)
# ============================================================================


class TestDeterministicROP:
    """Test suite for deterministic Reorder Point ROP = (d * L) + SS."""

    def test_reorder_point_with_and_without_safety_stock(self) -> None:
        """Verify ROP calculation under various safety stock buffers."""
        assert reorder_point(daily_demand=20.0, lead_time_days=5.0, safety_stock=0.0) == 100.0
        assert reorder_point(daily_demand=20.0, lead_time_days=5.0, safety_stock=30.0) == 130.0

    def test_reorder_point_zero_lead_time(self) -> None:
        """Zero lead time produces ROP = safety stock."""
        assert calculate_reorder_point(daily_demand=15.0, lead_time_days=0.0, safety_stock=10.0) == 10.0

    def test_reorder_point_negative_parameters_raise_validation_error(self) -> None:
        """Negative demand or lead time raises ValidationError."""
        with pytest.raises(ValidationError):
            reorder_point(daily_demand=-5.0, lead_time_days=4.0)
        with pytest.raises(ValidationError):
            reorder_point(daily_demand=5.0, lead_time_days=-1.0)


# ============================================================================
# 6. Acklam's Probit Approximation & Normal Distribution
# ============================================================================


class TestAcklamProbitAndNormalDist:
    """Test suite for Peter John Acklam's Chebyshev probit approximation."""

    @pytest.mark.parametrize(
        "p,expected_z",
        [
            (0.50, 0.0),
            (0.8413447, 1.0),
            (0.90, 1.28155),
            (0.95, 1.64485),
            (0.975, 1.95996),
            (0.99, 2.32635),
            (0.999, 3.09023),
            (0.9999, 3.71902),
        ],
    )
    def test_acklam_accuracy_across_service_levels(self, p: float, expected_z: float) -> None:
        """Verify Acklam inverse normal CDF against standard normal quantiles."""
        computed_z = acklam_norm_inv_cdf(p)
        assert computed_z == pytest.approx(expected_z, abs=0.005)

    def test_acklam_symmetry_property(self) -> None:
        """Verify symmetry: Phi^-1(p) == -Phi^-1(1 - p)."""
        for p in [0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.80, 0.95, 0.99]:
            z1 = norm_inv_cdf(p)
            z2 = norm_inv_cdf(1.0 - p)
            assert z1 == pytest.approx(-z2, abs=1e-5)

    def test_acklam_out_of_bounds_raises_validation_error(self) -> None:
        """Probabilities outside (0, 1) raise ValidationError."""
        for invalid_p in [-0.2, 0.0, 1.0, 1.05]:
            with pytest.raises(ValidationError):
                norm_inv_cdf(invalid_p)

    def test_standard_normal_pdf_and_cdf(self) -> None:
        """Verify standard normal PDF and CDF."""
        # PDF at 0: 1 / sqrt(2*pi)
        assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), rel=1e-5)
        # CDF at 0: 0.5
        assert norm_cdf(0.0) == pytest.approx(0.5, rel=1e-5)
        # Symmetry: CDF(z) + CDF(-z) = 1.0
        for z in [0.5, 1.0, 1.96, 2.58]:
            assert norm_cdf(z) + norm_cdf(-z) == pytest.approx(1.0, rel=1e-5)


# ============================================================================
# 7. Stochastic Safety Stock & Item Fill Rate
# ============================================================================


class TestSafetyStockAndFillRate:
    """Test suite for safety stock and fill rate calculations."""

    def test_safety_stock_constant_lead_time(self) -> None:
        """Verify safety stock under constant lead time: SS = z * sigma_d * sqrt(L)."""
        ss = calculate_safety_stock(
            avg_demand=20.0,
            std_demand=5.0,
            avg_lead_time=4.0,
            std_lead_time=0.0,
            service_level=0.95,
        )
        # z for 0.95 is ~1.64485; sigma_DL = 5 * sqrt(4) = 10; SS = 16.4485
        assert ss == pytest.approx(16.4485, rel=1e-3)

    def test_safety_stock_joint_stochastic_variance(self) -> None:
        """Verify safety stock with joint demand and lead-time uncertainty."""
        # avg_d = 20, std_d = 4, avg_L = 9, std_L = 1
        # sigma_DL = sqrt(9 * 16 + 400 * 1) = sqrt(144 + 400) = sqrt(544) ~ 23.3238
        # z for 0.99 is ~2.32635; SS ~ 54.260
        ss = safety_stock_joint(
            z_score=2.32635,
            avg_demand=20.0,
            std_demand=4.0,
            avg_lead_time=9.0,
            std_lead_time=1.0,
        )
        expected_sigma = math.sqrt(9.0 * 16.0 + 400.0 * 1.0)
        assert ss == pytest.approx(2.32635 * expected_sigma, rel=1e-3)

    def test_safety_stock_zero_variance_or_lead_time_returns_zero(self) -> None:
        """Zero demand variance or zero lead time produces zero safety stock."""
        assert calculate_safety_stock(daily_demand_std=0.0, lead_time_days=5.0, service_level=0.95) == 0.0
        assert calculate_safety_stock(daily_demand_std=5.0, lead_time_days=0.0, service_level=0.95) == 0.0

    def test_normal_loss_function_properties(self) -> None:
        """Verify standard normal loss function G_u(z) properties."""
        # G_u(0) = 1 / sqrt(2*pi) ~ 0.398942
        assert normal_loss_function(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), rel=1e-4)
        # G_u(1) ~ 0.0833
        assert normal_loss_function(1.0) == pytest.approx(0.0833, abs=0.005)
        # Monotonically decreasing
        assert normal_loss_function(1.0) > normal_loss_function(2.0) > normal_loss_function(3.0)

    def test_item_fill_rate_calculation(self) -> None:
        """Verify fill rate beta = 1 - (ESC / Q)."""
        fill = calculate_fill_rate(sigma_dl=10.0, lot_size=100.0, z_score=1.645)
        assert 0.99 <= fill <= 1.0
        assert fill == pytest.approx(0.9979, abs=0.002)

        # Larger lot size increases fill rate
        higher_lot_fill = calculate_fill_rate(sigma_dl=10.0, lot_size=200.0, z_score=1.645)
        assert higher_lot_fill > fill


# ============================================================================
# 8. Pareto ABC & XYZ Categorization
# ============================================================================


class TestABCXYZCategorization:
    """Test suite for ABC value, XYZ volatility, and 9-box matrix segmentation."""

    @pytest.fixture
    def test_portfolio(self) -> List[StockItem]:
        """Portfolio of SKUs with distinct annual values and volatility profiles."""
        return [
            StockItem(sku="SKU-A", on_hand=100, unit_cost=50.0, daily_demand_history=[20.0] * 30),  # High value, stable
            StockItem(sku="SKU-B", on_hand=50, unit_cost=25.0, daily_demand_history=[5.0, 6.0, 4.0, 5.0, 5.0]),  # Mid value
            StockItem(sku="SKU-C", on_hand=20, unit_cost=5.0, daily_demand_history=[0.0, 0.0, 5.0, 0.0, 0.0]),  # Low, sporadic
        ]

    def test_pareto_abc_classification(self, test_portfolio: List[StockItem]) -> None:
        """Verify items are partitioned into A, B, and C tiers."""
        abc = classify_abc(test_portfolio)
        assert abc["SKU-A"] == "A"
        assert abc["SKU-C"] == "C"

    def test_single_item_portfolio_classified_as_a(self) -> None:
        """A single item accounts for 100% of value and must be classified as 'A'."""
        single = [StockItem(sku="SOLO-1", on_hand=50, unit_cost=10.0, daily_demand_history=[5.0] * 10)]
        res = abc_classification(single)
        assert res["SOLO-1"] == "A"

    def test_xyz_volatility_classification(self) -> None:
        """Verify XYZ assignment based on Coefficient of Variation."""
        stable = StockItem(sku="STABLE", on_hand=50, daily_demand_history=[10.0, 10.0, 10.0, 10.0])  # CV = 0.0 -> X
        volatile = StockItem(sku="VOLATILE", on_hand=50, daily_demand_history=[0.0, 0.0, 20.0, 0.0, 1.0])  # CV > 1.0 -> Z
        xyz = classify_xyz([stable, volatile])
        assert xyz["STABLE"] == "X"
        assert xyz["VOLATILE"] == "Z"

    def test_nine_box_matrix_generation(self, test_portfolio: List[StockItem]) -> None:
        """Verify all 9 matrix cells exist and contain all items."""
        matrix = generate_abc_xyz_matrix(test_portfolio)
        expected_cells = {"AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"}
        assert expected_cells == set(matrix.keys())
        total_skus = sum(len(skus) for skus in matrix.values())
        assert total_skus == len(test_portfolio)


# ============================================================================
# 9. Turnover, DSI, Dead Stock, and Stockout Risk
# ============================================================================


class TestStockAnalyticsAndAlerts:
    """Test suite for financial inventory metrics and risk alerts."""

    def test_inventory_turnover_and_dsi(self) -> None:
        """Verify turnover ratio = COGS / Avg Inventory and DSI = 365 / Turnover."""
        cogs = 1200000.0
        avg_inv = 200000.0
        turnover = calculate_inventory_turnover(cogs=cogs, average_inventory=avg_inv)
        assert turnover == pytest.approx(6.0, rel=1e-5)

        dsi = calculate_days_sales_of_inventory(turnover=turnover, days=365)
        assert dsi == pytest.approx(365.0 / 6.0, rel=1e-4)

    def test_dead_stock_detection(self) -> None:
        """Verify dead stock detection flags dormant SKUs with zero consumption."""
        active = StockItem(sku="ACTIVE", on_hand=50, daily_demand_history=[10.0, 8.0, 12.0])
        dead = StockItem(sku="DEAD", on_hand=100, daily_demand_history=[0.0, 0.0, 0.0, 0.0])
        dead_list = detect_dead_stock([active, dead])
        assert "DEAD" in dead_list
        assert "ACTIVE" not in dead_list

    def test_stockout_risk_detection(self) -> None:
        """Verify detection of items with imminent stockout risk."""
        at_risk = StockItem(sku="RISK-1", on_hand=5, daily_demand_history=[10.0, 10.0])  # ~0.5 days of supply
        safe = StockItem(sku="SAFE-1", on_hand=200, daily_demand_history=[5.0, 5.0])  # 40 days of supply
        alerts = detect_stockout_risks([at_risk, safe], days_horizon=7.0)
        alert_skus = [a["sku"] for a in alerts]
        assert "RISK-1" in alert_skus
        assert "SAFE-1" not in alert_skus


# ============================================================================
# 10. ReplenishmentEventEngine & Auto-PO Workflow
# ============================================================================


class TestReplenishmentEventEngine:
    """Test suite for the reactive ReplenishmentEventEngine."""

    @pytest.fixture
    def monitor_item(self) -> StockItem:
        """Stock item for engine testing."""
        return StockItem(
            sku="SKU-MONITOR-01",
            on_hand=60,
            on_order=0,
            unit_cost=25.0,
            holding_cost_rate=0.20,
            lead_time_days=3.0,
            daily_demand_history=[8.0, 10.0, 9.0, 11.0, 12.0],
            allocated=0,
        )

    def test_evaluate_sku_status(self, monitor_item: StockItem) -> None:
        """Verify evaluate_sku generates populated ReplenishmentStatus."""
        engine = ReplenishmentEventEngine()
        status = engine.evaluate_sku(monitor_item, service_level=0.95, order_cost=50.0)

        assert isinstance(status, ReplenishmentStatus)
        assert status.sku == monitor_item.sku
        assert status.reorder_point > 0.0
        assert status.safety_stock >= 0.0
        assert status.eoq > 0.0

    def test_stock_decrement_below_rop_generates_purchase_order(self, monitor_item: StockItem) -> None:
        """Decrementing stock below ROP automatically issues a PurchaseOrder."""
        engine = ReplenishmentEventEngine()
        status = engine.evaluate_sku(monitor_item)
        rop = status.reorder_point

        # Position on_hand slightly above ROP, then decrement to breach ROP
        monitor_item.on_hand = int(math.ceil(rop)) + 2
        po = engine.process_stock_decrement(sku=monitor_item.sku, quantity=10)

        assert po is not None
        assert isinstance(po, PurchaseOrder)
        assert po.sku == monitor_item.sku
        assert po.quantity >= int(math.floor(status.eoq))
        assert po.status == "ISSUED"

    def test_on_order_deduplication(self, monitor_item: StockItem) -> None:
        """Outstanding on_order stock prevents redundant duplicate PO generation."""
        engine = ReplenishmentEventEngine()
        status = engine.evaluate_sku(monitor_item)
        monitor_item.on_hand = int(status.reorder_point) - 5

        # Initial trigger
        po1 = engine.process_stock_decrement(sku=monitor_item.sku, quantity=5)
        assert po1 is not None

        # Simulate receipt of PO into on_order pipeline
        monitor_item.on_order = po1.quantity

        # Subsequent small decrement should not trigger second PO because net inventory covers ROP
        po2 = engine.process_stock_decrement(sku=monitor_item.sku, quantity=2)
        assert po2 is None

    def test_eventbus_integration_publishes_reorder_event(self, monitor_item: StockItem) -> None:
        """Engine publishes replenishment event on EventBus when PO is generated."""
        bus = EventBus()
        captured_events: List[Event] = []

        bus.subscribe("inventory.replenishment_triggered", lambda evt: captured_events.append(evt))
        bus.subscribe("inventory.reorder_triggered", lambda evt: captured_events.append(evt))

        engine = ReplenishmentEventEngine(event_bus=bus)
        monitor_item.on_hand = 10  # Below ROP

        po = engine.process_stock_decrement(sku=monitor_item.sku, quantity=5)
        assert po is not None
        assert len(captured_events) >= 2
        assert any(e.topic == "inventory.replenishment_triggered" for e in captured_events)
        assert any(e.topic == "inventory.reorder_triggered" for e in captured_events)

    def test_stockout_error_on_excessive_decrement(self, monitor_item: StockItem) -> None:
        """Decrementing more than available on_hand raises StockoutError."""
        engine = ReplenishmentEventEngine()
        monitor_item.on_hand = 15
        with pytest.raises(StockoutError):
            engine.process_stock_decrement(sku=monitor_item.sku, quantity=20)
