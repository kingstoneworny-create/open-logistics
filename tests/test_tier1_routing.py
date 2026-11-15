"""Tier 1 Routing Tests: Features 5-9.

F5: Clarke-Wright CVRP Routing Heuristic (Parallel & Sequential)
F6: TSP Optimizers (Nearest Neighbor & 2-Opt Local Search)
F7: Time Windows & Schedule Propagation (VRPTW & Soft Penalty Models)
F8: Heterogeneous Fleet Dispatch & Multi-Dimensional Capacity
F9: AutoDispatchEngine Automated Workflow
"""

import math
import pytest
from typing import Any, Callable, Dict, List, Optional, Tuple

from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import Event, EventBus
from open_logistics.core.models import Depot, Order, OrderItem, Route, Stop, Vehicle

# Routing algorithms are implemented under open_logistics.routing in M2
try:
    from open_logistics.routing.clarke_wright import ClarkeWrightCVRP
    from open_logistics.routing.dispatch_engine import AutoDispatchEngine, DispatchResult
    from open_logistics.routing.distances import (
        euclidean_distance,
        haversine_distance,
        manhattan_distance,
    )
    from open_logistics.routing.time_windows import ScheduleResult, TimeWindowScheduler
    from open_logistics.routing.tsp import NearestNeighborTSP, TwoOptSearch
    HAS_ROUTING = True
except ImportError:
    HAS_ROUTING = False
    ClarkeWrightCVRP = None  # type: ignore
    NearestNeighborTSP = None  # type: ignore
    TwoOptSearch = None  # type: ignore
    TimeWindowScheduler = None  # type: ignore
    ScheduleResult = None  # type: ignore
    AutoDispatchEngine = None  # type: ignore
    DispatchResult = None  # type: ignore
    euclidean_distance = None  # type: ignore
    haversine_distance = None  # type: ignore
    manhattan_distance = None  # type: ignore

pytestmark = pytest.mark.skipif(not HAS_ROUTING, reason="open_logistics.routing modules implemented in Milestone M2")


# ==============================================================================
# Feature 5: Clarke-Wright CVRP Routing (F5)
# ==============================================================================


class TestFeature5ClarkeWrightCVRP:
    """Comprehensive test suite for Feature 5: Clarke-Wright CVRP Routing Heuristic."""

    def test_f5_clarke_wright_savings_matrix_calculation(self, sample_depot_cartesian: Depot) -> None:
        """Verify Clarke-Wright savings formula s(i,j) = d(0,i) + d(0,j) - d(i,j)."""
        cvrp = ClarkeWrightCVRP(depot=sample_depot_cartesian, vehicle_capacity=1000.0)

        # Depot at (0, 0); stop 1 at (3, 0); stop 2 at (0, 4)
        # d(0, 1) = 3; d(0, 2) = 4; d(1, 2) = sqrt(3^2 + 4^2) = 5
        # s(1, 2) = 3 + 4 - 5 = 2.0
        stop1 = Stop(location=(3.0, 0.0), order_id="ORD-1", demand_weight=100.0)
        stop2 = Stop(location=(0.0, 4.0), order_id="ORD-2", demand_weight=100.0)

        savings = cvrp.calculate_savings([stop1, stop2])
        # Expected savings between stop 1 and stop 2 is 2.0
        key = (0, 1) if (0, 1) in savings else (1, 0)
        assert math.isclose(savings[key], 2.0, abs_tol=1e-5), f"Expected saving of 2.0, got {savings.get(key)}"

    def test_f5_clarke_wright_parallel_savings_merging(self, sample_depot_cartesian: Depot) -> None:
        """Verify parallel savings merges customer stops into combined multi-stop routes."""
        cvrp = ClarkeWrightCVRP(depot=sample_depot_cartesian, vehicle_capacity=1000.0, mode="parallel")

        # 4 stops positioned in clusters
        stops = [
            Stop(location=(1.0, 1.0), order_id="ORD-1", demand_weight=100.0),
            Stop(location=(1.1, 1.1), order_id="ORD-2", demand_weight=150.0),
            Stop(location=(-5.0, -5.0), order_id="ORD-3", demand_weight=200.0),
            Stop(location=(-5.1, -5.1), order_id="ORD-4", demand_weight=250.0),
        ]

        routes = cvrp.solve(stops)
        assert len(routes) >= 1
        # Total orders served across all routes must equal 4
        served_order_ids = [s.order_id for r in routes for s in r.stops if not s.is_depot]
        assert sorted(served_order_ids) == ["ORD-1", "ORD-2", "ORD-3", "ORD-4"]

        # Close points (ORD-1, ORD-2) should ideally be clustered in the same route
        route_for_1 = next(r for r in routes if any(s.order_id == "ORD-1" for s in r.stops))
        assert any(s.order_id == "ORD-2" for s in route_for_1.stops)

    def test_f5_clarke_wright_sequential_route_generation(self, sample_depot_cartesian: Depot) -> None:
        """Verify sequential mode generates valid routes respecting vehicle capacity."""
        cvrp = ClarkeWrightCVRP(depot=sample_depot_cartesian, vehicle_capacity=500.0, mode="sequential")

        stops = [
            Stop(location=(2.0, 2.0), order_id="ORD-1", demand_weight=300.0),
            Stop(location=(2.1, 2.1), order_id="ORD-2", demand_weight=300.0),
            Stop(location=(3.0, 3.0), order_id="ORD-3", demand_weight=200.0),
        ]

        routes = cvrp.solve(stops)
        # 300 + 300 = 600 > 500, so ORD-1 and ORD-2 cannot be on the same route
        assert len(routes) >= 2
        for r in routes:
            assert r.total_load_weight <= 500.0

    def test_f5_clarke_wright_capacity_boundary_respect(self, sample_depot_cartesian: Depot) -> None:
        """Verify routes never exceed the specified vehicle weight capacity."""
        vehicle_cap = 800.0
        cvrp = ClarkeWrightCVRP(depot=sample_depot_cartesian, vehicle_capacity=vehicle_cap)

        # 4 stops of 500 kg each -> can NEVER combine 2 on one vehicle (500+500=1000 > 800)
        stops = [
            Stop(location=(1.0, 2.0), order_id=f"ORD-{i}", demand_weight=500.0)
            for i in range(1, 5)
        ]

        routes = cvrp.solve(stops)
        assert len(routes) == 4, f"Expected 4 separate routes due to capacity, got {len(routes)}"
        for r in routes:
            assert r.total_load_weight <= vehicle_cap
            customer_stops = [s for s in r.stops if not s.is_depot]
            assert len(customer_stops) == 1

    def test_f5_clarke_wright_single_vs_multi_vehicle_solutions(self, sample_depot_cartesian: Depot) -> None:
        """Verify demand under capacity produces 1 route; demand exceeding capacity creates multiple routes."""
        cvrp = ClarkeWrightCVRP(depot=sample_depot_cartesian, vehicle_capacity=1000.0)

        # Small demands that fit easily on 1 vehicle (total = 150 kg)
        light_stops = [
            Stop(location=(1.0, 0.0), order_id="L1", demand_weight=50.0),
            Stop(location=(2.0, 0.0), order_id="L2", demand_weight=50.0),
            Stop(location=(3.0, 0.0), order_id="L3", demand_weight=50.0),
        ]
        light_routes = cvrp.solve(light_stops)
        assert len(light_routes) == 1
        assert light_routes[0].total_load_weight == 150.0

        # Heavy demands requiring multiple vehicles (total = 2500 kg)
        heavy_stops = [
            Stop(location=(1.0, 0.0), order_id=f"H{i}", demand_weight=500.0)
            for i in range(1, 6)
        ]
        heavy_routes = cvrp.solve(heavy_stops)
        assert len(heavy_routes) >= 3

    def test_f5_clarke_wright_empty_and_single_stop_corner_cases(self, sample_depot_cartesian: Depot) -> None:
        """Verify boundary cases: 0 stops returns empty list; 1 stop returns single round-trip route."""
        cvrp = ClarkeWrightCVRP(depot=sample_depot_cartesian, vehicle_capacity=1000.0)

        # 0 stops
        empty_routes = cvrp.solve([])
        assert empty_routes == []

        # 1 stop
        single_stop = Stop(location=(5.0, 5.0), order_id="SOLO-1", demand_weight=120.0)
        single_routes = cvrp.solve([single_stop])
        assert len(single_routes) == 1
        route = single_routes[0]
        # Route should start at depot, visit customer, return to depot
        assert len(route.stops) >= 2
        assert any(s.order_id == "SOLO-1" for s in route.stops)
        assert route.total_load_weight == 120.0


# ==============================================================================
# Feature 6: TSP Optimizers (F6)
# ==============================================================================


class TestFeature6TSPOptimizers:
    """Comprehensive test suite for Feature 6: Nearest Neighbor TSP & 2-Opt Local Search."""

    def test_f6_nearest_neighbor_greedy_tour_validity(self) -> None:
        """Verify NearestNeighborTSP visits all stops exactly once without duplication."""
        nn_solver = NearestNeighborTSP()
        stops = [
            Stop(location=(0.0, 0.0), order_id="DEPOT", is_depot=True),
            Stop(location=(1.0, 0.0), order_id="S1"),
            Stop(location=(3.0, 0.0), order_id="S2"),
            Stop(location=(2.0, 0.0), order_id="S3"),
        ]

        tour = nn_solver.optimize(stops)
        assert len(tour) == len(stops)
        # All stop order_ids must be present
        visited_ids = [s.order_id for s in tour]
        assert set(visited_ids) == {"DEPOT", "S1", "S2", "S3"}

    def test_f6_two_opt_tour_edge_swapping_improvement(self) -> None:
        """Verify 2-opt uncrosses intersecting edges and improves or preserves tour length."""
        two_opt = TwoOptSearch(max_iterations=100)

        # Sub-optimal "hourglass" / self-crossing polygon:
        # A(0, 0) -> D(1, 1) -> B(0, 1) -> C(1, 0) has crossed diagonals
        crossed_tour = [
            Stop(location=(0.0, 0.0), order_id="A"),
            Stop(location=(1.0, 1.0), order_id="D"),
            Stop(location=(0.0, 1.0), order_id="B"),
            Stop(location=(1.0, 0.0), order_id="C"),
        ]

        def tour_distance(tour: List[Stop]) -> float:
            d = 0.0
            for i in range(len(tour)):
                p1 = tour[i].location
                p2 = tour[(i + 1) % len(tour)].location
                d += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            return d

        initial_distance = tour_distance(crossed_tour)
        optimized_tour = two_opt.optimize(crossed_tour)
        optimized_distance = tour_distance(optimized_tour)

        # 2-opt must uncross the edges and yield a strictly shorter perimeter
        assert optimized_distance <= initial_distance
        # The uncrossed square (0,0)->(1,0)->(1,1)->(0,1) perimeter is 4.0; hourglass is 2 + 2*sqrt(2) ~ 4.828
        assert math.isclose(optimized_distance, 4.0, abs_tol=0.1)

    def test_f6_two_opt_preserves_all_original_stops(self) -> None:
        """Verify 2-opt never adds, drops, or duplicates stops during tour swapping."""
        two_opt = TwoOptSearch(max_iterations=50)
        stops = [Stop(location=(float(i), float(i % 3)), order_id=f"ST-{i}") for i in range(8)]

        optimized = two_opt.optimize(stops)
        assert len(optimized) == len(stops)
        assert set(s.order_id for s in optimized) == set(s.order_id for s in stops)

    def test_f6_triangular_inequality_in_optimized_tours(self) -> None:
        """Verify optimized tour distances conform to Euclidean metric properties."""
        nn_solver = NearestNeighborTSP()
        two_opt = TwoOptSearch()

        stops = [
            Stop(location=(0.0, 0.0), order_id="S0"),
            Stop(location=(4.0, 0.0), order_id="S1"),
            Stop(location=(4.0, 3.0), order_id="S2"),
            Stop(location=(0.0, 3.0), order_id="S3"),
        ]

        tour = two_opt.optimize(nn_solver.optimize(stops))
        assert len(tour) == 4

        # Triangle inequality: dist(i, i+2) <= dist(i, i+1) + dist(i+1, i+2)
        for i in range(len(tour)):
            p0 = tour[i].location
            p1 = tour[(i + 1) % 4].location
            p2 = tour[(i + 2) % 4].location
            d01 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            d12 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            d02 = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
            assert d02 <= d01 + d12 + 1e-7

    def test_f6_three_stop_vs_ten_stop_tours(self) -> None:
        """Verify optimizer handles both trivial 3-stop tours and complex 10-stop tours."""
        two_opt = TwoOptSearch()

        # 3 stops (triangle is trivially optimal)
        three_stops = [
            Stop(location=(0.0, 0.0), order_id="T1"),
            Stop(location=(3.0, 0.0), order_id="T2"),
            Stop(location=(0.0, 4.0), order_id="T3"),
        ]
        opt_three = two_opt.optimize(three_stops)
        assert len(opt_three) == 3

        # 10 stops randomly distributed along a circle
        ten_stops = [
            Stop(
                location=(10.0 * math.cos(2 * math.pi * i / 10), 10.0 * math.sin(2 * math.pi * i / 10)),
                order_id=f"CIRC-{i}",
            )
            for i in [0, 5, 2, 8, 1, 9, 3, 7, 4, 6]  # Deliberately scrambled order
        ]
        opt_ten = two_opt.optimize(ten_stops)
        assert len(opt_ten) == 10
        # Perimeter of 10-gon inscribed in radius 10 circle is ~61.8
        # Initial scrambled path is significantly longer (> 100)
        d_opt = sum(
            math.hypot(
                opt_ten[(i + 1) % 10].location[0] - opt_ten[i].location[0],
                opt_ten[(i + 1) % 10].location[1] - opt_ten[i].location[1],
            )
            for i in range(10)
        )
        assert d_opt <= 65.0, f"Expected circle tour ~61.8, got {d_opt}"

    def test_f6_two_opt_max_iterations_and_convergence(self) -> None:
        """Verify 2-opt terminates cleanly within max_iterations without infinite looping."""
        two_opt = TwoOptSearch(max_iterations=10)
        stops = [Stop(location=(float(i * 2), float(i * 3)), order_id=f"I-{i}") for i in range(15)]

        result = two_opt.optimize(stops)
        assert len(result) == 15


# ==============================================================================
# Feature 7: Time Windows & Schedule Propagation (F7)
# ==============================================================================


class TestFeature7TimeWindowsAndSchedules:
    """Comprehensive test suite for Feature 7: VRPTW Feasibility & Schedules."""

    def test_f7_schedule_arrival_and_wait_time_calculation(self) -> None:
        """Verify wait time calculation when vehicle arrives before customer earliest window."""
        scheduler = TimeWindowScheduler(speed_kmh=60.0)

        # Stop distance = 60 km from depot -> arrival time = 1.0 hr (60/60)
        # Customer window opens at 2.5 hr -> vehicle must wait 1.5 hr
        # Service time = 0.5 hr -> departure time = 2.5 + 0.5 = 3.0 hr
        depot_stop = Stop(location=(0.0, 0.0), is_depot=True)
        customer_stop = Stop(
            location=(60.0, 0.0),
            order_id="ORD-WAIT",
            time_window=(2.5, 5.0),
            service_time=0.5,
        )
        route = Route(vehicle_id="V1", stops=[depot_stop, customer_stop])

        result: ScheduleResult = scheduler.propagate(route)
        assert result.is_feasible is True
        assert len(result.schedules) == 2

        cust_sched = result.schedules[1]
        assert math.isclose(cust_sched.arrival_time, 1.0, abs_tol=1e-3)
        assert math.isclose(cust_sched.wait_time, 1.5, abs_tol=1e-3)
        assert math.isclose(cust_sched.departure_time, 3.0, abs_tol=1e-3)

    def test_f7_schedule_departure_propagation_along_route(self) -> None:
        """Verify schedule propagation across multiple successive customer stops."""
        scheduler = TimeWindowScheduler(speed_kmh=100.0)

        # Stop 0: Depot at (0, 0), departure at t=0
        # Stop 1: (100, 0) -> 100km / 100kmh = 1 hr travel -> arrive t=1.0, window [0.5, 2.0], service 0.5 -> depart t=1.5
        # Stop 2: (200, 0) -> 100km travel -> arrive t=2.5, window [2.0, 4.0], service 0.5 -> depart t=3.0
        stops = [
            Stop(location=(0.0, 0.0), is_depot=True),
            Stop(location=(100.0, 0.0), order_id="S1", time_window=(0.5, 2.0), service_time=0.5),
            Stop(location=(200.0, 0.0), order_id="S2", time_window=(2.0, 4.0), service_time=0.5),
        ]
        route = Route(vehicle_id="V1", stops=stops)

        result = scheduler.propagate(route)
        assert result.is_feasible is True
        assert math.isclose(result.schedules[1].departure_time, 1.5, abs_tol=1e-3)
        assert math.isclose(result.schedules[2].arrival_time, 2.5, abs_tol=1e-3)
        assert math.isclose(result.schedules[2].departure_time, 3.0, abs_tol=1e-3)

    def test_f7_time_window_violation_detection(self) -> None:
        """Verify detection of arrival after latest customer delivery window."""
        scheduler = TimeWindowScheduler(speed_kmh=50.0)

        # Stop is 100 km away -> arrival at 2.0 hr
        # Customer window strictly closes at 1.0 hr -> late by 1.0 hr
        depot = Stop(location=(0.0, 0.0), is_depot=True)
        late_stop = Stop(
            location=(100.0, 0.0),
            order_id="ORD-LATE",
            time_window=(0.5, 1.0),
            service_time=0.25,
        )
        route = Route(vehicle_id="V1", stops=[depot, late_stop])

        result = scheduler.propagate(route)
        assert result.is_feasible is False
        assert result.total_delay > 0.0
        assert math.isclose(result.total_delay, 1.0, abs_tol=1e-3)

    def test_f7_soft_time_window_penalty_cost_computation(self) -> None:
        """Verify soft time window penalty cost proportional to lateness duration."""
        penalty_rate_per_hour = 50.0
        scheduler = TimeWindowScheduler(
            speed_kmh=50.0,
            penalty_cost_per_late_hour=penalty_rate_per_hour,
        )

        # Arrive at t=2.0 hr; window [0.0, 1.5] -> 0.5 hours late
        # Expected penalty = 0.5 * $50 = $25.0
        depot = Stop(location=(0.0, 0.0), is_depot=True)
        stop = Stop(location=(100.0, 0.0), order_id="ORD-SOFT", time_window=(0.0, 1.5))
        route = Route(vehicle_id="V1", stops=[depot, stop])

        result = scheduler.propagate(route)
        assert math.isclose(result.penalty_cost, 25.0, abs_tol=1e-2)

        # On-time stop produces 0 penalty
        on_time_stop = Stop(location=(100.0, 0.0), order_id="ORD-OK", time_window=(0.0, 3.0))
        route_ok = Route(vehicle_id="V1", stops=[depot, on_time_stop])
        result_ok = scheduler.propagate(route_ok)
        assert result_ok.penalty_cost == 0.0

    def test_f7_feasible_vs_infeasible_schedule_classification(self) -> None:
        """Verify scheduler accurately marks routes as feasible vs infeasible."""
        scheduler = TimeWindowScheduler(speed_kmh=60.0)

        depot = Stop(location=(0.0, 0.0), is_depot=True)
        # Feasible stop (arrive at t=1.0, window [0.0, 2.0])
        s_good = Stop(location=(60.0, 0.0), order_id="GOOD", time_window=(0.0, 2.0))
        # Infeasible stop (arrive at t=2.0, window [0.0, 1.2])
        s_bad = Stop(location=(120.0, 0.0), order_id="BAD", time_window=(0.0, 1.2))

        assert scheduler.propagate(Route(vehicle_id="V1", stops=[depot, s_good])).is_feasible is True
        assert scheduler.propagate(Route(vehicle_id="V1", stops=[depot, s_bad])).is_feasible is False

    def test_f7_zero_service_time_boundary(self) -> None:
        """Verify zero service time maintains departure equal to arrival (or arrival + wait)."""
        scheduler = TimeWindowScheduler(speed_kmh=50.0)
        depot = Stop(location=(0.0, 0.0), is_depot=True)
        stop = Stop(location=(50.0, 0.0), order_id="INSTANT", time_window=(0.0, 5.0), service_time=0.0)

        result = scheduler.propagate(Route(vehicle_id="V1", stops=[depot, stop]))
        cust_sched = result.schedules[1]
        assert cust_sched.arrival_time == cust_sched.departure_time == 1.0


# ==============================================================================
# Feature 8: Fleet Dispatch & Multi-Dimensional Capacity (F8)
# ==============================================================================


class TestFeature8FleetDispatchAndMultiDimCapacity:
    """Comprehensive test suite for Feature 8: Fleet Dispatch & Multi-Dimensional Capacity."""

    def test_f8_simultaneous_weight_and_volume_capacity_feasibility(
        self,
        sample_depot_cartesian: Depot,
    ) -> None:
        """Verify vehicle rejects cargo when weight is within limit but volume is exceeded."""
        vehicle = Vehicle(id="V1", capacity_weight=1000.0, capacity_volume=5.0)
        engine = AutoDispatchEngine()

        # Order 1: 300 kg, 3.0 m^3
        # Order 2: 300 kg, 3.5 m^3
        # Combined weight: 600 kg <= 1000 kg (FEASIBLE)
        # Combined volume: 6.5 m^3 > 5.0 m^3 (EXCEEDED)
        orders = [
            Order(id="ORD-V1", destination=(1.0, 1.0), weight=300.0, volume=3.0),
            Order(id="ORD-V2", destination=(1.1, 1.1), weight=300.0, volume=3.5),
        ]

        result = engine.dispatch(orders=orders, fleet=[vehicle], depot=sample_depot_cartesian)
        # Since only one vehicle is available and they can't fit together, at least 1 order must be unassigned
        # or separate routes must be required if multiple trips allowed
        total_assigned_orders = sum(len([s for s in r.stops if not s.is_depot]) for r in result.routes)
        assert len(result.unassigned_orders) >= 1 or len(result.routes) >= 2

    def test_f8_vehicle_selection_by_cost_efficiency(
        self,
        sample_depot_cartesian: Depot,
        sample_vehicle_van: Vehicle,
        sample_vehicle_semi: Vehicle,
    ) -> None:
        """Verify small order is dispatched with cost-effective van rather than expensive semi."""
        engine = AutoDispatchEngine()
        small_order = Order(id="ORD-SMALL", destination=(10.0, 0.0), weight=200.0, volume=1.0)

        # Fleet has both Van ($50 fixed, $1.2/km) and Semi ($250 fixed, $3.5/km)
        fleet = [sample_vehicle_semi, sample_vehicle_van]  # semi first in list to test smart selection

        result = engine.dispatch(orders=[small_order], fleet=fleet, depot=sample_depot_cartesian)
        assert len(result.routes) == 1
        dispatched_vehicle_id = result.routes[0].vehicle_id
        assert dispatched_vehicle_id == sample_vehicle_van.id, f"Expected Van, got {dispatched_vehicle_id}"

    def test_f8_priority_order_dispatch_assignment(
        self,
        sample_depot_cartesian: Depot,
    ) -> None:
        """Verify Priority 3 (urgent) orders are allocated before Priority 1 orders when capacity is tight."""
        # Vehicle can only carry 100 kg
        single_van = Vehicle(id="TINY-VAN", capacity_weight=100.0, capacity_volume=10.0)
        engine = AutoDispatchEngine()

        orders = [
            Order(id="ORD-LOW", destination=(5.0, 5.0), weight=80.0, priority=1),
            Order(id="ORD-HIGH", destination=(5.0, 5.0), weight=80.0, priority=3),
        ]

        result = engine.dispatch(orders=orders, fleet=[single_van], depot=sample_depot_cartesian)
        assert len(result.routes) == 1
        served_order_ids = [s.order_id for s in result.routes[0].stops if not s.is_depot]
        assert "ORD-HIGH" in served_order_ids
        assert any(o.id == "ORD-LOW" for o in result.unassigned_orders)

    def test_f8_unassigned_order_handling_when_capacity_exhausted(
        self,
        sample_depot_cartesian: Depot,
        sample_vehicle_van: Vehicle,
    ) -> None:
        """Verify orders exceeding total fleet capacity are returned in unassigned_orders."""
        engine = AutoDispatchEngine()

        # Van capacity = 1000 kg. Total orders = 2500 kg
        orders = [
            Order(id=f"HEAVY-{i}", destination=(float(i), 1.0), weight=500.0)
            for i in range(5)
        ]

        result = engine.dispatch(orders=orders, fleet=[sample_vehicle_van], depot=sample_depot_cartesian)
        assigned_count = sum(len([s for s in r.stops if not s.is_depot]) for r in result.routes)
        unassigned_count = len(result.unassigned_orders)

        assert assigned_count + unassigned_count == 5
        assert unassigned_count >= 3  # At least 3 orders cannot fit in 1000 kg van

    def test_f8_heterogeneous_fleet_multi_vehicle_load_balancing(
        self,
        sample_depot_cartesian: Depot,
        sample_heterogeneous_fleet: List[Vehicle],
    ) -> None:
        """Verify allocation across diverse vehicle types respects individual vehicle limits."""
        engine = AutoDispatchEngine()

        # Batch of orders totaling ~4500 kg (requires multiple vehicles)
        orders = [
            Order(id=f"O-{i}", destination=(float(i * 2), float(i)), weight=450.0, volume=1.5)
            for i in range(10)
        ]

        result = engine.dispatch(orders=orders, fleet=sample_heterogeneous_fleet, depot=sample_depot_cartesian)
        assert len(result.routes) >= 2

        # Verify no route exceeds its vehicle's capacity
        vehicle_map = {v.id: v for v in sample_heterogeneous_fleet}
        for r in result.routes:
            veh = vehicle_map[r.vehicle_id]
            assert r.total_load_weight <= veh.capacity_weight + 1e-5
            assert r.total_load_volume <= veh.capacity_volume + 1e-5

    def test_f8_vehicle_max_stops_constraint_respect(
        self,
        sample_depot_cartesian: Depot,
    ) -> None:
        """Verify vehicle max_stops constraint is respected even when weight capacity remains."""
        constrained_vehicle = Vehicle(
            id="CONSTRAINED",
            capacity_weight=10000.0,
            max_stops=3,  # Maximum 3 stops allowed
        )
        engine = AutoDispatchEngine()

        # 6 small orders of 10 kg each
        orders = [
            Order(id=f"STOP-{i}", destination=(float(i), 0.0), weight=10.0)
            for i in range(6)
        ]

        result = engine.dispatch(orders=orders, fleet=[constrained_vehicle], depot=sample_depot_cartesian)
        for r in result.routes:
            customer_stops = [s for s in r.stops if not s.is_depot]
            assert len(customer_stops) <= 3


# ==============================================================================
# Feature 9: AutoDispatchEngine Workflow (F9)
# ==============================================================================


class TestFeature9AutoDispatchEngine:
    """Comprehensive test suite for Feature 9: AutoDispatchEngine Workflow."""

    def test_f9_autodispatch_end_to_end_workflow(
        self,
        sample_orders_batch: List[Order],
        sample_heterogeneous_fleet: List[Vehicle],
        sample_depot_geo: Depot,
    ) -> None:
        """Verify end-to-end dispatch method produces complete DispatchResult."""
        engine = AutoDispatchEngine()
        result: DispatchResult = engine.dispatch(
            orders=sample_orders_batch,
            fleet=sample_heterogeneous_fleet,
            depot=sample_depot_geo,
        )

        assert isinstance(result, DispatchResult)
        assert isinstance(result.routes, list)
        assert isinstance(result.unassigned_orders, list)
        assert result.total_distance >= 0.0
        assert result.total_cost >= 0.0

        # All orders accounted for
        assigned_ids = {s.order_id for r in result.routes for s in r.stops if not s.is_depot}
        unassigned_ids = {o.id for o in result.unassigned_orders}
        all_order_ids = {o.id for o in sample_orders_batch}
        assert (assigned_ids | unassigned_ids) == all_order_ids
        assert len(assigned_ids & unassigned_ids) == 0

    def test_f9_autodispatch_event_bus_publishing(
        self,
        event_bus: EventBus,
        sample_orders_batch: List[Order],
        sample_vehicle_van: Vehicle,
        sample_depot_geo: Depot,
    ) -> None:
        """Verify dispatch completion triggers event publication on EventBus."""
        published_events: List[Event] = []
        event_bus.subscribe("dispatch.*", lambda e: published_events.append(e))

        engine = AutoDispatchEngine(event_bus=event_bus)
        engine.dispatch(
            orders=sample_orders_batch[:2],
            fleet=[sample_vehicle_van],
            depot=sample_depot_geo,
        )

        assert len(published_events) >= 1
        evt = published_events[0]
        assert "dispatch" in evt.topic
        assert "routes_count" in evt.payload or "total_distance" in evt.payload

    def test_f9_route_manifest_generation_structure(
        self,
        sample_order: Order,
        sample_vehicle_van: Vehicle,
        sample_depot_geo: Depot,
    ) -> None:
        """Verify generated route manifest has valid structure: depot start, customer, depot end."""
        engine = AutoDispatchEngine()
        result = engine.dispatch(
            orders=[sample_order],
            fleet=[sample_vehicle_van],
            depot=sample_depot_geo,
        )

        assert len(result.routes) == 1
        route = result.routes[0]
        assert route.vehicle_id == sample_vehicle_van.id
        assert len(route.stops) >= 2

        # First and last stop should be depot
        assert route.stops[0].is_depot is True
        assert route.stops[0].location == sample_depot_geo.location

        # Middle stop is customer
        cust_stop = route.stops[1]
        assert cust_stop.order_id == sample_order.id
        assert cust_stop.is_depot is False

    def test_f9_empty_fleet_dispatch_handling(
        self,
        sample_orders_batch: List[Order],
        sample_depot_geo: Depot,
    ) -> None:
        """Verify empty fleet causes all orders to be placed in unassigned_orders with 0 routes."""
        engine = AutoDispatchEngine()
        result = engine.dispatch(orders=sample_orders_batch, fleet=[], depot=sample_depot_geo)

        assert len(result.routes) == 0
        assert len(result.unassigned_orders) == len(sample_orders_batch)
        assert result.total_distance == 0.0
        assert result.total_cost == 0.0

    def test_f9_empty_orders_dispatch_handling(
        self,
        sample_heterogeneous_fleet: List[Vehicle],
        sample_depot_geo: Depot,
    ) -> None:
        """Verify empty orders list returns 0 routes and 0 unassigned orders."""
        engine = AutoDispatchEngine()
        result = engine.dispatch(orders=[], fleet=sample_heterogeneous_fleet, depot=sample_depot_geo)

        assert len(result.routes) == 0
        assert len(result.unassigned_orders) == 0
        assert result.total_distance == 0.0
        assert result.total_cost == 0.0

    def test_f9_single_order_dispatch_workflow(
        self,
        sample_order: Order,
        sample_vehicle_van: Vehicle,
        sample_depot_geo: Depot,
    ) -> None:
        """Verify dispatch with 1 order and 1 vehicle creates exactly 1 completed route."""
        engine = AutoDispatchEngine()
        result = engine.dispatch(orders=[sample_order], fleet=[sample_vehicle_van], depot=sample_depot_geo)

        assert len(result.routes) == 1
        assert len(result.unassigned_orders) == 0
        route = result.routes[0]
        assert route.total_load_weight == sample_order.weight
        assert route.total_distance > 0.0
