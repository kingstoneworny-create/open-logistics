"""Comprehensive Unit Test Suite for Milestone M2 Routing and Fleet Dispatch.

Covers:
1. Distance Calculators: Euclidean, Manhattan, Haversine accuracy, bounds, and symmetry.
2. TSP Optimizers: Nearest Neighbor greedy tour, 2-Opt edge swapping, convergence.
3. Clarke-Wright CVRP: Savings calculation, parallel/sequential heuristics, dual capacity.
4. VRPTW Scheduling: Arrival/wait/departure propagation, violations, soft penalties.
5. AutoDispatchEngine: Priority allocation, heterogeneous fleet matching, overflow.
6. Public Exports: Routing module symbols.
"""

from __future__ import annotations

import math
import pytest
from typing import Any, Dict, List

from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import Event, EventBus
from open_logistics.core.exceptions import RoutingError, ValidationError
from open_logistics.core.models import Depot, Location, Order, OrderItem, Route, Stop, Vehicle
from open_logistics.routing import (
    AutoDispatchEngine,
    ClarkeWrightCVRP,
    DispatchResult,
    NearestNeighborTSP,
    ScheduleResult,
    StopSchedule,
    TimeWindowScheduler,
    TwoOptSearch,
    calculate_distance_matrix,
    distance_matrix,
    euclidean_distance,
    get_distance_calculator,
    haversine_distance,
    manhattan_distance,
    validate_coordinates,
)


# ==============================================================================
# 1. Distance Calculators Tests
# ==============================================================================


class TestDistanceCalculators:
    """Tests for Euclidean, Manhattan, and Haversine distance functions."""

    def test_euclidean_distance_accuracy(self) -> None:
        """Verify exact Euclidean distance matches Pythagorean theorem."""
        # Standard 3-4-5 right triangle
        assert math.isclose(euclidean_distance((0.0, 0.0), (3.0, 4.0)), 5.0, abs_tol=1e-7)
        # Negative coordinate displacement: sqrt((2 - (-1))^2 + (2 - (-2))^2) = 5.0
        assert math.isclose(euclidean_distance((-1.0, -2.0), (2.0, 2.0)), 5.0, abs_tol=1e-7)
        # 1D line along x axis
        assert math.isclose(euclidean_distance((10.0, 5.0), (20.0, 5.0)), 10.0, abs_tol=1e-7)

    def test_euclidean_distance_polymorphic_inputs(self) -> None:
        """Verify coordinate extraction from tuples, lists, Location, Stop, Order, and Depot."""
        loc = Location(lat=3.0, lon=4.0)
        depot = Depot(id="DEPOT-1", location=(0.0, 0.0))
        stop = Stop(location=(3.0, 4.0), order_id="ORD-1")
        order = Order(id="ORD-1", destination=(3.0, 4.0), weight=10.0)

        assert math.isclose(euclidean_distance(depot, loc), 5.0, abs_tol=1e-7)
        assert math.isclose(euclidean_distance(depot, stop), 5.0, abs_tol=1e-7)
        assert math.isclose(euclidean_distance(depot, order), 5.0, abs_tol=1e-7)
        assert math.isclose(euclidean_distance([0.0, 0.0], [3.0, 4.0]), 5.0, abs_tol=1e-7)

    def test_manhattan_distance_accuracy(self) -> None:
        """Verify Manhattan distance matches L1 norm |dx| + |dy|."""
        assert math.isclose(manhattan_distance((0.0, 0.0), (3.0, 4.0)), 7.0, abs_tol=1e-7)
        assert math.isclose(manhattan_distance((-5.0, 10.0), (5.0, -10.0)), 30.0, abs_tol=1e-7)
        assert math.isclose(manhattan_distance((12.5, -3.2), (12.5, -3.2)), 0.0, abs_tol=1e-7)

    def test_haversine_distance_accuracy(self) -> None:
        """Verify spherical Haversine distance against known intercity geodetic distances."""
        # NYC (40.7128, -74.0060) to London (51.5074, -0.1278) ~5570 km
        d_nyc_lon = haversine_distance((40.7128, -74.0060), (51.5074, -0.1278))
        assert 5550.0 <= d_nyc_lon <= 5590.0

        # Paris (48.8566, 2.3522) to Berlin (52.5200, 13.4050) ~878 km
        d_par_ber = haversine_distance((48.8566, 2.3522), (52.5200, 13.4050))
        assert 870.0 <= d_par_ber <= 885.0

        # Tokyo (35.6762, 139.6503) to Sydney (-33.8688, 151.2093) ~7825 km
        d_tyo_syd = haversine_distance((35.6762, 139.6503), (-33.8688, 151.2093))
        assert 7800.0 <= d_tyo_syd <= 7850.0

    def test_distance_symmetry(self) -> None:
        """Verify dist(A, B) == dist(B, A) across all calculators."""
        p1 = (41.8781, -87.6298)  # Chicago
        p2 = (34.0522, -118.2437)  # Los Angeles

        assert math.isclose(euclidean_distance(p1, p2), euclidean_distance(p2, p1), rel_tol=1e-9)
        assert math.isclose(manhattan_distance(p1, p2), manhattan_distance(p2, p1), rel_tol=1e-9)
        assert math.isclose(haversine_distance(p1, p2), haversine_distance(p2, p1), rel_tol=1e-9)

    def test_distance_triangle_inequality(self) -> None:
        """Verify triangle inequality dist(A, C) <= dist(A, B) + dist(B, C)."""
        p_a = (40.7128, -74.0060)
        p_b = (41.8781, -87.6298)
        p_c = (34.0522, -118.2437)

        assert euclidean_distance(p_a, p_c) <= euclidean_distance(p_a, p_b) + euclidean_distance(p_b, p_c) + 1e-9
        assert manhattan_distance(p_a, p_c) <= manhattan_distance(p_a, p_b) + manhattan_distance(p_b, p_c) + 1e-9
        assert haversine_distance(p_a, p_c) <= haversine_distance(p_a, p_b) + haversine_distance(p_b, p_c) + 1e-9

    def test_identical_coordinates_zero_distance(self) -> None:
        """Verify distance between coincident coordinates is exactly 0.0."""
        p = (52.5200, 13.4050)
        assert euclidean_distance(p, p) == 0.0
        assert manhattan_distance(p, p) == 0.0
        assert haversine_distance(p, p) == 0.0

    def test_coordinate_validation_boundary_limits(self) -> None:
        """Verify coordinate validation enforces lat in [-90, 90] and lon in [-180, 180]."""
        # Valid extremes
        validate_coordinates(90.0, 180.0)
        validate_coordinates(-90.0, -180.0)

        # Invalid latitudes
        with pytest.raises(ValidationError):
            validate_coordinates(90.001, 0.0)
        with pytest.raises(ValidationError):
            validate_coordinates(-90.001, 0.0)

        # Invalid longitudes
        with pytest.raises(ValidationError):
            validate_coordinates(0.0, 180.001)
        with pytest.raises(ValidationError):
            validate_coordinates(0.0, -180.001)

        # Non-numeric or NaN/Inf
        with pytest.raises(ValidationError):
            validate_coordinates(float("nan"), 0.0)
        with pytest.raises(ValidationError):
            validate_coordinates(0.0, float("inf"))

    def test_haversine_out_of_bounds_raises_validation_error(self) -> None:
        """Verify haversine_distance raises ValidationError on out-of-bounds spherical coordinates."""
        with pytest.raises(ValidationError):
            haversine_distance((95.0, 0.0), (0.0, 0.0))
        with pytest.raises(ValidationError):
            haversine_distance((0.0, 0.0), (0.0, 200.0))

    def test_invalid_coordinate_extraction_types(self) -> None:
        """Verify coordinate extraction rejects non-coordinate types and improper sequences."""
        with pytest.raises(ValidationError):
            euclidean_distance((1.0,), (2.0, 3.0))
        with pytest.raises(ValidationError):
            euclidean_distance((1.0, 2.0, 3.0), (2.0, 3.0))
        with pytest.raises(ValidationError):
            euclidean_distance("invalid", (2.0, 3.0))
        with pytest.raises(ValidationError):
            euclidean_distance(("a", "b"), (2.0, 3.0))

    def test_get_distance_calculator_valid_and_invalid(self) -> None:
        """Verify metric lookup by name and error on unsupported metric."""
        assert get_distance_calculator("euclidean") == euclidean_distance
        assert get_distance_calculator("manhattan") == manhattan_distance
        assert get_distance_calculator("haversine") == haversine_distance
        assert get_distance_calculator("  HAVERSINE  ") == haversine_distance

        with pytest.raises(ValidationError, match="Unknown distance metric"):
            get_distance_calculator("chebyshev")

    def test_calculate_distance_matrix_and_alias(self) -> None:
        """Verify pairwise distance matrix calculation and symmetry."""
        pts = [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]
        mat = calculate_distance_matrix(pts, metric="euclidean")

        assert len(mat) == 3
        assert mat[0][0] == 0.0
        assert mat[1][1] == 0.0
        assert mat[2][2] == 0.0
        assert math.isclose(mat[0][1], 3.0)
        assert math.isclose(mat[1][0], 3.0)
        assert math.isclose(mat[0][2], 4.0)
        assert math.isclose(mat[1][2], 5.0)

        # Alias verification
        assert distance_matrix == calculate_distance_matrix
        assert calculate_distance_matrix([]) == []


# ==============================================================================
# 2. TSP Optimizers Tests
# ==============================================================================


class TestTSPOptimizers:
    """Tests for NearestNeighborTSP and TwoOptSearch."""

    def test_nearest_neighbor_tour_completeness(self) -> None:
        """Verify NearestNeighborTSP visits all stops without loss or duplication."""
        stops = [
            Stop(location=(0.0, 0.0), order_id="S0"),
            Stop(location=(10.0, 0.0), order_id="S1"),
            Stop(location=(5.0, 5.0), order_id="S2"),
            Stop(location=(1.0, 1.0), order_id="S3"),
        ]
        solver = NearestNeighborTSP(metric="euclidean")
        tour = solver.optimize(stops)

        assert len(tour) == 4
        assert {s.order_id for s in tour} == {"S0", "S1", "S2", "S3"}

    def test_nearest_neighbor_prefers_depot_start(self) -> None:
        """Verify NearestNeighborTSP starts at depot stop if present."""
        depot = Stop(location=(0.0, 0.0), is_depot=True)
        s1 = Stop(location=(10.0, 0.0), order_id="S1")
        s2 = Stop(location=(1.0, 0.0), order_id="S2")

        # Depot placed at the end of input list
        solver = NearestNeighborTSP(metric="euclidean")
        tour = solver.optimize([s1, s2, depot])

        assert tour[0].is_depot is True
        assert tour[1].order_id == "S2"  # S2 at (1, 0) is closer to depot (0, 0) than S1 at (10, 0)
        assert tour[2].order_id == "S1"

    def test_nearest_neighbor_explicit_start_index(self) -> None:
        """Verify start_index overrides depot preference."""
        s0 = Stop(location=(0.0, 0.0), order_id="S0")
        s1 = Stop(location=(10.0, 0.0), order_id="S1")
        solver = NearestNeighborTSP()
        tour = solver.optimize([s0, s1], start_index=1)
        assert tour[0].order_id == "S1"

    def test_nearest_neighbor_empty_and_small_stops(self) -> None:
        """Verify edge cases for 0, 1, and 2 stops."""
        solver = NearestNeighborTSP()
        assert solver.optimize([]) == []
        s1 = Stop(location=(0.0, 0.0), order_id="S1")
        assert len(solver.optimize([s1])) == 1
        assert solver.solve([s1]) == [s1]

    def test_two_opt_edge_swapping_hourglass_improvement(self) -> None:
        """Verify 2-opt uncrosses intersecting edges on an hourglass polygon."""
        # Square with crossed diagonals: A(0,0) -> D(1,1) -> B(0,1) -> C(1,0)
        # Perimeter = sqrt(2) + 1 + sqrt(2) + 1 = 2 + 2*sqrt(2) ~ 4.828
        stops = [
            Stop(location=(0.0, 0.0), order_id="A"),
            Stop(location=(1.0, 1.0), order_id="D"),
            Stop(location=(0.0, 1.0), order_id="B"),
            Stop(location=(1.0, 0.0), order_id="C"),
        ]
        two_opt = TwoOptSearch(metric="euclidean")
        optimized = two_opt.optimize(stops)

        # Compute perimeter of optimized tour
        n = len(optimized)
        opt_dist = sum(
            euclidean_distance(optimized[i].location, optimized[(i + 1) % n].location)
            for i in range(n)
        )
        # Uncrossed perimeter must be 4.0
        assert math.isclose(opt_dist, 4.0, abs_tol=1e-3)
        assert len(optimized) == 4
        assert {s.order_id for s in optimized} == {"A", "B", "C", "D"}

    def test_two_opt_fixed_depot_boundary_protection(self) -> None:
        """Verify fixed depot endpoints (index 0 and -1) remain invariant during 2-opt."""
        depot_start = Stop(location=(0.0, 0.0), is_depot=True)
        depot_end = Stop(location=(0.0, 0.0), is_depot=True)
        # Scrambled customer stops
        c1 = Stop(location=(0.0, 1.0), order_id="C1")
        c2 = Stop(location=(1.0, 1.0), order_id="C2")
        c3 = Stop(location=(1.0, 0.0), order_id="C3")

        # Initial order with crossed path
        closed_route = [depot_start, c2, c1, c3, depot_end]
        two_opt = TwoOptSearch(metric="euclidean")
        refined = two_opt.optimize(closed_route)

        # Depot endpoints must remain strictly invariant
        assert refined[0] is depot_start
        assert refined[-1] is depot_end
        assert {s.order_id for s in refined[1:-1]} == {"C1", "C2", "C3"}

    def test_two_opt_max_iterations_bound(self) -> None:
        """Verify TwoOptSearch respects max_iterations limit."""
        stops = [Stop(location=(float(i), float(i % 3)), order_id=f"S{i}") for i in range(12)]
        two_opt = TwoOptSearch(max_iterations=2)
        refined = two_opt.optimize(stops)
        assert len(refined) == 12

    def test_two_opt_small_inputs_and_alias(self) -> None:
        """Verify <= 3 stops are returned directly and solve() alias works."""
        two_opt = TwoOptSearch()
        assert two_opt.optimize([]) == []
        s1 = Stop(location=(0.0, 0.0))
        s2 = Stop(location=(1.0, 1.0))
        assert two_opt.optimize([s1, s2]) == [s1, s2]
        assert two_opt.solve([s1, s2]) == [s1, s2]


# ==============================================================================
# 3. Clarke-Wright CVRP Tests
# ==============================================================================


class TestClarkeWrightCVRP:
    """Tests for ClarkeWrightCVRP parallel and sequential heuristics."""

    def test_savings_matrix_formula(self) -> None:
        """Verify savings formula s(i, j) = d(0, i) + d(0, j) - d(i, j)."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        s1 = Stop(location=(3.0, 0.0), order_id="S1")
        s2 = Stop(location=(0.0, 4.0), order_id="S2")

        cvrp = ClarkeWrightCVRP(depot=depot, vehicle_capacity=1000.0)
        savings = cvrp.calculate_savings([s1, s2])

        # d(0, 1) = 3.0, d(0, 2) = 4.0, d(1, 2) = 5.0 -> s = 3 + 4 - 5 = 2.0
        assert math.isclose(savings[(0, 1)], 2.0, abs_tol=1e-5)
        assert math.isclose(savings[(1, 0)], 2.0, abs_tol=1e-5)

    def test_parallel_route_merging_clusters(self) -> None:
        """Verify parallel mode merges nearby stops into multi-stop routes."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        stops = [
            Stop(location=(1.0, 1.0), order_id="ORD-1", demand_weight=100.0),
            Stop(location=(1.1, 1.1), order_id="ORD-2", demand_weight=150.0),
            Stop(location=(-5.0, -5.0), order_id="ORD-3", demand_weight=200.0),
            Stop(location=(-5.1, -5.1), order_id="ORD-4", demand_weight=250.0),
        ]
        cvrp = ClarkeWrightCVRP(depot=depot, vehicle_capacity=1000.0, mode="parallel")
        routes = cvrp.solve(stops)

        # Should produce valid routes serving all 4 orders
        served_ids = [s.order_id for r in routes for s in r.stops if not s.is_depot]
        assert sorted(served_ids) == ["ORD-1", "ORD-2", "ORD-3", "ORD-4"]

        # Close points ORD-1 and ORD-2 should be clustered
        r1 = next(r for r in routes if any(s.order_id == "ORD-1" for s in r.stops))
        assert any(s.order_id == "ORD-2" for s in r1.stops)

    def test_sequential_route_generation(self) -> None:
        """Verify sequential mode creates capacity-respecting routes."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        stops = [
            Stop(location=(2.0, 2.0), order_id="O1", demand_weight=300.0),
            Stop(location=(2.1, 2.1), order_id="O2", demand_weight=300.0),
            Stop(location=(3.0, 3.0), order_id="O3", demand_weight=200.0),
        ]
        cvrp = ClarkeWrightCVRP(depot=depot, vehicle_capacity=500.0, mode="sequential")
        routes = cvrp.solve(stops)

        assert len(routes) >= 2
        for r in routes:
            assert r.total_load_weight <= 500.0 + 1e-6

    def test_dual_capacity_constraints(self) -> None:
        """Verify CVRP respects both weight and volume constraints."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        stops = [
            Stop(location=(1.0, 1.0), order_id="O1", demand_weight=100.0, demand_volume=5.0),
            Stop(location=(1.1, 1.1), order_id="O2", demand_weight=100.0, demand_volume=6.0),
        ]
        # Weight capacity is generous (1000 kg), but volume capacity is tight (8.0 m^3)
        cvrp = ClarkeWrightCVRP(
            depot=depot,
            capacity_weight=1000.0,
            capacity_volume=8.0,
        )
        routes = cvrp.solve(stops)
        # Combined volume 11.0 > 8.0 -> must split into 2 routes
        assert len(routes) == 2

    def test_max_stops_constraint(self) -> None:
        """Verify CVRP respects max_stops limit."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        stops = [
            Stop(location=(float(i), 0.0), order_id=f"O{i}", demand_weight=10.0)
            for i in range(5)
        ]
        cvrp = ClarkeWrightCVRP(depot=depot, vehicle_capacity=1000.0, max_stops=2)
        routes = cvrp.solve(stops)

        for r in routes:
            cust_stops = [s for s in r.stops if not s.is_depot]
            assert len(cust_stops) <= 2

    def test_cvrp_vehicle_dataclass_input(self) -> None:
        """Verify CVRP can be configured from a Vehicle dataclass instance."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        veh = Vehicle(id="VAN-X", capacity_weight=500.0, capacity_volume=10.0, max_stops=4)
        cvrp = ClarkeWrightCVRP(depot=depot, vehicle=veh)

        assert cvrp.capacity_weight == 500.0
        assert cvrp.capacity_volume == 10.0
        assert cvrp.max_stops == 4
        assert cvrp.vehicle_id == "VAN-X"

    def test_cvrp_invalid_inputs(self) -> None:
        """Verify validation errors for improper CVRP initialization."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        with pytest.raises(ValidationError):
            ClarkeWrightCVRP(depot="not-a-depot")  # type: ignore
        with pytest.raises(ValidationError):
            ClarkeWrightCVRP(depot=depot, capacity_weight=-100.0)
        with pytest.raises(ValidationError):
            ClarkeWrightCVRP(depot=depot, mode="invalid-mode")

    def test_cvrp_empty_and_single_stop(self) -> None:
        """Verify CVRP handles empty and single stop edge cases."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        cvrp = ClarkeWrightCVRP(depot=depot, vehicle_capacity=1000.0)

        assert cvrp.solve([]) == []

        s1 = Stop(location=(1.0, 1.0), order_id="SOLO", demand_weight=50.0)
        routes = cvrp.solve([s1])
        assert len(routes) == 1
        assert len(routes[0].stops) == 3
        assert routes[0].stops[0].is_depot is True
        assert routes[0].stops[1].order_id == "SOLO"
        assert routes[0].stops[2].is_depot is True


# ==============================================================================
# 4. VRPTW Scheduling Tests
# ==============================================================================


class TestTimeWindowScheduler:
    """Tests for VRPTW timeline propagation, wait times, and penalty calculations."""

    def test_arrival_and_wait_time_propagation(self) -> None:
        """Verify arrival and dwell wait time when vehicle arrives before window opens."""
        scheduler = TimeWindowScheduler(speed_kmh=60.0)
        depot_stop = Stop(location=(0.0, 0.0), is_depot=True)
        customer_stop = Stop(
            location=(60.0, 0.0),
            order_id="WAIT",
            time_window=(2.5, 5.0),
            service_time=0.5,
        )
        route = Route(vehicle_id="V1", stops=[depot_stop, customer_stop])
        result = scheduler.propagate(route)

        assert result.is_feasible is True
        assert len(result.schedules) == 2
        cust_sched = result.schedules[1]
        assert math.isclose(cust_sched.arrival_time, 1.0, abs_tol=1e-3)
        assert math.isclose(cust_sched.wait_time, 1.5, abs_tol=1e-3)
        assert math.isclose(cust_sched.service_start_time, 2.5, abs_tol=1e-3)
        assert math.isclose(cust_sched.departure_time, 3.0, abs_tol=1e-3)

    def test_departure_propagation_multi_stop(self) -> None:
        """Verify successive stops propagate departure times correctly."""
        scheduler = TimeWindowScheduler(speed_kmh=100.0)
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

    def test_lateness_and_soft_penalty_cost(self) -> None:
        """Verify late arrival triggers penalty cost proportional to lateness duration."""
        scheduler = TimeWindowScheduler(speed_kmh=50.0, penalty_cost_per_late_hour=40.0)
        depot = Stop(location=(0.0, 0.0), is_depot=True)
        # 100 km away -> arrival at t=2.0 hr; window [0.0, 1.0] -> 1.0 hr late
        late_stop = Stop(location=(100.0, 0.0), order_id="LATE", time_window=(0.0, 1.0))
        route = Route(vehicle_id="V1", stops=[depot, late_stop])
        result = scheduler.propagate(route)

        assert result.is_feasible is False
        assert math.isclose(result.total_delay, 1.0, abs_tol=1e-3)
        assert math.isclose(result.penalty_cost, 40.0, abs_tol=1e-3)

    def test_zero_service_time_boundary(self) -> None:
        """Verify zero service time maintains departure equal to arrival."""
        scheduler = TimeWindowScheduler(speed_kmh=50.0)
        depot = Stop(location=(0.0, 0.0), is_depot=True)
        stop = Stop(location=(50.0, 0.0), order_id="INSTANT", time_window=(0.0, 5.0), service_time=0.0)
        result = scheduler.propagate(Route(vehicle_id="V1", stops=[depot, stop]))

        sched = result.schedules[1]
        assert sched.arrival_time == sched.departure_time == 1.0

    def test_evaluate_stop_single_api(self) -> None:
        """Verify direct evaluate_stop API produces correct StopSchedule."""
        scheduler = TimeWindowScheduler(penalty_cost_per_late_hour=20.0)
        # Arrive at 3.0, window [1.0, 2.0] -> late by 1.0 hr, penalty $20.0
        sched = scheduler.evaluate_stop(arrival_time=3.0, time_window=(1.0, 2.0), service_time=0.25)
        assert sched.lateness == 1.0
        assert sched.is_feasible is False
        assert sched.penalty_cost == 20.0
        assert sched.departure_time == 3.25

    def test_invalid_scheduler_inputs(self) -> None:
        """Verify scheduler rejects invalid speed or negative penalties."""
        with pytest.raises(ValidationError):
            TimeWindowScheduler(speed_kmh=0.0)
        with pytest.raises(ValidationError):
            TimeWindowScheduler(speed_kmh=-10.0)
        with pytest.raises(ValidationError):
            TimeWindowScheduler(penalty_cost_per_late_hour=-5.0)

    def test_schedule_serialization_roundtrip(self) -> None:
        """Verify StopSchedule and ScheduleResult serialization."""
        sched = StopSchedule(
            arrival_time=1.0,
            wait_time=0.5,
            service_start_time=1.5,
            departure_time=2.0,
            service_time=0.5,
            order_id="ORD-1",
        )
        d = sched.to_dict()
        restored = StopSchedule.from_dict(d)
        assert restored.arrival_time == 1.0
        assert restored.order_id == "ORD-1"

        res = ScheduleResult(schedules=[sched], total_delay=0.0, penalty_cost=0.0)
        assert len(res.to_dict()["schedules"]) == 1


# ==============================================================================
# 5. AutoDispatchEngine Tests
# ==============================================================================


class TestAutoDispatchEngine:
    """Tests for AutoDispatchEngine workflow and fleet matching."""

    def test_dispatch_end_to_end(self, sample_orders_batch: List[Order], sample_heterogeneous_fleet: List[Vehicle], sample_depot_geo: Depot) -> None:
        """Verify complete dispatch workflow produces valid DispatchResult."""
        engine = AutoDispatchEngine()
        result = engine.dispatch(orders=sample_orders_batch, fleet=sample_heterogeneous_fleet, depot=sample_depot_geo)

        assert isinstance(result, DispatchResult)
        assert result.total_distance >= 0.0
        assert result.total_cost >= 0.0
        assigned_ids = {s.order_id for r in result.routes for s in r.stops if not s.is_depot}
        unassigned_ids = {o.id for o in result.unassigned_orders}
        assert (assigned_ids | unassigned_ids) == {o.id for o in sample_orders_batch}

    def test_express_orders_priority_allocation(self) -> None:
        """Verify Priority 3 orders are assigned ahead of Priority 1 orders under tight capacity."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        single_van = Vehicle(id="VAN", capacity_weight=100.0)
        engine = AutoDispatchEngine()

        orders = [
            Order(id="ORD-LOW", destination=(5.0, 5.0), weight=80.0, priority=1),
            Order(id="ORD-HIGH", destination=(5.0, 5.0), weight=80.0, priority=3),
        ]
        result = engine.dispatch(orders=orders, fleet=[single_van], depot=depot)

        assert len(result.routes) == 1
        served = [s.order_id for s in result.routes[0].stops if not s.is_depot]
        assert "ORD-HIGH" in served
        assert any(o.id == "ORD-LOW" for o in result.unassigned_orders)

    def test_express_orders_fastest_vehicle_matching(self) -> None:
        """Verify high priority orders are assigned to fastest vehicle."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        fleet = [
            Vehicle(id="FAST", capacity_weight=200.0, speed_kmh=100.0),
            Vehicle(id="SLOW", capacity_weight=500.0, speed_kmh=40.0),
        ]
        orders = [
            Order(id="EXPRESS-1", destination=(1.0, 1.0), weight=50.0, priority=3),
            Order(id="STD-1", destination=(2.0, 2.0), weight=50.0, priority=1),
        ]
        engine = AutoDispatchEngine()
        result = engine.dispatch(orders=orders, fleet=fleet, depot=depot)

        fast_route = next((r for r in result.routes if r.vehicle_id == "FAST"), None)
        assert fast_route is not None
        fast_orders = [s.order_id for s in fast_route.stops if s.order_id]
        assert "EXPRESS-1" in fast_orders

    def test_heterogeneous_fleet_cost_minimization(self) -> None:
        """Verify cheaper vehicle is chosen over expensive vehicle for small load."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        van = Vehicle(id="CHEAP-VAN", capacity_weight=1000.0, fixed_cost=50.0, cost_per_km=1.0)
        semi = Vehicle(id="EXPENSIVE-SEMI", capacity_weight=15000.0, fixed_cost=300.0, cost_per_km=4.0)

        order = Order(id="SMALL", destination=(10.0, 0.0), weight=100.0)
        engine = AutoDispatchEngine()
        result = engine.dispatch(orders=[order], fleet=[semi, van], depot=depot)

        assert len(result.routes) == 1
        assert result.routes[0].vehicle_id == "CHEAP-VAN"

    def test_volume_capacity_rejection(self) -> None:
        """Verify orders exceeding volume capacity are not packed together."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        van = Vehicle(id="VAN", capacity_weight=1000.0, capacity_volume=5.0)

        orders = [
            Order(id="O1", destination=(1.0, 1.0), weight=100.0, volume=3.0),
            Order(id="O2", destination=(1.1, 1.1), weight=100.0, volume=3.0),
        ]
        engine = AutoDispatchEngine()
        result = engine.dispatch(orders=orders, fleet=[van], depot=depot)

        # Combined volume 6.0 > 5.0 -> 1 order must be unassigned
        assert len(result.routes) == 1
        assert len(result.unassigned_orders) == 1

    def test_empty_fleet_and_empty_orders(self) -> None:
        """Verify graceful handling of empty fleet and empty orders list."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        engine = AutoDispatchEngine()

        # Empty orders
        res1 = engine.dispatch(orders=[], fleet=[Vehicle(id="V1", capacity_weight=1000.0)], depot=depot)
        assert len(res1.routes) == 0
        assert len(res1.unassigned_orders) == 0

        # Empty fleet
        order = Order(id="O1", destination=(1.0, 1.0), weight=10.0)
        res2 = engine.dispatch(orders=[order], fleet=[], depot=depot)
        assert len(res2.routes) == 0
        assert len(res2.unassigned_orders) == 1

    def test_event_bus_publishing(self, event_bus: EventBus) -> None:
        """Verify dispatch engine publishes dispatch.completed event."""
        events: List[Event] = []
        event_bus.subscribe("dispatch.*", lambda e: events.append(e))

        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        van = Vehicle(id="V1", capacity_weight=1000.0)
        order = Order(id="O1", destination=(1.0, 1.0), weight=10.0)

        engine = AutoDispatchEngine(event_bus=event_bus)
        engine.dispatch(orders=[order], fleet=[van], depot=depot)

        assert len(events) >= 1
        assert events[0].topic == "dispatch.completed"
        assert "routes_count" in events[0].payload

    def test_manifest_structure(self) -> None:
        """Verify manifest dictionary matches expected schema."""
        depot = Depot(id="DEPOT", location=(0.0, 0.0))
        van = Vehicle(id="V1", capacity_weight=1000.0)
        order = Order(id="O1", destination=(1.0, 1.0), weight=10.0)

        engine = AutoDispatchEngine()
        res = engine.dispatch(orders=[order], fleet=[van], depot=depot)

        manifest = res.manifest
        assert manifest["depot_id"] == "DEPOT"
        assert manifest["routes_count"] == 1
        assert len(manifest["routes"]) == 1
        assert manifest["routes"][0]["vehicle_id"] == "V1"


# ==============================================================================
# 6. Public Exports Tests
# ==============================================================================


class TestRoutingPublicExports:
    """Tests verifying public exports in open_logistics.routing."""

    def test_public_exports_all_symbols(self) -> None:
        """Verify open_logistics.routing exports all required classes and functions."""
        import open_logistics.routing as routing

        expected_symbols = [
            "euclidean_distance",
            "manhattan_distance",
            "haversine_distance",
            "distance_matrix",
            "calculate_distance_matrix",
            "validate_coordinates",
            "get_distance_calculator",
            "ClarkeWrightCVRP",
            "NearestNeighborTSP",
            "TwoOptSearch",
            "TimeWindowScheduler",
            "ScheduleResult",
            "StopSchedule",
            "AutoDispatchEngine",
            "DispatchResult",
        ]
        for sym in expected_symbols:
            assert hasattr(routing, sym), f"Missing public export: {sym}"
            assert sym in routing.__all__, f"{sym} not in __all__"
