"""Clarke-Wright Savings Heuristic for CVRP (Parallel and Sequential).

Implements the classical Clarke-Wright savings algorithm for capacitated vehicle
routing problems with multi-dimensional capacity constraints (weight and volume)
and vehicle max_stops limits.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from open_logistics.core.exceptions import RoutingError, ValidationError
from open_logistics.core.models import Depot, Route, Stop, Vehicle
from open_logistics.routing.distances import (
    euclidean_distance,
    haversine_distance,
    manhattan_distance,
)


class ClarkeWrightCVRP:
    """Clarke-Wright savings heuristic solver for Capacitated Vehicle Routing.

    Attributes:
        depot: Distribution center Depot model.
        capacity_weight: Maximum cargo weight limit.
        capacity_volume: Maximum cargo volume limit.
        max_stops: Maximum number of customer stops per route.
        vehicle_id: Vehicle identifier assigned to generated routes.
        mode: Route generation algorithm mode ('parallel' or 'sequential').
        distance_func: Callable calculating distance between two coordinates.
    """

    def __init__(
        self,
        depot: Depot,
        vehicle_capacity: Optional[float] = None,
        capacity_weight: Optional[float] = None,
        capacity_volume: Optional[float] = None,
        max_stops: Optional[int] = None,
        vehicle: Optional[Vehicle] = None,
        vehicle_id: Optional[str] = None,
        mode: str = "parallel",
        distance_func: Optional[Callable[[Any, Any], float]] = None,
        distance_metric: str = "euclidean",
    ) -> None:
        if not isinstance(depot, Depot):
            raise ValidationError(f"depot must be an instance of Depot, got {type(depot)}")
        self.depot = depot

        # Resolve vehicle capacity and constraints
        if vehicle is not None:
            self.capacity_weight = float(vehicle.capacity_weight)
            self.capacity_volume = float(vehicle.capacity_volume)
            self.max_stops = int(vehicle.max_stops)
            self.vehicle_id = str(vehicle.id)
        else:
            weight_cap = vehicle_capacity if vehicle_capacity is not None else capacity_weight
            self.capacity_weight = float(weight_cap if weight_cap is not None else 1e9)
            self.capacity_volume = float(capacity_volume if capacity_volume is not None else 1e9)
            self.max_stops = int(max_stops if max_stops is not None else 50)
            self.vehicle_id = str(vehicle_id if vehicle_id is not None else "V1")

        if self.capacity_weight <= 0:
            raise ValidationError(f"capacity_weight must be positive, got {self.capacity_weight}")
        if self.capacity_volume <= 0:
            raise ValidationError(f"capacity_volume must be positive, got {self.capacity_volume}")
        if self.max_stops <= 0:
            raise ValidationError(f"max_stops must be positive, got {self.max_stops}")

        mode_clean = mode.lower().strip()
        if mode_clean not in ("parallel", "sequential"):
            raise ValidationError(f"mode must be 'parallel' or 'sequential', got '{mode}'")
        self.mode = mode_clean

        # Resolve distance calculator
        if distance_func is not None:
            self.distance_func = distance_func
        else:
            metric = distance_metric.lower().strip()
            if metric == "haversine":
                self.distance_func = haversine_distance
            elif metric == "manhattan":
                self.distance_func = manhattan_distance
            else:
                self.distance_func = euclidean_distance

    def calculate_savings(self, stops: List[Stop]) -> Dict[Tuple[int, int], float]:
        """Compute Clarke-Wright savings matrix: s(i, j) = d(0, i) + d(0, j) - d(i, j).

        Args:
            stops: List of Stop instances (depots are filtered out).

        Returns:
            Dictionary mapping stop index pairs (i, j) to savings values.
        """
        customer_stops = [s for s in stops if not getattr(s, "is_depot", False)]
        n = len(customer_stops)
        savings: Dict[Tuple[int, int], float] = {}
        depot_loc = self.depot.location

        for i in range(n):
            d_0_i = self.distance_func(depot_loc, customer_stops[i].location)
            for j in range(i + 1, n):
                d_0_j = self.distance_func(depot_loc, customer_stops[j].location)
                d_i_j = self.distance_func(customer_stops[i].location, customer_stops[j].location)
                s = d_0_i + d_0_j - d_i_j
                savings[(i, j)] = s
                savings[(j, i)] = s

        return savings

    def solve(self, stops: List[Stop]) -> List[Route]:
        """Solve CVRP and return list of feasible Route objects.

        Args:
            stops: List of stops to service.

        Returns:
            List of generated Route objects.
        """
        customer_stops = [s for s in stops if not getattr(s, "is_depot", False)]
        if not customer_stops:
            return []

        if len(customer_stops) == 1:
            return [self._build_single_stop_route(customer_stops[0])]

        if self.mode == "parallel":
            return self._solve_parallel(customer_stops)
        else:
            return self._solve_sequential(customer_stops)

    def _build_single_stop_route(self, stop: Stop) -> Route:
        depot_start = Stop(location=self.depot.location, is_depot=True)
        depot_end = Stop(location=self.depot.location, is_depot=True)
        route_stops = [depot_start, stop, depot_end]
        dist = (
            self.distance_func(depot_start.location, stop.location)
            + self.distance_func(stop.location, depot_end.location)
        )
        return Route(
            vehicle_id=self.vehicle_id,
            stops=route_stops,
            total_load_weight=stop.demand_weight,
            total_load_volume=stop.demand_volume,
            total_distance=dist,
        )

    def _solve_parallel(self, customer_stops: List[Stop]) -> List[Route]:
        n = len(customer_stops)
        depot_loc = self.depot.location

        # Compute savings and sort descending
        savings_list: List[Tuple[float, int, int]] = []
        for i in range(n):
            d_0_i = self.distance_func(depot_loc, customer_stops[i].location)
            for j in range(i + 1, n):
                d_0_j = self.distance_func(depot_loc, customer_stops[j].location)
                d_i_j = self.distance_func(customer_stops[i].location, customer_stops[j].location)
                s = d_0_i + d_0_j - d_i_j
                savings_list.append((s, i, j))

        savings_list.sort(key=lambda x: x[0], reverse=True)

        # Initial single-customer routes
        routes: Dict[int, List[int]] = {i: [i] for i in range(n)}
        node_to_route: Dict[int, int] = {i: i for i in range(n)}
        route_weight: Dict[int, float] = {i: customer_stops[i].demand_weight for i in range(n)}
        route_volume: Dict[int, float] = {i: customer_stops[i].demand_volume for i in range(n)}

        for s, i, j in savings_list:
            if s <= 0:
                break  # Further pairs produce no positive savings

            r_i = node_to_route[i]
            r_j = node_to_route[j]
            if r_i == r_j:
                continue

            route_a = routes[r_i]
            route_b = routes[r_j]

            # Endpoint checks: both i and j must be exterior endpoints of their respective routes
            i_is_first = (i == route_a[0])
            i_is_last = (i == route_a[-1])
            if not (i_is_first or i_is_last):
                continue

            j_is_first = (j == route_b[0])
            j_is_last = (j == route_b[-1])
            if not (j_is_first or j_is_last):
                continue

            # Multi-dimensional capacity checks
            merged_weight = route_weight[r_i] + route_weight[r_j]
            merged_volume = route_volume[r_i] + route_volume[r_j]
            merged_stops = len(route_a) + len(route_b)

            if merged_weight > self.capacity_weight + 1e-6:
                continue
            if merged_volume > self.capacity_volume + 1e-6:
                continue
            if merged_stops > self.max_stops:
                continue

            # Determine merge orientation
            if i_is_last and j_is_first:
                merged = route_a + route_b
            elif i_is_last and j_is_last:
                merged = route_a + list(reversed(route_b))
            elif i_is_first and j_is_first:
                merged = list(reversed(route_a)) + route_b
            elif i_is_first and j_is_last:
                merged = route_b + route_a
            else:
                continue

            routes[r_i] = merged
            route_weight[r_i] = merged_weight
            route_volume[r_i] = merged_volume

            del routes[r_j]
            del route_weight[r_j]
            del route_volume[r_j]

            for node in route_b:
                node_to_route[node] = r_i

        # Construct Route domain objects
        result_routes: List[Route] = []
        for r_id, node_indices in routes.items():
            route_stops = [Stop(location=self.depot.location, is_depot=True)]
            for idx in node_indices:
                route_stops.append(customer_stops[idx])
            route_stops.append(Stop(location=self.depot.location, is_depot=True))

            tot_dist = sum(
                self.distance_func(route_stops[m].location, route_stops[m + 1].location)
                for m in range(len(route_stops) - 1)
            )

            result_routes.append(
                Route(
                    vehicle_id=self.vehicle_id,
                    stops=route_stops,
                    total_load_weight=route_weight[r_id],
                    total_load_volume=route_volume[r_id],
                    total_distance=tot_dist,
                )
            )

        return result_routes

    def _solve_sequential(self, customer_stops: List[Stop]) -> List[Route]:
        n = len(customer_stops)
        depot_loc = self.depot.location

        # Savings matrix lookup
        savings: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            d_0_i = self.distance_func(depot_loc, customer_stops[i].location)
            for j in range(n):
                if i != j:
                    d_0_j = self.distance_func(depot_loc, customer_stops[j].location)
                    d_i_j = self.distance_func(customer_stops[i].location, customer_stops[j].location)
                    savings[(i, j)] = d_0_i + d_0_j - d_i_j

        unrouted: Set[int] = set(range(n))
        routes_nodes: List[List[int]] = []

        while unrouted:
            # Find best initial pair among unrouted nodes
            best_pair: Optional[Tuple[int, int]] = None
            best_sav = 0.0

            for i in unrouted:
                for j in unrouted:
                    if i >= j:
                        continue
                    if (
                        customer_stops[i].demand_weight + customer_stops[j].demand_weight <= self.capacity_weight + 1e-6
                        and customer_stops[i].demand_volume + customer_stops[j].demand_volume <= self.capacity_volume + 1e-6
                        and self.max_stops >= 2
                    ):
                        s = savings.get((i, j), 0.0)
                        if s > best_sav:
                            best_sav = s
                            best_pair = (i, j)

            if best_pair is None:
                # No positive savings pair fits; create single-stop route
                solo = min(unrouted)
                routes_nodes.append([solo])
                unrouted.remove(solo)
                continue

            i, j = best_pair
            current_route = [i, j]
            unrouted.remove(i)
            unrouted.remove(j)
            curr_weight = customer_stops[i].demand_weight + customer_stops[j].demand_weight
            curr_volume = customer_stops[i].demand_volume + customer_stops[j].demand_volume

            # Greedily expand current route
            while unrouted and len(current_route) < self.max_stops:
                best_cand: Optional[int] = None
                best_cand_pos: Optional[str] = None
                best_cand_sav = 0.0

                head = current_route[0]
                tail = current_route[-1]

                for k in unrouted:
                    st = customer_stops[k]
                    if (
                        curr_weight + st.demand_weight > self.capacity_weight + 1e-6
                        or curr_volume + st.demand_volume > self.capacity_volume + 1e-6
                    ):
                        continue

                    s_front = savings.get((k, head), 0.0)
                    if s_front > best_cand_sav:
                        best_cand_sav = s_front
                        best_cand = k
                        best_cand_pos = "head"

                    s_back = savings.get((tail, k), 0.0)
                    if s_back > best_cand_sav:
                        best_cand_sav = s_back
                        best_cand = k
                        best_cand_pos = "tail"

                if best_cand is not None and best_cand_sav > 0.0:
                    if best_cand_pos == "head":
                        current_route.insert(0, best_cand)
                    else:
                        current_route.append(best_cand)
                    curr_weight += customer_stops[best_cand].demand_weight
                    curr_volume += customer_stops[best_cand].demand_volume
                    unrouted.remove(best_cand)
                else:
                    break

            routes_nodes.append(current_route)

        # Build Route objects
        result_routes: List[Route] = []
        for node_indices in routes_nodes:
            route_stops = [Stop(location=self.depot.location, is_depot=True)]
            w = 0.0
            v = 0.0
            for idx in node_indices:
                st = customer_stops[idx]
                route_stops.append(st)
                w += st.demand_weight
                v += st.demand_volume
            route_stops.append(Stop(location=self.depot.location, is_depot=True))

            tot_dist = sum(
                self.distance_func(route_stops[m].location, route_stops[m + 1].location)
                for m in range(len(route_stops) - 1)
            )

            result_routes.append(
                Route(
                    vehicle_id=self.vehicle_id,
                    stops=route_stops,
                    total_load_weight=w,
                    total_load_volume=v,
                    total_distance=tot_dist,
                )
            )

        return result_routes
