"""Traveling Salesperson Problem (TSP) route construction and refinement algorithms.

Provides NearestNeighborTSP greedy tour construction and TwoOptSearch local search
route refinement optimizer with fixed-depot boundary protection.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

from open_logistics.core.models import Stop
from open_logistics.routing.distances import euclidean_distance, get_distance_calculator


class NearestNeighborTSP:
    """Nearest Neighbor greedy TSP tour constructor.

    Sequentially builds a tour by greedily selecting the nearest unvisited stop.
    If a depot stop is present, tour begins at the depot.
    """

    def __init__(
        self,
        distance_func: Optional[Callable[[Any, Any], float]] = None,
        metric: str = "euclidean",
    ) -> None:
        """Initialize NearestNeighborTSP solver.

        Args:
            distance_func: Custom distance callable (p1, p2) -> float.
            metric: Metric name ('euclidean', 'haversine', 'manhattan') if distance_func is omitted.
        """
        if distance_func is not None:
            self.distance_func = distance_func
        else:
            self.distance_func = get_distance_calculator(metric)

    def optimize(
        self,
        stops: List[Stop],
        start_index: Optional[int] = None,
    ) -> List[Stop]:
        """Construct a greedy nearest neighbor tour across all stops.

        Args:
            stops: List of stops to visit.
            start_index: Explicit index of the starting stop. If None, prefers first depot stop.

        Returns:
            Ordered list of stops representing the initial tour.
        """
        if not stops:
            return []
        if len(stops) <= 1:
            return list(stops)

        # Select start position: explicit index, first depot, or index 0
        if start_index is not None:
            start_pos = start_index % len(stops)
        else:
            depot_indices = [i for i, s in enumerate(stops) if getattr(s, "is_depot", False)]
            start_pos = depot_indices[0] if depot_indices else 0

        unvisited = list(stops)
        current = unvisited.pop(start_pos)
        tour = [current]

        while unvisited:
            best_idx = 0
            best_dist = float("inf")
            for i, candidate in enumerate(unvisited):
                d = self.distance_func(current, candidate)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            current = unvisited.pop(best_idx)
            tour.append(current)

        return tour

    def solve(
        self,
        stops: List[Stop],
        start_index: Optional[int] = None,
    ) -> List[Stop]:
        """Alias for optimize()."""
        return self.optimize(stops, start_index=start_index)


class TwoOptSearch:
    """2-Opt Local Search route refinement optimizer.

    Iteratively uncrosses intersecting edges by 2-opt edge swaps until no further
    improvement exceeding improvement_threshold is found, or max_iterations is reached.
    Automatically respects fixed depot endpoints when a route begins and ends at a depot.
    """

    def __init__(
        self,
        max_iterations: int = 500,
        distance_func: Optional[Callable[[Any, Any], float]] = None,
        metric: str = "euclidean",
        improvement_threshold: float = 1e-6,
    ) -> None:
        """Initialize TwoOptSearch optimizer.

        Args:
            max_iterations: Maximum search passes before stopping (default: 500).
            distance_func: Custom distance callable (p1, p2) -> float.
            metric: Metric name ('euclidean', 'haversine', 'manhattan') if distance_func is omitted.
            improvement_threshold: Minimum tour reduction delta required to accept a swap.
        """
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        if distance_func is not None:
            self.distance_func = distance_func
        else:
            self.distance_func = get_distance_calculator(metric)

    def optimize(
        self,
        stops: List[Stop],
        max_iterations: Optional[int] = None,
    ) -> List[Stop]:
        """Refine a route by applying 2-opt edge swaps.

        Args:
            stops: Ordered list of stops to optimize.
            max_iterations: Optional iteration limit override.

        Returns:
            Optimized list of stops with preserved or reduced total distance.
        """
        if not stops or len(stops) <= 3:
            return list(stops)

        limit = max_iterations if max_iterations is not None else self.max_iterations
        tour = list(stops)
        n = len(tour)

        # Check if route has fixed depots at both start and end (e.g. CVRP closed route)
        fixed_endpoints = (
            getattr(tour[0], "is_depot", False)
            and getattr(tour[-1], "is_depot", False)
            and n >= 4
        )

        iteration = 0
        improved = True

        if fixed_endpoints:
            # Mode A: Endpoints tour[0] and tour[n-1] remain fixed at depot
            # Intermediate customers are in slice 1 .. n-2
            while improved and iteration < limit:
                improved = False
                iteration += 1
                for i in range(1, n - 2):
                    for j in range(i + 1, n - 1):
                        p_prev = tour[i - 1]
                        p_i = tour[i]
                        p_j = tour[j]
                        p_next = tour[j + 1]

                        old_cost = self.distance_func(p_prev, p_i) + self.distance_func(p_j, p_next)
                        new_cost = self.distance_func(p_prev, p_j) + self.distance_func(p_i, p_next)

                        if new_cost < old_cost - self.improvement_threshold:
                            tour[i : j + 1] = list(reversed(tour[i : j + 1]))
                            improved = True
                            break
                    if improved:
                        break
        else:
            # Mode B: Cyclic TSP tour (tour[0] remains invariant as anchor)
            while improved and iteration < limit:
                improved = False
                iteration += 1
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if i == 0 and j == n - 1:
                            continue

                        p_i = tour[i]
                        p_i1 = tour[i + 1]
                        p_j = tour[j]
                        p_j1 = tour[(j + 1) % n]

                        old_cost = self.distance_func(p_i, p_i1) + self.distance_func(p_j, p_j1)
                        new_cost = self.distance_func(p_i, p_j) + self.distance_func(p_i1, p_j1)

                        if new_cost < old_cost - self.improvement_threshold:
                            tour[i + 1 : j + 1] = list(reversed(tour[i + 1 : j + 1]))
                            improved = True
                            break
                    if improved:
                        break

        return tour

    def solve(
        self,
        stops: List[Stop],
        max_iterations: Optional[int] = None,
    ) -> List[Stop]:
        """Alias for optimize()."""
        return self.optimize(stops, max_iterations=max_iterations)
