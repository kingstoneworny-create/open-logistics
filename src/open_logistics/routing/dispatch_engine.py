"""Automated fleet dispatch engine and priority route allocation for Open Logistics.

Coordinates multi-dimensional capacity filtering, priority allocation,
heterogeneous fleet cost minimization, 2-opt TSP optimization, VRPTW scheduling,
event bus publishing, and structured route manifest generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import Event, EventBus
from open_logistics.core.exceptions import RoutingError, ValidationError
from open_logistics.core.models import Depot, Order, Route, Stop, Vehicle
from open_logistics.routing.distances import (
    _extract_coords,
    euclidean_distance,
    get_distance_calculator,
    haversine_distance,
    manhattan_distance,
)
from open_logistics.routing.time_windows import ScheduleResult, TimeWindowScheduler
from open_logistics.routing.tsp import NearestNeighborTSP, TwoOptSearch

logger = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """Result of an automated fleet dispatch execution.

    Attributes:
        routes: List of assigned delivery routes with vehicles and stop sequences.
        unassigned_orders: Orders that could not be assigned due to capacity or fleet exhaustion.
        total_distance: Aggregated travel distance across all dispatched routes (km).
        total_cost: Aggregated operational cost (fixed costs + distance * cost_per_km).
        manifest: Human-readable / JSON-serializable structured route manifest.
    """

    routes: List[Route] = field(default_factory=list)
    unassigned_orders: List[Order] = field(default_factory=list)
    total_distance: float = 0.0
    total_cost: float = 0.0
    manifest: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dispatch result to dictionary."""
        return {
            "routes_count": len(self.routes),
            "unassigned_count": len(self.unassigned_orders),
            "total_distance": self.total_distance,
            "total_cost": self.total_cost,
            "routes": [r.to_dict() for r in self.routes],
            "unassigned_orders": [o.to_dict() for o in self.unassigned_orders],
            "manifest": self.manifest,
        }


class AutoDispatchEngine:
    """Automated priority-aware fleet dispatch and route allocation engine.

    Orchestrates:
    1. Multi-dimensional capacity filtering (weight, volume, max_stops).
    2. Priority queue ordering (express orders with priority > 1 served first).
    3. Heterogeneous fleet cost minimization (fixed_cost + distance * cost_per_km).
    4. Express-to-speed allocation (matching urgent orders to fastest vehicles).
    5. Tour optimization via Nearest Neighbor and 2-opt local search.
    6. VRPTW schedule propagation across customer time windows.
    7. EventBus integration publishing 'dispatch.completed' notifications.
    8. Manifest generation for warehouse and driver execution.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[LogisticsConfig] = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config or LogisticsConfig()

    def _resolve_distance_func(
        self,
        locations: Sequence[Any],
        metric: str,
    ) -> Callable[[Any, Any], float]:
        """Select appropriate distance callable with boundary safety checks."""
        metric_lower = metric.lower().strip()
        if metric_lower == "haversine":
            for loc in locations:
                coords = _extract_coords(loc)
                if not (-90.0 <= coords[0] <= 90.0 and -180.0 <= coords[1] <= 180.0):
                    return euclidean_distance
            return haversine_distance
        return get_distance_calculator(metric_lower)

    def dispatch(
        self,
        orders: List[Order],
        fleet: List[Vehicle],
        depot: Depot,
        config: Optional[LogisticsConfig] = None,
    ) -> DispatchResult:
        """Execute automated order dispatch across available vehicle fleet.

        Args:
            orders: List of customer orders to fulfill.
            fleet: Available vehicles.
            depot: Dispatch origin and return depot.
            config: Optional config overriding engine defaults.

        Returns:
            DispatchResult containing assigned routes, unassigned orders, totals, and manifest.
        """
        active_config = config or self.config

        # 1. Boundary: empty orders
        if not orders:
            return DispatchResult(
                routes=[],
                unassigned_orders=[],
                total_distance=0.0,
                total_cost=0.0,
                manifest={
                    "depot_id": depot.id,
                    "depot_location": depot.location,
                    "routes_count": 0,
                    "unassigned_count": 0,
                    "total_distance_km": 0.0,
                    "total_cost": 0.0,
                    "routes": [],
                    "unassigned_orders": [],
                },
            )

        # 2. Boundary: empty fleet
        if not fleet:
            return DispatchResult(
                routes=[],
                unassigned_orders=list(orders),
                total_distance=0.0,
                total_cost=0.0,
                manifest={
                    "depot_id": depot.id,
                    "depot_location": depot.location,
                    "routes_count": 0,
                    "unassigned_count": len(orders),
                    "total_distance_km": 0.0,
                    "total_cost": 0.0,
                    "routes": [],
                    "unassigned_orders": [o.id for o in orders],
                },
            )

        # 3. Resolve distance calculator
        all_locs = [depot.location] + [o.destination for o in orders]
        dist_fn = self._resolve_distance_func(all_locs, active_config.distance_metric)

        # 4. Filter individually oversized orders
        max_fleet_weight = max(v.capacity_weight for v in fleet)
        max_fleet_volume = max(v.capacity_volume for v in fleet)

        valid_orders: List[Order] = []
        unassigned_orders: List[Order] = []

        for o in orders:
            if o.weight > max_fleet_weight + 1e-6 or o.volume > max_fleet_volume + 1e-6:
                unassigned_orders.append(o)
            else:
                valid_orders.append(o)

        # 5. Partition orders into express (priority > 1) and standard (priority <= 1)
        express_orders = [o for o in valid_orders if o.priority > 1]
        standard_orders = [o for o in valid_orders if o.priority <= 1]

        # Sort priority descending, earlier time window opening first
        express_orders.sort(
            key=lambda o: (o.priority, -(o.time_window[0] if o.time_window else 0.0)),
            reverse=True,
        )
        standard_orders.sort(
            key=lambda o: (o.priority, -(o.time_window[0] if o.time_window else 0.0)),
            reverse=True,
        )

        routes: List[Route] = []
        available_vehicles = list(fleet)
        assigned_vehicles_map: Dict[str, List[Order]] = {}

        # 6. Allocate Express Orders to fastest vehicles first
        if express_orders and available_vehicles:
            # Sort vehicles by speed descending
            available_vehicles.sort(key=lambda v: v.speed_kmh, reverse=True)
            unassigned_express: List[Order] = []

            for o in express_orders:
                placed = False
                for v in available_vehicles:
                    current_orders = assigned_vehicles_map.get(v.id, [])
                    curr_weight = sum(x.weight for x in current_orders)
                    curr_volume = sum(x.volume for x in current_orders)

                    if (
                        curr_weight + o.weight <= v.capacity_weight + 1e-6
                        and curr_volume + o.volume <= v.capacity_volume + 1e-6
                        and len(current_orders) < v.max_stops
                    ):
                        if v.id not in assigned_vehicles_map:
                            assigned_vehicles_map[v.id] = []
                        assigned_vehicles_map[v.id].append(o)
                        placed = True
                        break
                if not placed:
                    unassigned_express.append(o)

            unassigned_orders.extend(unassigned_express)

        # Remove vehicles that have been assigned express orders from unassigned pool
        unassigned_vehicles = [v for v in available_vehicles if v.id not in assigned_vehicles_map]

        # 7. Allocate Standard Orders
        # Sort unassigned vehicles by cost efficiency (fixed_cost, cost_per_km, capacity)
        unassigned_vehicles.sort(key=lambda v: (v.fixed_cost, v.cost_per_km, v.capacity_weight))

        unassigned_standard: List[Order] = []
        for o in standard_orders:
            placed = False
            # First attempt: assign to available unassigned vehicles (cheapest first)
            for v in unassigned_vehicles:
                current_orders = assigned_vehicles_map.get(v.id, [])
                curr_weight = sum(x.weight for x in current_orders)
                curr_volume = sum(x.volume for x in current_orders)

                if (
                    curr_weight + o.weight <= v.capacity_weight + 1e-6
                    and curr_volume + o.volume <= v.capacity_volume + 1e-6
                    and len(current_orders) < v.max_stops
                ):
                    if v.id not in assigned_vehicles_map:
                        assigned_vehicles_map[v.id] = []
                    assigned_vehicles_map[v.id].append(o)
                    placed = True
                    break

            # Second attempt: if no unassigned vehicle can take it, pack into already-assigned vehicles with spare capacity
            if not placed:
                for v in available_vehicles:
                    if v.id in assigned_vehicles_map:
                        current_orders = assigned_vehicles_map[v.id]
                        curr_weight = sum(x.weight for x in current_orders)
                        curr_volume = sum(x.volume for x in current_orders)

                        if (
                            curr_weight + o.weight <= v.capacity_weight + 1e-6
                            and curr_volume + o.volume <= v.capacity_volume + 1e-6
                            and len(current_orders) < v.max_stops
                        ):
                            current_orders.append(o)
                            placed = True
                            break

            if not placed:
                unassigned_standard.append(o)

        unassigned_orders.extend(unassigned_standard)

        # 8. Build Route objects for all assigned vehicles
        vehicle_by_id = {v.id: v for v in fleet}

        for veh_id, veh_orders in assigned_vehicles_map.items():
            if not veh_orders:
                continue

            v = vehicle_by_id[veh_id]

            # Construct Stop objects
            customer_stops = [
                Stop(
                    location=o.destination,
                    order_id=o.id,
                    demand_weight=o.weight,
                    demand_volume=o.volume,
                    time_window=o.time_window,
                    service_time=active_config.default_service_time_minutes / 60.0,
                    is_depot=False,
                )
                for o in veh_orders
            ]

            # Order stops: Nearest neighbor starting from point closest to depot
            if len(customer_stops) > 1:
                start_idx = min(
                    range(len(customer_stops)),
                    key=lambda i: dist_fn(depot.location, customer_stops[i].location),
                )
                ordered_stops = NearestNeighborTSP(distance_func=dist_fn).optimize(
                    customer_stops, start_index=start_idx
                )
            else:
                ordered_stops = list(customer_stops)

            # Enclose with depot start and depot return
            depot_start = Stop(
                location=depot.location,
                is_depot=True,
                time_window=depot.time_window,
            )
            depot_end = Stop(
                location=depot.location,
                is_depot=True,
                time_window=depot.time_window,
            )
            route_stops = [depot_start] + ordered_stops + [depot_end]

            # Optimize with 2-opt if enabled
            if active_config.enable_2opt and len(ordered_stops) >= 2:
                route_stops = TwoOptSearch(
                    max_iterations=active_config.max_iterations_2opt,
                    distance_func=dist_fn,
                ).optimize(route_stops)

            # Calculate route metrics
            tot_dist = sum(
                dist_fn(route_stops[k].location, route_stops[k + 1].location)
                for k in range(len(route_stops) - 1)
            )
            tot_w = sum(s.demand_weight for s in route_stops)
            tot_v = sum(s.demand_volume for s in route_stops)

            route = Route(
                vehicle_id=v.id,
                stops=route_stops,
                total_load_weight=tot_w,
                total_load_volume=tot_v,
                total_distance=tot_dist,
            )

            # Propagate VRPTW schedule
            scheduler = TimeWindowScheduler(
                speed_kmh=v.speed_kmh,
                distance_func=dist_fn,
                max_route_duration_hours=active_config.max_route_duration_hours,
            )
            scheduler.propagate(route)

            routes.append(route)

        # 9. Compute aggregated distance and cost
        total_distance = sum(r.total_distance for r in routes)
        total_cost = sum(
            vehicle_by_id[r.vehicle_id].fixed_cost
            + r.total_distance * vehicle_by_id[r.vehicle_id].cost_per_km
            for r in routes
        )

        # 10. Generate structured manifest
        manifest = {
            "depot_id": depot.id,
            "depot_location": depot.location,
            "routes_count": len(routes),
            "unassigned_count": len(unassigned_orders),
            "total_distance_km": round(total_distance, 4),
            "total_cost": round(total_cost, 2),
            "routes": [
                {
                    "vehicle_id": r.vehicle_id,
                    "stops_count": len(r.stops),
                    "customer_stops_count": len([s for s in r.stops if not s.is_depot]),
                    "total_weight": r.total_load_weight,
                    "total_volume": r.total_load_volume,
                    "total_distance_km": r.total_distance,
                    "stops": [
                        {
                            "location": s.location,
                            "order_id": s.order_id,
                            "is_depot": s.is_depot,
                            "demand_weight": s.demand_weight,
                            "demand_volume": s.demand_volume,
                        }
                        for s in r.stops
                    ],
                    "schedule": [
                        s.to_dict() if hasattr(s, "to_dict") else str(s)
                        for s in r.schedule
                    ] if r.schedule else [],
                }
                for r in routes
            ],
            "unassigned_orders": [o.id for o in unassigned_orders],
        }

        result = DispatchResult(
            routes=routes,
            unassigned_orders=unassigned_orders,
            total_distance=round(total_distance, 4),
            total_cost=round(total_cost, 2),
            manifest=manifest,
        )

        # 11. Publish event on EventBus if present
        if self.event_bus:
            try:
                self.event_bus.publish(
                    Event(
                        topic="dispatch.completed",
                        payload={
                            "routes_count": len(result.routes),
                            "total_distance": result.total_distance,
                            "total_cost": result.total_cost,
                            "unassigned_count": len(result.unassigned_orders),
                            "assigned_order_ids": [
                                s.order_id
                                for r in result.routes
                                for s in r.stops
                                if s.order_id
                            ],
                            "unassigned_order_ids": [o.id for o in result.unassigned_orders],
                        },
                        source="AutoDispatchEngine",
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to publish dispatch.completed event: {exc}")

        return result
