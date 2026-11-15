"""VRPTW Schedule Propagation & Time Window Penalty Models.

Provides timeline simulation, early wait time, tardiness penalties,
and feasibility checking for vehicle routing with customer time windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from open_logistics.core.exceptions import RoutingError, ValidationError
from open_logistics.core.models import Route, Stop
from open_logistics.routing.distances import (
    euclidean_distance,
    haversine_distance,
    manhattan_distance,
)


@dataclass
class StopSchedule:
    """Detailed timing entry for an individual route stop.

    Attributes:
        arrival_time: Clock time (hours) of vehicle arrival.
        wait_time: Dwell wait time (hours) if arriving before window opening.
        service_start_time: Clock time (hours) when unloading/service begins.
        departure_time: Clock time (hours) of vehicle departure.
        service_time: Service duration (hours).
        lateness: Hours arrived after latest allowed window (tardiness).
        is_feasible: True if arrival_time <= latest customer window, False if late.
        penalty_cost: Financial penalty cost accrued ($).
        time_window: Customer delivery window (earliest, latest).
        stop: Associated Stop model instance.
        order_id: Customer order ID (or None for depot).
    """

    arrival_time: float
    wait_time: float
    service_start_time: float
    departure_time: float
    service_time: float = 0.0
    lateness: float = 0.0
    is_feasible: bool = True
    penalty_cost: float = 0.0
    time_window: Optional[Tuple[float, float]] = None
    stop: Optional[Stop] = None
    order_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stop schedule to dictionary."""
        return {
            "arrival_time": self.arrival_time,
            "wait_time": self.wait_time,
            "service_start_time": self.service_start_time,
            "departure_time": self.departure_time,
            "service_time": self.service_time,
            "lateness": self.lateness,
            "is_feasible": self.is_feasible,
            "penalty_cost": self.penalty_cost,
            "time_window": self.time_window,
            "order_id": self.order_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StopSchedule:
        """Instantiate StopSchedule from dictionary."""
        return cls(
            arrival_time=data["arrival_time"],
            wait_time=data["wait_time"],
            service_start_time=data.get(
                "service_start_time", data["arrival_time"] + data["wait_time"]
            ),
            departure_time=data["departure_time"],
            service_time=data.get("service_time", 0.0),
            lateness=data.get("lateness", 0.0),
            is_feasible=data.get("is_feasible", True),
            penalty_cost=data.get("penalty_cost", 0.0),
            time_window=tuple(data["time_window"]) if data.get("time_window") else None,  # type: ignore[arg-type]
            order_id=data.get("order_id"),
        )


@dataclass
class ScheduleResult:
    """Aggregated route schedule propagation result across all stops.

    Attributes:
        schedules: Ordered list of StopSchedule entries for each stop.
        is_feasible: True if all stops satisfy time windows without tardiness.
        total_delay: Aggregated tardiness / delay duration (hours).
        penalty_cost: Total financial penalty cost incurred from tardiness ($).
        total_wait_time: Aggregated dwell wait duration (hours).
        total_travel_time: Aggregated en-route transit time (hours).
        total_service_time: Aggregated customer service duration (hours).
        total_duration: Overall route duration from start to final return (hours).
        route: Associated Route domain model instance (if provided).
    """

    schedules: List[StopSchedule] = field(default_factory=list)
    is_feasible: bool = True
    total_delay: float = 0.0
    penalty_cost: float = 0.0
    total_wait_time: float = 0.0
    total_travel_time: float = 0.0
    total_service_time: float = 0.0
    total_duration: float = 0.0
    route: Optional[Route] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schedule result to dictionary."""
        return {
            "schedules": [s.to_dict() for s in self.schedules],
            "is_feasible": self.is_feasible,
            "total_delay": self.total_delay,
            "penalty_cost": self.penalty_cost,
            "total_wait_time": self.total_wait_time,
            "total_travel_time": self.total_travel_time,
            "total_service_time": self.total_service_time,
            "total_duration": self.total_duration,
        }


class TimeWindowScheduler:
    """Timeline propagation and schedule penalty calculator for VRPTW.

    Attributes:
        speed_kmh: Vehicle travel speed in km/h.
        penalty_cost_per_late_hour: Financial cost rate per late hour ($/h).
        distance_func: Distance calculation callable.
        max_route_duration_hours: Maximum allowable route duration.
    """

    def __init__(
        self,
        speed_kmh: float = 40.0,
        penalty_cost_per_late_hour: float = 0.0,
        distance_func: Optional[Callable[[Any, Any], float]] = None,
        distance_metric: str = "euclidean",
        allow_soft_windows: bool = True,
        max_route_duration_hours: Optional[float] = None,
    ) -> None:
        self.speed_kmh = float(speed_kmh)
        if self.speed_kmh <= 0:
            raise ValidationError(f"speed_kmh must be positive, got {self.speed_kmh}")

        self.penalty_cost_per_late_hour = float(penalty_cost_per_late_hour)
        if self.penalty_cost_per_late_hour < 0:
            raise ValidationError(
                f"penalty_cost_per_late_hour cannot be negative, got {self.penalty_cost_per_late_hour}"
            )

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

        self.allow_soft_windows = allow_soft_windows
        self.max_route_duration_hours = max_route_duration_hours

    def evaluate_stop(
        self,
        arrival_time: float,
        time_window: Optional[Tuple[float, float]],
        service_time: float = 0.0,
        stop: Optional[Stop] = None,
        order_id: Optional[str] = None,
    ) -> StopSchedule:
        """Evaluate schedule feasibility and penalty for a single stop visit."""
        arr = float(arrival_time)
        serv = float(service_time)

        if time_window is not None:
            earliest = float(time_window[0])
            latest = float(time_window[1])
            wait = max(0.0, earliest - arr)
            lateness = max(0.0, arr - latest)
        else:
            wait = 0.0
            lateness = 0.0

        service_start = arr + wait
        departure = service_start + serv
        is_feasible = (lateness <= 1e-6)
        penalty = lateness * self.penalty_cost_per_late_hour

        return StopSchedule(
            arrival_time=arr,
            wait_time=wait,
            service_start_time=service_start,
            departure_time=departure,
            service_time=serv,
            lateness=lateness,
            is_feasible=is_feasible,
            penalty_cost=penalty,
            time_window=time_window,
            stop=stop,
            order_id=order_id or (stop.order_id if stop else None),
        )

    def calculate_schedule(
        self,
        stops: List[Stop],
        start_time: float = 0.0,
        route: Optional[Route] = None,
    ) -> ScheduleResult:
        """Propagate timeline through an ordered stop list starting at start_time."""
        if not stops:
            return ScheduleResult(
                schedules=[],
                is_feasible=True,
                total_delay=0.0,
                penalty_cost=0.0,
                total_wait_time=0.0,
                total_travel_time=0.0,
                total_service_time=0.0,
                total_duration=0.0,
                route=route,
            )

        schedules: List[StopSchedule] = []
        tot_travel = 0.0

        # Stop 0 (initial departure location, e.g. depot)
        first_stop = stops[0]
        first_tw = first_stop.time_window
        if first_tw is not None:
            e0, l0 = float(first_tw[0]), float(first_tw[1])
            wait0 = max(0.0, e0 - start_time)
            late0 = max(0.0, start_time - l0)
        else:
            wait0 = 0.0
            late0 = 0.0

        serv_start0 = start_time + wait0
        dep0 = serv_start0 + first_stop.service_time
        pen0 = late0 * self.penalty_cost_per_late_hour

        schedules.append(
            StopSchedule(
                arrival_time=start_time,
                wait_time=wait0,
                service_start_time=serv_start0,
                departure_time=dep0,
                service_time=first_stop.service_time,
                lateness=late0,
                is_feasible=(late0 <= 1e-6),
                penalty_cost=pen0,
                time_window=first_tw,
                stop=first_stop,
                order_id=first_stop.order_id,
            )
        )

        curr_dep = dep0

        # Subsequent stops
        for idx in range(1, len(stops)):
            p_prev = stops[idx - 1]
            p_curr = stops[idx]

            dist = self.distance_func(p_prev.location, p_curr.location)
            leg_travel = dist / self.speed_kmh
            tot_travel += leg_travel

            arr = curr_dep + leg_travel
            sched_entry = self.evaluate_stop(
                arrival_time=arr,
                time_window=p_curr.time_window,
                service_time=p_curr.service_time,
                stop=p_curr,
                order_id=p_curr.order_id,
            )
            schedules.append(sched_entry)
            curr_dep = sched_entry.departure_time

        tot_delay = sum(s.lateness for s in schedules)
        tot_penalty = sum(s.penalty_cost for s in schedules)
        tot_wait = sum(s.wait_time for s in schedules)
        tot_service = sum(s.service_time for s in schedules)
        tot_duration = curr_dep - start_time

        feasible = all(s.is_feasible for s in schedules)
        if self.max_route_duration_hours and tot_duration > self.max_route_duration_hours + 1e-6:
            feasible = False

        return ScheduleResult(
            schedules=schedules,
            is_feasible=feasible,
            total_delay=tot_delay,
            penalty_cost=tot_penalty,
            total_wait_time=tot_wait,
            total_travel_time=tot_travel,
            total_service_time=tot_service,
            total_duration=tot_duration,
            route=route,
        )

    def propagate(self, route: Route, start_time: Optional[float] = None) -> ScheduleResult:
        """Propagate schedule along Route stops and assign to route.schedule."""
        if not route.stops:
            res = self.calculate_schedule([], start_time=0.0, route=route)
            route.schedule = []
            return res

        if start_time is None:
            if route.stops[0].time_window is not None:
                resolved_start = float(route.stops[0].time_window[0])
            else:
                resolved_start = 0.0
        else:
            resolved_start = float(start_time)

        res = self.calculate_schedule(stops=route.stops, start_time=resolved_start, route=route)
        route.schedule = res.schedules
        return res
