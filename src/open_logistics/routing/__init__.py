"""Routing, vehicle routing problem (CVRP/VRPTW), TSP, and automated fleet dispatch.

Exports distance engines, CVRP heuristics, TSP local search optimizers,
VRPTW schedule propagation, and automated fleet dispatch.
"""

from __future__ import annotations

import math
from typing import Any

from open_logistics.core.exceptions import ValidationError
from open_logistics.core.models import Location, Stop
from open_logistics.routing.clarke_wright import ClarkeWrightCVRP
from open_logistics.routing.dispatch_engine import AutoDispatchEngine, DispatchResult
from open_logistics.routing.distances import (
    calculate_distance_matrix,
    distance_matrix,
    euclidean_distance,
    get_distance_calculator,
    haversine_distance,
    manhattan_distance,
    validate_coordinates,
)
from open_logistics.routing.time_windows import (
    ScheduleResult,
    StopSchedule,
    TimeWindowScheduler,
)
from open_logistics.routing.tsp import NearestNeighborTSP, TwoOptSearch

# Patch Stop.__post_init__ to support both Cartesian (x, y) and Geographic (lat, lon) coordinates
def _stop_post_init(self: Stop) -> None:
    if isinstance(self.location, Location):
        self.location = self.location.as_tuple()
    elif isinstance(self.location, (list, tuple)) and len(self.location) == 2:
        try:
            self.location = (float(self.location[0]), float(self.location[1]))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Stop location elements must be numeric, got {self.location!r}") from exc
    else:
        raise ValidationError(f"Stop location must be a 2-tuple (lat, lon) or (x, y), got {self.location!r}")

    lat, lon = self.location
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise ValidationError(f"Stop location coordinates must be numbers, got ({lat!r}, {lon!r})")
    if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
        raise ValidationError(f"Stop coordinates must be finite real numbers, got ({lat}, {lon})")

    self.demand_weight = float(self.demand_weight)
    self.demand_volume = float(self.demand_volume)
    self.service_time = float(self.service_time)

    if self.demand_weight < 0:
        raise ValidationError(f"Stop demand_weight cannot be negative, got {self.demand_weight}")
    if self.demand_volume < 0:
        raise ValidationError(f"Stop demand_volume cannot be negative, got {self.demand_volume}")
    if self.service_time < 0:
        raise ValidationError(f"Stop service_time cannot be negative, got {self.service_time}")


Stop.__post_init__ = _stop_post_init  # type: ignore[method-assign]

__all__ = [
    # Distances
    "euclidean_distance",
    "manhattan_distance",
    "haversine_distance",
    "distance_matrix",
    "calculate_distance_matrix",
    "validate_coordinates",
    "get_distance_calculator",
    # CVRP
    "ClarkeWrightCVRP",
    # TSP
    "NearestNeighborTSP",
    "TwoOptSearch",
    # VRPTW
    "TimeWindowScheduler",
    "ScheduleResult",
    "StopSchedule",
    # Automated Fleet Dispatch
    "AutoDispatchEngine",
    "DispatchResult",
]
