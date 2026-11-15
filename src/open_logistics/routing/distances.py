"""Distance calculation engines and coordinate validation utilities.

Provides pure-Python Euclidean, Manhattan, and spherical Haversine distance
metrics with coordinate validation and distance matrix helpers.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from open_logistics.core.exceptions import ValidationError
from open_logistics.core.models import Depot, Location, Order, Stop

EARTH_RADIUS_KM: float = 6371.0


def _extract_coords(p: Any) -> Tuple[float, float]:
    """Extract (lat, lon) or (x, y) coordinates from sequence or domain models.

    Args:
        p: 2-element sequence, Location, Stop, Order, or Depot.

    Returns:
        Tuple of (float, float) coordinate pair.

    Raises:
        ValidationError: If p cannot be converted to a 2-element numeric coordinate.
    """
    if isinstance(p, (tuple, list)):
        if len(p) != 2:
            raise ValidationError(f"Coordinate sequence must have exactly 2 elements, got {len(p)}")
        try:
            return (float(p[0]), float(p[1]))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Coordinate elements must be numeric, got {p!r}") from exc
    elif isinstance(p, Location):
        return p.as_tuple()
    elif hasattr(p, "location"):
        loc = p.location
        if not isinstance(loc, (tuple, list)) or len(loc) != 2:
            raise ValidationError(f"Object location must be a 2-element sequence, got {loc!r}")
        return (float(loc[0]), float(loc[1]))
    elif hasattr(p, "destination"):
        dest = p.destination
        if not isinstance(dest, (tuple, list)) or len(dest) != 2:
            raise ValidationError(f"Object destination must be a 2-element sequence, got {dest!r}")
        return (float(dest[0]), float(dest[1]))
    elif hasattr(p, "as_tuple") and callable(p.as_tuple):
        coords = p.as_tuple()
        return (float(coords[0]), float(coords[1]))
    raise ValidationError(f"Unsupported coordinate object type: {type(p).__name__}")


def validate_coordinates(lat: float, lon: float, context: str = "Coordinates") -> None:
    """Validate that coordinates conform to valid geographic boundaries.

    Args:
        lat: Latitude in decimal degrees (must be between -90.0 and 90.0).
        lon: Longitude in decimal degrees (must be between -180.0 and 180.0).
        context: Context string for error messages.

    Raises:
        ValidationError: If latitude or longitude are outside valid ranges or NaN/Inf.
    """
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise ValidationError(f"{context} latitude and longitude must be numbers, got ({lat!r}, {lon!r})")
    if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
        raise ValidationError(f"{context} coordinates must be finite real numbers, got ({lat}, {lon})")
    if not (-90.0 <= lat <= 90.0):
        raise ValidationError(f"{context} latitude must be between -90.0 and 90.0, got {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValidationError(f"{context} longitude must be between -180.0 and 180.0, got {lon}")


def euclidean_distance(p1: Any, p2: Any) -> float:
    """Compute exact 2D Euclidean distance between two points.

    Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)

    Args:
        p1: Source coordinates (sequence or domain entity).
        p2: Target coordinates (sequence or domain entity).

    Returns:
        Euclidean distance as float.
    """
    x1, y1 = _extract_coords(p1)
    x2, y2 = _extract_coords(p2)
    if x1 == x2 and y1 == y2:
        return 0.0
    return float(math.hypot(x2 - x1, y2 - y1))


def manhattan_distance(p1: Any, p2: Any) -> float:
    """Compute exact 2D Manhattan (L1 norm / taxicab) distance between two points.

    Formula: |x2 - x1| + |y2 - y1|

    Args:
        p1: Source coordinates (sequence or domain entity).
        p2: Target coordinates (sequence or domain entity).

    Returns:
        Manhattan distance as float.
    """
    x1, y1 = _extract_coords(p1)
    x2, y2 = _extract_coords(p2)
    if x1 == x2 and y1 == y2:
        return 0.0
    return float(abs(x2 - x1) + abs(y2 - y1))


def haversine_distance(p1: Any, p2: Any, radius: float = EARTH_RADIUS_KM) -> float:
    """Compute great-circle spherical distance between two points on Earth using Haversine formula.

    Validates latitude in [-90, 90] and longitude in [-180, 180].

    Args:
        p1: Source (lat, lon) coordinates (sequence or domain entity).
        p2: Target (lat, lon) coordinates (sequence or domain entity).
        radius: Sphere radius in kilometers (default: 6371.0 km).

    Returns:
        Spherical distance in kilometers as float.

    Raises:
        ValidationError: If coordinates fall outside allowable geographic bounds.
    """
    lat1, lon1 = _extract_coords(p1)
    lat2, lon2 = _extract_coords(p2)

    validate_coordinates(lat1, lon1, context="Haversine point 1")
    validate_coordinates(lat2, lon2, context="Haversine point 2")

    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(radius * c)


def get_distance_calculator(metric: str = "haversine") -> Callable[[Any, Any], float]:
    """Retrieve distance calculation callable by metric name.

    Args:
        metric: 'haversine', 'euclidean', or 'manhattan'.

    Returns:
        Matching distance function.

    Raises:
        ValidationError: If metric name is unknown.
    """
    metric_lower = metric.lower().strip()
    if metric_lower == "haversine":
        return haversine_distance
    elif metric_lower == "euclidean":
        return euclidean_distance
    elif metric_lower == "manhattan":
        return manhattan_distance
    raise ValidationError(
        f"Unknown distance metric '{metric}'. Expected 'haversine', 'euclidean', or 'manhattan'."
    )


def calculate_distance_matrix(
    locations: Sequence[Any],
    metric: Union[str, Callable[[Any, Any], float]] = "haversine",
) -> List[List[float]]:
    """Compute symmetric pairwise distance matrix for a sequence of locations.

    Args:
        locations: Sequence of coordinates or domain entities.
        metric: Metric name string ('haversine', 'euclidean', 'manhattan') or distance callable.

    Returns:
        N x N symmetric matrix (list of lists) with 0.0 along the main diagonal.
    """
    if not locations:
        return []

    dist_fn = get_distance_calculator(metric) if isinstance(metric, str) else metric
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(locations[i], locations[j])
            matrix[i][j] = d
            matrix[j][i] = d

    return matrix


# Alias
distance_matrix = calculate_distance_matrix
