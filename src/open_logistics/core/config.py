"""Configuration management for Open Logistics framework.

Provides unified configuration settings for routing, inventory optimization,
warehouse management, and event bus behaviors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

from open_logistics.core.exceptions import ConfigurationError


@dataclass
class LogisticsConfig:
    """Comprehensive settings for Open Logistics operations.

    Attributes:
        default_vehicle_speed_kmh: Default speed of vehicles in fleet (km/h).
        default_service_time_minutes: Default delivery dwell time at stop (minutes).
        max_route_duration_hours: Maximum allowable driver route duration (hours).
        distance_metric: Calculation method ('haversine', 'euclidean', 'manhattan').
        cvrp_algorithm: Clarke-Wright algorithm mode ('parallel' or 'sequential').
        enable_2opt: Whether to run 2-opt local search route optimization.
        max_iterations_2opt: Iteration ceiling for 2-opt search.
        default_service_level: Target inventory cycle service level (e.g. 0.95 for 95%).
        default_holding_cost_rate: Annual holding cost fraction of unit cost (e.g. 0.20 = 20%).
        default_order_cost: Fixed procurement order setup cost ($/order).
        default_lead_time_days: Default replenishment lead time in days.
        work_days_per_year: Operational calendar days per year for annualized metrics.
        abc_cutoffs: Cumulative revenue thresholds (A_cutoff, B_cutoff).
        xyz_cutoffs: Coefficient of variation cutoffs (X_cutoff, Y_cutoff).
        default_picking_strategy: Warehouse routing method ('serpentine', 'return', 'midpoint').
        max_wave_orders: Upper limit of orders batched into a single picking wave.
        max_wave_weight: Maximum total cargo weight per picker cart/wave (kg).
        max_wave_volume: Maximum total volume per picker cart/wave (m^3).
        log_level: Default logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        event_bus_isolated_errors: Whether EventBus suppresses subscriber callback errors.
        enable_metrics: Whether to calculate summary KPIs and metrics.
    """

    # Routing settings
    default_vehicle_speed_kmh: float = 40.0
    default_service_time_minutes: float = 15.0
    max_route_duration_hours: float = 8.0
    distance_metric: str = "haversine"
    cvrp_algorithm: str = "parallel"
    enable_2opt: bool = True
    max_iterations_2opt: int = 500

    # Inventory settings
    default_service_level: float = 0.95
    default_holding_cost_rate: float = 0.20
    default_order_cost: float = 50.0
    default_lead_time_days: float = 3.0
    work_days_per_year: int = 365
    abc_cutoffs: Tuple[float, float] = (0.80, 0.95)
    xyz_cutoffs: Tuple[float, float] = (0.10, 0.25)

    # Warehouse settings
    default_picking_strategy: str = "serpentine"
    max_wave_orders: int = 25
    max_wave_weight: float = 250.0
    max_wave_volume: float = 1.5

    # System settings
    log_level: str = "INFO"
    event_bus_isolated_errors: bool = True
    enable_metrics: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate all configuration attributes and raise ConfigurationError if invalid."""
        if self.default_vehicle_speed_kmh <= 0:
            raise ConfigurationError(
                f"default_vehicle_speed_kmh must be > 0, got {self.default_vehicle_speed_kmh}"
            )
        if self.default_service_time_minutes < 0:
            raise ConfigurationError(
                f"default_service_time_minutes cannot be negative, got {self.default_service_time_minutes}"
            )
        if self.max_route_duration_hours <= 0:
            raise ConfigurationError(
                f"max_route_duration_hours must be > 0, got {self.max_route_duration_hours}"
            )

        valid_metrics = {"haversine", "euclidean", "manhattan"}
        if self.distance_metric.lower() not in valid_metrics:
            raise ConfigurationError(
                f"distance_metric must be one of {valid_metrics}, got '{self.distance_metric}'"
            )

        valid_cvrp = {"parallel", "sequential"}
        if self.cvrp_algorithm.lower() not in valid_cvrp:
            raise ConfigurationError(
                f"cvrp_algorithm must be one of {valid_cvrp}, got '{self.cvrp_algorithm}'"
            )

        if not (0.0 < self.default_service_level < 1.0):
            raise ConfigurationError(
                f"default_service_level must be between 0.0 and 1.0 (exclusive), got {self.default_service_level}"
            )
        if self.default_holding_cost_rate <= 0:
            raise ConfigurationError(
                f"default_holding_cost_rate must be > 0, got {self.default_holding_cost_rate}"
            )
        if self.default_order_cost < 0:
            raise ConfigurationError(
                f"default_order_cost cannot be negative, got {self.default_order_cost}"
            )
        if self.default_lead_time_days < 0:
            raise ConfigurationError(
                f"default_lead_time_days cannot be negative, got {self.default_lead_time_days}"
            )
        if self.work_days_per_year <= 0:
            raise ConfigurationError(
                f"work_days_per_year must be > 0, got {self.work_days_per_year}"
            )

        if len(self.abc_cutoffs) != 2 or not (0.0 < self.abc_cutoffs[0] < self.abc_cutoffs[1] < 1.0):
            raise ConfigurationError(
                f"abc_cutoffs must be a tuple of two fractions (0 < A < B < 1.0), got {self.abc_cutoffs}"
            )
        if len(self.xyz_cutoffs) != 2 or not (0.0 < self.xyz_cutoffs[0] < self.xyz_cutoffs[1]):
            raise ConfigurationError(
                f"xyz_cutoffs must be a tuple of two positive cutoffs (0 < X < Y), got {self.xyz_cutoffs}"
            )

        valid_picking = {"serpentine", "return", "midpoint"}
        if self.default_picking_strategy.lower() not in valid_picking:
            raise ConfigurationError(
                f"default_picking_strategy must be one of {valid_picking}, got '{self.default_picking_strategy}'"
            )

        if self.max_wave_orders <= 0:
            raise ConfigurationError(f"max_wave_orders must be > 0, got {self.max_wave_orders}")
        if self.max_wave_weight <= 0:
            raise ConfigurationError(f"max_wave_weight must be > 0, got {self.max_wave_weight}")
        if self.max_wave_volume <= 0:
            raise ConfigurationError(f"max_wave_volume must be > 0, got {self.max_wave_volume}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogisticsConfig:
        """Instantiate LogisticsConfig from a dictionary with fallback to defaults."""
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in fields}
        if "abc_cutoffs" in filtered and isinstance(filtered["abc_cutoffs"], (list, tuple)):
            filtered["abc_cutoffs"] = tuple(filtered["abc_cutoffs"])
        if "xyz_cutoffs" in filtered and isinstance(filtered["xyz_cutoffs"], (list, tuple)):
            filtered["xyz_cutoffs"] = tuple(filtered["xyz_cutoffs"])
        return cls(**filtered)
