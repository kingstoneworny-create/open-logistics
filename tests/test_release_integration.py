"""Regression coverage for the public SDK, CLI and shared payload handlers."""

import json
from io import StringIO

import pytest

from open_logistics import EventBus, LogisticsConfig, LogisticsEngine, Order, OrderItem, PickResult, StockItem
from open_logistics.api import LogisticsEngine as APILogisticsEngine
from open_logistics.api import PickResult as APIPickResult
from open_logistics.cli.main import main
from open_logistics.core.exceptions import CapacityExceededError, InventoryError, ValidationError
from open_logistics.warehouse.layout import Bin, WarehouseLayout


def test_public_engine_preserves_empty_injected_bus() -> None:
    bus = EventBus()
    engine = LogisticsEngine(event_bus=bus)
    topics = []
    bus.subscribe("warehouse.#", lambda event: topics.append(event.topic))

    assert LogisticsEngine is APILogisticsEngine
    assert PickResult is APIPickResult
    assert engine.event_bus is bus
    assert isinstance(engine.allocate_and_pick(["ORDER-1"]), PickResult)
    assert topics == ["warehouse.wave_created", "warehouse.manifest_generated"]


def test_stock_consumption_requires_registration_with_this_engine() -> None:
    stock = StockItem(sku="SHARED-SKU", on_hand=12)
    first_engine = LogisticsEngine()
    first_engine.evaluate_inventory([stock])
    with pytest.raises(InventoryError, match="not registered"):
        LogisticsEngine().consume_stock(stock.sku, 1)
    assert stock.on_hand == 12
    first_engine.consume_stock(stock.sku, 1)
    assert stock.on_hand == 11


def test_successive_wave_requests_preserve_existing_manifests() -> None:
    engine = LogisticsEngine()
    first = engine.allocate_and_pick(["ORDER-1"])
    second = engine.allocate_and_pick(["ORDER-2", "ORDER-3"], max_orders_per_wave=1)
    manual = engine.warehouse.create_wave(order_ids=["ORDER-4"])

    assert [*engine.warehouse.waves] == ["WAVE-001", "WAVE-002", "WAVE-003", "WAVE-004"]
    assert manual.id not in first.tasks and manual.id not in second.tasks
    assert engine.warehouse.tasks["WAVE-001"] == first.all_tasks
    assert [task.order_id for task in second.all_tasks] == ["ORDER-2", "ORDER-3"]


@pytest.mark.parametrize("options", [
    {"max_orders": 0},
    {"max_orders": -1},
    {"max_orders": True},
    {"max_orders": "2"},
    {"strategy": ""},
    {"strategy": "unsupported"},
    {"strategy": 1},
])
def test_invalid_wave_options_leave_no_warehouse_state(options: dict) -> None:
    engine = LogisticsEngine()
    with pytest.raises(ValidationError):
        engine.wave_from_payload({"order_ids": ["ORDER-1"], **options})
    assert not engine.warehouse.waves
    assert not engine.warehouse.order_fsms


@pytest.mark.parametrize("weight, volume", [(6, 0.1), (1, 0.6)])
def test_wave_capacity_limits_split_orders_and_reject_oversized_orders(weight: float, volume: float) -> None:
    engine = LogisticsEngine(LogisticsConfig(max_wave_orders=3, max_wave_weight=5, max_wave_volume=0.5))
    with pytest.raises(CapacityExceededError, match="OVERSIZED"):
        engine.allocate_and_pick([Order("OVERSIZED", (0, 0), weight, volume)])
    assert not engine.warehouse.waves

    first = Order("ORDER-1", (0, 0), weight=4, volume=0.1)
    second = Order("ORDER-2", (0, 0), weight=2, volume=0.3)
    third = Order("ORDER-3", (0, 0), weight=1, volume=0.3)
    result = engine.allocate_and_pick([first, second, third])
    assert [wave.order_ids for wave in result.waves] == [["ORDER-1"], ["ORDER-2"], ["ORDER-3"]]


def test_configured_midpoint_router_sequences_front_bins_first() -> None:
    layout = WarehouseLayout(bins=[
        Bin("REAR", aisle=1, x=1, y=15, sku="REAR-SKU"),
        Bin("FRONT", aisle=2, x=2, y=3, sku="FRONT-SKU"),
    ])
    order = Order("ORDER-1", (0, 0), weight=2, items=[
        OrderItem("REAR-SKU", 1),
        OrderItem("FRONT-SKU", 1),
    ])
    engine = LogisticsEngine(LogisticsConfig(default_picking_strategy="midpoint"), layout=layout)
    assert [task.bin_id for task in engine.allocate_and_pick([order]).all_tasks] == ["FRONT", "REAR"]


def test_cli_dispatch_file_returns_structured_schedule(tmp_path, capsys) -> None:
    path = tmp_path / "dispatch.json"
    payload = {
        "orders": [{"id": "ORDER-1", "destination": [41.88, -87.62], "weight": 5}],
        "fleet": [{"id": "VAN-1", "capacity_weight": 50}],
        "depot": {"id": "DEPOT-1", "location": [41.878, -87.629]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["dispatch", "--input", str(path)]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["routes"][0]["schedule"] == result["manifest"]["routes"][0]["schedule"]
    assert "arrival_time" in result["routes"][0]["schedule"][0]
    assert captured.err == ""


def test_cli_malformed_json_reports_error_without_stdout(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", StringIO("{broken"))
    with pytest.raises(SystemExit) as error:
        main(["wave", "--stdin"])
    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "error:" in captured.err
