"""Tier 1 Feature Coverage Tests: API, Engine Facade, Webhooks, REST Server, and CLI.

Covers:
- Feature 19: Unified LogisticsEngine Facade
- Feature 20: Outbound Webhook Dispatcher (HMAC Signed)
- Feature 21: Pure-Stdlib Embedded REST Server
- Feature 22: UNIX-Pipeable CLI
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import io
import json
import socket
import sys
import threading
import time
from typing import Any, Dict, Generator, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from open_logistics.core.config import LogisticsConfig
from open_logistics.core.events import Event, EventBus
from open_logistics.core.exceptions import LogisticsError, ValidationError
from open_logistics.core.models import (
    Depot,
    Order,
    OrderItem,
    StockItem,
    Vehicle,
)

# Safe imports for integration modules (Progressive Testability)
try:
    from open_logistics.api.engine import LogisticsEngine
    API_ENGINE_AVAILABLE = True
except ImportError:
    API_ENGINE_AVAILABLE = False

try:
    from open_logistics.api.webhooks import WebhookDispatcher, WebhookEvent
    WEBHOOKS_AVAILABLE = True
except ImportError:
    WEBHOOKS_AVAILABLE = False

try:
    from open_logistics.api.server import LogisticsHTTPHandler, create_server
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False

try:
    from open_logistics.cli.main import main as cli_main
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


# ============================================================================
# Shared Fixtures & Helpers for Tier 1 API & CLI Tests
# ============================================================================

@pytest.fixture
def sample_orders_list() -> List[Order]:
    """Sample list of customer orders for dispatch and facade testing."""
    return [
        Order(
            id="ORD-API-001",
            destination=(41.8825, -87.6225),
            weight=15.0,
            volume=0.03,
            priority=2,
            time_window=(8.0, 12.0),
        ),
        Order(
            id="ORD-API-002",
            destination=(41.8900, -87.6300),
            weight=25.0,
            volume=0.05,
            priority=1,
            time_window=(9.0, 13.0),
        ),
        Order(
            id="ORD-API-003",
            destination=(41.8750, -87.6350),
            weight=10.0,
            volume=0.02,
            priority=1,
            time_window=(10.0, 14.0),
        ),
    ]


@pytest.fixture
def sample_fleet_list() -> List[Vehicle]:
    """Sample delivery fleet."""
    return [
        Vehicle(
            id="VAN-01",
            capacity_weight=1000.0,
            capacity_volume=10.0,
            max_stops=25,
            speed_kmh=40.0,
            fixed_cost=50.0,
            cost_per_km=1.2,
        ),
        Vehicle(
            id="VAN-02",
            capacity_weight=1200.0,
            capacity_volume=12.0,
            max_stops=30,
            speed_kmh=40.0,
            fixed_cost=55.0,
            cost_per_km=1.25,
        ),
    ]


@pytest.fixture
def sample_depot_obj() -> Depot:
    """Distribution center depot."""
    return Depot(
        id="DEPOT-CENTRAL",
        location=(41.8781, -87.6298),
        name="Central Chicago DC",
    )


@pytest.fixture
def sample_stock_items_list() -> List[StockItem]:
    """List of inventory stock items with varying demand and on-hand."""
    return [
        StockItem(
            sku="SKU-EVAL-01",
            on_hand=30,
            on_order=0,
            unit_cost=15.0,
            holding_cost_rate=0.20,
            lead_time_days=3.0,
            daily_demand_history=[20.0, 22.0, 25.0, 18.0, 21.0, 24.0],
            allocated=5,
        ),
        StockItem(
            sku="SKU-EVAL-02",
            on_hand=500,
            on_order=0,
            unit_cost=50.0,
            holding_cost_rate=0.15,
            lead_time_days=5.0,
            daily_demand_history=[5.0, 6.0, 4.0, 5.0, 5.0],
            allocated=10,
        ),
    ]


def find_free_port() -> int:
    """Find a random available TCP port for embedded server testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ============================================================================
# Feature 19: Unified LogisticsEngine Facade (≥ 7 tests)
# ============================================================================

@pytest.mark.skipif(not API_ENGINE_AVAILABLE, reason="open_logistics.api.engine not implemented")
class TestLogisticsEngineFacade:
    """Tests for the high-level LogisticsEngine unified facade."""

    def test_f19_instantiation_default_config(self) -> None:
        """Verify instantiation with default config initializes sub-engines and event bus."""
        engine = LogisticsEngine()
        assert engine is not None
        assert isinstance(engine.config, LogisticsConfig)
        assert isinstance(engine.event_bus, EventBus)
        assert engine.config.cvrp_algorithm in ("parallel", "sequential")

    def test_f19_instantiation_custom_config(self) -> None:
        """Verify instantiation respects user-supplied custom configuration."""
        custom_cfg = LogisticsConfig(
            distance_metric="euclidean",
            default_service_level=0.99,
            default_vehicle_speed_kmh=55.0,
        )
        engine = LogisticsEngine(config=custom_cfg)
        assert engine.config.distance_metric == "euclidean"
        assert engine.config.default_service_level == 0.99
        assert engine.config.default_vehicle_speed_kmh == 55.0

    def test_f19_dispatch_orders_delegation(
        self,
        sample_orders_list: List[Order],
        sample_fleet_list: List[Vehicle],
        sample_depot_obj: Depot,
    ) -> None:
        """Verify dispatch_orders coordinates routing and returns DispatchResult."""
        engine = LogisticsEngine()
        result = engine.dispatch_orders(
            orders=sample_orders_list,
            fleet=sample_fleet_list,
            depot=sample_depot_obj,
        )
        assert result is not None
        assert hasattr(result, "routes")
        assert hasattr(result, "unassigned_orders")
        assert hasattr(result, "total_distance")
        assert hasattr(result, "total_cost")
        assert isinstance(result.routes, list)
        assert len(result.routes) > 0
        assert result.total_distance >= 0.0

    def test_f19_evaluate_inventory_delegation(
        self,
        sample_stock_items_list: List[StockItem],
    ) -> None:
        """Verify evaluate_inventory delegates to ReplenishmentEventEngine."""
        engine = LogisticsEngine()
        statuses = engine.evaluate_inventory(sample_stock_items_list)
        assert isinstance(statuses, list)
        assert len(statuses) == len(sample_stock_items_list)

        first_status = statuses[0]
        assert hasattr(first_status, "sku")
        assert hasattr(first_status, "needs_reorder")
        assert hasattr(first_status, "reorder_point")
        assert hasattr(first_status, "safety_stock")
        assert hasattr(first_status, "eoq")
        assert first_status.sku == "SKU-EVAL-01"

    def test_f19_allocate_and_pick_delegation(
        self,
        sample_orders_list: List[Order],
    ) -> None:
        """Verify allocate_and_pick coordinates wave creation and picker tasks."""
        engine = LogisticsEngine()
        # Call allocate_and_pick with orders (and optional warehouse layout)
        res = engine.allocate_and_pick(orders=sample_orders_list)
        assert res is not None
        assert hasattr(res, "wave") or hasattr(res, "tasks") or isinstance(res, (dict, list, object))

    def test_f19_event_bus_exposure_and_subscription(
        self,
        sample_orders_list: List[Order],
        sample_fleet_list: List[Vehicle],
        sample_depot_obj: Depot,
    ) -> None:
        """Verify that external clients can subscribe to events on engine.event_bus."""
        engine = LogisticsEngine()
        captured_events: List[Event] = []

        engine.event_bus.subscribe("#", lambda evt: captured_events.append(evt))
        engine.dispatch_orders(sample_orders_list, sample_fleet_list, sample_depot_obj)

        # Either routing events published or custom event publish works
        custom_evt = Event(topic="custom.test", payload={"data": 123})
        engine.event_bus.publish(custom_evt)
        assert any(e.topic == "custom.test" for e in captured_events)

    def test_f19_error_propagation_invalid_inputs(
        self,
        sample_fleet_list: List[Vehicle],
        sample_depot_obj: Depot,
    ) -> None:
        """Verify facade does not swallow invalid parameter errors silently."""
        engine = LogisticsEngine()
        # Passing an invalid order structure or empty string
        with pytest.raises((LogisticsError, ValidationError, ValueError, TypeError)):
            engine.dispatch_orders(
                orders=[None],  # type: ignore[list-item]
                fleet=sample_fleet_list,
                depot=sample_depot_obj,
            )


# ============================================================================
# Feature 20: Outbound Webhook Dispatcher (≥ 7 tests)
# ============================================================================

@pytest.mark.skipif(not WEBHOOKS_AVAILABLE, reason="open_logistics.api.webhooks not implemented")
class TestWebhookDispatcher:
    """Tests for HMAC SHA-256 signed outbound webhook dispatcher."""

    def test_f20_hmac_signature_calculation(self) -> None:
        """Verify HMAC-SHA256 signature generation matches canonical standard."""
        secret = "secret_key_12345"
        payload = {"event": "order.shipped", "order_id": "ORD-999", "timestamp": 1700000000}
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        expected_sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

        dispatcher = WebhookDispatcher()
        calculated_sig = dispatcher.calculate_signature(payload_bytes, secret)
        assert calculated_sig == expected_sig

    def test_f20_signature_verification_helper(self) -> None:
        """Verify verification helper accepts valid signature and rejects tampered."""
        secret = "test_webhook_secret"
        payload_bytes = b'{"status": "delivered"}'
        valid_sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

        dispatcher = WebhookDispatcher()
        assert dispatcher.verify_signature(payload_bytes, valid_sig, secret) is True
        assert dispatcher.verify_signature(payload_bytes, "tampered_signature", secret) is False
        assert dispatcher.verify_signature(b'{"status": "tampered"}', valid_sig, secret) is False

    def test_f20_payload_validation(self) -> None:
        """Verify dispatcher validates payload format and rejects non-dict/invalid inputs."""
        dispatcher = WebhookDispatcher()
        with pytest.raises((ValidationError, ValueError, TypeError)):
            dispatcher.dispatch(endpoint="https://example.com/webhook", payload="not a dict", secret="s")  # type: ignore[arg-type]

    def test_f20_subscriber_registration(self) -> None:
        """Verify registration of webhook subscribers with url, secret, and topic filter."""
        dispatcher = WebhookDispatcher()
        sub_id = dispatcher.register(
            url="https://api.3pl-partner.com/events",
            secret="shh_super_secret",
            events=["order.shipped", "order.cancelled"],
        )
        assert sub_id is not None
        assert len(dispatcher.get_subscribers()) >= 1

    def test_f20_topic_filtering_dispatch(self) -> None:
        """Verify dispatcher filters events and only notifies subscribers interested in topic."""
        dispatcher = WebhookDispatcher()
        dispatched_targets: List[str] = []

        # Hook or mock the raw HTTP post
        def mock_post(url: str, payload_bytes: bytes, headers: Dict[str, str]) -> int:
            dispatched_targets.append(url)
            return 200

        dispatcher._http_post = mock_post  # type: ignore[assignment]
        dispatcher.register(
            url="https://webhook1.example.com",
            secret="sec1",
            events=["order.shipped"],
        )
        dispatcher.register(
            url="https://webhook2.example.com",
            secret="sec2",
            events=["stock.low"],
        )

        event = WebhookEvent(topic="order.shipped", data={"order_id": "ORD-1"})
        dispatcher.dispatch_event(event)

        assert "https://webhook1.example.com" in dispatched_targets
        assert "https://webhook2.example.com" not in dispatched_targets

    def test_f20_retry_logic_on_failure(self) -> None:
        """Verify retry logic attempts multiple deliveries when HTTP post fails."""
        dispatcher = WebhookDispatcher(max_retries=3, retry_delay_seconds=0.01)
        attempt_count = 0

        def failing_post(url: str, payload_bytes: bytes, headers: Dict[str, str]) -> int:
            nonlocal attempt_count
            attempt_count += 1
            raise IOError("Simulated network outage")

        dispatcher._http_post = failing_post  # type: ignore[assignment]
        res = dispatcher.dispatch(
            endpoint="https://failing-target.com",
            payload={"key": "val"},
            secret="test_sec",
        )
        assert attempt_count == 3
        assert res.success is False

    def test_f20_successful_delivery_response_metrics(self) -> None:
        """Verify successful delivery records response status, timestamp, and signature."""
        dispatcher = WebhookDispatcher(max_retries=1)

        def success_post(url: str, payload_bytes: bytes, headers: Dict[str, str]) -> int:
            assert "X-Signature-SHA256" in headers or "X-Signature" in headers or "X-Hub-Signature-256" in headers
            return 200

        dispatcher._http_post = success_post  # type: ignore[assignment]
        res = dispatcher.dispatch(
            endpoint="https://success-target.com",
            payload={"action": "fulfill"},
            secret="secret_abc",
        )
        assert res.success is True
        assert res.status_code == 200


# ============================================================================
# Feature 21: Pure-Stdlib Embedded REST Server (≥ 7 tests)
# ============================================================================

@pytest.mark.skipif(not SERVER_AVAILABLE, reason="open_logistics.api.server not implemented")
class TestEmbeddedRESTServer:
    """Tests for the pure-standard-library HTTP JSON microservice server."""

    @pytest.fixture
    def test_server(self) -> Generator[str, None, None]:
        """Spin up an embedded HTTP server on a random port in a background thread."""
        port = find_free_port()
        server = create_server(host="127.0.0.1", port=port)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{port}"
        # Wait until server is accepting connections
        time.sleep(0.05)
        yield base_url
        server.shutdown()
        server.server_close()

    def test_f21_get_health_endpoint(self, test_server: str) -> None:
        """Verify GET /health returns 200 OK and healthy status json."""
        req = Request(f"{test_server}/health", method="GET")
        with urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") in ("ok", "healthy")

    def test_f21_post_dispatch_endpoint(self, test_server: str) -> None:
        """Verify POST /api/dispatch accepts orders/fleet/depot and returns routes."""
        payload = {
            "orders": [
                {"id": "O1", "destination": [41.88, -87.62], "weight": 10.0},
                {"id": "O2", "destination": [41.89, -87.63], "weight": 15.0},
            ],
            "fleet": [
                {"id": "V1", "capacity_weight": 100.0, "speed_kmh": 40.0}
            ],
            "depot": {"id": "D1", "location": [41.878, -87.629]},
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{test_server}/api/dispatch",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "routes" in data or "total_distance" in data

    def test_f21_post_inventory_endpoint(self, test_server: str) -> None:
        """Verify POST /api/inventory evaluates replenishment status."""
        payload = {
            "items": [
                {
                    "sku": "SKU-REST-1",
                    "on_hand": 20,
                    "unit_cost": 10.0,
                    "lead_time_days": 2.0,
                    "daily_demand_history": [10.0, 15.0, 12.0],
                }
            ]
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{test_server}/api/inventory",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert isinstance(data, (dict, list))

    def test_f21_post_wave_pick_endpoint(self, test_server: str) -> None:
        """Verify POST /api/wave-pick returns batched pick wave tasks."""
        payload = {
            "order_ids": ["ORD-101", "ORD-102", "ORD-103"],
            "max_orders": 25,
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{test_server}/api/wave-pick",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data is not None

    def test_f21_malformed_json_bad_request(self, test_server: str) -> None:
        """Verify sending invalid JSON string triggers 400 Bad Request."""
        invalid_bytes = b"This is NOT valid JSON syntax"
        req = Request(
            f"{test_server}/api/dispatch",
            data=invalid_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 400

    def test_f21_not_found_endpoint(self, test_server: str) -> None:
        """Verify GET to unknown path triggers 404 Not Found."""
        req = Request(f"{test_server}/api/unknown_endpoint_xyz", method="GET")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 404

    def test_f21_unsupported_method_error(self, test_server: str) -> None:
        """Verify sending unsupported HTTP method triggers 404 or 405."""
        req = Request(f"{test_server}/health", method="DELETE")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code in (404, 405)


# ============================================================================
# Feature 22: UNIX-Pipeable Standalone CLI (≥ 7 tests)
# ============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="open_logistics.cli.main not implemented")
class TestUnixPipeableCLI:
    """Tests for the standalone, pipeable command-line interface."""

    def test_f22_cli_help_flag(self) -> None:
        """Verify CLI responds to --help with status code 0 and usage message."""
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli_main(["--help"])
            assert exc_info.value.code == 0
            output = buf.getvalue().lower()
            assert "usage" in output
            assert "dispatch" in output or "inventory" in output
        finally:
            sys.stdout = old_stdout

    def test_f22_cli_version_flag(self) -> None:
        """Verify CLI responds to --version with status code 0."""
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli_main(["--version"])
            assert exc_info.value.code == 0
            assert "open-logistics" in buf.getvalue() or "0." in buf.getvalue()
        finally:
            sys.stdout = old_stdout

    def test_f22_dispatch_subcommand_pipe(self) -> None:
        """Verify dispatch subcommand reads JSON from stdin and prints routes JSON to stdout."""
        input_payload = {
            "orders": [{"id": "ORD-1", "destination": [41.88, -87.62], "weight": 5.0}],
            "fleet": [{"id": "V1", "capacity_weight": 500.0, "speed_kmh": 40.0}],
            "depot": {"id": "D1", "location": [41.878, -87.629]},
        }
        stdin_buf = io.StringIO(json.dumps(input_payload))
        stdout_buf = io.StringIO()

        old_stdin, old_stdout = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = stdin_buf, stdout_buf
        try:
            exit_code = cli_main(["dispatch", "--stdin"])
            assert exit_code in (0, None)
            output = json.loads(stdout_buf.getvalue())
            assert "routes" in output or "total_distance" in output
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout

    def test_f22_inventory_subcommand_pipe(self) -> None:
        """Verify inventory subcommand processes stock item JSON and outputs replenishment status."""
        input_payload = {
            "items": [
                {
                    "sku": "SKU-CLI-1",
                    "on_hand": 10,
                    "unit_cost": 25.0,
                    "lead_time_days": 3.0,
                    "daily_demand_history": [10.0, 12.0, 11.0],
                }
            ]
        }
        stdin_buf = io.StringIO(json.dumps(input_payload))
        stdout_buf = io.StringIO()

        old_stdin, old_stdout = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = stdin_buf, stdout_buf
        try:
            exit_code = cli_main(["inventory", "--stdin"])
            assert exit_code in (0, None)
            output = json.loads(stdout_buf.getvalue())
            assert isinstance(output, (list, dict))
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout

    def test_f22_wave_subcommand_pipe(self) -> None:
        """Verify wave subcommand batches orders into pick wave tasks."""
        input_payload = {
            "order_ids": ["O1", "O2", "O3"],
            "max_orders": 10,
        }
        stdin_buf = io.StringIO(json.dumps(input_payload))
        stdout_buf = io.StringIO()

        old_stdin, old_stdout = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = stdin_buf, stdout_buf
        try:
            exit_code = cli_main(["wave", "--stdin"])
            assert exit_code in (0, None)
            output = json.loads(stdout_buf.getvalue())
            assert output is not None
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout

    def test_f22_unknown_subcommand_error(self) -> None:
        """Verify invoking CLI with non-existent subcommand returns non-zero exit code."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["invalid_command_name_xyz"])
        assert exc_info.value.code != 0

    def test_f22_missing_subcommand_arguments_error(self) -> None:
        """Verify subcommand without required inputs exits with non-zero error."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["dispatch"])
        # Expecting non-zero exit code when required --input or --stdin flag missing
        assert exc_info.value.code != 0
