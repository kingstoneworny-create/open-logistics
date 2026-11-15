"""Outbound webhook dispatcher with HMAC-SHA256 request signing.

Lets you push logistics events (``order.shipped``, ``replenishment.triggered``,
...) to partner systems such as an ERP, a 3PL portal, or a Slack bridge.

Every request carries an ``X-Signature-SHA256`` header containing the hex
HMAC-SHA256 digest of the raw request body, keyed by the subscriber's shared
secret. Receivers verify it with :meth:`WebhookDispatcher.verify_signature` or
the equivalent four lines in any language.

Example::

    hooks = WebhookDispatcher()
    hooks.register("https://erp.example.com/hooks", secret="s3cr3t",
                   events=["order.*", "replenishment.#"])
    hooks.attach(engine.event_bus)   # forward bus events automatically
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from open_logistics.core.events import Event, EventBus, topic_matches
from open_logistics.core.exceptions import ValidationError

__all__ = [
    "WebhookEvent",
    "WebhookSubscription",
    "DeliveryResult",
    "WebhookDispatcher",
    "SIGNATURE_HEADER",
]

SIGNATURE_HEADER = "X-Signature-SHA256"


@dataclass
class WebhookEvent:
    """An event to deliver to webhook subscribers."""

    topic: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.topic, str) or not self.topic:
            raise ValidationError("WebhookEvent topic must be a non-empty string")
        if not isinstance(self.data, dict):
            raise ValidationError("WebhookEvent data must be a dict")

    def to_payload(self) -> Dict[str, Any]:
        """Body sent to subscribers."""
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclass
class WebhookSubscription:
    """A registered webhook endpoint."""

    id: str
    url: str
    secret: Optional[str]
    events: List[str] = field(default_factory=list)
    active: bool = True

    def matches(self, topic: str) -> bool:
        """True if this subscription wants ``topic`` (no filters = everything)."""
        if not self.active:
            return False
        if not self.events:
            return True
        return any(topic_matches(pattern, topic) for pattern in self.events)


@dataclass
class DeliveryResult:
    """Outcome of one webhook delivery (after retries)."""

    endpoint: str
    success: bool
    status_code: Optional[int] = None
    attempts: int = 0
    signature: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


def _encode(payload: Dict[str, Any]) -> bytes:
    # Canonical form: sorted keys, compact separators, UTF-8 (non-ASCII preserved).
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode(
        "utf-8"
    )


class WebhookDispatcher:
    """Signs and delivers JSON webhooks with retry and exponential backoff.

    Args:
        secret: Default signing secret used when a call/subscription has none.
        max_retries: Total delivery attempts per endpoint (>= 1).
        retry_delay_seconds: Delay before the second attempt.
        backoff_factor: Multiplier applied to the delay after each failure.
        timeout_seconds: Socket timeout per HTTP request.
        history_size: Number of recent :class:`DeliveryResult` objects kept in :attr:`history`.
    """

    def __init__(
        self,
        secret: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.5,
        backoff_factor: float = 2.0,
        timeout_seconds: float = 5.0,
        history_size: int = 1000,
    ) -> None:
        if int(max_retries) < 1:
            raise ValidationError("max_retries must be >= 1")
        if retry_delay_seconds < 0 or backoff_factor < 1 or timeout_seconds <= 0:
            raise ValidationError("retry_delay_seconds >= 0, backoff_factor >= 1, timeout_seconds > 0 required")
        self.secret = secret
        self.max_retries = int(max_retries)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.backoff_factor = float(backoff_factor)
        self.timeout_seconds = float(timeout_seconds)
        self.history: Deque[DeliveryResult] = deque(maxlen=history_size)
        self._subscribers: Dict[str, WebhookSubscription] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- signatures
    @staticmethod
    def calculate_signature(payload_bytes: bytes, secret: str) -> str:
        """Hex HMAC-SHA256 of ``payload_bytes`` keyed by ``secret``."""
        if not isinstance(payload_bytes, (bytes, bytearray)):
            raise ValidationError("payload_bytes must be bytes")
        if not isinstance(secret, str) or not secret:
            raise ValidationError("secret must be a non-empty string")
        return hmac.new(secret.encode("utf-8"), bytes(payload_bytes), hashlib.sha256).hexdigest()

    def generate_signature(self, payload_bytes: bytes, secret: Optional[str] = None) -> str:
        """Sign with ``secret`` or the dispatcher's default secret."""
        key = secret or self.secret
        if not key:
            raise ValidationError("No signing secret provided and no default secret configured")
        return self.calculate_signature(payload_bytes, key)

    def verify_signature(self, payload_bytes: bytes, signature: str, secret: Optional[str] = None) -> bool:
        """Constant-time check of ``signature`` (optionally prefixed ``sha256=``)."""
        key = secret or self.secret
        if not key or not isinstance(signature, str):
            return False
        sig = signature[7:] if signature.startswith("sha256=") else signature
        expected = self.calculate_signature(payload_bytes, key)
        return hmac.compare_digest(expected, sig)

    # ------------------------------------------------------------ subscribers
    def register(self, url: str, secret: Optional[str] = None, events: Optional[List[str]] = None) -> str:
        """Register an endpoint; ``events`` are topic patterns (``*``/``#`` wildcards)."""
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValidationError(f"Webhook url must start with http:// or https://, got {url!r}")
        patterns = list(events or [])
        if not all(isinstance(p, str) and p for p in patterns):
            raise ValidationError("events must be a list of non-empty topic patterns")
        sub = WebhookSubscription(id=f"WH-{uuid.uuid4().hex[:12]}", url=url, secret=secret, events=patterns)
        with self._lock:
            self._subscribers[sub.id] = sub
        return sub.id

    def unregister(self, subscription_id: str) -> bool:
        """Remove a subscription. Returns False if it did not exist."""
        with self._lock:
            return self._subscribers.pop(subscription_id, None) is not None

    def get_subscribers(self, topic: Optional[str] = None) -> List[WebhookSubscription]:
        """All subscriptions, or only those matching ``topic``."""
        with self._lock:
            subs = list(self._subscribers.values())
        return [s for s in subs if topic is None or s.matches(topic)]

    # --------------------------------------------------------------- delivery
    def _http_post(self, url: str, payload_bytes: bytes, headers: Dict[str, str]) -> int:
        """Send one POST and return the HTTP status code. Replaceable in tests."""
        req = Request(url, data=payload_bytes, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:  # noqa: S310 (http/https enforced)
                return int(resp.status)
        except HTTPError as exc:
            return int(exc.code)

    def dispatch(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        secret: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> DeliveryResult:
        """Sign and POST ``payload`` to ``endpoint``, retrying on network errors and non-2xx."""
        if not isinstance(payload, dict):
            raise ValidationError(f"Webhook payload must be a dict, got {type(payload).__name__}")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValidationError("endpoint must be a non-empty URL string")

        body = _encode(payload)
        key = secret or self.secret
        signature = self.calculate_signature(body, key) if key else None
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "open-logistics-webhooks/0.1",
            "X-Webhook-Timestamp": str(int(time.time())),
        }
        if signature:
            headers[SIGNATURE_HEADER] = signature
        if topic:
            headers["X-Webhook-Topic"] = topic

        result = DeliveryResult(endpoint=endpoint, success=False, signature=signature)
        delay = self.retry_delay_seconds
        for attempt in range(1, self.max_retries + 1):
            result.attempts = attempt
            try:
                status = self._http_post(endpoint, body, headers)
                result.status_code = status
                if 200 <= status < 300:
                    result.success = True
                    result.error = None
                    break
                result.error = f"HTTP {status}"
            except Exception as exc:  # network failures of any kind are retried
                result.error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries and delay > 0:
                time.sleep(delay)
                delay *= self.backoff_factor

        result.timestamp = time.time()
        self.history.append(result)
        return result

    def dispatch_event(self, event: WebhookEvent) -> List[DeliveryResult]:
        """Deliver ``event`` to every subscriber whose filters match its topic."""
        if not isinstance(event, WebhookEvent):
            raise ValidationError("dispatch_event expects a WebhookEvent")
        payload = event.to_payload()
        return [
            self.dispatch(sub.url, payload, secret=sub.secret, topic=event.topic)
            for sub in self.get_subscribers(event.topic)
        ]

    def attach(self, event_bus: EventBus, topic_pattern: str = "#") -> str:
        """Forward every matching bus event to webhook subscribers. Returns the bus subscription ID."""

        def _forward(evt: Event) -> None:
            data = evt.payload if isinstance(evt.payload, dict) else {"value": evt.payload}
            self.dispatch_event(
                WebhookEvent(
                    topic=evt.topic,
                    data=data,
                    event_id=evt.event_id or uuid.uuid4().hex,
                    timestamp=evt.timestamp or time.time(),
                )
            )

        return event_bus.subscribe(topic_pattern, _forward)
