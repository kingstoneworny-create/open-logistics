"""Thread-safe publish/subscribe EventBus with wildcard topic matching.

Implements decoupled in-memory event dispatching supporting single-level (*)
and multi-level (#, **) wildcards, with subscriber error isolation.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def topic_matches(pattern: str, topic: str) -> bool:
    """Evaluate whether an event topic matches a subscription pattern.

    Wildcard syntax:
        `*`    Matches exactly one topic segment between dots.
        `#` or `**` Matches zero or more topic segments.
        Exact  Matches the exact topic string.

    Examples:
        - `order.*` matches `order.created`, `order.updated`
        - `order.*` does NOT match `order` or `order.item.added`
        - `order.#` matches `order`, `order.created`, `order.item.added`
        - `#.alert` matches `alert`, `inventory.alert`, `warehouse.shelf.alert`
        - `#` matches any topic

    Args:
        pattern: Subscription topic pattern.
        topic: Actual event topic name.

    Returns:
        True if the topic matches the pattern, False otherwise.
    """
    if pattern == topic or pattern in ("#", "**"):
        return True

    p_tokens = pattern.split(".")
    t_tokens = topic.split(".")

    def _match(p_idx: int, t_idx: int) -> bool:
        while p_idx < len(p_tokens) and t_idx < len(t_tokens):
            token = p_tokens[p_idx]
            if token in ("#", "**"):
                next_p = p_idx + 1
                if next_p == len(p_tokens):
                    return True
                for next_t in range(t_idx, len(t_tokens) + 1):
                    if _match(next_p, next_t):
                        return True
                return False
            elif token == "*" or token == t_tokens[t_idx]:
                p_idx += 1
                t_idx += 1
            else:
                return False

        while p_idx < len(p_tokens) and p_tokens[p_idx] in ("#", "**"):
            p_idx += 1

        return p_idx == len(p_tokens) and t_idx == len(t_tokens)

    return _match(0, 0)


@dataclass
class Event:
    """Logistics event message for asynchronous or decoupled notifications.

    Attributes:
        topic: Dot-delimited topic path (e.g. 'order.created', 'stock.replenished').
        payload: Event payload dictionary containing contextual data.
        source: Component or service identifier producing the event.
        timestamp: Epoch timestamp when event occurred.
        event_id: Unique event identifier (UUID string).
    """

    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = 0.0
    event_id: str = ""

    def __post_init__(self) -> None:
        if not self.topic or not isinstance(self.topic, str):
            raise ValueError(f"Event topic must be a non-empty string, got {self.topic!r}")
        if not self.event_id:
            self.event_id = uuid.uuid4().hex
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not isinstance(self.payload, dict):
            raise ValueError(f"Event payload must be a dict, got {type(self.payload).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to a dictionary."""
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Instantiate event from a dictionary."""
        return cls(
            topic=data["topic"],
            payload=data.get("payload", {}),
            source=data.get("source", ""),
            timestamp=data.get("timestamp", 0.0),
            event_id=data.get("event_id", ""),
        )


@dataclass
class Subscriber:
    """Registered event subscriber binding a topic pattern to a callback.

    Attributes:
        callback: Callable invoked upon matching event dispatch.
        subscription_id: Unique identifier for unsubscription.
        topic_pattern: Subscribed topic pattern.
    """

    callback: Callable[[Event], Any]
    subscription_id: str
    topic_pattern: str


class EventBus:
    """Thread-safe in-memory publish/subscribe event bus.

    Features:
        - Thread-safety via `threading.RLock()`.
        - Single-level (*) and multi-level (#, **) topic pattern wildcards.
        - Subscriber error isolation: exceptions raised inside a handler
          are captured, logged, and recorded without disrupting remaining
          subscribers or failing the publisher.
    """

    def __init__(self, isolated_errors: bool = True) -> None:
        self._subscribers: Dict[str, Subscriber] = {}
        self._lock = threading.RLock()
        self._isolated_errors = isolated_errors
        self._failed_dispatches: List[Dict[str, Any]] = []

    def subscribe(self, topic_pattern: str, handler: Callable[[Event], Any]) -> str:
        """Register a subscriber handler for a topic pattern.

        Args:
            topic_pattern: Exact topic or pattern containing `*` or `#`/`**`.
            handler: Callable taking a single `Event` parameter.

        Returns:
            Unique subscription ID for subsequent unsubscription.
        """
        if not topic_pattern or not isinstance(topic_pattern, str):
            raise ValueError("topic_pattern must be a non-empty string")
        if not callable(handler):
            raise ValueError("handler must be a callable object")

        sub_id = uuid.uuid4().hex
        subscriber = Subscriber(
            callback=handler,
            subscription_id=sub_id,
            topic_pattern=topic_pattern,
        )
        with self._lock:
            self._subscribers[sub_id] = subscriber
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscriber by its subscription ID.

        Args:
            subscription_id: ID returned from `subscribe()`.

        Returns:
            True if subscriber was found and removed, False otherwise.
        """
        with self._lock:
            if subscription_id in self._subscribers:
                del self._subscribers[subscription_id]
                return True
            return False

    def publish(self, event: Event) -> int:
        """Publish an event to all matching subscribers.

        Args:
            event: Event object to dispatch.

        Returns:
            Count of matching subscribers invoked.
        """
        if not isinstance(event, Event):
            raise TypeError(f"Expected Event instance, got {type(event).__name__}")

        with self._lock:
            matching = [
                sub
                for sub in self._subscribers.values()
                if topic_matches(sub.topic_pattern, event.topic)
            ]

        notified = 0
        for sub in matching:
            try:
                sub.callback(event)
                notified += 1
            except Exception as exc:
                notified += 1
                logger.exception(
                    "Error executing subscriber %s for event %s on topic '%s': %s",
                    sub.subscription_id,
                    event.event_id,
                    event.topic,
                    exc,
                )
                with self._lock:
                    self._failed_dispatches.append({
                        "subscription_id": sub.subscription_id,
                        "topic_pattern": sub.topic_pattern,
                        "event_id": event.event_id,
                        "topic": event.topic,
                        "error": str(exc),
                        "timestamp": time.time(),
                    })
                if not self._isolated_errors:
                    raise

        return notified

    def clear(self) -> None:
        """Remove all active subscribers and recorded failures."""
        with self._lock:
            self._subscribers.clear()
            self._failed_dispatches.clear()

    def get_subscribers(self, topic: Optional[str] = None) -> List[Subscriber]:
        """Return active subscribers, optionally filtered by matching a specific topic."""
        with self._lock:
            if topic is None:
                return list(self._subscribers.values())
            return [
                sub
                for sub in self._subscribers.values()
                if topic_matches(sub.topic_pattern, topic)
            ]

    def get_failed_dispatches(self) -> List[Dict[str, Any]]:
        """Return list of captured subscriber dispatch failures."""
        with self._lock:
            return list(self._failed_dispatches)

    def clear_failed_dispatches(self) -> None:
        """Clear the failure dispatch log."""
        with self._lock:
            self._failed_dispatches.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._subscribers)
