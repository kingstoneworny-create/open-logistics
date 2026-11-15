"""Unit tests for thread-safe publish/subscribe EventBus and topic wildcard matching."""

import threading
import time
from typing import List

import pytest

from open_logistics.core.events import Event, EventBus, Subscriber, topic_matches


class TestTopicMatching:
    """Tests for MQTT/AMQP-style topic wildcard matching."""

    def test_exact_matches(self) -> None:
        assert topic_matches("order.created", "order.created")
        assert topic_matches("inventory.stock.depleted", "inventory.stock.depleted")
        assert not topic_matches("order.created", "order.updated")
        assert not topic_matches("order.created", "order")
        assert not topic_matches("order", "order.created")

    def test_single_level_wildcard(self) -> None:
        pattern = "order.*"
        assert topic_matches(pattern, "order.created")
        assert topic_matches(pattern, "order.updated")
        assert topic_matches(pattern, "order.cancelled")
        assert not topic_matches(pattern, "order")
        assert not topic_matches(pattern, "order.item.added")
        assert not topic_matches(pattern, "other.created")

        pattern_middle = "warehouse.*.status"
        assert topic_matches(pattern_middle, "warehouse.zoneA.status")
        assert topic_matches(pattern_middle, "warehouse.dock1.status")
        assert not topic_matches(pattern_middle, "warehouse.status")
        assert not topic_matches(pattern_middle, "warehouse.zoneA.aisle1.status")

    def test_multi_level_wildcard_hash(self) -> None:
        pattern = "order.#"
        assert topic_matches(pattern, "order")
        assert topic_matches(pattern, "order.created")
        assert topic_matches(pattern, "order.item.added")
        assert topic_matches(pattern, "order.item.line.priority.high")
        assert not topic_matches(pattern, "inventory.order")

        # Global match
        assert topic_matches("#", "anything")
        assert topic_matches("#", "deeply.nested.topic.name.here")

        # Suffix multi-level match
        pattern_suffix = "#.alert"
        assert topic_matches(pattern_suffix, "alert")
        assert topic_matches(pattern_suffix, "order.alert")
        assert topic_matches(pattern_suffix, "warehouse.temperature.sensor.alert")
        assert not topic_matches(pattern_suffix, "alert.resolved")

    def test_multi_level_wildcard_double_star(self) -> None:
        pattern = "order.**"
        assert topic_matches(pattern, "order")
        assert topic_matches(pattern, "order.created")
        assert topic_matches(pattern, "order.item.added")
        assert topic_matches("**", "any.topic")

    def test_mixed_wildcards(self) -> None:
        pattern = "facility.*.rack.#"
        assert topic_matches(pattern, "facility.east.rack")
        assert topic_matches(pattern, "facility.east.rack.shelf1")
        assert topic_matches(pattern, "facility.west.rack.shelf2.bin3")
        assert not topic_matches(pattern, "facility.rack")
        assert not topic_matches(pattern, "facility.east.zone.rack")


class TestEventModel:
    """Tests for Event dataclass."""

    def test_event_auto_fields(self) -> None:
        e = Event(topic="order.created", payload={"order_id": "ORD-1"})
        assert e.topic == "order.created"
        assert e.payload == {"order_id": "ORD-1"}
        assert len(e.event_id) > 0
        assert e.timestamp > 0.0

    def test_invalid_event_topic_or_payload(self) -> None:
        with pytest.raises(ValueError, match="topic"):
            Event(topic="", payload={})
        with pytest.raises(ValueError, match="payload"):
            Event(topic="order.created", payload="not_a_dict")  # type: ignore[arg-type]

    def test_event_serialization_roundtrip(self) -> None:
        e = Event(
            topic="stock.replenished",
            payload={"sku": "SKU-A", "qty": 50},
            source="ReplenishmentEngine",
            timestamp=1700000000.0,
            event_id="evt-123",
        )
        d = e.to_dict()
        restored = Event.from_dict(d)
        assert restored.topic == e.topic
        assert restored.payload == e.payload
        assert restored.source == e.source
        assert restored.timestamp == e.timestamp
        assert restored.event_id == e.event_id


class TestEventBus:
    """Tests for EventBus pub/sub, subscriptions, error isolation, and thread safety."""

    def test_subscribe_and_publish_exact(self, event_bus: EventBus) -> None:
        received: List[Event] = []

        sub_id = event_bus.subscribe("order.created", lambda evt: received.append(evt))
        assert len(event_bus) == 1

        event = Event(topic="order.created", payload={"id": "O1"})
        notified = event_bus.publish(event)

        assert notified == 1
        assert len(received) == 1
        assert received[0].payload == {"id": "O1"}

    def test_wildcard_dispatch(self, event_bus: EventBus) -> None:
        exact_events: List[Event] = []
        wildcard_events: List[Event] = []
        all_events: List[Event] = []

        event_bus.subscribe("order.created", lambda e: exact_events.append(e))
        event_bus.subscribe("order.*", lambda e: wildcard_events.append(e))
        event_bus.subscribe("#", lambda e: all_events.append(e))

        evt1 = Event(topic="order.created", payload={"step": 1})
        evt2 = Event(topic="order.cancelled", payload={"step": 2})
        evt3 = Event(topic="inventory.depleted", payload={"step": 3})

        event_bus.publish(evt1)
        event_bus.publish(evt2)
        event_bus.publish(evt3)

        assert len(exact_events) == 1
        assert exact_events[0].payload["step"] == 1

        assert len(wildcard_events) == 2
        assert [e.payload["step"] for e in wildcard_events] == [1, 2]

        assert len(all_events) == 3

    def test_unsubscribe(self, event_bus: EventBus) -> None:
        events: List[Event] = []
        sub_id = event_bus.subscribe("test.topic", lambda e: events.append(e))
        assert len(event_bus) == 1

        event_bus.publish(Event(topic="test.topic", payload={"val": 1}))
        assert len(events) == 1

        assert event_bus.unsubscribe(sub_id) is True
        assert len(event_bus) == 0
        assert event_bus.unsubscribe("non-existent-id") is False

        event_bus.publish(Event(topic="test.topic", payload={"val": 2}))
        assert len(events) == 1  # No new events received

    def test_clear_bus(self, event_bus: EventBus) -> None:
        event_bus.subscribe("topic1", lambda e: None)
        event_bus.subscribe("topic2", lambda e: None)
        assert len(event_bus) == 2

        event_bus.clear()
        assert len(event_bus) == 0

    def test_get_subscribers_filtered(self, event_bus: EventBus) -> None:
        s1 = event_bus.subscribe("order.*", lambda e: None)
        s2 = event_bus.subscribe("warehouse.#", lambda e: None)
        s3 = event_bus.subscribe("#", lambda e: None)

        all_subs = event_bus.get_subscribers()
        assert len(all_subs) == 3

        order_subs = event_bus.get_subscribers("order.created")
        assert len(order_subs) == 2  # matches order.* and #
        matching_ids = {s.subscription_id for s in order_subs}
        assert s1 in matching_ids
        assert s3 in matching_ids
        assert s2 not in matching_ids

    def test_subscriber_error_isolation(self, event_bus: EventBus) -> None:
        successful_events: List[Event] = []

        def failing_handler(evt: Event) -> None:
            raise RuntimeError("Database connection timed out during event processing")

        def normal_handler(evt: Event) -> None:
            successful_events.append(evt)

        event_bus.subscribe("critical.event", failing_handler)
        event_bus.subscribe("critical.event", normal_handler)

        evt = Event(topic="critical.event", payload={"action": "test"})
        notified = event_bus.publish(evt)

        # Both subscribers were invoked
        assert notified == 2
        # Normal handler received the event despite previous handler failure
        assert len(successful_events) == 1

        failed = event_bus.get_failed_dispatches()
        assert len(failed) == 1
        assert "Database connection timed out" in failed[0]["error"]
        assert failed[0]["topic"] == "critical.event"

        event_bus.clear_failed_dispatches()
        assert len(event_bus.get_failed_dispatches()) == 0

    def test_subscriber_error_propagation_when_not_isolated(self) -> None:
        bus = EventBus(isolated_errors=False)

        def failing_handler(evt: Event) -> None:
            raise ValueError("Direct failure")

        bus.subscribe("fail.topic", failing_handler)
        with pytest.raises(ValueError, match="Direct failure"):
            bus.publish(Event(topic="fail.topic", payload={}))

    def test_concurrent_publish_and_subscribe(self, event_bus: EventBus) -> None:
        """Verify thread-safety under concurrent load."""
        counter = {"count": 0}
        lock = threading.Lock()

        def handler(e: Event) -> None:
            with lock:
                counter["count"] += 1

        event_bus.subscribe("concurrent.event", handler)

        num_threads = 8
        events_per_thread = 50

        def worker() -> None:
            for i in range(events_per_thread):
                event_bus.publish(Event(topic="concurrent.event", payload={"i": i}))

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter["count"] == num_threads * events_per_thread
