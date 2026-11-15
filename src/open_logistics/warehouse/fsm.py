"""Pick-Pack-Ship 7-Stage Order Lifecycle Finite State Machine (FSM).

Stages:
CREATED -> ALLOCATED -> RELEASED -> PICKING -> PICKED -> PACKED -> SHIPPED
Also supports CANCELLED as a safe terminal state from any non-shipped status,
and atomic rollback to safely unwind reservations upon mid-fulfillment disruption.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from open_logistics.core.exceptions import FSMStateError, InvalidTransitionError  # noqa: F401  (re-exported)


class OrderStatus(str, Enum):
    """Seven formal lifecycle stages plus CANCELLED terminal state."""

    CREATED = "CREATED"
    ALLOCATED = "ALLOCATED"
    RELEASED = "RELEASED"
    PICKING = "PICKING"
    PICKED = "PICKED"
    PACKED = "PACKED"
    SHIPPED = "SHIPPED"
    CANCELLED = "CANCELLED"


class OrderFSM:
    """Order lifecycle state machine enforcing strict sequential forward transitions,

    terminal immutability, and atomic cancellation rollback guards.
    """

    # Allowed forward transitions
    _FORWARD_TRANSITIONS: Dict[OrderStatus, OrderStatus] = {
        OrderStatus.CREATED: OrderStatus.ALLOCATED,
        OrderStatus.ALLOCATED: OrderStatus.RELEASED,
        OrderStatus.RELEASED: OrderStatus.PICKING,
        OrderStatus.PICKING: OrderStatus.PICKED,
        OrderStatus.PICKED: OrderStatus.PACKED,
        OrderStatus.PACKED: OrderStatus.SHIPPED,
    }

    def __init__(
        self,
        order_id: str,
        initial_state: Union[OrderStatus, str] = OrderStatus.CREATED,
    ) -> None:
        self.order_id = str(order_id)
        self._state: OrderStatus = self._coerce_status(initial_state)
        self.history: List[Dict[str, Any]] = [
            {
                "from_state": None,
                "to_state": self._state.value,
                "timestamp": time.time(),
                "context": {"action": "init"},
            }
        ]

    @property
    def current_state(self) -> OrderStatus:
        """Current order lifecycle state."""
        return self._state

    @property
    def state(self) -> OrderStatus:
        """Alias for current_state."""
        return self._state

    @staticmethod
    def _coerce_status(status: Union[OrderStatus, str]) -> OrderStatus:
        """Convert string or OrderStatus enum to OrderStatus."""
        if isinstance(status, OrderStatus):
            return status
        try:
            return OrderStatus(str(status).upper())
        except ValueError:
            raise InvalidTransitionError(
                current_state="UNKNOWN",
                attempted_state=str(status),
                message=f"Invalid order status {status!r}",
            )

    def transition(
        self,
        new_status: Union[OrderStatus, str],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Advance the order FSM to a new state with atomic transition validation.

        Args:
            new_status: Target OrderStatus or status string.
            context: Optional diagnostic context dictionary (user, timestamp, wave_id, etc.).

        Raises:
            InvalidTransitionError: If the transition violates sequential guards or immutability.
        """
        target = self._coerce_status(new_status)

        # Terminal state check: SHIPPED cannot transition to anything
        if self._state == OrderStatus.SHIPPED:
            raise InvalidTransitionError(
                current_state=self._state.value,
                attempted_state=target.value,
                message=f"Order {self.order_id} is SHIPPED and cannot transition to {target.value}",
                context=context,
            )

        # Terminal state check: CANCELLED cannot transition
        if self._state == OrderStatus.CANCELLED:
            raise InvalidTransitionError(
                current_state=self._state.value,
                attempted_state=target.value,
                message=f"Order {self.order_id} is CANCELLED and cannot transition to {target.value}",
                context=context,
            )

        # Cancellation transition is permitted from any non-shipped, non-cancelled state
        if target == OrderStatus.CANCELLED:
            self._record_transition(target, context)
            return

        # Sequential forward transition check
        expected_forward = self._FORWARD_TRANSITIONS.get(self._state)
        if expected_forward != target:
            raise InvalidTransitionError(
                current_state=self._state.value,
                attempted_state=target.value,
                message=(
                    f"Illegal state transition for order {self.order_id}: "
                    f"cannot transition from '{self._state.value}' to '{target.value}'. "
                    f"Expected next state: '{expected_forward.value if expected_forward else 'NONE'}'"
                ),
                context=context,
            )

        self._record_transition(target, context)

    def _record_transition(self, target: OrderStatus, context: Optional[Dict[str, Any]] = None) -> None:
        old = self._state
        self._state = target
        self.history.append(
            {
                "from_state": old.value,
                "to_state": target.value,
                "timestamp": time.time(),
                "context": context or {},
            }
        )

    def rollback(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Rollback recent state transition upon cancellation or operational disruption.

        Reverts picking/release back to ALLOCATED, or allocated back to CREATED.
        Raises InvalidTransitionError if order is already SHIPPED or CANCELLED.
        """
        if self._state == OrderStatus.SHIPPED:
            raise InvalidTransitionError(
                current_state=self._state.value,
                attempted_state="ROLLBACK",
                message=f"Order {self.order_id} is SHIPPED and cannot be rolled back",
                context=context,
            )
        if self._state == OrderStatus.CANCELLED:
            raise InvalidTransitionError(
                current_state=self._state.value,
                attempted_state="ROLLBACK",
                message=f"Order {self.order_id} is already CANCELLED",
                context=context,
            )

        old_state = self._state
        # Determine target safe rollback state
        if old_state in (OrderStatus.PICKING, OrderStatus.RELEASED, OrderStatus.PICKED):
            new_state = OrderStatus.ALLOCATED
        elif old_state == OrderStatus.PACKED:
            new_state = OrderStatus.PICKED
        elif old_state == OrderStatus.ALLOCATED:
            new_state = OrderStatus.CREATED
        else:
            new_state = OrderStatus.CREATED

        self._state = new_state
        ctx = dict(context or {})
        ctx.update({"action": "rollback", "from_state": old_state.value, "to_state": new_state.value})
        self.history.append(
            {
                "from_state": old_state.value,
                "to_state": new_state.value,
                "timestamp": time.time(),
                "context": ctx,
            }
        )


class OrderLifecycleManager:
    """Coordinates order allocations, state transitions, and inventory reservation rollback."""

    def __init__(self) -> None:
        self.fsms: Dict[str, OrderFSM] = {}
        self.reservations: Dict[str, Dict[str, Any]] = {}

    def get_or_create_fsm(self, order_id: str) -> OrderFSM:
        """Retrieve or initialize an OrderFSM for an order ID."""
        if order_id not in self.fsms:
            self.fsms[order_id] = OrderFSM(order_id=order_id)
        return self.fsms[order_id]

    def allocate_order(
        self,
        order_id: str,
        sku: str,
        quantity: int,
        stock_item: Any,
    ) -> None:
        """Allocate inventory stock and advance order FSM to ALLOCATED."""
        stock_item.allocate(quantity)
        self.reservations[order_id] = {"sku": sku, "quantity": quantity}
        fsm = self.get_or_create_fsm(order_id)
        fsm.transition(OrderStatus.ALLOCATED, context={"sku": sku, "quantity": quantity})

    def rollback_order(
        self,
        order_id: str,
        stock_item: Optional[Any] = None,
    ) -> None:
        """Rollback order state and restore allocated stock units."""
        res = self.reservations.pop(order_id, None)
        if stock_item is not None and res is not None:
            stock_item.deallocate(res["quantity"])

        if order_id in self.fsms:
            self.fsms[order_id].rollback()
