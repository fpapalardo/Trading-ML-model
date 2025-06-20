"""
Order Management System with OCO (One Cancels Other) functionality

This module handles order lifecycle management including:
- Tracking related orders (entry, TP, SL)
- Automatically cancelling opposite orders when one fills
- Order status monitoring
"""

import asyncio
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import timezone, timedelta

from projectx_connector import ProjectXClient
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    WORKING = 0
    FILLED = 1
    CANCELLED = 2
    EXPIRED = 3
    REJECTED = 4


@dataclass
class OrderGroup:
    """Represents a group of related orders (entry, TP, SL)."""
    entry_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None
    sl_order_id: Optional[int] = None
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def get_exit_orders(self) -> List[int]:
        """Get list of exit order IDs (TP and SL)."""
        orders = []
        if self.tp_order_id:
            orders.append(self.tp_order_id)
        if self.sl_order_id:
            orders.append(self.sl_order_id)
        return orders
    
    def has_exit_orders(self) -> bool:
        """Check if this group has any exit orders."""
        return bool(self.tp_order_id or self.sl_order_id)


class OrderManager:
    """
    Manages order lifecycle and OCO (One Cancels Other) functionality.
    
    This class monitors order status and automatically cancels related orders
    when one side of a bracket is filled.
    """
    
    def __init__(self, px_client: ProjectXClient, check_interval: float = 2.0):
        """
        Initialize the order manager.
        
        Args:
            px_client: ProjectX API client instance
            check_interval: How often to check order status (seconds)
        """
        self.px = px_client
        self.check_interval = check_interval
        
        # Order tracking
        self.active_groups: Dict[int, OrderGroup] = {}  # entry_order_id -> OrderGroup
        self.order_to_group: Dict[int, int] = {}  # any_order_id -> entry_order_id
        
        # Thread control
        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Record when we started watching orders
        self.startup_time = datetime.now(timezone.utc)
        
        # Callbacks
        self.on_order_filled = None
        self.on_order_cancelled = None
        self.on_order_error = None

        self.filled_orders: Set[int] = set()  # Track orders we know are filled
        self.known_statuses: Dict[int, OrderStatus] = {}
        
    def start(self):
        """Start the order monitoring thread."""
        if self._running:
            logger.warning("Order manager already running")
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Order manager started")
    
    def stop(self):
        """Stop the order monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Order manager stopped")
    
    def register_order_group(
        self,
        entry_order_id: int,
        tp_order_id: Optional[int] = None,
        sl_order_id: Optional[int] = None
    ) -> OrderGroup:
        """
        Register a new order group for OCO management.
        
        Args:
            entry_order_id: Market entry order ID
            tp_order_id: Take profit order ID
            sl_order_id: Stop loss order ID
            
        Returns:
            Created OrderGroup instance
        """
        with self._lock:
            # Create order group
            group = OrderGroup(
                entry_order_id=entry_order_id,
                tp_order_id=tp_order_id,
                sl_order_id=sl_order_id
            )
            
            # Track in both directions
            self.active_groups[entry_order_id] = group
            self.order_to_group[entry_order_id] = entry_order_id
            
            if tp_order_id:
                self.order_to_group[tp_order_id] = entry_order_id
            if sl_order_id:
                self.order_to_group[sl_order_id] = entry_order_id
            
            logger.info(f"Registered order group: Entry={entry_order_id}, TP={tp_order_id}, SL={sl_order_id}")
            return group
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in separate thread."""
        logger.info("Order monitoring loop started")
        
        while self._running:
            try:
                # Get current open orders
                open_orders = self.px.search_open_orders()
                open_order_ids = {o.get('id') or o.get('orderId') for o in open_orders}
                
                # Check each active group
                groups_to_check = []
                with self._lock:
                    groups_to_check = list(self.active_groups.values())
                
                for group in groups_to_check:
                    if not group.is_active:
                        continue
                        
                    self._check_order_group(group, open_order_ids)
                
                # Clean up old groups
                self._cleanup_old_groups()
                
            except Exception as e:
                logger.error(f"Error in order monitor loop: {e}")
                if self.on_order_error:
                    self.on_order_error(e)
            
            # Wait before next check
            threading.Event().wait(self.check_interval)
        
        logger.info("Order monitoring loop stopped")
    
    def _check_order_group(self, group: OrderGroup, open_order_ids: Set[int]):
        """
        Check status of an order group and handle OCO logic.
        Modified to prevent false cancellations.
        """
        # Skip groups created before manager started
        if group.created_at < self.startup_time:
            return
            
        # Skip recently created groups to avoid false positives
        order_age = (datetime.now(timezone.utc) - group.created_at).total_seconds()
        if order_age < 60:  # Wait 60 seconds to ensure order is in system
            return
            
        # Check if we already know the status from real-time updates
        entry_status = self.known_statuses.get(group.entry_order_id)
        
        # Check if TP was filled
        if group.tp_order_id:
            if group.tp_order_id in self.filled_orders:
                # We know it's filled from real-time update
                return  # Already handled
            elif group.tp_order_id not in open_order_ids:
                # Double-check it was actually filled
                if self._was_order_filled(group.tp_order_id):
                    logger.info(f"TP order {group.tp_order_id} was filled (REST API check)")
                    self._handle_tp_filled(group)
                return
        
        # Check if SL was filled
        if group.sl_order_id:
            if group.sl_order_id in self.filled_orders:
                # We know it's filled from real-time update
                return  # Already handled
            elif group.sl_order_id not in open_order_ids:
                # Double-check it was actually filled
                if self._was_order_filled(group.sl_order_id):
                    logger.info(f"SL order {group.sl_order_id} was filled (REST API check)")
                    self._handle_sl_filled(group)
                return
        
        # Check if entry was cancelled/rejected (NOT just missing from open orders)
        if group.entry_order_id:
            # If we know the status from real-time, use that
            if entry_status == OrderStatus.FILLED:
                # Entry is filled, don't cancel anything
                return
            elif entry_status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                # Entry was truly cancelled/rejected
                self._handle_entry_cancelled(group)
                return
            
            # Only check REST API if we don't have real-time status
            if group.entry_order_id not in open_order_ids:
                # Give it more time before checking - entry might have just filled
                order_age = (datetime.now(timezone.utc) - group.created_at).total_seconds()
                if order_age < 10:  # Wait 10 seconds before checking
                    return
                
                # Now check if it was filled or cancelled
                was_filled = self._was_order_filled(group.entry_order_id)
                if was_filled:
                    logger.info(f"Entry order {group.entry_order_id} was filled (REST API check)")
                    self.filled_orders.add(group.entry_order_id)
                    self.known_statuses[group.entry_order_id] = OrderStatus.FILLED
                else:
                    logger.info(f"Entry order {group.entry_order_id} was cancelled/rejected (REST API check)")
                    self._handle_entry_cancelled(group)
    
    def _was_order_filled(self, order_id: int) -> bool:
        """
        Check if an order was filled by searching recent orders.
        Enhanced to be more reliable.
        """
        try:
            # First check our cache
            if order_id in self.filled_orders:
                return True
            
            # Search recent orders with a longer window
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=2)  # Increased from 1 hour
            
            orders = self.px.search_orders(start=start_time, end=end_time)
            
            for order in orders:
                oid = order.get('id') or order.get('orderId')
                if oid == order_id:
                    status = order.get('status')
                    filled = status == OrderStatus.FILLED.value
                    
                    # Update our cache
                    self.known_statuses[order_id] = OrderStatus(status)
                    if filled:
                        self.filled_orders.add(order_id)
                    
                    return filled
                    
        except Exception as e:
            logger.error(f"Error checking if order {order_id} was filled: {e}")
            # In case of error, assume not filled to be safe
            
        return False
    
    def _handle_realtime_order_update(self, order_id: int, status: int, order_data: dict):
        """
        Handle real-time order update from SignalR.
        This provides immediate updates instead of waiting for polling.
        """
        # Update our known status cache
        self.known_statuses[order_id] = OrderStatus(status)
        
        # Track filled orders
        if status == OrderStatus.FILLED.value:
            self.filled_orders.add(order_id)
        
        # Find which group this order belongs to
        with self._lock:
            entry_id = self.order_to_group.get(order_id)
            if not entry_id:
                return  # Not tracking this order
            
            group = self.active_groups.get(entry_id)
            if not group or not group.is_active:
                return  # Group no longer active
            
            # Handle based on status
            if status == OrderStatus.FILLED.value:
                logger.info(f"Real-time update: Order {order_id} filled")
                
                # Determine which order was filled
                if order_id == group.tp_order_id:
                    self._handle_tp_filled(group)
                elif order_id == group.sl_order_id:
                    self._handle_sl_filled(group)
                elif order_id == group.entry_order_id:
                    logger.info(f"Entry order {order_id} filled - bracket is now active")
                    # Entry filled - DO NOT cancel exit orders!
                    
            elif status in (OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value):
                logger.info(f"Real-time update: Order {order_id} cancelled/rejected")
                
                if order_id == group.entry_order_id:
                    # Only cancel exits if entry was truly cancelled/rejected
                    self._handle_entry_cancelled(group)
    
    def _handle_tp_filled(self, group: OrderGroup):
        """Handle take profit order being filled."""
        with self._lock:
            group.is_active = False
            
            # Cancel the stop loss order
            if group.sl_order_id:
                try:
                    self.px.cancel_order(group.sl_order_id)
                    logger.info(f"Cancelled SL order {group.sl_order_id} (TP was filled)")
                    
                    if self.on_order_cancelled:
                        self.on_order_cancelled(group.sl_order_id, "TP filled")
                        
                except Exception as e:
                    logger.error(f"Failed to cancel SL order {group.sl_order_id}: {e}")
            
            # Notify callback
            if self.on_order_filled:
                self.on_order_filled(group.tp_order_id, "TP", group)
    
    def _handle_sl_filled(self, group: OrderGroup):
        """Handle stop loss order being filled."""
        with self._lock:
            group.is_active = False
            
            # Cancel the take profit order
            if group.tp_order_id:
                try:
                    self.px.cancel_order(group.tp_order_id)
                    logger.info(f"Cancelled TP order {group.tp_order_id} (SL was filled)")
                    
                    if self.on_order_cancelled:
                        self.on_order_cancelled(group.tp_order_id, "SL filled")
                        
                except Exception as e:
                    logger.error(f"Failed to cancel TP order {group.tp_order_id}: {e}")
            
            # Notify callback
            if self.on_order_filled:
                self.on_order_filled(group.sl_order_id, "SL", group)
    
    def _handle_entry_cancelled(self, group: OrderGroup):
        """Handle entry order being cancelled/rejected."""
        with self._lock:
            group.is_active = False
            
            # Cancel both TP and SL orders
            for order_id in group.get_exit_orders():
                try:
                    self.px.cancel_order(order_id)
                    logger.info(f"Cancelled order {order_id} (entry was cancelled)")
                    
                    if self.on_order_cancelled:
                        self.on_order_cancelled(order_id, "Entry cancelled")
                        
                except Exception as e:
                    logger.error(f"Failed to cancel order {order_id}: {e}")
    
    def _cleanup_old_groups(self):
        """Remove inactive groups older than 1 hour."""
        with self._lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            
            groups_to_remove = []
            for entry_id, group in self.active_groups.items():
                if not group.is_active and group.created_at < cutoff_time:
                    groups_to_remove.append(entry_id)
            
            for entry_id in groups_to_remove:
                group = self.active_groups[entry_id]
                
                # Clean up caches
                for order_id in [entry_id, group.tp_order_id, group.sl_order_id]:
                    if order_id:
                        self.filled_orders.discard(order_id)
                        self.known_statuses.pop(order_id, None)
                
                # Remove from mappings
                del self.active_groups[entry_id]
                self.order_to_group.pop(entry_id, None)
                self.order_to_group.pop(group.tp_order_id, None)
                self.order_to_group.pop(group.sl_order_id, None)
                
            if groups_to_remove:
                logger.debug(f"Cleaned up {len(groups_to_remove)} old order groups")
    
    def get_active_groups(self) -> List[OrderGroup]:
        """Get list of currently active order groups."""
        with self._lock:
            return [g for g in self.active_groups.values() if g.is_active]
    
    def cancel_all_orders(self):
        """Cancel all orders in all active groups."""
        groups = self.get_active_groups()
        
        for group in groups:
            with self._lock:
                group.is_active = False
                
            # Cancel all orders in the group
            all_orders = [group.entry_order_id] + group.get_exit_orders()
            
            for order_id in all_orders:
                if order_id:
                    try:
                        self.px.cancel_order(order_id)
                        logger.info(f"Cancelled order {order_id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order_id}: {e}")


# Enhanced place_order function that returns order response
def place_bracket_order(
    px_client: ProjectXClient,
    order_manager: OrderManager,
    contract_id: str,
    side: str,
    quantity: int,
    tp_price: float,
    sl_price: float
) -> Optional[OrderGroup]:
    """
    Place a bracket order (market entry with TP and SL).
    
    Args:
        px_client: ProjectX API client
        order_manager: Order manager instance
        contract_id: Contract to trade
        side: 'Buy' or 'Sell'
        quantity: Number of contracts
        tp_price: Take profit price
        sl_price: Stop loss price
        
    Returns:
        OrderGroup if successful, None otherwise
    """
    try:
        # Place market order
        entry_response = px_client.place_order(
            contract_id,
            side,
            quantity,
            order_type=2,  # Market order
        )
        
        if not entry_response.get('success'):
            logger.error(f"Failed to place entry order: {entry_response}")
            return None
            
        entry_order_id = entry_response.get('orderId')
        logger.info(f"Placed market {side} order: {entry_order_id}")
        
        # Place TP order (limit order, opposite side)
        tp_side = 'Sell' if side == 'Buy' else 'Buy'
        tp_response = px_client.place_order(
            contract_id,
            tp_side,
            quantity,
            order_type=1,  # Limit order
            limit_price=tp_price,
            linked_order_id=entry_order_id
        )
        
        tp_order_id = None
        if tp_response.get('success'):
            tp_order_id = tp_response.get('orderId')
            logger.info(f"Placed TP order at {tp_price}: {tp_order_id}")
        else:
            logger.error(f"Failed to place TP order: {tp_response}")
        
        # Place SL order (stop order, opposite side)
        sl_response = px_client.place_order(
            contract_id,
            tp_side,  # Same side as TP
            quantity,
            order_type=4,  # Stop order
            stop_price=sl_price,
            linked_order_id=entry_order_id
        )
        
        sl_order_id = None
        if sl_response.get('success'):
            sl_order_id = sl_response.get('orderId')
            logger.info(f"Placed SL order at {sl_price}: {sl_order_id}")
        else:
            logger.error(f"Failed to place SL order: {sl_response}")
        
        # Register with order manager
        group = order_manager.register_order_group(
            entry_order_id=entry_order_id,
            tp_order_id=tp_order_id,
            sl_order_id=sl_order_id
        )
        
        return group
        
    except Exception as e:
        logger.error(f"Failed to place bracket order: {e}")
        return None