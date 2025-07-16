# trade_manager.py

import time
from . import config
from . import binance_client 
import math
import threading
import queue

class TradeManager:
    def __init__(self):
        self.in_trade = False
        self.position = {}
        self.simulated_balance = 1000.0
        self.simulated_trades_log = []
        self.sl_order_id = None
        self.tp_order_id = None
        self.sl_is_conditional = True  # Portfolio Margin uses conditional orders
        self.tp_is_conditional = True
        self.monitoring_orders = False
        
        # Continuous monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.order_executed_queue = queue.Queue()
        self.monitoring_active = False

    def enter_trade(self, side, entry_price, position_size, atr):
        if self.in_trade:
            print("Error: Already in a trade.")
            return

        # Validate inputs
        if entry_price <= 0:
            print(f"Error: Invalid entry price {entry_price}")
            return
        if atr <= 0:
            print(f"Error: Invalid ATR {atr}")
            return

        self.in_trade = True
        side_code = 1 if side == 'long' else 2
        
        # Calculate SL and TP prices
        sl_price = entry_price - config.SL_ATR_MULT * atr if side == 'long' else entry_price + config.SL_ATR_MULT * atr
        tp_price = entry_price + config.TP_ATR_MULT * atr if side == 'long' else entry_price - config.TP_ATR_MULT * atr

        # Validate calculated prices
        if sl_price <= 0 or tp_price <= 0:
            print(f"Error: Invalid SL ({sl_price}) or TP ({tp_price}) prices calculated")
            print(f"Entry: {entry_price}, ATR: {atr}, SL mult: {config.SL_ATR_MULT}, TP mult: {config.TP_ATR_MULT}")
            self.in_trade = False
            return

        # Trailing stop setup
        use_trailing = (config.TRAIL_START_MULT > 0.0) and (config.TRAIL_STOP_MULT > 0.0)
        if use_trailing:
            trail_trigger = entry_price + config.TRAIL_START_MULT * atr if side == 'long' else entry_price - config.TRAIL_START_MULT * atr
        else:
            trail_trigger = 0.0

        self.position = {
            "side": side,
            "side_code": side_code,
            "entry_price": entry_price,
            "position_size": position_size,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "atr_at_entry": atr,
            "max_p": entry_price,
            "min_p": entry_price,
            "use_trailing": use_trailing,
            "trailing_on": False,
            "trail_trigger": trail_trigger,
            "trail_stop": 0.0,
            "entry_time": time.time()
        }

        print(f"--- ENTERING TRADE ---")
        print(f"Side: {side}, Size: {position_size:.4f}, Entry: ${entry_price:.2f}")
        print(f"SL: ${sl_price:.2f}, TP: ${tp_price:.2f}")
        if use_trailing:
            print(f"Trailing will trigger at: ${trail_trigger:.2f}")

        # Place SL and TP orders for live trading
        if not config.SIMULATED:
            self._place_sl_tp_orders()
            # Start continuous monitoring if orders were placed
            if self.monitoring_orders:
                self.start_continuous_monitoring()

    def is_position_still_open(self):
        """Check if we still have an open position"""
        if config.SIMULATED:
            return self.in_trade
        
        try:
            position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
            if position_info:
                position_amt = float(position_info.get('positionAmt', 0))
                return position_amt != 0
            return False
        except Exception as e:
            print(f"Error checking position: {e}")
            return True  # Assume position is still open if we can't check
        
    def sync_trade_state_with_position(self):
        """
        Sync the bot's trade state with the actual position.
        This helps recover from cases where orders were manually cancelled
        or the position was closed outside the bot.
        """
        if config.SIMULATED:
            return None, 0.0
        
        if not self.in_trade:
            return None, 0.0
        
        try:
            # Check actual position
            position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
            
            if not position_info:
                print("Could not get position info")
                return None, 0.0
            
            position_amt = float(position_info.get('positionAmt', 0))
            
            if position_amt == 0:
                print(f"SYNC: No position found but bot thinks it's in trade - closing trade")
                print(f"Current SL ID: {self.sl_order_id}, TP ID: {self.tp_order_id}")
                
                # Cancel any remaining orders
                if self.sl_order_id:
                    try:
                        binance_client.client.cancel_order(config.TRADING_SYMBOL, self.sl_order_id, self.sl_is_conditional)
                    except:
                        pass
                if self.tp_order_id:
                    try:
                        binance_client.client.cancel_order(config.TRADING_SYMBOL, self.tp_order_id, self.tp_is_conditional)
                    except:
                        pass
                
                self.monitoring_orders = False
                
                # Determine exit reason based on order status
                exit_reason = "UNKNOWN_CLOSE"
                exit_price = self.position.get('entry_price', 0)
                
                # Check if we can determine which order was hit
                if self.sl_order_id:
                    sl_status = binance_client.client.get_order_status(config.TRADING_SYMBOL, self.sl_order_id, self.sl_is_conditional)
                    if sl_status and sl_status.get('strategyStatus') in ['TRIGGERED', 'FILLED']:
                        exit_reason = "SL"
                        exit_price = self.position.get('sl_price', exit_price)
                
                if self.tp_order_id:
                    tp_status = binance_client.client.get_order_status(config.TRADING_SYMBOL, self.tp_order_id, self.tp_is_conditional)
                    if tp_status and tp_status.get('strategyStatus') in ['TRIGGERED', 'FILLED']:
                        exit_reason = "TP"
                        exit_price = self.position.get('tp_price', exit_price)
                
                return exit_reason, exit_price
            else:
                print(f"SYNC: Position still open: {position_amt}")
                
                # Position is still open, check if we have monitoring orders
                if not self.sl_order_id and not self.tp_order_id:
                    print("SYNC: Position open but no orders - this is unusual")
                    # Could recreate orders here if needed, but for now just log it
                
                return None, 0.0
                
        except Exception as e:
            print(f"Error in sync_trade_state_with_position: {e}")
            return None, 0.0
    
    def _place_sl_tp_orders(self):
        if config.SIMULATED:
            return

        side = self.position['side']
        qty = self.position['position_size']
        sl = self.position['sl_price']
        tp = self.position['tp_price']

        # 1. Fetch futures filters from Binance
        info = binance_client.client.futures_exchange_info()
        sym = next(s for s in info['symbols'] if s['symbol'] == config.TRADING_SYMBOL)
        tick = float(next(f['tickSize'] for f in sym['filters'] if f['filterType']=='PRICE_FILTER'))
        step = float(next(f['stepSize'] for f in sym['filters'] if f['filterType']=='LOT_SIZE'))
        min_qty = float(next(f['minQty'] for f in sym['filters'] if f['filterType']=='LOT_SIZE'))

        # 2. Calculate decimal places for proper formatting
        tick_precision = len(str(tick).split('.')[-1]) if '.' in str(tick) else 0
        step_precision = len(str(step).split('.')[-1]) if '.' in str(step) else 0

        # 3. Round quantity and prices to allowed precision
        qty = round(qty / step) * step
        qty = round(qty, step_precision)  # Ensure proper decimal places
        
        sl = round(sl / tick) * tick
        sl = round(sl, tick_precision)  # Ensure proper decimal places
        
        tp = round(tp / tick) * tick
        tp = round(tp, tick_precision)  # Ensure proper decimal places

        if qty < min_qty:
            raise ValueError(f"Rounded qty {qty} is below minimum {min_qty}")

        # print(f"DEBUG: qty={qty}, sl={sl}, tp={tp}")
        # print(f"DEBUG: tick_precision={tick_precision}, step_precision={step_precision}")

        close_side = 'SELL' if side == 'long' else 'BUY'

        # 4. Submit Stop-Loss order
        try:
            sl_order = binance_client.client.create_stop_loss_order(config.TRADING_SYMBOL, close_side, qty, sl)
            
            if sl_order and isinstance(sl_order, dict):
                # For conditional orders, look for strategyId instead of orderId
                if 'strategyId' in sl_order:
                    self.sl_order_id = sl_order['strategyId']
                    self.sl_is_conditional = True
                elif 'orderId' in sl_order:
                    self.sl_order_id = sl_order['orderId']
                    self.sl_is_conditional = False
                elif 'id' in sl_order:
                    self.sl_order_id = sl_order['id']
                    self.sl_is_conditional = False
                else:
                    print(f"WARNING: Could not find order ID in SL response: {sl_order}")
                    self.sl_order_id = None
                    
                print(f"Stop-Loss order ID: {self.sl_order_id} (conditional: {self.sl_is_conditional})")
            else:
                print("ERROR: SL order creation failed or returned invalid response")
                self.sl_order_id = None
                
        except Exception as e:
            print(f"SL error: {e}")
            self.sl_order_id = None

        # 5. Submit Take-Profit order
        try:
            tp_order = binance_client.client.create_take_profit_order(config.TRADING_SYMBOL, close_side, qty, tp)
            
            if tp_order and isinstance(tp_order, dict):
                # For conditional orders, look for strategyId instead of orderId
                if 'strategyId' in tp_order:
                    self.tp_order_id = tp_order['strategyId']
                    self.tp_is_conditional = True
                elif 'orderId' in tp_order:
                    self.tp_order_id = tp_order['orderId']
                    self.tp_is_conditional = False
                elif 'id' in tp_order:
                    self.tp_order_id = tp_order['id']
                    self.tp_is_conditional = False
                else:
                    print(f"WARNING: Could not find order ID in TP response: {tp_order}")
                    self.tp_order_id = None
                    
                print(f"Take-Profit order ID: {self.tp_order_id} (conditional: {self.tp_is_conditional})")
            else:
                print("ERROR: TP order creation failed or returned invalid response")
                self.tp_order_id = None
                
        except Exception as e:
            print(f"TP error: {e}")
            self.tp_order_id = None

        # Only set monitoring to True if at least one order was placed successfully
        if self.sl_order_id or self.tp_order_id:
            self.monitoring_orders = True
            print(f"Order monitoring enabled. SL ID: {self.sl_order_id}, TP ID: {self.tp_order_id}")
        else:
            print("WARNING: No SL or TP orders were placed successfully")
            self.monitoring_orders = False

    def _cancel_remaining_order(self, executed_order_type):
        """Cancel the remaining order when one executes"""
        if config.SIMULATED:
            return
            
        if executed_order_type == 'SL' and self.tp_order_id:
            if binance_client.client.cancel_order(config.TRADING_SYMBOL, self.tp_order_id, self.tp_is_conditional):
                print(f"Cancelled remaining TP order: {self.tp_order_id}")
            self.tp_order_id = None
        elif executed_order_type == 'TP' and self.sl_order_id:
            if binance_client.client.cancel_order(config.TRADING_SYMBOL, self.sl_order_id, self.sl_is_conditional):
                print(f"Cancelled remaining SL order: {self.sl_order_id}")
            self.sl_order_id = None

    def start_continuous_monitoring(self):
        """Start continuous monitoring of orders in a background thread"""
        if config.SIMULATED or not self.monitoring_orders:
            return
        
        if self.monitoring_active:
            print("Monitoring already active")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._continuous_order_monitor, daemon=True)
        self.monitoring_thread.start()
        print("Started continuous order monitoring (5-second intervals)")

    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        if not self.monitoring_active:
            return
        
        self.stop_monitoring.set()
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        print("Stopped continuous order monitoring")

    def _continuous_order_monitor(self):
        """Background thread that monitors orders every 5 seconds"""
        while not self.stop_monitoring.is_set() and self.monitoring_orders:
            try:
                # Check if any orders have been executed
                exit_reason, exit_price = self.check_order_status()
                
                if exit_reason:
                    # Clear monitoring flag and order IDs since we've handled the cancellation
                    self.monitoring_orders = False
                    # Note: order IDs are already cleared in check_order_status when it calls _cancel_remaining_order
                    
                    # Put the result in the queue for the main thread to process
                    self.order_executed_queue.put((exit_reason, exit_price))
                    print(f"🔔 CONTINUOUS MONITOR: {exit_reason} order executed at ${exit_price:.2f}")
                    break
                
                # Wait 5 seconds or until stop signal
                if self.stop_monitoring.wait(5.0):
                    break
                    
            except Exception as e:
                print(f"Error in continuous monitoring: {e}")
                # Continue monitoring even if there's an error
                if self.stop_monitoring.wait(5.0):
                    break
        
        print("Continuous monitoring thread stopped")
        self.monitoring_active = False

    def check_for_executed_orders(self):
        """Check if any orders were executed in the background monitor"""
        try:
            # Non-blocking check for executed orders
            exit_reason, exit_price = self.order_executed_queue.get_nowait()
            return exit_reason, exit_price
        except queue.Empty:
            return None, 0.0

    def check_order_status(self):
        """Check if any of our SL/TP orders have been executed"""
        if config.SIMULATED or not self.monitoring_orders:
            return None, 0.0

        # Check SL order
        if self.sl_order_id:
            sl_status = binance_client.client.get_order_status(config.TRADING_SYMBOL, self.sl_order_id, self.sl_is_conditional)
            
            if sl_status:
                # print(f"DEBUG: SL order status: {sl_status}")
                
                # For conditional orders, check strategyStatus
                if self.sl_is_conditional:
                    status_field = 'strategyStatus'
                    order_status = sl_status.get(status_field, 'UNKNOWN')
                    
                    # Check if order was triggered/filled/cancelled/finished
                    if order_status in ['FILLED', 'TRIGGERED', 'EXPIRED', 'CANCELLED', 'FINISHED']:
                        print(f"SL order {order_status}!")
                        
                        # For cancelled orders, check if position is still open
                        if order_status == 'CANCELLED':
                            position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
                            if position_info:
                                position_amt = float(position_info.get('positionAmt', 0))
                                if position_amt == 0:
                                    print("SL was cancelled and position is closed - trade was executed")
                                    self._cancel_remaining_order('SL')
                                    self.monitoring_orders = False
                                    return 'SL', self.position['sl_price']
                                else:
                                    print(f"SL was cancelled but position still open ({position_amt}) - manual cancellation")
                                    # Don't close the trade yet, just remove the SL order ID
                                    self.sl_order_id = None
                                    self.sl_is_conditional = False
                                    return None, 0.0
                        else:
                            # Order was filled/triggered/expired/finished
                            # Try to get fill price
                            fill_price = 0.0
                            
                            # For FINISHED orders, check both avgPrice in the main response and in the nested status
                            if order_status == 'FINISHED':
                                # Check nested status first
                                if 'avgPrice' in sl_status and sl_status['avgPrice'] and float(sl_status['avgPrice']) > 0:
                                    fill_price = float(sl_status['avgPrice'])
                                elif sl_status.get('status') == 'FILLED':
                                    # Sometimes avgPrice is not in the conditional response, use stop price
                                    if 'stopPrice' in sl_status and sl_status['stopPrice']:
                                        fill_price = float(sl_status['stopPrice'])
                            else:
                                # For other statuses, try the usual fields
                                if 'avgPrice' in sl_status and sl_status['avgPrice']:
                                    fill_price = float(sl_status['avgPrice'])
                                elif 'price' in sl_status and sl_status['price']:
                                    fill_price = float(sl_status['price'])
                                elif 'stopPrice' in sl_status and sl_status['stopPrice']:
                                    fill_price = float(sl_status['stopPrice'])
                            
                            if fill_price <= 0:
                                fill_price = self.position['sl_price']
                                print(f"Could not determine SL fill price, using target: {fill_price}")
                            
                            self._cancel_remaining_order('SL')
                            self.monitoring_orders = False
                            return 'SL', fill_price
                else:
                    # Regular order
                    status_field = 'status'
                    order_status = sl_status.get(status_field, 'UNKNOWN')
                    
                    if order_status in ['FILLED', 'CANCELED', 'EXPIRED']:
                        if order_status == 'FILLED':
                            fill_price = float(sl_status.get('avgPrice', self.position['sl_price']))
                            self._cancel_remaining_order('SL')
                            self.monitoring_orders = False
                            return 'SL', fill_price
                        else:
                            # Order was cancelled/expired - check position
                            position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
                            if not position_info or float(position_info.get('positionAmt', 0)) == 0:
                                self._cancel_remaining_order('SL')
                                self.monitoring_orders = False
                                return 'SL', self.position['sl_price']
            else:
                # If we can't find the SL order, it might have been executed
                print(f"WARNING: Could not find SL order {self.sl_order_id}. It may have been executed.")
                
                # Check if we still have a position
                position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
                if not position_info or float(position_info.get('positionAmt', 0)) == 0:
                    print("No position found - SL may have been executed")
                    self._cancel_remaining_order('SL')
                    self.monitoring_orders = False
                    return 'SL', self.position['sl_price']

        # Check TP order
        if self.tp_order_id:
            tp_status = binance_client.client.get_order_status(config.TRADING_SYMBOL, self.tp_order_id, self.tp_is_conditional)
            
            if tp_status:
                # print(f"DEBUG: TP order status: {tp_status}")
                
                # For conditional orders, check strategyStatus
                if self.tp_is_conditional:
                    status_field = 'strategyStatus'
                    order_status = tp_status.get(status_field, 'UNKNOWN')
                    
                    # Check if order was triggered/filled/cancelled/finished
                    if order_status in ['FILLED', 'TRIGGERED', 'EXPIRED', 'CANCELLED', 'FINISHED']:
                        print(f"TP order {order_status}!")
                        
                        # For cancelled orders, check if position is still open
                        if order_status == 'CANCELLED':
                            position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
                            if position_info:
                                position_amt = float(position_info.get('positionAmt', 0))
                                if position_amt == 0:
                                    print("TP was cancelled and position is closed - trade was executed")
                                    self._cancel_remaining_order('TP')
                                    self.monitoring_orders = False
                                    return 'TP', self.position['tp_price']
                                else:
                                    print(f"TP was cancelled but position still open ({position_amt}) - manual cancellation")
                                    # Don't close the trade yet, just remove the TP order ID
                                    self.tp_order_id = None
                                    self.tp_is_conditional = False
                                    return None, 0.0
                        else:
                            # Order was filled/triggered/expired/finished
                            # Try to get fill price
                            fill_price = 0.0
                            
                            # For FINISHED orders, check both avgPrice in the main response and in the nested status
                            if order_status == 'FINISHED':
                                # Check nested status first
                                if 'avgPrice' in tp_status and tp_status['avgPrice'] and float(tp_status['avgPrice']) > 0:
                                    fill_price = float(tp_status['avgPrice'])
                                elif tp_status.get('status') == 'FILLED':
                                    # Sometimes avgPrice is not in the conditional response, use stop price
                                    if 'stopPrice' in tp_status and tp_status['stopPrice']:
                                        fill_price = float(tp_status['stopPrice'])
                            else:
                                # For other statuses, try the usual fields
                                if 'avgPrice' in tp_status and tp_status['avgPrice']:
                                    fill_price = float(tp_status['avgPrice'])
                                elif 'price' in tp_status and tp_status['price']:
                                    fill_price = float(tp_status['price'])
                                elif 'stopPrice' in tp_status and tp_status['stopPrice']:
                                    fill_price = float(tp_status['stopPrice'])
                            
                            if fill_price <= 0:
                                fill_price = self.position['tp_price']
                                print(f"Could not determine TP fill price, using target: {fill_price}")
                            
                            self._cancel_remaining_order('TP')
                            self.monitoring_orders = False
                            return 'TP', fill_price
                else:
                    # Regular order
                    status_field = 'status'
                    order_status = tp_status.get(status_field, 'UNKNOWN')
                    
                    if order_status in ['FILLED', 'CANCELED', 'EXPIRED']:
                        if order_status == 'FILLED':
                            fill_price = float(tp_status.get('avgPrice', self.position['tp_price']))
                            self._cancel_remaining_order('TP')
                            self.monitoring_orders = False
                            return 'TP', fill_price
                        else:
                            # Order was cancelled/expired - check position
                            position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
                            if not position_info or float(position_info.get('positionAmt', 0)) == 0:
                                self._cancel_remaining_order('TP')
                                self.monitoring_orders = False
                                return 'TP', self.position['tp_price']
            else:
                # If we can't find the TP order, it might have been executed
                print(f"WARNING: Could not find TP order {self.tp_order_id}. It may have been executed.")
                
                # Check if we still have a position
                position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
                if not position_info or float(position_info.get('positionAmt', 0)) == 0:
                    print("No position found - TP may have been executed")
                    self._cancel_remaining_order('TP')
                    self.monitoring_orders = False
                    return 'TP', self.position['tp_price']

        # Additional safety check - if both orders are gone but we're still "in trade"
        # Check if position actually exists
        if not self.sl_order_id and not self.tp_order_id and self.in_trade:
            print("Both SL and TP orders are missing - checking position status...")
            position_info = binance_client.client.get_position_info(config.TRADING_SYMBOL)
            if not position_info or float(position_info.get('positionAmt', 0)) == 0:
                print("No position found and no orders - trade must have been closed")
                self.monitoring_orders = False
                # Return a generic close signal with current price
                return 'MANUAL_CLOSE', self.position.get('entry_price', 0)

        return None, 0.0

    def check_exit_conditions(self, high, low):
        if not self.in_trade:
            return None, 0.0

        # First check if any live orders have been executed
        if not config.SIMULATED:
            exit_reason, exit_price = self.check_order_status()
            if exit_reason:
                return exit_reason, exit_price

        # For simulated trading, continue with the original logic
        pos = self.position
        
        # 1. Check Stop-Loss (simulated only)
        if config.SIMULATED:
            if (pos['side_code'] == 1 and low <= pos['sl_price']) or \
            (pos['side_code'] == 2 and high >= pos['sl_price']):
                print(f"--- EXIT: Stop Loss Hit ---")
                return 'SL', pos['sl_price']

            # 2. Check Take-Profit (simulated only)
            if (pos['side_code'] == 1 and high >= pos['tp_price']) or \
            (pos['side_code'] == 2 and low <= pos['tp_price']):
                print(f"--- EXIT: Take Profit Hit ---")
                return 'TP', pos['tp_price']

        # 3. Check Trailing-Stop (works for both simulated and live)
        if pos['use_trailing']:
            # Track peak prices
            if high > pos['max_p']: pos['max_p'] = high
            if low < pos['min_p']: pos['min_p'] = low

            # Activate trailing if trigger is hit
            if not pos['trailing_on']:
                if (pos['side_code'] == 1 and high >= pos['trail_trigger']) or \
                (pos['side_code'] == 2 and low <= pos['trail_trigger']):
                    pos['trailing_on'] = True
                    
                    # Set initial trail stop based on trail_stop_mult from current peak/trough
                    if pos['side_code'] == 1:  # Long
                        pos['trail_stop'] = pos['max_p'] - config.TRAIL_STOP_MULT * pos['atr_at_entry']
                    else:  # Short
                        pos['trail_stop'] = pos['min_p'] + config.TRAIL_STOP_MULT * pos['atr_at_entry']
                    
                    print(f"*** Trailing Stop Activated ***")
                    print(f"    Peak/Trough: ${pos['max_p'] if pos['side_code'] == 1 else pos['min_p']:.2f}")
                    print(f"    Trail stop set at: ${pos['trail_stop']:.2f} ({config.TRAIL_STOP_MULT} ATR from peak)")

                    # For live trading, update the SL order to the new trailing stop level
                    if not config.SIMULATED:
                        self._update_trailing_stop(pos['trail_stop'])

            # If trailing is active, update and check the stop
            if pos['trailing_on']:
                if pos['side_code'] == 1:  # Long position
                    new_trail = pos['max_p'] - config.TRAIL_STOP_MULT * pos['atr_at_entry']
                    if new_trail > pos['trail_stop']:
                        pos['trail_stop'] = new_trail
                        print(f"    Trailing stop updated to: ${new_trail:.2f}")
                        # For live trading, update the stop order
                        if not config.SIMULATED:
                            self._update_trailing_stop(new_trail)
                    
                    # Check if trailing stop hit (simulated only)
                    if config.SIMULATED and low <= pos['trail_stop']:
                        print(f"--- EXIT: Trailing Stop Hit ---")
                        return 'TRAIL', pos['trail_stop']
                else:  # Short position
                    new_trail = pos['min_p'] + config.TRAIL_STOP_MULT * pos['atr_at_entry']
                    if new_trail < pos['trail_stop']:
                        pos['trail_stop'] = new_trail
                        print(f"    Trailing stop updated to: ${new_trail:.2f}")
                        # For live trading, update the stop order
                        if not config.SIMULATED:
                            self._update_trailing_stop(new_trail)
                    
                    # Check if trailing stop hit (simulated only)
                    if config.SIMULATED and high >= pos['trail_stop']:
                        print(f"--- EXIT: Trailing Stop Hit ---")
                        return 'TRAIL', pos['trail_stop']

        return None, 0.0

    def _update_trailing_stop(self, new_stop_price):
        """Update the trailing stop order for live trading"""
        if config.SIMULATED:
            return
            
        # Cancel existing SL order
        if self.sl_order_id:
            binance_client.client.cancel_order(config.TRADING_SYMBOL, self.sl_order_id, self.sl_is_conditional)
            
        # Get precision info for proper formatting
        info = binance_client.client.futures_exchange_info()
        sym = next(s for s in info['symbols'] if s['symbol'] == config.TRADING_SYMBOL)
        tick = float(next(f['tickSize'] for f in sym['filters'] if f['filterType']=='PRICE_FILTER'))
        step = float(next(f['stepSize'] for f in sym['filters'] if f['filterType']=='LOT_SIZE'))
        
        # Calculate decimal places
        tick_precision = len(str(tick).split('.')[-1]) if '.' in str(tick) else 0
        step_precision = len(str(step).split('.')[-1]) if '.' in str(step) else 0
        
        # Round values to proper precision
        new_stop_price = round(new_stop_price / tick) * tick
        new_stop_price = round(new_stop_price, tick_precision)
        
        quantity = round(self.position['position_size'] / step) * step
        quantity = round(quantity, step_precision)
        
        # Place new trailing stop order
        side = self.position['side']
        close_side = 'SELL' if side == 'long' else 'BUY'
        
        try:
            sl_order = binance_client.client.create_stop_loss_order(
                config.TRADING_SYMBOL,
                close_side,
                quantity,
                new_stop_price
            )
            if sl_order:
                # Handle conditional orders properly
                if 'strategyId' in sl_order:
                    self.sl_order_id = sl_order['strategyId']
                    self.sl_is_conditional = True
                elif 'orderId' in sl_order:
                    self.sl_order_id = sl_order['orderId']
                    self.sl_is_conditional = False
                else:
                    print("WARNING: Could not find order ID in trailing stop response")
                    self.sl_order_id = None
                    return
                    
                print(f"Updated trailing stop to ${new_stop_price:.{tick_precision}f}, Order: {self.sl_order_id} (conditional: {self.sl_is_conditional})")
        except Exception as e:
            print(f"Error updating trailing stop: {e}")

    def close_trade(self, exit_price, reason):
        if not self.in_trade:
            return

        # Stop continuous monitoring first
        self.stop_continuous_monitoring()

        # Cancel any remaining orders for live trading
        # Only try to cancel if we still have order IDs (they might have been cleared by continuous monitor)
        if not config.SIMULATED:
            orders_to_cancel = []
            
            if self.sl_order_id:
                orders_to_cancel.append(("SL", self.sl_order_id, self.sl_is_conditional))
            
            if self.tp_order_id:
                orders_to_cancel.append(("TP", self.tp_order_id, self.tp_is_conditional))
            
            if orders_to_cancel:
                print(f"Cancelling {len(orders_to_cancel)} remaining orders...")
                for order_type, order_id, is_conditional in orders_to_cancel:
                    try:
                        success = binance_client.client.cancel_order(config.TRADING_SYMBOL, order_id, is_conditional)
                        if success:
                            print(f"✅ Cancelled {order_type} order: {order_id}")
                        else:
                            print(f"⚠️ {order_type} order {order_id} cancellation unclear")
                    except Exception as e:
                        print(f"❌ Failed to cancel {order_type} order {order_id}: {e}")
            else:
                print("No orders to cancel (already handled by continuous monitor)")
            
            # Clear order IDs and monitoring flag
            self.sl_order_id = None
            self.tp_order_id = None
            self.monitoring_orders = False

        # Calculate PnL
        pnl = 0.0
        if self.position['side'] == 'long':
            pnl = (exit_price - self.position['entry_price']) * self.position['position_size']
        else: # Short
            pnl = (self.position['entry_price'] - exit_price) * self.position['position_size']
        
        print(f"Trade Closed. Reason: {reason}, Exit Price: ${exit_price:.2f}, PnL: ${pnl:.2f}")

        if config.SIMULATED:
            self.simulated_balance += pnl
            print(f"Simulated Balance Updated: ${self.simulated_balance:.2f}")
            self.simulated_trades_log.append({**self.position, "exit_price": exit_price, "pnl": pnl, "reason": reason})

        # Reset state
        self.in_trade = False
        self.position = {}
        self.sl_order_id = None
        self.tp_order_id = None
        self.sl_is_conditional = True
        self.tp_is_conditional = True
        self.monitoring_orders = False