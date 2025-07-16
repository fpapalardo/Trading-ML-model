# binance_client.py

import pandas as pd
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
from binance.client import Client
from binance.enums import *
from . import config
import math

class BinanceClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.papi_base_url = "https://papi.binance.com"
        
        try:
            self.client = Client(api_key, api_secret)
            # Test connection by making a simple request
            self._make_request("GET", "/papi/v1/ping")
            print("Successfully connected to Binance Portfolio Margin.")
        except Exception as e:
            print(f"Error connecting to Binance: {e}")
            self.client = None

    def _get_timestamp(self):
        return int(time.time() * 1000)

    def _generate_signature(self, query_string):
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, method, endpoint, params=None):
        """Make a request to the PAPI endpoints"""
        if params is None:
            params = {}
        
        # Add timestamp
        params['timestamp'] = self._get_timestamp()
        
        # Generate query string
        query_string = urlencode(params)
        
        # Generate signature
        signature = self._generate_signature(query_string)
        query_string += f"&signature={signature}"
        
        # Prepare headers
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        # Make request
        url = f"{self.papi_base_url}{endpoint}"
        
        # Handle all HTTP methods
        if method == "GET":
            response = requests.get(f"{url}?{query_string}", headers=headers)
        elif method == "POST":
            response = requests.post(f"{url}?{query_string}", headers=headers)
        elif method == "DELETE":
            response = requests.delete(f"{url}?{query_string}", headers=headers)
        elif method == "PUT":
            response = requests.put(f"{url}?{query_string}", headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def futures_exchange_info(self):
        """Get futures exchange info"""
        if not self.client:
            return None
        try:
            return self.client.futures_exchange_info()
        except Exception as e:
            print(f"Error fetching futures exchange info: {e}")
            return None
    
    def get_historical_candles(self, symbol, interval, start_str, end_str=None):
        """Get historical candles - this still works via regular client for market data"""
        if not self.client: 
            return pd.DataFrame()
        try:
            # Market data endpoints are still accessible via regular API
            klines = self.client.futures_historical_klines(symbol, interval, start_str, end_str=end_str)
            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error fetching historical candles: {e}")
            return pd.DataFrame()

    def get_portfolio_balance(self, asset='USDT'):
        """Get balance from Portfolio Margin account"""
        try:
            # Get all balances
            response = self._make_request("GET", "/papi/v1/balance")
            
            # If specific asset requested
            if asset:
                for balance in response:
                    if balance['asset'] == asset:
                        # Return the total wallet balance for the asset
                        return float(balance['totalWalletBalance'])
                return 0.0
            else:
                return response
        except Exception as e:
            print(f"Error fetching portfolio balance: {e}")
            return 0.0

    def get_futures_balance(self, asset='USDT'):
        """Wrapper method that now uses Portfolio Margin balance"""
        return self.get_portfolio_balance(asset)
            
    def set_leverage(self, symbol, leverage):
        """Set leverage for futures trading in Portfolio Margin"""
        if not self.client: 
            return
        try:
            leverage = int(leverage)
            # Try to use the UM futures endpoint through PAPI
            params = {
                'symbol': symbol,
                'leverage': leverage
            }
            response = self._make_request("POST", "/papi/v1/um/leverage", params)
            print(f"Leverage for {symbol} set to {leverage}x.")
        except Exception as e:
            print(f"Error setting leverage: {e}")
            # Try using the regular client as fallback
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                print(f"Leverage for {symbol} set to {leverage}x (via futures API).")
            except Exception as e2:
                print(f"Failed to set leverage via both methods: {e}, {e2}")

    def create_market_order(self, symbol, side, quantity):
        """Create a market order through Portfolio Margin"""
        if not self.client: 
            return None
        if quantity <= 0:
            print("Order quantity must be positive.")
            return None
        
        # Get precision for proper formatting
        info = self.client.futures_exchange_info()
        sym = next(s for s in info['symbols'] if s['symbol'] == symbol)
        step = float(next(f['stepSize'] for f in sym['filters'] if f['filterType']=='LOT_SIZE'))
        step_precision = len(str(step).split('.')[-1]) if '.' in str(step) else 0
        
        # Round quantity to proper precision
        quantity = round(quantity / step) * step
        quantity = round(quantity, step_precision)
        
        try:
            print(f"Placing MARKET order: {side} {quantity} {symbol}")
            
            # Use PAPI endpoint for UM futures orders
            params = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': f"{quantity:.{step_precision}f}"
            }
            
            order = self._make_request("POST", "/papi/v1/um/order", params)
            # print(f"DEBUG - PAPI Order Response: {order}")
            
            # For market orders, wait a moment and check if it executed
            if order and 'orderId' in order:
                time.sleep(0.5)  # Give it a moment to execute
                
                # Check order status to get execution details
                status_params = {
                    'symbol': symbol,
                    'orderId': order['orderId']
                }
                
                try:
                    updated_order = self._make_request("GET", "/papi/v1/um/order", status_params)
                    if updated_order:
                        # print(f"DEBUG - Updated Order Status: {updated_order}")
                        return updated_order
                except Exception as e:
                    print(f"Could not get updated order status: {e}")
                    
            return order
                
        except Exception as e:
            print(f"Error creating market order via PAPI: {e}")
            # Try using regular futures API as fallback
            try:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                return order
            except Exception as e2:
                print(f"Failed to create order via both methods: {e}, {e2}")
                return None

    def create_stop_loss_order(self, symbol, side, quantity, stop_price, limit_price_offset=None):
        """Create a stop-loss limit order through Portfolio Margin using conditional orders"""
        if not self.client: 
            return None
        
        # Use config value if offset not specified
        if limit_price_offset is None:
            limit_price_offset = config.SL_LIMIT_OFFSET if hasattr(config, 'SL_LIMIT_OFFSET') else 0.001
        
        # Validate stop price
        if stop_price <= 0:
            print(f"ERROR: Invalid stop price {stop_price}. Stop price must be positive!")
            return None
        
        # Get precision info for proper formatting
        info = self.client.futures_exchange_info()
        sym = next(s for s in info['symbols'] if s['symbol'] == symbol)
        tick = float(next(f['tickSize'] for f in sym['filters'] if f['filterType']=='PRICE_FILTER'))
        step = float(next(f['stepSize'] for f in sym['filters'] if f['filterType']=='LOT_SIZE'))
        
        # Calculate decimal places
        tick_precision = len(str(tick).split('.')[-1]) if '.' in str(tick) else 0
        step_precision = len(str(step).split('.')[-1]) if '.' in str(step) else 0
        
        # Calculate limit price (slightly worse than stop price to ensure execution)
        if side == 'SELL':  # Long position SL
            limit_price = stop_price * (1 - limit_price_offset)
        else:  # Short position SL
            limit_price = stop_price * (1 + limit_price_offset)
        
        # Round limit price to proper precision
        limit_price = round(limit_price / tick) * tick
        limit_price = round(limit_price, tick_precision)
        
        # Round stop price to proper precision (in case it wasn't already)
        stop_price = round(stop_price / tick) * tick
        stop_price = round(stop_price, tick_precision)
        
        # Round quantity to proper precision
        quantity = round(quantity / step) * step
        quantity = round(quantity, step_precision)
            
        try:
            print(f"Placing Stop Limit order: {side} {quantity} {symbol}")
            print(f"  Stop trigger: ${stop_price:.{tick_precision}f}, Limit price: ${limit_price:.{tick_precision}f}")
            
            # Use PAPI conditional order endpoint for stop limit
            params = {
                'symbol': symbol,
                'side': side,
                'type': 'STOP',  # STOP is stop limit order
                'quantity': f"{quantity:.{step_precision}f}",  # Format to proper precision
                'positionSide': 'BOTH',
                'price': f"{limit_price:.{tick_precision}f}",  # Format to proper precision
                'stopPrice': f"{stop_price:.{tick_precision}f}",  # Format to proper precision
                'workingType': 'MARK_PRICE',  # Use mark price for triggering
                'timeInForce': 'GTC',
                'reduceOnly': 'true',  # Use string instead of boolean
                'strategyType': 'STOP'
            }
            
            # print(f"DEBUG: Order params: {params}")
            order = self._make_request("POST", "/papi/v1/um/conditional/order", params)
            
            if order:
                # print(f"DEBUG: SL Order Response: {order}")
                return order
            else:
                print("ERROR: No response from SL order API")
                return None
            
        except Exception as e:
            print(f"Error creating stop limit order via PAPI: {e}")
            # Try without reduceOnly as fallback
            try:
                params.pop('reduceOnly', None)
                # print(f"DEBUG: Retry params without reduceOnly: {params}")
                order = self._make_request("POST", "/papi/v1/um/conditional/order", params)
                
                if order:
                    # print(f"DEBUG: SL Order Response (retry): {order}")
                    return order
                else:
                    print("ERROR: No response from SL order API (retry)")
                    return None
                    
            except Exception as e2:
                print(f"Failed to create stop limit: {e}, {e2}")
                return None

    def create_take_profit_order(self, symbol, side, quantity, stop_price, limit_price_offset=None):
        """Create a take-profit limit order through Portfolio Margin using conditional orders"""
        if not self.client: 
            return None
        
        # Use config value if offset not specified
        if limit_price_offset is None:
            limit_price_offset = config.TP_LIMIT_OFFSET if hasattr(config, 'TP_LIMIT_OFFSET') else 0.001
            
        # Validate stop price
        if stop_price <= 0:
            print(f"ERROR: Invalid take profit price {stop_price}. Price must be positive!")
            return None
        
        # Get precision info for proper formatting
        info = self.client.futures_exchange_info()
        sym = next(s for s in info['symbols'] if s['symbol'] == symbol)
        tick = float(next(f['tickSize'] for f in sym['filters'] if f['filterType']=='PRICE_FILTER'))
        step = float(next(f['stepSize'] for f in sym['filters'] if f['filterType']=='LOT_SIZE'))
        
        # Calculate decimal places
        tick_precision = len(str(tick).split('.')[-1]) if '.' in str(tick) else 0
        step_precision = len(str(step).split('.')[-1]) if '.' in str(step) else 0
        
        # Calculate limit price (slightly worse than stop price to ensure execution)
        if side == 'SELL':  # Long position TP
            limit_price = stop_price * (1 - limit_price_offset)
        else:  # Short position TP
            limit_price = stop_price * (1 + limit_price_offset)
        
        # Round limit price to proper precision
        limit_price = round(limit_price / tick) * tick
        limit_price = round(limit_price, tick_precision)
        
        # Round stop price to proper precision (in case it wasn't already)
        stop_price = round(stop_price / tick) * tick
        stop_price = round(stop_price, tick_precision)
        
        # Round quantity to proper precision
        quantity = round(quantity / step) * step
        quantity = round(quantity, step_precision)
            
        try:
            print(f"Placing Take Profit Limit order: {side} {quantity} {symbol}")
            print(f"  Trigger price: ${stop_price:.{tick_precision}f}, Limit price: ${limit_price:.{tick_precision}f}")
            
            # Use PAPI conditional order endpoint for take profit limit
            params = {
                'symbol': symbol,
                'side': side,
                'type': 'TAKE_PROFIT',  # TAKE_PROFIT is take profit limit order
                'quantity': f"{quantity:.{step_precision}f}",  # Format to proper precision
                'positionSide': 'BOTH',
                'price': f"{limit_price:.{tick_precision}f}",  # Format to proper precision
                'stopPrice': f"{stop_price:.{tick_precision}f}",  # Format to proper precision
                'workingType': 'MARK_PRICE',  # Use mark price for triggering
                'timeInForce': 'GTC',
                'reduceOnly': 'true',  # Use string instead of boolean
                'strategyType': 'TAKE_PROFIT'
            }
            
            # print(f"DEBUG: Order params: {params}")
            order = self._make_request("POST", "/papi/v1/um/conditional/order", params)
            
            if order:
                # print(f"DEBUG: TP Order Response: {order}")
                return order
            else:
                print("ERROR: No response from TP order API")
                return None
            
        except Exception as e:
            print(f"Error creating take profit limit order via PAPI: {e}")
            # Try without reduceOnly as fallback
            try:
                params.pop('reduceOnly', None)
                # print(f"DEBUG: Retry params without reduceOnly: {params}")
                order = self._make_request("POST", "/papi/v1/um/conditional/order", params)
                
                if order:
                    # print(f"DEBUG: TP Order Response (retry): {order}")
                    return order
                else:
                    print("ERROR: No response from TP order API (retry)")
                    return None
                    
            except Exception as e2:
                print(f"Failed to create take profit limit: {e}, {e2}")
                return None
            
    def get_futures_filters(self, symbol='BTCUSDT'):
        info = self.client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                f_ps = next(f for f in s['filters'] if f['filterType']=='PRICE_FILTER')
                f_ls = next(f for f in s['filters'] if f['filterType']=='LOT_SIZE')
                return float(f_ps['tickSize']), float(f_ls['stepSize']), float(f_ls['minQty'])
        raise ValueError(f"{symbol} not found")

    def round_price(self, price, tick):
        return math.floor(price / tick) * tick

    def round_quantity(self, qty, step):
        precision = int(round(-math.log(step, 10), 0))
        return round(qty, precision)
    
    def debug_all_orders(self, symbol):
        """Debug method to show all open orders for troubleshooting"""
        if not self.client:
            return
        
        try:
            print(f"\n=== DEBUG: All orders for {symbol} ===")
            
            # Get regular open orders
            try:
                regular_orders = self._make_request("GET", "/papi/v1/um/openOrders", {'symbol': symbol})
                print(f"Regular open orders: {len(regular_orders) if regular_orders else 0}")
                if regular_orders:
                    for order in regular_orders:
                        print(f"  Regular Order ID: {order.get('orderId')}, Status: {order.get('status')}, Side: {order.get('side')}")
            except Exception as e:
                print(f"Error getting regular orders: {e}")
            
            # Get conditional open orders  
            try:
                conditional_orders = self._make_request("GET", "/papi/v1/um/conditional/openOrders", {'symbol': symbol})
                print(f"Conditional open orders: {len(conditional_orders) if conditional_orders else 0}")
                if conditional_orders:
                    for order in conditional_orders:
                        print(f"  Conditional Strategy ID: {order.get('strategyId')}, Status: {order.get('strategyStatus')}, Side: {order.get('side')}")
            except Exception as e:
                print(f"Error getting conditional orders: {e}")
                
            print("=== END DEBUG ===\n")
            
        except Exception as e:
            print(f"Error in debug_all_orders: {e}")

    def cancel_order(self, symbol, order_id, is_conditional=False):
        """Cancel an existing order through Portfolio Margin"""
        if not self.client: 
            return False
        try:
            if is_conditional:
                # Cancel conditional order using strategyId
                params = {
                    'symbol': symbol,
                    'strategyId': order_id  # Use strategyId for conditional orders
                }
                result = self._make_request("DELETE", "/papi/v1/um/conditional/order", params)
            else:
                # Cancel regular order using orderId
                params = {
                    'symbol': symbol,
                    'orderId': order_id
                }
                result = self._make_request("DELETE", "/papi/v1/um/order", params)
            
            print(f"Successfully cancelled order {order_id} (conditional: {is_conditional})")
            return True
            
        except Exception as e:
            error_message = str(e)
            
            # These are acceptable errors that mean the order is already gone
            acceptable_errors = [
                "Order does not exist",
                "Unknown order sent",
                "order does not exist",
                "-2013",  # Order does not exist error code
                "-2011"   # Unknown order sent error code
            ]
            
            if any(acceptable_error in error_message for acceptable_error in acceptable_errors):
                print(f"Order {order_id} already cancelled/executed (conditional: {is_conditional})")
                return True  # Treat as success since the order is gone
            else:
                print(f"Error cancelling order {order_id}: {e}")
                return False

    def get_order_status(self, symbol, order_id, is_conditional=False):
        """Get the status of a specific order through Portfolio Margin"""
        if not self.client: 
            return None
        try:
            if is_conditional:
                # For conditional orders, we need to get all orders and filter by strategyId
                # since there's no single conditional order endpoint
                params = {'symbol': symbol}
                
                # First try to get from open conditional orders (most efficient)
                try:
                    open_orders = self._make_request("GET", "/papi/v1/um/conditional/openOrders", params)
                    if open_orders:
                        for order in open_orders:
                            if order.get('strategyId') == order_id:
                                return order
                except Exception as e:
                    print(f"Error checking open conditional orders: {e}")
                
                # If not found in open orders, check all conditional orders (recent ones)
                try:
                    all_orders = self._make_request("GET", "/papi/v1/um/conditional/allOrders", params)
                    if all_orders:
                        # Find the order with matching strategyId
                        for order in all_orders:
                            if order.get('strategyId') == order_id:
                                return order
                except Exception as e:
                    print(f"Error checking all conditional orders: {e}")
                    
                return None
            else:
                # Get regular order status using orderId
                params = {
                    'symbol': symbol,
                    'orderId': order_id
                }
                order = self._make_request("GET", "/papi/v1/um/order", params)
                return order
            
        except Exception as e:
            print(f"Error fetching order status for {order_id}: {e}")
            return None

    def get_open_orders(self, symbol):
        """Get all open orders for a symbol through Portfolio Margin"""
        if not self.client: 
            return []
        try:
            all_orders = []
            
            # Get regular open orders
            params = {'symbol': symbol}
            regular_orders = self._make_request("GET", "/papi/v1/um/openOrders", params)
            if regular_orders:
                for order in regular_orders:
                    order['isConditional'] = False
                all_orders.extend(regular_orders)
            
            # Get conditional open orders
            conditional_orders = self._make_request("GET", "/papi/v1/um/conditional/openOrders", params)
            if conditional_orders:
                for order in conditional_orders:
                    order['isConditional'] = True
                all_orders.extend(conditional_orders)
            
            return all_orders
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return []

    def get_position_info(self, symbol):
        """Get current position information through Portfolio Margin"""
        if not self.client: 
            return None
        try:
            # Get all positions
            positions = self._make_request("GET", "/papi/v1/um/positionRisk")
            
            # Filter for the specific symbol
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                    return pos
            return None
        except Exception as e:
            print(f"Error fetching position info via PAPI: {e}")
            # Fallback to regular API
            try:
                positions = self.client.futures_position_information(symbol=symbol)
                for pos in positions:
                    if float(pos['positionAmt']) != 0:
                        return pos
                return None
            except Exception as e2:
                print(f"Failed to get position info via both methods: {e}, {e2}")
                return None

# Singleton instance
client = BinanceClient(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)