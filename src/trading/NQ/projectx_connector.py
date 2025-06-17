"""
ProjectX Gateway API Client

Optimized client for interacting with TopStepX trading platform.
Provides REST API access for account management, order placement,
and historical data retrieval.

Features:
- Connection pooling for performance
- Token caching to reduce authentication calls
- Automatic retries for transient failures
- Support for both numeric and string contract IDs

Author: Trading Bot
Date: 2024
"""

import os
import json
import logging
import sys
import threading
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import List, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout, # Log to the console
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ProjectXClient:
    """
    Optimized ProjectX Gateway API client with connection pooling,
    caching, and reduced latency optimizations.
    
    This client handles authentication, order management, and market data
    retrieval from the TopStepX platform.
    """

    def __init__(
        self,
        username: str,
        api_key: str,
        token_file: Optional[str] = None,
        request_timeout: float = 5.0
    ):
        """
        Initialize the ProjectX client.
        
        Args:
            username: TopStepX username
            api_key: API key for authentication
            token_file: Path to cache authentication token
            request_timeout: Request timeout in seconds
        """
        self.username = username
        self.api_key = api_key
        self.base_url = "https://api.topstepx.com"
        
        # Pre-build URLs for efficiency
        self.urls = {
            'login': f"{self.base_url}/api/Auth/loginKey",
            'accounts': f"{self.base_url}/api/Account/search",
            'search': f"{self.base_url}/api/Contract/search",
            'bars': f"{self.base_url}/api/History/retrieveBars",
            'place_order': f"{self.base_url}/api/Order/place",
            'search_orders': f"{self.base_url}/api/Order/search",
            'search_open': f"{self.base_url}/api/Order/searchOpen",
            'cancel_order': f"{self.base_url}/api/Order/cancel"
        }

        # Create optimized HTTP session
        self._session = self._create_session()
        
        # Authentication state
        self.token: Optional[str] = None
        self.account_id: Optional[int] = None
        self.request_timeout = request_timeout
        
        # Thread-safe token management
        self._token_lock = threading.Lock()
        
        # Token caching
        self.token_file = token_file or os.path.expanduser("~/.projectx_token.json")
        self._load_cached_token()
        
        # Data caches
        self._contract_cache = {}
        self._last_orders_check = None
        self._cached_open_orders = []

    def _create_session(self) -> requests.Session:
        """
        Create optimized session with connection pooling and retries.
        
        Returns:
            Configured requests Session
        """
        session = requests.Session()
        
        # Retry strategy for transient failures
        retry_strategy = Retry(
            total=10,  # Reduced retries for speed
            backoff_factor=0.5,  # Faster backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        
        # Connection pooling with more connections
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Default headers for all requests
        session.headers.update({
            'User-Agent': 'ProjectX-Client/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Connection': 'keep-alive'
        })
        
        return session

    def _load_cached_token(self):
        """Load authentication token from cache file."""
        if not os.path.isfile(self.token_file):
            return
            
        try:
            with open(self.token_file, 'r') as f:
                data = json.load(f)
            
            token = data.get('token')
            ts_str = data.get('timestamp')
            
            if not token or not ts_str:
                return
                
            ts = datetime.fromisoformat(ts_str)
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            
            # Use token if less than 23 hours old (safer margin)
            if age_hours < 23:
                self.token = token
                logger.info("Loaded cached token (age: %.1fh)", age_hours)
            else:
                logger.info("Cached token expired (age: %.1fh)", age_hours)
                
        except Exception as e:
            logger.warning("Failed to load cached token: %s", e)

    def _save_cached_token(self):
        """Save authentication token to cache file."""
        if not self.token:
            return
            
        try:
            data = {
                'token': self.token,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Atomic write
            temp_file = self.token_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f)
            os.replace(temp_file, self.token_file)
            
            logger.debug("Saved token to cache")
        except Exception as e:
            logger.warning("Failed to save token: %s", e)

    def _clear_token_cache(self):
        """Clear cached authentication token."""
        try:
            if os.path.exists(self.token_file):
                os.remove(self.token_file)
        except Exception:
            pass
        self.token = None

    def _headers(self) -> Dict[str, str]:
        """
        Get authorization headers for API requests.
        
        Returns:
            Headers dict with authorization token
            
        Raises:
            RuntimeError: If not authenticated
        """
        if not self.token:
            raise RuntimeError("Not authenticated")
        return {"Authorization": f"Bearer {self.token}"}

    def authenticate(
        self,
        force: bool = False,
        preferred_account_name: Optional[str] = None
    ) -> None:
        """
        Authenticate with TopStepX and select trading account.
        
        Args:
            force: Force re-authentication even if token exists
            preferred_account_name: Specific account to select
            
        Raises:
            RuntimeError: If authentication fails
        """
        with self._token_lock:
            # Skip auth if token exists and not forced
            if self.token and not force and self.account_id:
                logger.debug("Using existing authentication")
                return

            # Login if needed
            if not self.token or force:
                login_payload = {
                    "userName": self.username,
                    "apiKey": self.api_key
                }
                
                try:
                    resp = self._session.post(
                        self.urls['login'],
                        json=login_payload,
                        timeout=self.request_timeout
                    )
                    
                    # Handle auth failure
                    if resp.status_code == 400:
                        logger.warning("Auth failed, clearing cache and retrying")
                        self._clear_token_cache()
                        
                        resp = self._session.post(
                            self.urls['login'],
                            json=login_payload,
                            timeout=self.request_timeout
                        )
                    
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if not data.get("success", False):
                        raise RuntimeError(f"Auth failed: {data.get('errorMessage')}")
                    
                    self.token = data["token"]
                    self._save_cached_token()
                    logger.info("Authentication successful")
                    
                except Exception as e:
                    logger.error("Authentication failed: %s", e)
                    raise

            # Get account if needed
            if not self.account_id:
                self._select_account(preferred_account_name)

    def _select_account(self, preferred_name: Optional[str] = None):
        """
        Select trading account.
        
        Args:
            preferred_name: Specific account name to select
            
        Raises:
            RuntimeError: If no accounts found or preferred account not found
        """
        resp = self._session.post(
            self.urls['accounts'],
            json={"onlyActiveAccounts": True},
            headers=self._headers(),
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        
        accounts = resp.json().get("accounts", [])
        if not accounts:
            raise RuntimeError("No active accounts found")

        # Select account
        if preferred_name:
            account = next((a for a in accounts if a.get('name') == preferred_name), None)
            if not account:
                raise RuntimeError(f"Account '{preferred_name}' not found")
        else:
            account = accounts[0]

        self.account_id = account['id']
        logger.info("Selected account: %s (ID: %s)", account.get('name', 'Unknown'), self.account_id)

    @lru_cache(maxsize=32)
    def search_contracts(self, search_text: str, live: bool = False) -> tuple:
        """
        Search for contracts by text.
        
        Args:
            search_text: Contract search term (e.g., 'NQ')
            live: Whether to search live contracts only
            
        Returns:
            Tuple of contract dictionaries
            
        Raises:
            RuntimeError: If search fails
        """
        resp = self._session.post(
            self.urls['search'],
            json={"searchText": search_text, "live": live},
            headers=self._headers(),
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(f"Contract search failed: {data.get('errorMessage')}")
        
        # Return as tuple for caching (lists aren't hashable)
        contracts = data.get("contracts", [])
        return tuple(contracts)

    def get_contract_info(self, contract_search: str) -> Dict:
        """
        Get full contract information including both numeric and string IDs.
        
        Args:
            contract_search: Contract search term
            
        Returns:
            Dictionary with contract information
            
        Raises:
            RuntimeError: If no contract found
        """
        contracts = list(self.search_contracts(contract_search))
        if not contracts:
            raise RuntimeError(f"No contract found for '{contract_search}'")
            
        contract = contracts[0]
        # Extract all possible ID fields
        return {
            'id': contract.get('id'),  # Numeric ID for REST API
            'symbol': contract.get('symbol'),  # String ID for SignalR (e.g., 'CON.F.US.NQ.H25')
            'contractId': contract.get('contractId'),  # Alternative field
            'name': contract.get('name'),
            'exchange': contract.get('exchange'),
            'raw': contract  # Full contract data
        }

    def get_bars(
        self,
        contract_id: Union[int, str],
        start: datetime,
        end: datetime,
        unit: int = 2,
        unit_number: int = 5,
        limit: int = 1000,
        include_partial: bool = False
    ) -> List[Dict]:
        """
        Retrieve historical OHLCV bars.
        
        Args:
            contract_id: Contract ID (numeric)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            unit: Time unit (2=minutes)
            unit_number: Number of units per bar
            limit: Maximum bars to retrieve
            include_partial: Whether to include incomplete bars
            
        Returns:
            List of bar dictionaries
            
        Raises:
            RuntimeError: If request fails
        """
        # Convert string to int if needed for REST API
        if isinstance(contract_id, str) and contract_id.isdigit():
            contract_id = int(contract_id)
            
        payload = {
            "contractId": contract_id,
            "live": False,
            "startTime": start.astimezone(timezone.utc).isoformat(),
            "endTime": end.astimezone(timezone.utc).isoformat(),
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": include_partial,
        }
        
        resp = self._session.post(
            self.urls['bars'],
            json=payload,
            headers=self._headers(),
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(f"Bar request failed: {data.get('errorMessage')}")
        
        return data.get("bars", [])

    def place_order(
        self,
        contract_id: Union[int, str],
        side: str,
        quantity: int,
        order_type: int = 2,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_price: Optional[float] = None,
        linked_order_id: Optional[int] = None
    ) -> Dict:
        """
        Place a new order.
        
        Args:
            contract_id: Contract ID
            side: 'Buy' or 'Sell'
            quantity: Number of contracts
            order_type: Order type (2=stop-limit)
            limit_price: Take profit price
            stop_price: Stop loss price
            trail_price: Trailing stop price
            linked_order_id: ID of linked order
            
        Returns:
            Order response dictionary
            
        Raises:
            RuntimeError: If order placement fails
        """
        
        # Convert string to int if needed for REST API
        if isinstance(contract_id, str) and contract_id.isdigit():
            contract_id = int(contract_id)
        
        # Fast tick rounding
        if limit_price is not None:
            limit_price = round(limit_price * 4) * 0.25
        if stop_price is not None:
            stop_price = round(stop_price * 4) * 0.25

        payload = {
            "accountId": self.account_id,
            "contractId": contract_id,
            "type": order_type,
            "side": 0 if side.lower().startswith("buy") else 1,
            "size": quantity,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": trail_price,
            "linkedOrderId": linked_order_id,
        }

        logger.info("Placing order: %s %d @ limit=%s stop=%s", 
                   side, quantity, limit_price, stop_price)

        resp = self._session.post(
            self.urls['place_order'],
            json=payload,
            headers=self._headers(),
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        
        data = resp.json()
        if not data.get("success", False):
            error_msg = data.get("errorMessage", "Unknown error")
            logger.error("Order placement failed: %s", error_msg)
            raise RuntimeError(f"Order placement failed: {error_msg}")
        
        return data

    def cancel_order(self, order_id: int) -> None:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Raises:
            RuntimeError: If cancellation fails
        """
        payload = {"accountId": self.account_id, "orderId": order_id}
        
        resp = self._session.post(
            self.urls['cancel_order'],
            json=payload,
            headers=self._headers(),
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        
        data = resp.json()
        if not data.get("success", False):
            error_msg = data.get("errorMessage", "Unknown error")
            raise RuntimeError(f"Order cancellation failed: {error_msg}")

    def search_open_orders(self) -> List[Dict]:
        """
        Search for open orders with caching.
        
        Returns:
            List of open order dictionaries
            
        Raises:
            RuntimeError: If search fails
        """
        now = datetime.now()
        
        # Use cache if recent (within 5 seconds)
        if (self._last_orders_check and 
            (now - self._last_orders_check).total_seconds() < 5):
            return self._cached_open_orders
        
        payload = {"accountId": self.account_id}
        
        resp = self._session.post(
            self.urls['search_open'],
            json=payload,
            headers=self._headers(),
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        
        data = resp.json()
        if not data.get("success", False):
            error_msg = data.get("errorMessage", "Unknown error")
            raise RuntimeError(f"Open orders search failed: {error_msg}")
        
        # Update cache
        self._cached_open_orders = data.get("orders", [])
        self._last_orders_check = now
        
        return self._cached_open_orders

    def search_orders(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Search historical orders.
        
        Args:
            start: Start datetime for search
            end: End datetime for search
            
        Returns:
            List of order dictionaries
            
        Raises:
            RuntimeError: If search fails
        """
        payload = {"accountId": self.account_id}
        
        if start:
            payload["startTimestamp"] = start.astimezone(timezone.utc).isoformat()
        if end:
            payload["endTimestamp"] = end.astimezone(timezone.utc).isoformat()

        resp = self._session.post(
            self.urls['search_orders'],
            json=payload,
            headers=self._headers(),
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        
        data = resp.json()
        if not data.get("success", False):
            error_msg = data.get("errorMessage", "Unknown error")
            raise RuntimeError(f"Order search failed: {error_msg}")
        
        return data.get("orders", [])

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_session'):
            self._session.close()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()