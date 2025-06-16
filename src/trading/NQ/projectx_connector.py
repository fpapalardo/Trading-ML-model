import os
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)

class ProjectXClient:
    """
    Thin wrapper around the ProjectX Gateway API at https://api.topstepx.com
    Implements token caching to avoid excessive authenticate calls.
    """

    def __init__(self, username: str, api_key: str, token_file: Optional[str] = None, request_timeout: float = 10.0):
        self.username = username
        self.api_key = api_key
        self.base_url = "https://api.topstepx.com"
        self.login_url = f"{self.base_url}/api/Auth/loginKey"
        self.accounts_url = f"{self.base_url}/api/Account/search"
        self.search_url = f"{self.base_url}/api/Contract/search"
        self.bars_url = f"{self.base_url}/api/History/retrieveBars"
        self.order_place_url = f"{self.base_url}/api/Order/place"
        self.order_search_url = f"{self.base_url}/api/Order/search"
        self.order_search_open_url = f"{self.base_url}/api/Order/searchOpen"
        self.order_cancel_url = f"{self.base_url}/api/Order/cancel"

        # token and account
        self.token: Optional[str] = None
        self.account_id: Optional[int] = None

        # timeout for all HTTP calls
        self.request_timeout = request_timeout

        # cache file for session token
        self.token_file = token_file or os.path.expanduser("~/.projectx_token.json")
        self._load_cached_token()

    def _load_cached_token(self):
        """Load token from file if present and not expired (24h)."""
        if not os.path.isfile(self.token_file):
            return
        try:
            with open(self.token_file, 'r') as f:
                data = json.load(f)
            token = data.get('token')
            ts = datetime.fromisoformat(data.get('timestamp'))
            if datetime.now(timezone.utc) - ts < timedelta(hours=24):
                self.token = token
                logger.info("Loaded cached token from %s", self.token_file)
        except Exception as e:
            logger.warning("Failed to load cached token: %s", e)

    def _save_cached_token(self):
        """Save current token and timestamp."""
        try:
            with open(self.token_file, 'w') as f:
                json.dump({
                    'token': self.token,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }, f)
            logger.info("Saved token to cache %s", self.token_file)
        except Exception as e:
            logger.warning("Failed to save token to cache: %s", e)

    def authenticate(self, force: bool = False) -> None:
        """
        Authenticate and store Bearer token + first active account ID.
        If token already loaded and not expired, skip unless force=True.
        """
        if self.token and not force:
            # assume cached token is valid
            logger.info("Using existing bearer token")
        else:
            # fetch new token
            resp = requests.post(
                self.login_url,
                json={"userName": self.username, "apiKey": self.api_key},
                headers={"Accept": "text/plain", "Content-Type": "application/json"},
                timeout=self.request_timeout
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success", False):
                raise RuntimeError("Auth failed: " + str(data.get("errorMessage")))
            self.token = data["token"]
            logger.info("Authenticated, received new token")
            self._save_cached_token()

        # fetch accounts
        resp = requests.post(
            self.accounts_url,
            json={"onlyActiveAccounts": True},
            headers=self._headers(),
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        acct_data = resp.json()
        accts = acct_data.get("accounts", [])
        if not accts:
            raise RuntimeError("No active accounts found")
        self.account_id = accts[0]["id"]
        logger.info("Using account ID %s", self.account_id)

    def _headers(self) -> Dict[str, str]:
        if not self.token:
            raise RuntimeError("Not authenticated; call authenticate() first.")
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def search_contracts(self, search_text: str, live: bool=False) -> List[Dict]:
        body = {"searchText": search_text, "live": live}
        resp = requests.post(self.search_url, json=body, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError("Search failed: " + str(data.get("errorMessage")))
        return data.get("contracts", [])

    def get_bars(
        self,
        contract_id: int,
        start: datetime,
        end: datetime,
        unit: int = 2,
        unit_number: int = 5,
        limit: int = 1000,
        include_partial: bool = False
    ) -> List[Dict]:
        body = {
            "contractId":        contract_id,
            "live":              False,
            "startTime":         start.astimezone(timezone.utc).isoformat(),
            "endTime":           end.astimezone(timezone.utc).isoformat(),
            "unit":              unit,
            "unitNumber":        unit_number,
            "limit":             limit,
            "includePartialBar": include_partial,
        }
        resp = requests.post(self.bars_url, json=body, headers=self._headers(), timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError("Bar request failed: " + str(data.get("errorMessage")))
        return data.get("bars", [])

    # ─── Order Status ─────────────────────────────────────────
    def search_orders(self, start: Optional[datetime]=None, end: Optional[datetime]=None) -> List[Dict]:
        """Search all orders (optionally between start/end)."""
        body: Dict = {"accountId": self.account_id}
        if start:
            body["startTimestamp"] = start.astimezone(timezone.utc).isoformat()
        if end:
            body["endTimestamp"]   = end  .astimezone(timezone.utc).isoformat()
        resp = requests.post(self.order_search_url, json=body, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError("Order search failed: " + str(data.get("errorMessage")))
        return data.get("orders", [])

    def search_open_orders(self) -> List[Dict]:
        """Fetch only open/pending orders."""
        body = {"accountId": self.account_id}
        resp = requests.post(self.order_search_open_url, json=body, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError("Open order search failed: " + str(data.get("errorMessage")))
        return data.get("orders", [])

    # ─── Place / Cancel ────────────────────────────────────────
    def place_order(
        self,
        contract_id: int,
        side: str,
        quantity: int,
        order_type: int = 2,           # 2 = Market by default
        limit_price: Optional[float] = None,
        stop_price:  Optional[float] = None,
        linked_order_id: Optional[int] = None
    ) -> Dict:
        """
        Place a single order. 'type' values:
          1 = Limit, 2 = Market, 4 = Stop, 5 = TrailingStop, etc.
        side: "Buy"  → 0 (Bid)
              "Sell" → 1 (Ask)
        """
        side_val = 0 if side.lower().startswith("buy") else 1
        body = {
            "accountId":     self.account_id,
            "contractId":    contract_id,
            "type":          order_type,
            "side":          side_val,
            "size":          quantity,
            "limitPrice":    limit_price,
            "stopPrice":     stop_price,
            "linkedOrderId": linked_order_id,
        }
        resp = requests.post(self.order_place_url, json=body, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError("Place order failed: " + str(data.get("errorMessage")))
        return data

    def cancel_order(self, order_id: int) -> None:
        """Cancel an existing order by its ID."""
        body = {"accountId": self.account_id, "orderId": order_id}
        resp = requests.post(self.order_cancel_url, json=body, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError("Cancel failed: " + str(data.get("errorMessage")))

    # ─── OCO Wrapper ───────────────────────────────────────────
    def place_oco_exit(
        self,
        contract_id: int,
        quantity: int,
        take_profit: float,
        stop_loss:   float,
        side: str
    ) -> Dict[str, Dict]:
        """
        Place a TP (limit) + SL (stop) as a One-Cancels-the-Other pair.
        Returns a dict with both legs’ responses.
        """
        logger.info(
            f"Starting OCO exit for acct={self.account_id}, "
            f"contract={contract_id}, qty={quantity}, "
            f"TP={take_profit:.2f}, SL={stop_loss:.2f}"
        )

        # 1) Place TP limit sell
        try:
            tp_resp = self.place_order(
                contract_id=contract_id,
                side=side,
                quantity=quantity,
                order_type=1,           # Limit
                limit_price=take_profit
            )
            logger.info(f"TP leg placed successfully, response: {tp_resp}")
        except Exception as e:
            logger.error(f"Failed to place TP leg: {e}", exc_info=True)
            raise

        tp_id = tp_resp.get("orderId") or tp_resp.get("id")
        if not tp_id:
            logger.error(f"TP response missing order ID: {tp_resp}")
            raise RuntimeError("Could not retrieve TP order ID for OCO link")

        # 2) Place SL stop sell with link back to TP leg
        try:
            sl_resp = self.place_order(
                contract_id=contract_id,
                side=side,
                quantity=quantity,
                order_type=4,           # Stop
                stop_price=stop_loss,
                linked_order_id=tp_id
            )
            logger.info(f"SL leg placed successfully, response: {sl_resp}")
        except Exception as e:
            logger.error(f"Failed to place SL leg: {e}", exc_info=True)
            # You might want to cancel the TP here if the SL fails:
            # self.cancel_order(tp_id)
            raise

        logger.info("OCO exit complete")
        return {"take_profit": tp_resp, "stop_loss": sl_resp}
