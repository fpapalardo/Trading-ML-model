import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional

class ProjectXClient:
    """
    Thin wrapper around the ProjectX Gateway API at https://api.topstepx.com
    """

    def __init__(self, username: str, api_key: str):
        self.username = username
        self.api_key = api_key

        self.base_url        = "https://api.topstepx.com"
        self.login_url       = f"{self.base_url}/api/Auth/loginKey"
        self.accounts_url    = f"{self.base_url}/api/Account/search"
        self.search_url      = f"{self.base_url}/api/Contract/search"
        self.bars_url        = f"{self.base_url}/api/History/retrieveBars"
        self.order_place_url = f"{self.base_url}/api/Order/place"
        self.order_search_url     = f"{self.base_url}/api/Order/search"      # :contentReference[oaicite:0]{index=0}
        self.order_search_open_url= f"{self.base_url}/api/Order/searchOpen"  # :contentReference[oaicite:1]{index=1}
        self.order_cancel_url     = f"{self.base_url}/api/Order/cancel"      # :contentReference[oaicite:2]{index=2}

        self.token:      Optional[str] = None
        self.account_id: Optional[int] = None

    def authenticate(self) -> None:
        """Authenticate and store Bearer token + first active account ID."""
        # 1) loginKey → get token
        resp = requests.post(
            self.login_url,
            json={"userName": self.username, "apiKey": self.api_key},
            headers={"Accept": "text/plain", "Content-Type": "application/json"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError("Auth failed: " + str(data.get("errorMessage")))
        self.token = data["token"]

        # 2) fetch accounts → pick first active
        resp = requests.post(self.accounts_url, json={"onlyActiveAccounts": True}, headers=self._headers())
        resp.raise_for_status()
        acct_data = resp.json()
        accts = acct_data.get("accounts", [])
        if not accts:
            raise RuntimeError("No active accounts found")
        self.account_id = accts[0]["id"]

    def _headers(self) -> Dict[str,str]:
        if not self.token:
            raise RuntimeError("Not authenticated; call authenticate() first.")
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept":        "application/json",
            "Content-Type":  "application/json",
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
        resp = requests.post(self.bars_url, json=body, headers=self._headers(), timeout=10)
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
        stop_loss:   float
    ) -> Dict[str,Dict]:
        """
        Place a TP (limit) + SL (stop) as a One-Cancels-the-Other pair.
        Returns a dict with both legs’ responses.
        """
        # 1) Place TP limit sell
        tp_resp = self.place_order(
            contract_id=contract_id,
            side="Sell",
            quantity=quantity,
            order_type=1,           # Limit
            limit_price=take_profit
        )
        tp_id = tp_resp.get("orderId") or tp_resp.get("id")

        # 2) Place SL stop sell with link back to TP leg
        sl_resp = self.place_order(
            contract_id=contract_id,
            side="Sell",
            quantity=quantity,
            order_type=4,           # Stop
            stop_price=stop_loss,
            linked_order_id=tp_id
        )

        return {"take_profit": tp_resp, "stop_loss": sl_resp}

# # ────────────────────────────────────────────────────────────
# # Example usage:
# if __name__ == "__main__":
#     from datetime import timedelta

#     client = ProjectXClient(username="YOUR_USER", api_key="YOUR_KEY")
#     client.authenticate()

#     # — accountId is now set —
#     print("Account ID:", client.account_id)

#     # — get open orders —
#     open_ords = client.search_open_orders()
#     print("Open orders:", open_ords)

#     # — place a simple market buy —
#     buy = client.place_order(contract_id=12345, side="Buy", quantity=1)
#     print("Bought:", buy)

#     # — place OCO exit legs for a long position —
#     oco = client.place_oco_exit(
#         contract_id=12345,
#         quantity=1,
#         take_profit=20000.50,
#         stop_loss=  19950.25
#     )
#     print("OCO legs:", oco)
