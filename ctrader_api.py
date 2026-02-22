import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# ============================================================
# CONFIG CTRADER / PEPPERSTONE
# ============================================================

load_dotenv()

CTRADER_CLIENT_ID = os.getenv("CTRADER_CLIENT_ID")
CTRADER_CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET")
CTRADER_ACCOUNT_ID = os.getenv("CTRADER_ACCOUNT_ID")
CTRADER_ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN")
CTRADER_REFRESH_TOKEN = os.getenv("CTRADER_REFRESH_TOKEN")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Endpoints officiels Spotware (cTrader Open API)
CTRADER_AUTH_URL = "https://api.spotware.com/connect/token"
CTRADER_DATA_URL = "https://api.spotware.com/connect/trading"


# ============================================================
# TELEGRAM — ENVOI DE MESSAGES
# ============================================================

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Token ou Chat ID manquant.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}

    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print("[Telegram] Erreur d’envoi :", e)


class CTraderAPI:
    """
    Client cTrader / Pepperstone PRO.
    """

    def __init__(self):
        self.client_id = CTRADER_CLIENT_ID
        self.client_secret = CTRADER_CLIENT_SECRET
        self.account_id = CTRADER_ACCOUNT_ID
        self.access_token = CTRADER_ACCESS_TOKEN
        self.refresh_token = CTRADER_REFRESH_TOKEN
        self.token_expiry: Optional[datetime] = None

    # ========================================================
    # AUTH / TOKENS
    # ========================================================

    def is_configured(self) -> bool:
        return bool(
            self.client_id
            and self.client_secret
            and self.account_id
            and self.access_token
            and self.refresh_token
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def refresh_access_token(self) -> bool:
        if not self.refresh_token:
            print("[cTrader] Aucun refresh_token configuré.")
            return False

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        try:
            resp = requests.post(CTRADER_AUTH_URL, data=data, timeout=10)
            if resp.status_code != 200:
                print("[cTrader] Échec refresh token :", resp.status_code, resp.text)
                return False

            payload = resp.json()
            self.access_token = payload.get("access_token")
            self.refresh_token = payload.get("refresh_token", self.refresh_token)
            expires_in = payload.get("expires_in", 3600)
            self.token_expiry = datetime.utcnow() + timedelta(seconds=expires_in - 60)

            print("[cTrader] Token rafraîchi avec succès.")
            return True

        except Exception as e:
            print("[cTrader] Erreur refresh token :", e)
            return False

    def ensure_token_valid(self) -> bool:
        if not self.access_token:
            print("[cTrader] Aucun access_token configuré.")
            return False

        if self.token_expiry and datetime.utcnow() < self.token_expiry:
            return True

        return self.refresh_access_token()

    # ========================================================
    # FETCH CANDLES
    # ========================================================

    def fetch_candles_api(self, symbol: str, timeframe: str, count: int = 300) -> Optional[pd.DataFrame]:
        if not self.is_configured():
            print("[cTrader] API non configurée.")
            return None

        if not self.ensure_token_valid():
            print("[cTrader] Token invalide.")
            return None

        url = f"{CTRADER_DATA_URL}/candles"
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count,
            "accountId": self.account_id,
        }

        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=10)
            if resp.status_code != 200:
                print("[cTrader] Erreur fetch_candles_api :", resp.status_code, resp.text)
                return None

            data = resp.json()
            candles = data.get("candles")
            if not candles:
                print("[cTrader] Aucune donnée renvoyée.")
                return None

            df = pd.DataFrame(candles)
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)

            return df

        except Exception as e:
            print("[cTrader] Exception fetch_candles_api :", e)
            return None

    def fetch_candles(self, symbol="XAUUSD", timeframe="M5", count=300):
        return self.fetch_candles_api(symbol, timeframe, count)

    # ========================================================
    # ORDRE MARKET
    # ========================================================

    def place_market_order(self, symbol: str, volume: float, side: str):
        if not self.ensure_token_valid():
            return None

        url = f"{CTRADER_DATA_URL}/orders/market"

        payload = {
            "accountId": self.account_id,
            "symbol": symbol,
            "volume": volume,
            "side": side.upper(),
        }

        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=10)
            if resp.status_code != 200:
                print("[cTrader] Erreur MARKET ORDER :", resp.status_code, resp.text)
                return None

            return resp.json()

        except Exception as e:
            print("[cTrader] Exception MARKET ORDER :", e)
            return None

    # ========================================================
    # SL / TP
    # ========================================================

    def set_sl_tp(self, position_id: str, sl: float, tp: float):
        if not self.ensure_token_valid():
            return None

        url = f"{CTRADER_DATA_URL}/positions/sltp"

        payload = {
            "accountId": self.account_id,
            "positionId": position_id,
            "stopLoss": sl,
            "takeProfit": tp,
        }

        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=10)
            if resp.status_code != 200:
                print("[cTrader] Erreur SL/TP :", resp.status_code, resp.text)
                return None

            return resp.json()

        except Exception as e:
            print("[cTrader] Exception SL/TP :", e)
            return None

    # ========================================================
    # POSITIONS OUVERTES
    # ========================================================

    def get_open_positions(self):
        if not self.ensure_token_valid():
            return None

        url = f"{CTRADER_DATA_URL}/positions"
        params = {"accountId": self.account_id}

        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=10)
            if resp.status_code != 200:
                print("[cTrader] Erreur GET POSITIONS :", resp.status_code, resp.text)
                return None

            return resp.json().get("positions", [])

        except Exception as e:
            print("[cTrader] Exception GET POSITIONS :", e)
            return None

    # ========================================================
    # POSITIONS FERMÉES
    # ========================================================

    def get_closed_positions(self):
        if not self.ensure_token_valid():
            return None

        url = f"{CTRADER_DATA_URL}/positions/history"
        params = {"accountId": self.account_id}

        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=10)
            if resp.status_code != 200:
                print("[cTrader] Erreur GET CLOSED POSITIONS :", resp.status_code, resp.text)
                return None

            return resp.json().get("positions", [])

        except Exception as e:
            print("[cTrader] Exception GET CLOSED POSITIONS :", e)
            return None

    # ========================================================
    # INFOS COMPTE / MARGE
    # ========================================================

    def get_account_info(self):
        if not self.ensure_token_valid():
            return None

        url = f"{CTRADER_DATA_URL}/account"
        params = {"accountId": self.account_id}

        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=10)
            if resp.status_code != 200:
                print("[cTrader] Erreur GET ACCOUNT INFO :", resp.status_code, resp.text)
                return None

            return resp.json()

        except Exception as e:
            print("[cTrader] Exception GET ACCOUNT INFO :", e)
            return None

    # ========================================================
    # CALCUL LOT = 100% DU CAPITAL DISPONIBLE
    # ========================================================

    def calculate_lot_size(self, free_margin: float, leverage: int = 30) -> float:
        lot_max_margin = (free_margin * leverage) / 100000.0

        if lot_max_margin <= 0:
            print("[cTrader] Marge insuffisante.")
            return 0.0

        return round(lot_max_margin, 2)

    # ========================================================
    # NOTIFICATION FERMETURE POSITION
    # ========================================================

    def notify_closed_position(self, position):
        position_id = position.get("positionId")
        entry_price = position.get("entryPrice")
        close_price = position.get("closePrice")
        profit = position.get("profit")
        close_time = position.get("closeTime")

        result = "🎯 TP TOUCHÉ" if profit > 0 else "❌ SL TOUCHÉ"

        send_telegram_message(
            f"📉 POSITION FERMÉE\n"
            f"Résultat : {result}\n"
            f"PnL : {profit}\n"
            f"Entrée : {entry_price}\n"
            f"Sortie : {close_price}\n"
            f"Heure : {close_time}"
        )

    # ========================================================
    # EXÉCUTION SIGNAL + SL + TP + TELEGRAM
    # ========================================================

    def execute_signal_with_atr(self, symbol, signal, last_price, atr_value):
        account = self.get_account_info()
        if not account:
            return {"error": "Impossible de lire le compte."}

        balance = float(account.get("balance", 0))
        equity = float(account.get("equity", balance))
        margin_used = float(account.get("marginUsed", 0))
        free_margin = equity - margin_used

        print(f"[cTrader] Balance={balance}, Equity={equity}, FreeMargin={free_margin}")

        # SL basé sur ATR
        sl_pips = atr_value * 2

        # Lot = 100% du capital dispo
        lot = self.calculate_lot_size(free_margin=free_margin, leverage=30)

        if lot <= 0:
            return {"error": "Marge insuffisante."}

        side = "BUY" if signal == "ACHAT" else "SELL"

        # 🔔 Telegram — ouverture trade
        send_telegram_message(
            f"📈 NOUVEAU TRADE OUVERT\n"
            f"Type : {side}\n"
            f"Volume : {lot} lots\n"
            f"Prix d'entrée : {last_price}\n"
            f"Heure : {datetime.utcnow()} UTC"
        )

        print(f"[cTrader] Envoi ordre {side} {lot} lots")

        order = self.place_market_order(symbol=symbol, volume=lot, side=side)

        if not order:
            return {"error": "Erreur lors de l’envoi de l’ordre."}

        # ====================================================
        # SL / TP AUTOMATIQUES
        # ====================================================

        position_id = order.get("positionId") or order.get("position", {}).get("id")

        if not position_id:
            print("[cTrader] Impossible de récupérer l'ID de la position.")
            return {"status": "OK", "order": order, "lot": lot}

        rr = 2.0
        tp_pips = sl_pips * rr

        pip_value = 0.1  # XAUUSD = 0.1 par pip

        if side == "BUY":
            sl_price = last_price - sl_pips * pip_value
            tp_price = last_price + tp_pips * pip_value
        else:
            sl_price = last_price + sl_pips * pip_value
            tp_price = last_price - tp_pips * pip_value

        self.set_sl_tp(
            position_id=str(position_id),
            sl=round(sl_price, 2),
            tp=round(tp_price, 2),
        )

        # 🔔 Telegram — SL/TP placés
        send_telegram_message(
            f"🔒 SL/TP PLACÉS\n"
            f"SL : {round(sl_price, 2)}\n"
            f"TP : {round(tp_price, 2)}\n"
            f"Heure : {datetime.utcnow()} UTC"
        )

        print(f"[cTrader] SL placé à {sl_price}, TP placé à {tp_price}")

        return {
            "status": "OK",
            "order": order,
            "lot": lot,
            "sl_price": sl_price,
            "tp_price": tp_price,
        }


# ============================================================
# SINGLETON
# ============================================================

_api_singleton: Optional[CTraderAPI] = None


def get_api() -> CTraderAPI:
    global _api_singleton
    if _api_singleton is None:
        _api_singleton = CTraderAPI()
    return _api_singleton


def fetch_candles_ctrader(instrument="XAUUSD", timeframe="M5", count=300):
    api = get_api()
    return api.fetch_candles(symbol=instrument, timeframe=timeframe, count=count)
