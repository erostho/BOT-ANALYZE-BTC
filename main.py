import os
import json
import math
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import requests
import pandas as pd
from dateutil import tz
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# ========================
#  Config
# ========================

OKX_BASE_URL = "https://www.okx.com"
OKX_INST_ID = os.getenv("OKX_INST_ID", "BTC-USDT-SWAP")

GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "OKX_BOT")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

EXNESS_PRICE_URL = os.getenv("EXNESS_PRICE_URL")  # endpoint trả JSON giá Exness

TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",      # KHUNG TRADE CHÍNH
    "1H": "1H",
    "2H": "2H",
    "4H": "4H",
}

VN_TZ = tz.gettz("Asia/Ho_Chi_Minh")


# ========================
#  Helpers
# ========================

def _log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] {msg}", flush=True)


def connect_gsheet():
    if not GOOGLE_SHEET_ID or not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("Missing GOOGLE_SHEET_ID or GOOGLE_SERVICE_ACCOUNT_JSON env")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(GOOGLE_SHEET_ID)


def get_or_create_worksheet(sh, title: str, rows: int = 100, cols: int = 20):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def fetch_okx_candles(tf: str, limit: int = 120) -> pd.DataFrame:
    """
    Lấy dữ liệu nến OKX cho 1 timeframe.
    Trả về DataFrame với index = datetime (UTC) & cột: open, high, low, close, volume.
    """
    url = f"{OKX_BASE_URL}/api/v5/market/candles"
    params = {
        "instId": OKX_INST_ID,
        "bar": tf,
        "limit": str(limit),
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError(f"Empty candles from OKX for {tf}")

    # OKX trả newest first -> đảo lại
    records = []
    for row in reversed(data):
        ts_ms = int(row[0])
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        o, h, l, c, vol = map(float, row[1:6])
        records.append(
            {
                "time": dt,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": vol,
            }
        )

    df = pd.DataFrame(records).set_index("time")
    return df


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def detect_trend_from_ema(last_row: pd.Series) -> str:
    ema20 = last_row["ema20"]
    ema50 = last_row["ema50"]
    close = last_row["close"]
    if close > ema50 and ema20 > ema50:
        return "UP"
    if close < ema50 and ema20 < ema50:
        return "DOWN"
    return "SIDE"


def _detect_swings(
    df: pd.DataFrame,
    lookback: int = 60,
    left: int = 2,
    right: int = 2,
) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
    """
    Tìm swing high / swing low dạng fractal:
    - swing high: high[i] > high[i-k] & high[i] > high[i+k] (k=1..left/right)
    - swing low  : low[i]  < low[i-k] & low[i]  < low[i+k]
    Chỉ dùng phần đuôi 'lookback' để nhẹ.
    """
    sub = df.tail(lookback)
    highs = sub["high"]
    lows = sub["low"]
    idx = list(sub.index)

    swing_highs: List[Tuple[pd.Timestamp, float]] = []
    swing_lows: List[Tuple[pd.Timestamp, float]] = []

    n = len(sub)
    for i in range(left, n - right):
        h = highs.iloc[i]
        l = lows.iloc[i]
        ok_high = True
        ok_low = True
        for k in range(1, left + 1):
            if h <= highs.iloc[i - k]:
                ok_high = False
                break
        for k in range(1, right + 1):
            if h <= highs.iloc[i + k]:
                ok_high = False
                break
        for k in range(1, left + 1):
            if l >= lows.iloc[i - k]:
                ok_low = False
                break
        for k in range(1, right + 1):
            if l >= lows.iloc[i + k]:
                ok_low = False
                break

        ts = idx[i]
        if ok_high:
            swing_highs.append((ts, float(h)))
        if ok_low:
            swing_lows.append((ts, float(l)))

    return swing_highs, swing_lows


def classify_market_structure(df: pd.DataFrame, lookback: int = 80) -> str:
    """
    Phân loại cấu trúc thị trường bằng swing high/low:
    - Tăng (HH–HL): ít nhất 3 swing high & 3 swing low, cả hai đều tăng dần ở 3 điểm cuối
    - Giảm (LH–LL): tương tự nhưng giảm dần
    - Ngược lại: Sideway / lẫn lộn
    """
    swing_highs, swing_lows = _detect_swings(df, lookback=lookback)

    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "Không rõ (thiếu swing)"

    last_highs = [p for _, p in swing_highs[-3:]]
    last_lows = [p for _, p in swing_lows[-3:]]

    def _is_increasing(vals: List[float]) -> bool:
        return vals[0] < vals[1] < vals[2]

    def _is_decreasing(vals: List[float]) -> bool:
        return vals[0] > vals[1] > vals[2]

    if _is_increasing(last_highs) and _is_increasing(last_lows):
        return "Tăng (HH–HL)"
    if _is_decreasing(last_highs) and _is_decreasing(last_lows):
        return "Giảm (LH–LL)"
    return "Sideway / lẫn lộn"


def classify_atr(atr_value: float) -> str:
    if pd.isna(atr_value):
        return "Chưa đủ dữ liệu ATR"
    if atr_value < 80:
        return "Biến động rất thấp / sideway chặt"
    if atr_value < 150:
        return "Sideway nhẹ, dao động nhỏ"
    if atr_value < 250:
        return "Biến động vừa"
    if atr_value < 350:
        return "Thị trường bắt đầu mạnh"
    if atr_value < 600:
        return "Trend mạnh, breakout mạnh"
    return "Biến động cực mạnh (thường khi có tin tức)"


def get_exness_price() -> Optional[float]:
    if not EXNESS_PRICE_URL:
        return None
    try:
        r = requests.get(EXNESS_PRICE_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            for key in ["price", "last", "ask", "bid"]:
                if key in data and isinstance(data[key], (int, float)):
                    return float(data[key])
            if "data" in data and isinstance(data["data"], dict):
                d2 = data["data"]
                for key in ["price", "last", "ask", "bid"]:
                    if key in d2 and isinstance(d2[key], (int, float)):
                        return float(d2[key])
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return float(data[0])
    except Exception as e:
        _log(f"get_exness_price error: {e}")
    return None


def to_exness_price(okx_price: float, diff: float) -> float:
    return okx_price + diff


def get_session_note(now_utc: datetime) -> str:
    vn_time = now_utc.astimezone(VN_TZ)
    hour = vn_time.hour
    if 7 <= hour < 14:
        return f"Giờ VN {vn_time.strftime('%H:%M')} – phiên Á, thường dao động vừa phải."
    if 14 <= hour < 20:
        return f"Giờ VN {vn_time.strftime('%H:%M')} – phiên Âu, thị trường sôi động dần."
    return f"Giờ VN {vn_time.strftime('%H:%M')} – phiên Mỹ, thị trường thường sôi động mạnh."


def get_session_type(now_utc: datetime) -> str:
    """
    Trả về: 'ASIA' / 'EU' / 'US'
    """
    vn_time = now_utc.astimezone(VN_TZ)
    hour = vn_time.hour
    if 7 <= hour < 14:
        return "ASIA"
    if 14 <= hour < 20:
        return "EU"
    return "US"


def get_retrace_zones(direction: str, last_close: float, atr: float) -> Dict[str, Any]:
    """
    Tính vùng hồi / điều chỉnh dựa trên ATR quanh giá hiện tại.
    direction: "up" (hồi lên) hoặc "down" (điều chỉnh xuống)
    """
    if pd.isna(atr) or atr <= 0:
        return {"direction": direction, "zones": []}

    zones = []
    if direction == "up":
        zones.append(("Vùng 1", last_close + 0.3 * atr, last_close + 0.6 * atr))
        zones.append(("Vùng 2", last_close + 0.6 * atr, last_close + 0.9 * atr))
        zones.append(("Vùng 3 (thấp)", last_close + 0.1 * atr, last_close + 0.3 * atr))
    else:
        zones.append(("Vùng 1", last_close - 0.6 * atr, last_close - 0.3 * atr))
        zones.append(("Vùng 2", last_close - 0.9 * atr, last_close - 0.6 * atr))
        zones.append(("Vùng 3 (cao)", last_close - 0.3 * atr, last_close - 0.1 * atr))

    return {"direction": direction, "zones": zones}


def detect_regime(rsi_val: float, atr: float) -> str:
    """
    Xác định chế độ: TREND / SIDEWAY / MIXED.
    ATR lớn + RSI xa 50 -> TREND, ngược lại SIDEWAY.
    """
    if pd.isna(atr) or pd.isna(rsi_val):
        return "UNKNOWN"
    if atr > 250 and (rsi_val > 60 or rsi_val < 40):
        return "TREND"
    if atr < 150 and 45 <= rsi_val <= 55:
        return "SIDEWAY"
    return "MIXED"


def build_trade_suggestion(trade_signal: str, last_row: pd.Series, atr: float) -> Optional[Dict[str, Any]]:
    """
    trade_signal:
      - "SHORT mạnh" / "LONG mạnh"  -> trend-follow, ATR-based (khung 30m)
      - "LONG hồi kỹ thuật" / "SHORT hồi kỹ thuật" -> counter-trend, TP gần / SL chặt
    """
    close = float(last_row["close"])
    if pd.isna(atr) or atr <= 0:
        return None

    # Trend-follow: dùng ATR rộng hơn
    if trade_signal == "SHORT mạnh":
        entry = close
        tp = close - 1.2 * atr
        sl = close + 0.8 * atr
        return {"side": "SHORT", "entry": entry, "tp": tp, "sl": sl}

    if trade_signal == "LONG mạnh":
        entry = close
        tp = close + 1.2 * atr
        sl = close - 0.8 * atr
        return {"side": "LONG", "entry": entry, "tp": tp, "sl": sl}

    # Hồi kỹ thuật: TP gần, SL chặt (ngược trend chính)
    rr = 1.1  # risk reward cho hồi kỹ thuật
    if trade_signal == "LONG hồi kỹ thuật":
        entry = close
        sl = close - 0.5 * atr
        tp = entry + rr * (entry - sl)
        return {"side": "LONG", "entry": entry, "tp": tp, "sl": sl}

    if trade_signal == "SHORT hồi kỹ thuật":
        entry = close
        sl = close + 0.5 * atr
        tp = entry - rr * (sl - entry)
        return {"side": "SHORT", "entry": entry, "tp": tp, "sl": sl}

    return None


def sheet_read_last_message_hash(ws_cache) -> Optional[str]:
    try:
        val = ws_cache.acell("A1").value
        return val or None
    except Exception:
        return None


def sheet_write_last_message_hash(ws_cache, h: str) -> None:
    ws_cache.update_acell("A1", h)


def compute_message_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        _log("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID, skip telegram")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=10)
    if not r.ok:
        _log(f"Telegram send error: {r.status_code} {r.text}")


# ========================
#  Trend Reliability & News filter
# ========================

def compute_trend_reliability(
    main_trend: str,
    trend_main: str,          # trend khung 30m (EMA)
    ms_main: str,             # market structure 30m
    ms_sub: str,              # market structure 15m
    tf_trends: Dict[str, Dict[str, Any]],
    last_main: pd.Series,     # nến 30m cuối
    atr_main: float,
    rsi_main: float,
    vol_main: float,
    vol_ma20_main: float,
) -> Tuple[int, str]:
    """
    Trend Reliability Index (TRI) 0–100 cho khung 30m.
    Dựa trên:
    - Đồng hướng đa khung
    - EMA20–EMA50 spread
    - RSI lệch khỏi 50
    - Volume ủng hộ xu hướng
    """
    tri = 0

    # 30m cùng hướng trend chính
    if main_trend in ("UP", "DOWN") and trend_main == main_trend:
        tri += 15

    # Market structure 30m
    if ("Tăng" in ms_main and main_trend == "UP") or ("Giảm" in ms_main and main_trend == "DOWN"):
        tri += 15

    # 15m phụ cùng hướng
    if ("Tăng" in ms_sub and main_trend == "UP") or ("Giảm" in ms_sub and main_trend == "DOWN"):
        tri += 10

    # 1H cùng hướng
    t1h = tf_trends.get("1H", {}).get("trend")
    if t1h == main_trend:
        tri += 10

    # EMA spread
    if atr_main > 0:
        ema_spread = abs(float(last_main["ema20"] - last_main["ema50"]))
        if ema_spread >= 0.4 * atr_main:
            tri += 20

    # RSI 30m
    if not math.isnan(rsi_main):
        if main_trend == "UP" and rsi_main >= 55:
            tri += 15
        elif main_trend == "DOWN" and rsi_main <= 45:
            tri += 15

    # Volume 30m
    if vol_ma20_main > 0 and vol_main >= 1.2 * vol_ma20_main:
        tri += 15

    tri = max(0, min(100, tri))

    if tri < 40:
        desc = "Trend yếu / dễ nhiễu"
    elif tri < 60:
        desc = "Trend trung bình"
    elif tri < 80:
        desc = "Trend khá tin cậy"
    else:
        desc = "Trend rất mạnh & tin cậy"

    return tri, desc


def detect_news_like_bar(
    df_main: pd.DataFrame,   # 30m
    atr_main: float,
    df_sub: pd.DataFrame,    # 15m
    atr_sub: float,
) -> bool:
    """
    Nến "giống nến tin" khi biên độ > 3×ATR trên khung chính (30m) hoặc phụ (15m).
    """
    if atr_main <= 0 and atr_sub <= 0:
        return False

    # 30m
    last_main = df_main.iloc[-1]
    prev_main = df_main.iloc[-2]
    tr_last_main = float(last_main["high"] - last_main["low"])
    tr_prev_main = float(prev_main["high"] - prev_main["low"])
    news_main = False
    if atr_main > 0:
        if tr_last_main > 3 * atr_main or tr_prev_main > 3 * atr_main:
            news_main = True

    # 15m
    last_sub = df_sub.iloc[-1]
    prev_sub = df_sub.iloc[-2]
    tr_last_sub = float(last_sub["high"] - last_sub["low"])
    tr_prev_sub = float(prev_sub["high"] - prev_sub["low"])
    news_sub = False
    if atr_sub > 0:
        if tr_last_sub > 3 * atr_sub or tr_prev_sub > 3 * atr_sub:
            news_sub = True

    return news_main or news_sub


# ========================
#  Signal quality scoring
# ========================

def compute_signal_score(
    main_trend: str,
    trend_main: str,
    ms_main: str,
    ms_sub: str,
    rsi_main: float,
    atr_main: float,
    last_main: pd.Series,
    prev1_main: pd.Series,
    prev2_main: pd.Series,
    vol_ma20_main: float,
    trade_signal: Optional[str],
    is_ma5_up: bool,
    is_ma5_down: bool,
    tri_score: int,
    session_type: str,
    news_like: bool,
) -> Tuple[int, int, int, int]:
    """
    Chấm điểm chất lượng tín hiệu (khung 30m):
    - Trend score  (0–40)
    - Momentum     (0–30)
    - Location     (0–30)
    + điều chỉnh bởi:
      - TRI (trend reliability)
      - phiên (ASIA/EU/US)
      - news-like bar
    Tổng: 0–100
    """
    trend_score = 0
    momentum_score = 0
    location_score = 0

    # --- Trend score cơ bản ---
    if main_trend in ("UP", "DOWN") and trend_main == main_trend:
        trend_score += 15

    if ("Tăng" in ms_main and main_trend == "UP") or ("Giảm" in ms_main and main_trend == "DOWN"):
        trend_score += 10
    if ("Tăng" in ms_sub and main_trend == "UP") or ("Giảm" in ms_sub and main_trend == "DOWN"):
        trend_score += 10

    if not math.isnan(rsi_main):
        if main_trend == "UP" and rsi_main >= 55:
            trend_score += 5
        elif main_trend == "DOWN" and rsi_main <= 45:
            trend_score += 5

    # --- Momentum score ---
    true_range = float(last_main["high"] - last_main["low"]) if not math.isnan(last_main["high"] - last_main["low"]) else 0.0
    if atr_main > 0:
        if true_range >= 0.8 * atr_main:
            momentum_score += 10
    vol_main = float(last_main["volume"])
    if vol_ma20_main > 0 and vol_main >= 1.2 * vol_ma20_main:
        momentum_score += 10

    prev_highs = max(prev1_main["high"], prev2_main["high"])
    prev_lows = min(prev1_main["low"], prev2_main["low"])
    broke_high = last_main["high"] > prev_highs
    broke_low = last_main["low"] < prev_lows

    if trade_signal in ("LONG mạnh", "LONG hồi kỹ thuật") and broke_high:
        momentum_score += 10
    elif trade_signal in ("SHORT mạnh", "SHORT hồi kỹ thuật") and broke_low:
        momentum_score += 10
    elif broke_high or broke_low:
        momentum_score += 5  # có phá range nhưng không khớp hẳn hướng trade

    # Momentum từ MA5
    if trade_signal in ("LONG mạnh", "LONG hồi kỹ thuật") and is_ma5_up:
        momentum_score += 5
    if trade_signal in ("SHORT mạnh", "SHORT hồi kỹ thuật") and is_ma5_down:
        momentum_score += 5

    # --- Location score ---
    if atr_main > 0:
        dist_ema20 = abs(float(last_main["close"] - last_main["ema20"]))
        # càng gần EMA20 càng tốt
        if dist_ema20 <= 0.7 * atr_main:
            location_score += 15
        elif dist_ema20 <= 1.0 * atr_main:
            location_score += 8

    # ưu tiên tín hiệu hồi kỹ thuật có vị trí đẹp (sau pha kéo/rơi mạnh)
    if trade_signal in ("LONG hồi kỹ thuật", "SHORT hồi kỹ thuật"):
        location_score += 10

    total = trend_score + momentum_score + location_score

    # --- Điều chỉnh theo Trend Reliability Index ---
    if tri_score >= 60:
        total += 10
    elif tri_score < 40:
        total -= 10

    # --- Điều chỉnh theo phiên giao dịch ---
    if session_type == "ASIA" and trade_signal in ("LONG mạnh", "SHORT mạnh"):
        # phiên Á trend thường yếu hơn
        total -= 10

    # --- Điều chỉnh theo nến "giống tin tức" ---
    if news_like:
        total -= 15

    total = max(0, min(100, total))
    return trend_score, momentum_score, location_score, total


# ========================
#  Core analysis (MAIN TF = 30m)
# ========================

def analyze_and_build_message() -> (str, str):
    now_utc = datetime.now(timezone.utc)
    session_type = get_session_type(now_utc)

    # 1) Lấy nến 30m (khung trade chính)
    df30 = fetch_okx_candles(TIMEFRAMES["30m"], limit=200)
    df30["ema20"] = ema(df30["close"], 20)
    df30["ema50"] = ema(df30["close"], 50)
    df30["atr14"] = calc_atr(df30, 14)
    df30["rsi14"] = rsi(df30["close"], 14)
    df30["vol_ma20"] = df30["volume"].rolling(window=20).mean()
    # Momentum layer: MA5
    df30["ma5"] = ema(df30["close"], 5)
    df30["ma5_slope"] = df30["ma5"].diff()

    last30 = df30.iloc[-1]
    prev30_1 = df30.iloc[-2]
    prev30_2 = df30.iloc[-3]

    atr_30 = float(last30["atr14"])
    atr_text = classify_atr(atr_30)
    rsi_30 = float(last30["rsi14"]) if not math.isnan(last30["rsi14"]) else float("nan")
    prev_rsi_30 = float(df30["rsi14"].iloc[-2]) if not math.isnan(df30["rsi14"].iloc[-2]) else float("nan")
    regime = detect_regime(rsi_30, atr_30)
    trend_30 = detect_trend_from_ema(last30)

    ma5_val = float(last30["ma5"]) if not math.isnan(last30["ma5"]) else float("nan")
    ma5_slope = float(last30["ma5_slope"]) if not math.isnan(last30["ma5_slope"]) else 0.0
    is_ma5_up = (ma5_slope > 0) and (not math.isnan(ma5_val)) and (last30["close"] > ma5_val)
    is_ma5_down = (ma5_slope < 0) and (not math.isnan(ma5_val)) and (last30["close"] < ma5_val)

    # Độ tuổi nến 30m (để tránh vào lệnh hồi quá trễ)
    last30_ts = df30.index[-1]
    frame_seconds_30 = 30 * 60
    age_seconds_30 = max(0.0, (now_utc - last30_ts).total_seconds())
    bar_age_ratio_30 = min(1.0, age_seconds_30 / frame_seconds_30)

    # 1b) Lấy thêm khung 15m để phát hiện hồi kỹ thuật SỚM + news-like
    df15 = fetch_okx_candles(TIMEFRAMES["15m"], limit=200)
    df15["rsi14"] = rsi(df15["close"], 14)
    df15["atr14"] = calc_atr(df15, 14)
    last15 = df15.iloc[-1]
    prev15_1 = df15.iloc[-2]
    prev15_2 = df15.iloc[-3]
    rsi_15 = float(last15["rsi14"]) if not math.isnan(last15["rsi14"]) else float("nan")
    atr_15 = float(last15["atr14"]) if not math.isnan(last15["atr14"]) else float("nan")

    # 2) Lấy nến higher TF & trend
    tf_trends = {}
    for name in ["30m", "1H", "2H", "4H"]:
        if name == "30m":
            df = df30
        else:
            df = fetch_okx_candles(TIMEFRAMES[name], limit=120)
        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        tf_trends[name] = {
            "trend": detect_trend_from_ema(df.iloc[-1]),
            "close": float(df.iloc[-1]["close"]),
        }

    # chọn trend chính: ưu tiên 4H, rồi 2H, 1H, 30m
    main_trend = trend_30
    for key in ["4H", "2H", "1H", "30m"]:
        t = tf_trends.get(key, {}).get("trend")
        if t in ["UP", "DOWN"]:
            main_trend = t
            break

    # 3) Market structure 30m (chính) & 15m (phụ)
    ms_30m = classify_market_structure(df30)
    ms_15m = classify_market_structure(df15)

    ms_30m_is_down = "Giảm" in ms_30m
    ms_30m_is_up = "Tăng" in ms_30m
    ms_15m_is_down = "Giảm" in ms_15m
    ms_15m_is_up = "Tăng" in ms_15m

    # BOS: phá swing high/low 30m
    swing_highs_30, swing_lows_30 = _detect_swings(df30, lookback=80)
    bos_up = False
    bos_down = False
    close_30 = float(last30["close"])
    if swing_highs_30:
        last_sh_price = swing_highs_30[-1][1]
        if close_30 > last_sh_price * 1.001:  # phá swing high rõ ràng
            bos_up = True
    if swing_lows_30:
        last_sl_price = swing_lows_30[-1][1]
        if close_30 < last_sl_price * 0.999:  # phá swing low rõ ràng
            bos_down = True

    # 4) Exness alignment
    okx_last_price = float(last30["close"])
    exness_last = get_exness_price()
    if exness_last is None:
        diff = 0.0
        exness_last = okx_last_price
    else:
        diff = exness_last - okx_last_price

    # 5) Một số flag nến cho 30m
    def is_bull(row):
        return row["close"] > row["open"]

    def is_bear(row):
        return row["close"] < row["open"]

    three_bull_30 = (
        is_bull(last30) and is_bull(prev30_1) and is_bull(prev30_2)
        and last30["close"] > prev30_1["close"] > prev30_2["close"]
    )
    three_bear_30 = (
        is_bear(last30) and is_bear(prev30_1) and is_bear(prev30_2)
        and last30["close"] < prev30_1["close"] < prev30_2["close"]
    )

    true_range_30 = last30["high"] - last30["low"]
    big_move_30 = (not math.isnan(atr_30)) and (true_range_30 > 1.0 * atr_30)
    moderate_move_30 = (not math.isnan(atr_30)) and (true_range_30 > 0.8 * atr_30)

    vol_30 = float(last30["volume"])
    vol_ma20_30 = float(last30["vol_ma20"]) if not math.isnan(last30["vol_ma20"]) else 0.0
    vol_ok_30 = (vol_ma20_30 == 0) or (vol_30 > 1.1 * vol_ma20_30)

    # 5b) Cờ cho 15m (phát hiện hồi sớm)
    last3_15 = [last15, prev15_1, prev15_2]
    bull_count_15 = sum(1 for r in last3_15 if is_bull(r))
    bear_count_15 = sum(1 for r in last3_15 if is_bear(r))
    change_15 = float(last15["close"] - prev15_2["close"])

    # News-like bar (30m & 15m)
    news_like = detect_news_like_bar(df30, atr_30, df15, atr_15)

    # Trend Reliability Index cho khung 30m
    tri_score, tri_desc = compute_trend_reliability(
        main_trend=main_trend,
        trend_main=trend_30,
        ms_main=ms_30m,
        ms_sub=ms_15m,
        tf_trends=tf_trends,
        last_main=last30,
        atr_main=atr_30,
        rsi_main=rsi_30,
        vol_main=vol_30,
        vol_ma20_main=vol_ma20_30,
    )

    # =========
    #  Logic tín hiệu: LONG/SHORT MẠNH & HỒI KỸ THUẬT (khung 30m, early 15m)
    # =========
    force = "Trung lập"
    signal = "Không rõ"
    rsi_val = rsi_30

    # chỉ cho phép gọi là "MẠNH" khi:
    # - regime = TREND
    # - ATR đủ lớn (>= 250)
    # - market structure 30m & 15m cùng hướng
    can_strong_short = (
        main_trend == "DOWN"
        and regime == "TREND"
        and atr_30 >= 250
        and ms_30m_is_down
        and ms_15m_is_down
    )

    can_strong_long = (
        main_trend == "UP"
        and regime == "TREND"
        and atr_30 >= 250
        and ms_30m_is_up
        and ms_15m_is_up
    )

    # ========== DOWN TREND ==========
    if main_trend == "DOWN":
        # kiểm tra rơi xa EMA20 để tránh short đuổi đáy
        extended_down = False
        if not math.isnan(atr_30):
            dist_from_ema20 = last30["ema20"] - last30["close"]
            extended_down = dist_from_ema20 > 0.8 * atr_30

        # điều kiện HỒI KỸ THUẬT (30m)
        strong_two_bull_30 = (
            is_bull(last30)
            and is_bull(prev30_1)
            and ((last30["high"] - last30["low"]) > 0.8 * atr_30)
            and ((prev30_1["high"] - prev30_1["low"]) > 0.8 * atr_30)
            and vol_ok_30
            and (not math.isnan(rsi_val) and rsi_val > 40)
            and (not math.isnan(prev_rsi_30) and prev_rsi_30 < 35)
        )
        three_bull_retrace_30 = (
            three_bull_30
            and last30["close"] >= last30["ema20"]
        )

        # điều kiện HỒI KỸ THUẬT SỚM (15m)
        early_long_retrace_15m = (
            bull_count_15 >= 2
            and not math.isnan(rsi_15)
            and rsi_15 > 45
            and (atr_30 > 0 and change_15 > 0.4 * atr_30)
        )

        is_tech_retrace_long = strong_two_bull_30 or three_bull_retrace_30 or early_long_retrace_15m

        if is_tech_retrace_long:
            if early_long_retrace_15m and not (strong_two_bull_30 or three_bull_retrace_30):
                force = "Nhịp hồi kỹ thuật SỚM trong Downtrend (dựa trên khung 15m)."
            else:
                force = "Nhịp hồi kỹ thuật rõ ràng trong Downtrend (3 nến hoặc 2 nến 30m mạnh)."
            signal = "LONG hồi kỹ thuật"

        else:
            # nếu không phải hồi rõ, xét SHORT mạnh nếu đủ điều kiện
            if can_strong_short and is_bear(last30) and last30["close"] < last30["ema20"] < last30["ema50"] and big_move_30 and vol_ok_30:
                if extended_down or (not math.isnan(rsi_val) and rsi_val < 25):
                    force = "Giá đã rơi sâu xa EMA, dễ có nhịp hồi kỹ thuật"
                    signal = "Chờ SHORT lại"
                else:
                    force = "Lực bán chiếm ưu thế, Downtrend mạnh (khung 30m)"
                    signal = "SHORT mạnh"
            else:
                if extended_down or (not math.isnan(rsi_val) and rsi_val < 30):
                    force = "Nhịp hồi/sideway sau pha rơi sâu – có thể đánh LONG hồi nhỏ"
                    signal = "LONG hồi kỹ thuật"
                else:
                    force = "Thị trường đang nhiễu trong Downtrend yếu/sideway"
                    signal = "Không rõ"

    # ========== UP TREND ==========
    elif main_trend == "UP":
        # kiểm tra kéo xa EMA
        extended_up = False
        if not math.isnan(atr_30):
            dist_from_ema20 = last30["close"] - last30["ema20"]
            extended_up = dist_from_ema20 > 0.8 * atr_30

        # điều kiện HỒI KỸ THUẬT (30m)
        strong_two_bear_30 = (
            is_bear(last30)
            and is_bear(prev30_1)
            and ((last30["high"] - last30["low"]) > 0.8 * atr_30)
            and ((prev30_1["high"] - prev30_1["low"]) > 0.8 * atr_30)
            and vol_ok_30
            and (not math.isnan(rsi_val) and rsi_val < 60)
            and (not math.isnan(prev_rsi_30) and prev_rsi_30 > 65)
        )
        three_bear_retrace_30 = (
            three_bear_30
            and last30["close"] <= last30["ema20"]
        )

        # điều kiện HỒI KỸ THUẬT SỚM (15m)
        early_short_retrace_15m = (
            bear_count_15 >= 2
            and not math.isnan(rsi_15)
            and rsi_15 < 55
            and (atr_30 > 0 and -change_15 > 0.4 * atr_30)
        )

        is_tech_retrace_short = strong_two_bear_30 or three_bear_retrace_30 or early_short_retrace_15m

        if is_tech_retrace_short:
            if early_short_retrace_15m and not (strong_two_bear_30 or three_bear_retrace_30):
                force = "Nhịp điều chỉnh giảm SỚM trong Uptrend (dựa trên khung 15m)."
            else:
                force = "Nhịp điều chỉnh giảm (hồi kỹ thuật) rõ ràng trong Uptrend (khung 30m)."
            signal = "SHORT hồi kỹ thuật"

        else:
            # không phải hồi rõ -> xét LONG mạnh nếu đủ điều kiện
            if can_strong_long and is_bull(last30) and last30["close"] > last30["ema20"] > last30["ema50"] and big_move_30 and vol_ok_30:
                if extended_up or (not math.isnan(rsi_val) and rsi_val > 75):
                    force = "Giá đã kéo xa EMA, dễ có nhịp điều chỉnh giảm"
                    signal = "Chờ LONG lại"
                else:
                    force = "Lực mua chiếm ưu thế, Uptrend mạnh (khung 30m)"
                    signal = "LONG mạnh"
            else:
                if extended_up or (not math.isnan(rsi_val) and rsi_val > 70):
                    force = "Nhịp điều chỉnh/sideway sau pha tăng mạnh – có thể SHORT hồi nhỏ"
                    signal = "SHORT hồi kỹ thuật"
                else:
                    force = "Thị trường đang nhiễu trong Uptrend yếu/sideway"
                    signal = "Không rõ"

    # ========== Không rõ trend (SIDE / MIXED) ==========
    else:
        force = "Thị trường sideway, không có xu hướng rõ trên khung lớn"
        signal = "Không rõ"

    # BOS override: nếu vừa phá cấu trúc thì ưu tiên báo đảo chiều, tránh gọi hồi kỹ thuật sai
    if main_trend == "DOWN" and bos_up:
        force = "Giá vừa phá swing high quan trọng trên 30m – có dấu hiệu đảo chiều từ Downtrend sang Uptrend, hạn chế coi đây là nhịp hồi kỹ thuật."
        signal = "Không rõ"
    elif main_trend == "UP" and bos_down:
        force = "Giá vừa phá swing low quan trọng trên 30m – có dấu hiệu đảo chiều từ Uptrend sang Downtrend, hạn chế coi đây là nhịp hồi kỹ thuật."
        signal = "Không rõ"

    # 6) Khả năng hồi / điều chỉnh (EXNESS) dùng ATR 30m
    if "LONG" in signal and "hồi" in signal:
        retrace_info = get_retrace_zones("up", exness_last, atr_30)
    elif "SHORT" in signal and "hồi" in signal:
        retrace_info = get_retrace_zones("down", exness_last, atr_30)
    elif signal == "Chờ SHORT lại":
        retrace_info = get_retrace_zones("up", exness_last, atr_30)
    elif signal == "Chờ LONG lại":
        retrace_info = get_retrace_zones("down", exness_last, atr_30)
    else:
        retrace_info = {"direction": None, "zones": []}

    # 7) Gợi ý lệnh: map signal hiển thị -> trade_signal thực sự
    trade_signal: Optional[str] = None
    if signal in ["SHORT mạnh", "LONG mạnh", "LONG hồi kỹ thuật", "SHORT hồi kỹ thuật"]:
        trade_signal = signal
    elif signal == "Chờ SHORT lại":
        trade_signal = "LONG hồi kỹ thuật"
    elif signal == "Chờ LONG lại":
        trade_signal = "SHORT hồi kỹ thuật"

    # Bảo vệ: nếu là hồi kỹ thuật nhưng nến 30m đã chạy > 70% thời gian
    late_retrace = False
    if trade_signal in ["LONG hồi kỹ thuật", "SHORT hồi kỹ thuật"] and bar_age_ratio_30 > 0.7:
        late_retrace = True
        force += " – Nhịp hồi đã đi được phần lớn cây nến 30m, hạn chế vào lệnh mới (tránh vào trễ)."

    # 7b) Tính Signal Score (trend/momentum/location + TRI + phiên + news)
    trend_score = momentum_score = location_score = total_score = 0
    if trade_signal is not None:
        trend_score, momentum_score, location_score, total_score = compute_signal_score(
            main_trend=main_trend,
            trend_main=trend_30,
            ms_main=ms_30m,
            ms_sub=ms_15m,
            rsi_main=rsi_30,
            atr_main=atr_30,
            last_main=last30,
            prev1_main=prev30_1,
            prev2_main=prev30_2,
            vol_ma20_main=vol_ma20_30,
            trade_signal=trade_signal,
            is_ma5_up=is_ma5_up,
            is_ma5_down=is_ma5_down,
            tri_score=tri_score,
            session_type=session_type,
            news_like=news_like,
        )

    # 7c) Gating: chỉ tạo lệnh khi score >= 60 và không bị late_retrace
    trade: Optional[Dict[str, Any]] = None
    score_comment = ""
    if trade_signal is not None and not late_retrace and total_score >= 60:
        trade = build_trade_suggestion(trade_signal, last30, atr_30)
        if total_score < 75:
            score_comment = (
                f"Điểm chất lượng tín hiệu: {total_score}/100 "
                f"(Trend: {trend_score}, Momentum: {momentum_score}, Vị trí: {location_score}) – "
                f"*tín hiệu KHÁ*, nên vào size vừa phải."
            )
        else:
            score_comment = (
                f"Điểm chất lượng tín hiệu: {total_score}/100 "
                f"(Trend: {trend_score}, Momentum: {momentum_score}, Vị trí: {location_score}) – "
                f"*tín hiệu MẠNH*, có thể cân nhắc vào lệnh chuẩn size."
            )
    elif trade_signal is not None and not late_retrace and total_score < 60:
        score_comment = (
            f"Điểm chất lượng tín hiệu: {total_score}/100 "
            f"(Trend: {trend_score}, Momentum: {momentum_score}, Vị trí: {location_score}) – "
            f"*dưới ngưỡng 60*, ưu tiên QUAN SÁT (NO TRADE)."
        )

    # 8) Build message
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    msg_lines: List[str] = []
    msg_lines.append("✅✅✅ *BTC UPDATE INFO (BTC-USDT)*")
    msg_lines.append(f"Tín hiệu: {signal}")
    if score_comment:
        msg_lines.append(f"- {score_comment}") 
    msg_lines.append(f"Thời gian: `{now_str}`")
    msg_lines.append(f"Giá EXNESS: {exness_last:,.2f} (lệch {diff:+.2f})")
    msg_lines.append("")
    msg_lines.append("*Trend higher timeframe:*")
    msg_lines.append(f"- Trend 30m: {tf_trends['30m']['trend']} (Close: {tf_trends['30m']['close']:,.2f})")
    msg_lines.append(f"- 1H: {tf_trends['1H']['trend']} (Close: {tf_trends['1H']['close']:,.2f})")
    msg_lines.append(f"- 2H: {tf_trends['2H']['trend']} (Close: {tf_trends['2H']['close']:,.2f})")
    msg_lines.append(f"- 4H: {tf_trends['4H']['trend']} (Close: {tf_trends['4H']['close']:,.2f})")
    msg_lines.append(f"→ *Trend chính (ưu tiên 4H)*: {main_trend}")
    msg_lines.append(f"→ Trend Reliability Index (TRI): {tri_score}/100 – {tri_desc}")
    msg_lines.append("")
    msg_lines.append("*Market structure:*")
    msg_lines.append(f"- 15m: {ms_15m}")
    msg_lines.append(f"- 30m: {ms_30m}  *(khung trade chính)*")
    msg_lines.append("")
    msg_lines.append("*Khung 30m (khung trade chính):*")
    msg_lines.append(f"- Xu hướng EMA 30m: {trend_30}")
    msg_lines.append(f"- {force}")
    msg_lines.append(f"- ATR14 30m: {atr_30:.2f}")
    msg_lines.append(f"  → {atr_text}")
    if not math.isnan(rsi_30):
        msg_lines.append(f"- RSI14 30m: {rsi_30:.1f} – Chế độ thị trường: {regime}")
    if news_like:
        msg_lines.append("⚠ Có nến biến động >3×ATR (giống nến tin tức) trong 1–2 nến gần đây (30m hoặc 15m) – nên cẩn trọng với tín hiệu.")
    msg_lines.append("")
    msg_lines.append(f"- {get_session_note(now_utc)}")
    #msg_lines.append(f"- Phiên hiện tại: {session_type}")
    msg_lines.append("")

    if retrace_info["zones"]:
        if retrace_info["direction"] == "up":
            msg_lines.append("*📌 Khả năng hồi lên các vùng (EXNESS – ATR 30m):*")
        else:
            msg_lines.append("*📌 Khả năng điều chỉnh về các vùng (EXNESS – ATR 30m):*")
        for label, z_low, z_high in retrace_info["zones"]:
            msg_lines.append(f"• {label}: {z_low:,.2f} – {z_high:,.2f}")
        msg_lines.append("")

    if trade:
        ex_entry = to_exness_price(trade["entry"], diff)
        ex_tp = to_exness_price(trade["tp"], diff)
        ex_sl = to_exness_price(trade["sl"], diff)

        msg_lines.append("🎯 *Gợi ý lệnh (30m – trend & hồi kỹ thuật):*")
        msg_lines.append(f"- Lệnh: *{trade['side']}* ({trade_signal})")
        #msg_lines.append("")
        #msg_lines.append(f"- Entry OKX: {trade['entry']:,.1f}")
        #msg_lines.append(f"- TP OKX: {trade['tp']:,.1f}")
        #msg_lines.append(f"- SL OKX: {trade['sl']:,.1f}")
        msg_lines.append("")
        msg_lines.append(f"- Entry EXNESS: {ex_entry:,.1f}")
        msg_lines.append(f"- TP EXNESS: {ex_tp:,.1f}")
        msg_lines.append(f"- SL EXNESS: {ex_sl:,.1f}")
    else:
        if "NO TRADE" in score_comment or "quan sát" in score_comment:
            msg_lines.append("⚠ Dù có tín hiệu, *điểm chất lượng thấp* hoặc bối cảnh nhiễu nên ưu tiên QUAN SÁT, chưa gợi ý lệnh cụ thể.")
        else:
            msg_lines.append("⚠ Hiện tín hiệu chưa đủ rõ để gợi ý lệnh (NO TRADE hoặc tránh vào trễ).")

    # === TẠO state_key cho logic chống spam ===
    state_parts = [
        main_trend,
        ms_30m,
        ms_15m,
        trend_30,
        force,
        signal,
        regime,
        atr_text,
        session_type,
        int(tri_score / 10),
        int(trend_score / 5),
        int(momentum_score / 5),
        int(location_score / 5),
        int(news_like),
    ]

    if trade:
        state_parts += [
            trade_signal,
            trade["side"],
            round(trade["entry"] / 10) * 10,
            round(trade["tp"] / 10) * 10,
            round(trade["sl"] / 10) * 10,
        ]

    state_key = "|".join(map(str, state_parts))

    return "\n".join(msg_lines), state_key


def main():
    _log("Start BTC analyzer bot (MAIN TF = 30m)...")

    # build message + state_key
    try:
        text, state_key = analyze_and_build_message()
    except Exception as e:
        _log(f"Analyze error: {e}")
        return
        
    # ⛔ Nếu tín hiệu là "Không rõ" thì bỏ qua, không gửi Telegram
    # (dựa vào dòng "- *Tín hiệu:* Không rõ" trong message)
    if "Tín hiệu:* Không rõ" in text or "Tín hiệu: Không rõ" in text:
        _log("Signal = 'Không rõ' -> skip Telegram để tránh spam.")
        return
    # connect sheet for anti-spam
    try:
        sh = connect_gsheet()
        ws_cache = get_or_create_worksheet(sh, "BT_CACHE_BTC", rows=10, cols=2)
    except Exception as e:
        _log(f"Google Sheet error: {e}")
        ws_cache = None

    new_hash = compute_message_hash(state_key)
    old_hash = None
    if ws_cache is not None:
        old_hash = sheet_read_last_message_hash(ws_cache)

    if old_hash == new_hash:
        _log("State unchanged from last run -> skip Telegram (avoid spam).")
        return

    send_telegram_message(text)
    _log("Message sent to Telegram.")

    if ws_cache is not None:
        sheet_write_last_message_hash(ws_cache, new_hash)
        _log("Updated state hash in BT_CACHE_BTC.")


if __name__ == "__main__":
    main()
