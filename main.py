import os
import json
import math
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

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

EXNESS_PRICE_URL = os.getenv("EXNESS_PRICE_URL")  # REST endpoint trả JSON giá Exness hiện tại

TIMEFRAMES = {
    "15m": "15m",
    "30m": "30m",
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


def classify_market_structure(df: pd.DataFrame, lookback: int = 30) -> str:
    """
    Rất đơn giản: so sánh đỉnh/đáy gần đây.
    Trả về: "Tăng (HH–HL)", "Giảm (LH–LL)", "Sideway / lẫn lộn"
    """
    sub = df.tail(lookback)
    highs = sub["high"]
    lows = sub["low"]

    n = len(sub)
    if n < 10:
        return "Không rõ (thiếu dữ liệu)"

    block = n // 3
    h1 = highs.iloc[:block].max()
    h2 = highs.iloc[block:2 * block].max()
    h3 = highs.iloc[2 * block:].max()

    l1 = lows.iloc[:block].min()
    l2 = lows.iloc[block:2 * block].min()
    l3 = lows.iloc[2 * block:].min()

    if h3 > h2 > h1 and l3 > l2 > l1:
        return "Tăng (HH–HL)"
    elif h3 < h2 < h1 and l3 < l2 < l1:
        return "Giảm (LH–LL)"
    else:
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
        # cố gắng tìm key giá
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


def get_retrace_zones(direction: str, last_close: float, atr: float) -> Dict[str, Any]:
    """
    Tính vùng hồi / điều chỉnh dựa trên ATR quanh giá hiện tại.
    direction: "up" (hồi lên) hoặc "down" (điều chỉnh xuống)
    """
    if pd.isna(atr) or atr <= 0:
        return {"direction": direction, "zones": []}

    zones = []
    if direction == "up":
        # 3 vùng kháng cự phía trên
        zones.append(("Vùng 1", last_close + 0.3 * atr, last_close + 0.6 * atr))
        zones.append(("Vùng 2", last_close + 0.6 * atr, last_close + 0.9 * atr))
        zones.append(("Vùng 3 (thấp)", last_close + 0.1 * atr, last_close + 0.3 * atr))
    else:
        # 3 vùng hỗ trợ phía dưới
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
      - "SHORT mạnh" / "LONG mạnh"  -> trend-follow, ATR-based
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
#  Core analysis
# ========================

def analyze_and_build_message() -> str:
    now_utc = datetime.now(timezone.utc)

    # 1) Lấy nến 15m
    df15 = fetch_okx_candles(TIMEFRAMES["15m"], limit=200)
    df15["ema20"] = ema(df15["close"], 20)
    df15["ema50"] = ema(df15["close"], 50)
    df15["atr14"] = calc_atr(df15, 14)
    df15["rsi14"] = rsi(df15["close"], 14)
    df15["vol_ma20"] = df15["volume"].rolling(window=20).mean()

    last15 = df15.iloc[-1]
    prev1 = df15.iloc[-2]
    prev2 = df15.iloc[-3]

    atr_15 = float(last15["atr14"])
    atr_text = classify_atr(atr_15)
    rsi_15 = float(last15["rsi14"]) if not math.isnan(last15["rsi14"]) else float("nan")
    prev_rsi_15 = float(df15["rsi14"].iloc[-2]) if not math.isnan(df15["rsi14"].iloc[-2]) else float("nan")
    regime = detect_regime(rsi_15, atr_15)
    trend_15 = detect_trend_from_ema(last15)

    # 2) Lấy nến higher TF & trend
    tf_trends = {}
    for name in ["30m", "1H", "2H", "4H"]:
        df = fetch_okx_candles(TIMEFRAMES[name], limit=120)
        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        tf_trends[name] = {
            "trend": detect_trend_from_ema(df.iloc[-1]),
            "close": float(df.iloc[-1]["close"]),
        }

    # chọn trend chính: ưu tiên 4H, rồi 2H, 1H, 30m
    main_trend = trend_15
    for key in ["4H", "2H", "1H", "30m"]:
        t = tf_trends.get(key, {}).get("trend")
        if t in ["UP", "DOWN"]:
            main_trend = t
            break

    # 3) Market structure 15m & 30m
    ms_15m = classify_market_structure(df15)
    df30 = fetch_okx_candles(TIMEFRAMES["30m"], limit=120)
    ms_30m = classify_market_structure(df30)

    ms_15m_is_down = "Giảm" in ms_15m
    ms_15m_is_up = "Tăng" in ms_15m
    ms_30m_is_down = "Giảm" in ms_30m
    ms_30m_is_up = "Tăng" in ms_30m

    # 4) Exness alignment
    okx_last_price = float(last15["close"])
    exness_last = get_exness_price()
    if exness_last is None:
        diff = 0.0
        exness_last = okx_last_price
    else:
        diff = exness_last - okx_last_price

    # 5) Một số flag nến 15m
    def is_bull(row):
        return row["close"] > row["open"]

    def is_bear(row):
        return row["close"] < row["open"]

    three_bull = (
        is_bull(last15) and is_bull(prev1) and is_bull(prev2)
        and last15["close"] > prev1["close"] > prev2["close"]
    )
    three_bear = (
        is_bear(last15) and is_bear(prev1) and is_bear(prev2)
        and last15["close"] < prev1["close"] < prev2["close"]
    )

    true_range = last15["high"] - last15["low"]
    big_move = (not math.isnan(atr_15)) and (true_range > 1.0 * atr_15)
    moderate_move = (not math.isnan(atr_15)) and (true_range > 0.8 * atr_15)

    vol = float(last15["volume"])
    vol_ma20 = float(last15["vol_ma20"]) if not math.isnan(last15["vol_ma20"]) else 0.0
    vol_ok = (vol_ma20 == 0) or (vol > 1.1 * vol_ma20)

    rsi_val = rsi_15

    # =========
    #  Logic tín hiệu: LONG/SHORT MẠNH & HỒI KỸ THUẬT
    # =========
    force = "Trung lập"
    signal = "Không rõ"

    # chỉ cho phép gọi là "MẠNH" khi:
    # - regime = TREND
    # - ATR đủ lớn (>= 250)
    # - market structure 15m & 30m cùng hướng
    can_strong_short = (
        main_trend == "DOWN"
        and regime == "TREND"
        and atr_15 >= 250
        and ms_15m_is_down
        and ms_30m_is_down
    )

    can_strong_long = (
        main_trend == "UP"
        and regime == "TREND"
        and atr_15 >= 250
        and ms_15m_is_up
        and ms_30m_is_up
    )

    # ========== DOWN TREND ==========
    if main_trend == "DOWN":
        # kiểm tra rơi xa EMA20 để tránh short đuổi đáy
        extended_down = False
        if not math.isnan(atr_15):
            dist_from_ema20 = last15["ema20"] - last15["close"]
            extended_down = dist_from_ema20 > 0.8 * atr_15

        # điều kiện HỒI KỸ THUẬT (LONG)
        strong_two_bull = (
            is_bull(last15)
            and is_bull(prev1)
            and ( (last15["high"] - last15["low"]) > 0.8 * atr_15 )
            and ( (prev1["high"] - prev1["low"]) > 0.8 * atr_15 )
            and vol_ok
            and (not math.isnan(rsi_val) and rsi_val > 40)
            and (not math.isnan(prev_rsi_15) and prev_rsi_15 < 35)
        )
        three_bull_retrace = (
            three_bull
            and last15["close"] >= last15["ema20"]
        )

        is_tech_retrace_long = strong_two_bull or three_bull_retrace

        if is_tech_retrace_long:
            force = "Nhịp hồi kỹ thuật rõ ràng trong Downtrend (3 nến hoặc 2 nến mạnh)"
            signal = "LONG hồi kỹ thuật"

        else:
            # nếu không phải hồi rõ, xét SHORT mạnh nếu đủ điều kiện
            if can_strong_short and is_bear(last15) and last15["close"] < last15["ema20"] < last15["ema50"] and big_move and vol_ok:
                if extended_down or (not math.isnan(rsi_val) and rsi_val < 25):
                    # đủ điều kiện trend nhưng đã rơi quá sâu / quá bán -> chờ hồi
                    force = "Giá đã rơi sâu xa EMA, dễ có nhịp hồi kỹ thuật"
                    signal = "Chờ SHORT lại"
                else:
                    force = "Lực bán chiếm ưu thế, Downtrend mạnh"
                    signal = "SHORT mạnh"
            else:
                # không đủ điều kiện mạnh -> chỉ nhận diện hồi nhẹ hoặc nhiễu
                if extended_down or (not math.isnan(rsi_val) and rsi_val < 30):
                    force = "Nhịp hồi/sideway sau pha rơi sâu – có thể đánh LONG hồi nhỏ"
                    signal = "LONG hồi kỹ thuật"
                else:
                    force = "Downtrend nhưng tín hiệu 15m chưa rõ – nên quan sát thêm"
                    signal = "Không rõ"

    # ========== UP TREND ==========
    elif main_trend == "UP":
        # kiểm tra kéo xa EMA
        extended_up = False
        if not math.isnan(atr_15):
            dist_from_ema20 = last15["close"] - last15["ema20"]
            extended_up = dist_from_ema20 > 0.8 * atr_15

        # điều kiện HỒI KỸ THUẬT (SHORT)
        strong_two_bear = (
            is_bear(last15)
            and is_bear(prev1)
            and ( (last15["high"] - last15["low"]) > 0.8 * atr_15 )
            and ( (prev1["high"] - prev1["low"]) > 0.8 * atr_15 )
            and vol_ok
            and (not math.isnan(rsi_val) and rsi_val < 60)
            and (not math.isnan(prev_rsi_15) and prev_rsi_15 > 65)
        )
        three_bear_retrace = (
            three_bear
            and last15["close"] <= last15["ema20"]
        )

        is_tech_retrace_short = strong_two_bear or three_bear_retrace

        if is_tech_retrace_short:
            force = "Nhịp điều chỉnh giảm (hồi kỹ thuật) rõ ràng trong Uptrend"
            signal = "SHORT hồi kỹ thuật"

        else:
            # không phải hồi rõ -> xét LONG mạnh nếu đủ điều kiện
            if can_strong_long and is_bull(last15) and last15["close"] > last15["ema20"] > last15["ema50"] and big_move and vol_ok:
                if extended_up or (not math.isnan(rsi_val) and rsi_val > 75):
                    force = "Giá đã kéo xa EMA, dễ có nhịp điều chỉnh giảm"
                    signal = "Chờ LONG lại"
                else:
                    force = "Lực mua chiếm ưu thế, Uptrend mạnh"
                    signal = "LONG mạnh"
            else:
                if extended_up or (not math.isnan(rsi_val) and rsi_val > 70):
                    force = "Nhịp điều chỉnh/sideway sau pha tăng mạnh – có thể SHORT hồi nhỏ"
                    signal = "SHORT hồi kỹ thuật"
                else:
                    force = "Uptrend nhưng tín hiệu 15m chưa rõ – nên quan sát thêm"
                    signal = "Không rõ"

    # ========== Không rõ trend (SIDE / MIXED) ==========
    else:
        force = "Thị trường sideway, không có xu hướng rõ trên khung lớn"
        signal = "Không rõ"

    # 6) Khả năng hồi / điều chỉnh (EXNESS)
    if "LONG" in signal and "hồi" in signal:
        retrace_info = get_retrace_zones("up", exness_last, atr_15)
    elif "SHORT" in signal and "hồi" in signal:
        retrace_info = get_retrace_zones("down", exness_last, atr_15)
    elif signal == "Chờ SHORT lại":
        retrace_info = get_retrace_zones("up", exness_last, atr_15)
    elif signal == "Chờ LONG lại":
        retrace_info = get_retrace_zones("down", exness_last, atr_15)
    else:
        retrace_info = {"direction": None, "zones": []}

    # 7) Gợi ý lệnh: map signal hiển thị -> trade_signal thực sự
    trade_signal = None
    if signal in ["SHORT mạnh", "LONG mạnh", "LONG hồi kỹ thuật", "SHORT hồi kỹ thuật"]:
        trade_signal = signal
    elif signal == "Chờ SHORT lại":
        # view downtrend, nhưng lệnh thực tế ưu tiên LONG hồi kỹ thuật
        trade_signal = "LONG hồi kỹ thuật"
    elif signal == "Chờ LONG lại":
        trade_signal = "SHORT hồi kỹ thuật"

    trade = None
    if trade_signal is not None:
        trade = build_trade_suggestion(trade_signal, last15, atr_15)

    # 8) Build message
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    msg_lines: List[str] = []
    msg_lines.append("✅✅✅ *BTC UPDATE INFO (BTC-USDT)*")
    msg_lines.append(f"- *Tín hiệu:* {signal}")
    msg_lines.append(f"Thời gian: `{now_str}`")
    msg_lines.append(f"Giá EXNESS: {exness_last:,.2f} (lệch {diff:+.2f})")
    msg_lines.append("")
    msg_lines.append("*Trend higher timeframe:*")
    msg_lines.append(f"- Trend 30m: {tf_trends['30m']['trend']} (Close: {tf_trends['30m']['close']:,.2f})")
    msg_lines.append(f"- 1H: {tf_trends['1H']['trend']} (Close: {tf_trends['1H']['close']:,.2f})")
    msg_lines.append(f"- 2H: {tf_trends['2H']['trend']} (Close: {tf_trends['2H']['close']:,.2f})")
    msg_lines.append(f"- 4H: {tf_trends['4H']['trend']} (Close: {tf_trends['4H']['close']:,.2f})")
    msg_lines.append(f"→ *Trend chính (ưu tiên 4H)*: {main_trend}")
    msg_lines.append("")
    msg_lines.append("*Market structure:*")
    msg_lines.append(f"- 15m: {ms_15m}")
    msg_lines.append(f"- 30m: {ms_30m}")
    msg_lines.append("")
    msg_lines.append("*Khung 15m (khung trade chính):*")
    msg_lines.append(f"- Xu hướng EMA 15m: {trend_15}")
    msg_lines.append(f"- {force}")
    msg_lines.append(f"- ATR14 15m: {atr_15:.2f}")
    msg_lines.append(f"  → {atr_text}")
    if not math.isnan(rsi_15):
        msg_lines.append(f"- RSI14 15m: {rsi_15:.1f} – Chế độ thị trường: {regime}")
    msg_lines.append("")
    msg_lines.append(f"- {get_session_note(now_utc)}")
    msg_lines.append("")

    if retrace_info["zones"]:
        if retrace_info["direction"] == "up":
            msg_lines.append("*📌 Khả năng hồi lên các vùng (EXNESS):*")
        else:
            msg_lines.append("*📌 Khả năng điều chỉnh về các vùng (EXNESS):*")
        for label, z_low, z_high in retrace_info["zones"]:
            msg_lines.append(f"• {label}: {z_low:,.2f} – {z_high:,.2f}")
        msg_lines.append("")

        # ... (đoạn build msg_lines giữ nguyên như cũ)

    if trade:
        ex_entry = to_exness_price(trade["entry"], diff)
        ex_tp = to_exness_price(trade["tp"], diff)
        ex_sl = to_exness_price(trade["sl"], diff)

        msg_lines.append("🎯 *Gợi ý lệnh (15m – trend & hồi kỹ thuật):*")
        msg_lines.append(f"- Lệnh: *{trade['side']}* ({trade_signal})")
        msg_lines.append("")
        msg_lines.append(f"- Entry OKX: {trade['entry']:,.1f}")
        msg_lines.append(f"- TP OKX: {trade['tp']:,.1f}")
        msg_lines.append(f"- SL OKX: {trade['sl']:,.1f}")
        msg_lines.append("")
        msg_lines.append(f"- Entry EXNESS: {ex_entry:,.1f}")
        msg_lines.append(f"- TP EXNESS: {ex_tp:,.1f}")
        msg_lines.append(f"- SL EXNESS: {ex_sl:,.1f}")
    else:
        msg_lines.append("⚠ Hiện tín hiệu chưa đủ rõ để gợi ý lệnh cụ thể (NO TRADE).")

    # === TẠO state_key cho logic chống spam ===
    state_parts = [
        main_trend,
        ms_15m,
        ms_30m,
        trend_15,
        force,
        signal,
        regime,
        atr_text,
    ]

    if trade:
        # làm tròn cho đỡ nhạy với vài đô
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
    _log("Start BTC analyzer bot...")

    # build message + state_key
    try:
        text, state_key = analyze_and_build_message()
    except Exception as e:
        _log(f"Analyze error: {e}")
        return


    # connect sheet for anti-spam
    try:
        sh = connect_gsheet()
        ws_cache = get_or_create_worksheet(sh, "BT_CACHE", rows=10, cols=2)
    except Exception as e:
        _log(f"Google Sheet error: {e}")
        ws_cache = None

    new_hash = compute_message_hash(state_key)
    old_hash = None
    if ws_cache is not None:
        old_hash = sheet_read_last_message_hash(ws_cache)
    if old_hash == new_hash:
        _log("Message unchanged from last run -> skip Telegram (avoid spam).")
        return

    send_telegram_message(text)
    _log("Message sent to Telegram.")

    if ws_cache is not None:
        sheet_write_last_message_hash(ws_cache, new_hash)
        _log("Updated message hash in BT_CACHE.")


if __name__ == "__main__":
    main()
