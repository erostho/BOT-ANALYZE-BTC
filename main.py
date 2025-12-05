import os
import json
import requests
import gspread
import pandas as pd

from datetime import datetime, timedelta, timezone
from oauth2client.service_account import ServiceAccountCredentials


# =========================
# CONFIG TỪ BIẾN MÔI TRƯỜNG
# =========================

OKX_BASE = "https://www.okx.com"
OKX_SYMBOL = os.environ.get("OKX_SYMBOL", "BTC-USDT")

GOOGLE_SA_JSON = os.environ.get("GOOGLE_SA_JSON")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_SHEET_WORKSHEET = os.environ.get("GOOGLE_SHEET_WORKSHEET", "CANDLES")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Map timeframe nội bộ -> OKX "bar"
TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1H": "1H",
    "2H": "2H",
    "4H": "4H",
    "1D": "1D",
}


# =========================
# GOOGLE SHEET (CACHE)
# =========================

def get_cache_sheet():
    """
    Kết nối Google Sheet bằng Service Account.
    Dùng:
      - GOOGLE_SA_JSON
      - GOOGLE_SHEET_ID
      - GOOGLE_SHEET_WORKSHEET
    """
    if not GOOGLE_SA_JSON or not GOOGLE_SHEET_ID:
        raise RuntimeError("Missing GOOGLE_SA_JSON or GOOGLE_SHEET_ID in env")

    sa_info = json.loads(GOOGLE_SA_JSON)
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
    client = gspread.authorize(creds)

    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sh.worksheet(GOOGLE_SHEET_WORKSHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=GOOGLE_SHEET_WORKSHEET, rows=2000, cols=10)
        ws.append_row([
            "timeframe", "close_time",
            "open", "high", "low", "close", "volume",
            "updated_at",
        ])
    return ws


def read_cache_row(ws, tf, close_time_str):
    """
    Đọc 1 dòng cache theo (timeframe, close_time).
    """
    rows = ws.get_all_records()
    for row in rows:
        if str(row.get("timeframe")) == tf and str(row.get("close_time")) == close_time_str:
            return row
    return None


def upsert_cache_row(ws, tf, close_time_str, o, h, l, c, v):
    """
    Ghi nến vào sheet:
      - Nếu đã có (timeframe, close_time) -> UPDATE (ghi đè)
      - Nếu chưa có -> APPEND dòng mới

    Giúp tránh bị nhân bản cùng 1 nến như hiện tại.
    """
    rows = ws.get_all_records()
    target_row_index = None  # index thực trên sheet (bắt đầu từ 1)

    # get_all_records bắt đầu từ row 2, nên index thực = i + 2
    for i, row in enumerate(rows, start=2):
        if str(row.get("timeframe")) == tf and str(row.get("close_time")) == close_time_str:
            target_row_index = i
            break

    values = [
        tf,
        close_time_str,
        o, h, l, c, v,
        datetime.utcnow().isoformat()
    ]

    if target_row_index:
        # update đè lên dòng cũ
        cell_range = f"A{target_row_index}:H{target_row_index}"
        ws.update(cell_range, [values])
    else:
        # chưa có -> thêm mới
        ws.append_row(values)


# =========================
# OKX API
# =========================

def get_okx_candle_latest(inst_id, bar, limit=1):
    """
    Lấy nến mới nhất từ OKX (có thể là nến đang chạy nếu chưa đóng).
    Trả về dict: {close_time, open, high, low, close, volume}
    """
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("No candle data from OKX")

    row = data[0]
    ts, o, h, l, c, vol, *_ = row
    ts = int(ts)

    return {
        "close_time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "volume": float(vol),
    }


def get_lower_tf_df(tf: str, limit=200) -> pd.DataFrame:
    """
    Lấy dãy nến lower timeframe từ OKX (M5, M15, M30).
    Không dùng cache vì cần cả chuỗi để tính EMA, ATR.
    """
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {tf}")

    bar = TIMEFRAMES[tf]
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {"instId": OKX_SYMBOL, "bar": bar, "limit": limit}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])

    records = []
    for row in data:
        ts, o, h, l, c, vol, *_ = row
        records.append({
            "time": datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(vol),
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"Empty dataframe for {tf}")

    df = df.sort_values("time")
    return df


# =========================
# HIGH TF CACHE
# =========================

def compute_latest_close_time(tf: str, now_utc: datetime) -> datetime:
    """
    Tính close_time gần nhất cho các TF lớn.
    Dùng làm key cache để tránh ghi trùng.
    """
    if tf == "1H":
        base = now_utc.replace(minute=0, second=0, microsecond=0)
        return base if now_utc >= base else base - timedelta(hours=1)

    if tf == "2H":
        h = (now_utc.hour // 2) * 2
        base = now_utc.replace(hour=h, minute=0, second=0, microsecond=0)
        return base if now_utc >= base else base - timedelta(hours=2)

    if tf == "4H":
        h = (now_utc.hour // 4) * 4
        base = now_utc.replace(hour=h, minute=0, second=0, microsecond=0)
        return base if now_utc >= base else base - timedelta(hours=4)

    if tf == "1D":
        base = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        return base if now_utc >= base else base - timedelta(days=1)

    raise ValueError("Unsupported TF in compute_latest_close_time")


def get_higher_tf_candle(tf: str, ws) -> dict:
    """
    Lấy nến high TF (1H,2H,4H,1D) với cache:
      - Key: (timeframe, close_time)
      - Nếu đã có trong sheet: đọc ra
      - Nếu chưa có: gọi OKX, rồi upsert vào sheet
    """
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    close_time = compute_latest_close_time(tf, now_utc)
    close_time_str = close_time.isoformat()

    cached = read_cache_row(ws, tf, close_time_str)
    if cached:
        return {
            "close_time": close_time,
            "open": float(cached["open"]),
            "high": float(cached["high"]),
            "low": float(cached["low"]),
            "close": float(cached["close"]),
            "volume": float(cached["volume"]),
        }

    # chưa có cache -> gọi OKX 1 lần
    bar = TIMEFRAMES[tf]
    candle = get_okx_candle_latest(OKX_SYMBOL, bar)

    upsert_cache_row(
        ws,
        tf,
        close_time_str,
        candle["open"],
        candle["high"],
        candle["low"],
        candle["close"],
        candle["volume"],
    )

    return candle


# =========================
# TELEGRAM
# =========================

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram env missing, message below:")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}

    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Error sending Telegram:", e)
        print("Message:", text)


# =========================
# TREND & TRADE LOGIC
# =========================

def detect_simple_trend_from_candle(candle: dict) -> str:
    if candle["close"] > candle["open"]:
        return "UP"
    if candle["close"] < candle["open"]:
        return "DOWN"
    return "SIDEWAY"


def detect_tf_trend(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    if last["ema20"] > last["ema50"] and last["close"] > last["ema20"]:
        return "UP"
    if last["ema20"] < last["ema50"] and last["close"] < last["ema20"]:
        return "DOWN"
    return "SIDEWAY"


def build_trade_suggestion(signal: str, last: pd.Series):
    """
    Tạo đề xuất lệnh dựa trên tín hiệu + ATR M5 (volatility).
    TP/SL tính theo:
      - SL = 1 * ATR
      - TP = 2 * SL
    """
    atr = last.get("atr14", None)
    if atr is None or pd.isna(atr):
        return None

    price = last["close"]
    ema20 = last["ema20"]

    SL_ATR = 1.0
    TP_RR = 2.0

    if "LONG" in signal:
        if "Gần" in signal:
            entry = ema20
        else:
            entry = price

        sl = entry - SL_ATR * atr
        tp = entry + SL_ATR * TP_RR * atr
        side = "LONG"

    elif "SHORT" in signal:
        if "Gần" in signal:
            entry = ema20
        else:
            entry = price

        sl = entry + SL_ATR * atr
        tp = entry - SL_ATR * TP_RR * atr
        side = "SHORT"

    else:
        return None

    return {
        "side": side,
        "entry": round(entry, 2),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "atr": round(atr, 2),
    }


# =========================
# MAIN ANALYSIS
# =========================

def analyze_and_build_message() -> str:
    ws = get_cache_sheet()

    # ---- Higher TF với cache ----
    c1h = get_higher_tf_candle("1H", ws)
    c2h = get_higher_tf_candle("2H", ws)
    c4h = get_higher_tf_candle("4H", ws)
    c1d = get_higher_tf_candle("1D", ws)

    t1h = detect_simple_trend_from_candle(c1h)
    t2h = detect_simple_trend_from_candle(c2h)
    t4h = detect_simple_trend_from_candle(c4h)
    t1d = detect_simple_trend_from_candle(c1d)

    main_trend = t1d
    if main_trend == "SIDEWAY":
        main_trend = t4h
    if main_trend == "SIDEWAY":
        main_trend = t2h
    if main_trend == "SIDEWAY":
        main_trend = t1h

    # ---- Lower TF: M5 (tín hiệu chính) ----
    df5 = get_lower_tf_df("5m", 200)
    df5["ema20"] = df5["close"].ewm(span=20).mean()
    df5["ema50"] = df5["close"].ewm(span=50).mean()
    df5["vol_ma20"] = df5["volume"].rolling(20).mean()

    # ATR14
    df5["prev_close"] = df5["close"].shift(1)
    df5["tr1"] = df5["high"] - df5["low"]
    df5["tr2"] = (df5["high"] - df5["prev_close"]).abs()
    df5["tr3"] = (df5["low"] - df5["prev_close"]).abs()
    df5["tr"] = df5[["tr1", "tr2", "tr3"]].max(axis=1)
    df5["atr14"] = df5["tr"].rolling(14).mean()

    last5 = df5.iloc[-1]
    prev5 = df5.iloc[-2]

    price = last5["close"]
    body = abs(last5["close"] - last5["open"])
    body_prev = abs(prev5["close"] - prev5["open"])
    vol = last5["volume"]
    vol_ma20 = last5["vol_ma20"] if not pd.isna(last5["vol_ma20"]) else 0

    is_bull = last5["close"] > last5["open"]
    is_bear = last5["close"] < last5["open"]
    vol_strong = vol_ma20 > 0 and vol > 1.5 * vol_ma20 and body > body_prev

    force = "Trung lập"
    base_signal = "Không rõ"

    if main_trend == "UP":
        if is_bull and last5["close"] > last5["ema20"]:
            force = "Lực mua chiếm ưu thế"
            base_signal = "LONG mạnh"
        elif is_bull and last5["close"] > last5["ema50"]:
            force = "Lực mua chiếm ưu thế"
            base_signal = "Gần LONG"
        elif is_bear:
            force = "Nhịp điều chỉnh trong Uptrend"
            base_signal = "Chờ LONG lại"

        if "LONG mạnh" in base_signal and not vol_strong:
            base_signal = "Gần LONG"

    elif main_trend == "DOWN":
        if is_bear and last5["close"] < last5["ema20"]:
            force = "Lực bán chiếm ưu thế"
            base_signal = "SHORT mạnh"
        elif is_bear and last5["close"] < last5["ema50"]:
            force = "Lực bán chiếm ưu thế"
            base_signal = "Gần SHORT"
        elif is_bull:
            force = "Nhịp hồi kỹ thuật trong Downtrend"
            base_signal = "Chờ SHORT lại"

        if "SHORT mạnh" in base_signal and not vol_strong:
            base_signal = "Gần SHORT"

    else:
        force = "Sideway"
        base_signal = "Sideway – ưu tiên đứng ngoài"

    # ---- M15 & M30 để lọc nhiễu ----
    df15 = get_lower_tf_df("15m", 200)
    df15["ema20"] = df15["close"].ewm(span=20).mean()
    df15["ema50"] = df15["close"].ewm(span=50).mean()
    trend_m15 = detect_tf_trend(df15)

    df30 = get_lower_tf_df("30m", 200)
    df30["ema20"] = df30["close"].ewm(span=20).mean()
    df30["ema50"] = df30["close"].ewm(span=50).mean()
    trend_m30 = detect_tf_trend(df30)

    # Ghi nến M15/M30 cuối cùng vào sheet (nếu bạn muốn nhìn trong cache)
    last15 = df15.iloc[-1]
    upsert_cache_row(
        ws,
        "15m",
        last15["time"].isoformat(),
        last15["open"],
        last15["high"],
        last15["low"],
        last15["close"],
        last15["volume"],
    )

    last30 = df30.iloc[-1]
    upsert_cache_row(
        ws,
        "30m",
        last30["time"].isoformat(),
        last30["open"],
        last30["high"],
        last30["low"],
        last30["close"],
        last30["volume"],
    )

    # ---- Lọc tín hiệu theo M15 & M30 ----
    filtered_signal = base_signal

    if "LONG" in base_signal:
        if trend_m30 == "DOWN":
            filtered_signal = "BỎ QUA – M30 DOWN ngược xu hướng LONG"
        else:
            if trend_m15 == "DOWN":
                filtered_signal += " ⚠️ (M15 DOWN ngược hướng)"
            elif trend_m15 == "SIDEWAY":
                filtered_signal += " ⚠️ (M15 SIDEWAY – tín hiệu yếu)"

    elif "SHORT" in base_signal:
        if trend_m30 == "UP":
            filtered_signal = "BỎ QUA – M30 UP ngược xu hướng SHORT"
        else:
            if trend_m15 == "UP":
                filtered_signal += " ⚠️ (M15 UP ngược hướng)"
            elif trend_m15 == "SIDEWAY":
                filtered_signal += " ⚠️ (M15 SIDEWAY – tín hiệu yếu)"

    # ---- Gợi ý lệnh ATR ----
    if filtered_signal.startswith("BỎ QUA"):
        trade = None
        recommendation = "Khuyến cáo: M30 đi ngược tín hiệu M5, ưu tiên đứng ngoài."
    else:
        trade = build_trade_suggestion(base_signal, last5)

        if "LONG mạnh" in base_signal and main_trend == "UP":
            recommendation = "Khuyến cáo: Có thể vào LONG theo trend, quản lý vốn chặt."
        elif "SHORT mạnh" in base_signal and main_trend == "DOWN":
            recommendation = "Khuyến cáo: Có thể vào SHORT theo trend, quản lý vốn chặt."
        elif "Gần LONG" in base_signal or "Gần SHORT" in base_signal:
            recommendation = "Khuyến cáo: Tín hiệu đang hình thành, có thể vào lệnh nhỏ hoặc chờ thêm 1–2 nến xác nhận."
        elif "Sideway" in base_signal:
            recommendation = "Khuyến cáo: Thị trường sideway, ưu tiên NO TRADE."
        else:
            recommendation = "Khuyến cáo: Quan sát thêm, chưa phải điểm vào lệnh đẹp."

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    msg = f"""
*BTC UPDATE (OKX: {OKX_SYMBOL})*
Thời gian: `{now_str}`
Giá hiện tại (M5): `{price:.2f}`

*Trend higher timeframe (dựa trên nến cache):*
- 1H: `{t1h}` (Close: {c1h['close']:.2f})
- 2H: `{t2h}` (Close: {c2h['close']:.2f})
- 4H: `{t4h}` (Close: {c4h['close']:.2f})
- 1D: `{t1d}` (Close: {c1d['close']:.2f})
→ *Trend chính:* `{main_trend}`

*Khung M5:*
- {force}
- Tín hiệu gốc (M5): *{base_signal}*
- ATR14 M5: `{last5['atr14']:.2f}`

*Khung xác nhận:*
- M15 trend: `{trend_m15}`
- M30 trend: `{trend_m30}`

*Tín hiệu sau khi lọc:* {filtered_signal}

{recommendation}
"""

    if trade:
        msg += f"""
*🎯 Gợi ý lệnh (ATR-based M5):*
- Lệnh: **{trade['side']}**
- Entry: `{trade['entry']}`
- TP: `{trade['tp']}`
- SL: `{trade['sl']}`
(ATR14 M5 ≈ `{trade['atr']}`)
"""

    return msg


# =========================
# ENTRYPOINT
# =========================

def main():
    try:
        msg = analyze_and_build_message()
        send_telegram(msg)
    except Exception as e:
        print("Error in main():", repr(e))


if __name__ == "__main__":
    main()
