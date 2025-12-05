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
    "10m": "5m",   # OKX không có 10m, tạm dùng lại 5m (nếu muốn 10m thật thì build từ 5m)
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
    Kết nối Google Sheet thông qua:
      - GOOGLE_SA_JSON: nội dung file service account JSON
      - GOOGLE_SHEET_ID: ID của file
      - GOOGLE_SHEET_WORKSHEET: tên sheet (ví dụ: CANDLES)
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
            "timeframe",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "updated_at",
        ])
    return ws


def read_cache_row(ws, tf, close_time_str):
    """
    Tìm 1 dòng cache theo:
      - timeframe (tf)
      - close_time (ISO string)
    """
    records = ws.get_all_records()
    for row in records:
        if str(row.get("timeframe")) == tf and str(row.get("close_time")) == close_time_str:
            return row
    return None


def write_cache_row(ws, tf, close_time_str, o, h, l, c, v):
    """
    Ghi thêm 1 dòng nến vào sheet (append)
    """
    ws.append_row([
        tf,
        close_time_str,
        o, h, l, c, v,
        datetime.utcnow().isoformat(),
    ])


# =========================
# OKX API
# =========================

def get_okx_candle_latest(inst_id, bar, limit=1):
    """
    Gọi OKX để lấy nến (candles).
    Trả về nến mới nhất (data[0]).
    """
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("No candle data from OKX")

    row = data[0]
    # OKX format:
    # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm, ...]
    ts, o, h, l, c, vol, *_ = row
    ts = int(ts)  # ms
    close_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    return {
        "close_time": close_time,
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "volume": float(vol),
    }


def get_lower_tf_df(tf: str, limit=200) -> pd.DataFrame:
    """
    Lấy nến lower timeframe (5m, 15m, 30m...) trực tiếp từ OKX (không cache).
    Dùng cho phân tích tín hiệu realtime.
    """
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {tf}")

    bar = TIMEFRAMES[tf]
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {
        "instId": OKX_SYMBOL,
        "bar": bar,
        "limit": limit,
    }
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
        raise RuntimeError("Empty dataframe from OKX lower tf")

    df = df.sort_values("time")  # từ cũ đến mới
    return df


# =========================
# CACHE CHO HIGHER TF
# =========================

def compute_latest_close_time(tf: str, now_utc: datetime) -> datetime:
    """
    Tính thời điểm close_time gần nhất (đã hoặc đang diễn ra) theo timeframe & UTC.
    Không gọi API, chỉ tính bằng toán học.
    """
    if tf == "1H":
        closed = now_utc.replace(minute=0, second=0, microsecond=0)
        if now_utc < closed:
            closed -= timedelta(hours=1)
        return closed

    if tf == "2H":
        hour_block = (now_utc.hour // 2) * 2
        closed = now_utc.replace(hour=hour_block, minute=0, second=0, microsecond=0)
        if now_utc < closed:
            closed -= timedelta(hours=2)
        return closed

    if tf == "4H":
        hour_block = (now_utc.hour // 4) * 4
        closed = now_utc.replace(hour=hour_block, minute=0, second=0, microsecond=0)
        if now_utc < closed:
            closed -= timedelta(hours=4)
        return closed

    if tf == "1D":
        closed = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        if now_utc < closed:
            closed -= timedelta(days=1)
        return closed

    raise ValueError(f"Unsupported higher TF for cache: {tf}")


def get_higher_tf_candle(tf: str, ws) -> dict:
    """
    Lấy nến higher timeframe (1H, 2H, 4H, 1D) với cơ chế cache:
      - Tính close_time gần nhất (UTC).
      - Nếu đã có trong sheet -> lấy ra dùng, không gọi OKX.
      - Nếu chưa có -> gọi OKX 1 lần, lưu vào sheet, rồi dùng.
    """
    if tf not in ("1H", "2H", "4H", "1D"):
        raise ValueError("get_higher_tf_candle chỉ dùng cho 1H/2H/4H/1D")

    if tf not in TIMEFRAMES:
        raise ValueError(f"TIMEFRAMES không có mapping cho {tf}")

    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    close_time = compute_latest_close_time(tf, now_utc)
    close_time_str = close_time.isoformat()

    # 1) thử đọc cache
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

    # 2) chưa có -> gọi OKX 1 lần, lưu cache
    bar = TIMEFRAMES[tf]
    candle = get_okx_candle_latest(OKX_SYMBOL, bar)
    write_cache_row(
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
    """
    Gửi tin nhắn Telegram.
    Nếu thiếu env thì chỉ in ra log.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM env missing, skip sending. Message:")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Error sending Telegram:", e)
        print("Message was:")
        print(text)


# =========================
# PHÂN TÍCH TREND & TÍN HIỆU
# =========================

def detect_simple_trend_from_candle(candle: dict) -> str:
    """
    Trend đơn giản cho higher TF:
      - close > open: UP
      - close < open: DOWN
      - còn lại: SIDEWAY
    """
    if candle["close"] > candle["open"]:
        return "UP"
    if candle["close"] < candle["open"]:
        return "DOWN"
    return "SIDEWAY"


def analyze_and_build_message():
    """
    Chạy 1 lần:
      - Lấy cache high TF (1H,2H,4H,1D)
      - Lấy low TF (5m) realtime
      - Tính EMA, lực mua/bán & tín hiệu
      - Build message text gửi Telegram
    """
    ws = get_cache_sheet()

    # --- Higher TF: dùng cache
    c1h = get_higher_tf_candle("1H", ws)
    c2h = get_higher_tf_candle("2H", ws)
    c4h = get_higher_tf_candle("4H", ws)
    c1d = get_higher_tf_candle("1D", ws)

    trend_1h = detect_simple_trend_from_candle(c1h)
    trend_2h = detect_simple_trend_from_candle(c2h)
    trend_4h = detect_simple_trend_from_candle(c4h)
    trend_1d = detect_simple_trend_from_candle(c1d)

    # Ưu tiên trend chính: 1D > 4H > 2H > 1H
    main_trend = trend_1d
    if main_trend == "SIDEWAY":
        main_trend = trend_4h
    if main_trend == "SIDEWAY":
        main_trend = trend_2h
    if main_trend == "SIDEWAY":
        main_trend = trend_1h

    # --- Low TF: dùng M5 để bắt nhịp lực & tín hiệu
    df5 = get_lower_tf_df("5m", limit=200)
    df5["ema20"] = df5["close"].ewm(span=20).mean()
    df5["ema50"] = df5["close"].ewm(span=50).mean()
    df5["vol_ma20"] = df5["volume"].rolling(20).mean()

    last = df5.iloc[-1]
    prev = df5.iloc[-2]

    price = last["close"]
    body = abs(last["close"] - last["open"])
    body_prev = abs(prev["close"] - prev["open"])
    vol = last["volume"]
    vol_ma20 = last["vol_ma20"] if not pd.isna(last["vol_ma20"]) else 0

    is_bull = last["close"] > last["open"]
    is_bear = last["close"] < last["open"]
    vol_strong = vol_ma20 > 0 and vol > 1.5 * vol_ma20 and body > body_prev

    force = "Trung lập"
    signal = "Không rõ"

    # Logic tín hiệu theo main_trend + M5
    if main_trend == "UP":
        if is_bull and last["close"] > last["ema20"]:
            force = "Lực mua chiếm ưu thế"
            signal = "LONG mạnh" if vol_strong else "Gần LONG (pullback nhẹ / tiếp diễn)"
        elif is_bear and last["close"] < last["ema20"]:
            force = "Nhịp điều chỉnh trong uptrend"
            signal = "Chờ tín hiệu LONG lại, tránh FOMO SHORT"

    elif main_trend == "DOWN":
        if is_bear and last["close"] < last["ema20"]:
            force = "Lực bán chiếm ưu thế"
            signal = "SHORT mạnh" if vol_strong else "Gần SHORT (pullback nhẹ / tiếp diễn)"
        elif is_bull and last["close"] > last["ema20"]:
            force = "Nhịp hồi kỹ thuật trong downtrend"
            signal = "Chờ tín hiệu SHORT lại, tránh FOMO LONG"

    else:  # SIDEWAY
        if vol_strong:
            force = "Biến động mạnh trong vùng sideway"
            signal = "Có thể scalping nhỏ nhưng rủi ro cao"
        else:
            force = "Biến động yếu, thị trường lình xình"
            signal = "Sideway, ưu tiên NO TRADE"

    # Khuyến cáo
    if "LONG mạnh" in signal and main_trend == "UP":
        recommendation = "Khuyến cáo: Có thể vào LONG theo trend, quản lý vốn chặt."
    elif "SHORT mạnh" in signal and main_trend == "DOWN":
        recommendation = "Khuyến cáo: Có thể vào SHORT theo trend, quản lý vốn chặt."
    elif "Gần LONG" in signal or "Gần SHORT" in signal:
        recommendation = "Khuyến cáo: Tín hiệu đang hình thành, có thể vào lệnh nhỏ hoặc chờ thêm 1–2 nến xác nhận."
    elif "Sideway" in signal:
        recommendation = "Khuyến cáo: Thị trường sideway, ưu tiên đứng ngoài để tránh nhiễu."
    else:
        recommendation = "Khuyến cáo: Quan sát thêm, chưa phải điểm vào lệnh đẹp."

    now = datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    msg = f"""
*BTC UPDATE (OKX: {OKX_SYMBOL})*
Thời gian: `{now}`
Giá hiện tại (M5): `{price:.2f}`

*Trend higher timeframe (dựa trên nến cache):*
- 1H: `{trend_1h}` (Close: {c1h['close']:.2f})
- 2H: `{trend_2h}` (Close: {c2h['close']:.2f})
- 4H: `{trend_4h}` (Close: {c4h['close']:.2f})
- 1D: `{trend_1d}` (Close: {c1d['close']:.2f})
→ *Trend chính:* `{main_trend}`

*Khung M5:*
- {force}
- Tín hiệu: *{signal}*

{recommendation}
"""
    return msg


# =========================
# ENTRYPOINT CHO CRON
# =========================

def main():
    try:
        msg = analyze_and_build_message()
        send_telegram(msg)
    except Exception as e:
        # để log ra cho Render log, dễ debug
        print("Error in main():", repr(e))


if __name__ == "__main__":
    main()
