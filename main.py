import os
import json
import requests
import gspread
import pandas as pd

from datetime import datetime, timedelta, timezone
from oauth2client.service_account import ServiceAccountCredentials


# =========================
# CONFIG
# =========================

OKX_BASE = "https://www.okx.com"
OKX_SYMBOL = os.environ.get("OKX_SYMBOL", "BTC-USDT")

GOOGLE_SA_JSON = os.environ.get("GOOGLE_SA_JSON")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_SHEET_WORKSHEET = os.environ.get("GOOGLE_SHEET_WORKSHEET", "CANDLES")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
EXNESS_PRICE_OFFSET = float(os.environ.get("EXNESS_PRICE_OFFSET", "0"))

TIMEFRAMES = {
    "15m": "15m",
    "30m": "30m",
    "1H": "1H",
    "2H": "2H",
    "4H": "4H",
}


# =========================
# GOOGLE SHEETS
# =========================
def to_exness_price(px: float) -> float:
    """Quy đổi giá OKX sang giá tương đương trên Exness bằng offset cố định."""
    return round(px + EXNESS_PRICE_OFFSET, 2)

def _get_gsheet_client():
    if not GOOGLE_SA_JSON or not GOOGLE_SHEET_ID:
        raise RuntimeError("Missing GOOGLE_SA_JSON or GOOGLE_SHEET_ID in env")

    sa_info = json.loads(GOOGLE_SA_JSON)
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
    client = gspread.authorize(creds)
    return client


def get_cache_sheet():
    client = _get_gsheet_client()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sh.worksheet(GOOGLE_SHEET_WORKSHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=GOOGLE_SHEET_WORKSHEET, rows=50, cols=10)
        ws.append_row([
            "timeframe", "close_time",
            "open", "high", "low", "close", "volume",
            "updated_at",
        ])
    return ws


def get_state_sheet():
    client = _get_gsheet_client()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    name = "STATE"
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=10, cols=3)
        ws.append_row(["key", "signature"])
    return ws


def read_cache_row(ws, tf):
    rows = ws.get_all_records()
    for row in rows:
        if str(row.get("timeframe")) == tf:
            return row
    return None


def upsert_cache_row(ws, tf, close_time_str, o, h, l, c, v):
    """
    Mỗi timeframe chỉ 1 dòng:
      - Nếu đã có TF -> update đè
      - Nếu chưa có -> append
    """
    rows = ws.get_all_records()
    target_idx = None

    for i, row in enumerate(rows, start=2):
        if str(row.get("timeframe")) == tf:
            target_idx = i
            break

    values = [
        tf,
        close_time_str,
        o, h, l, c, v,
        datetime.utcnow().isoformat(),
    ]

    if target_idx:
        ws.update(f"A{target_idx}:H{target_idx}", [values])
    else:
        ws.append_row(values)


def get_last_signature():
    ws = get_state_sheet()
    rows = ws.get_all_records()
    for i, row in enumerate(rows, start=2):
        if row.get("key") == "last":
            return row.get("signature"), ws, i
    return None, ws, None


def update_last_signature(ws, row_index, signature):
    if row_index:
        ws.update(f"A{row_index}:B{row_index}", [["last", signature]])
    else:
        ws.append_row(["last", signature])


# =========================
# OKX API
# =========================

def get_okx_candle_latest(inst_id, bar, limit=1):
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
# HIGH TF CACHE (1H,2H,4H)
# =========================

def compute_latest_close_time(tf: str, now_utc: datetime) -> datetime:
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

    raise ValueError("Unsupported TF in compute_latest_close_time")


def get_higher_tf_candle(tf: str, ws) -> dict:
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    close_time = compute_latest_close_time(tf, now_utc)
    close_time_str = close_time.isoformat()

    cached = read_cache_row(ws, tf)
    if cached and str(cached.get("close_time")) == close_time_str:
        return {
            "close_time": close_time,
            "open": float(cached["open"]),
            "high": float(cached["high"]),
            "low": float(cached["low"]),
            "close": float(cached["close"]),
            "volume": float(cached["volume"]),
        }

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
        print("Telegram ENV missing, message below:")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Error sending Telegram:", e)
        print("Message:", text)


# =========================
# TREND & SIGNAL LOGIC
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
    atr = last.get("atr14")
    if atr is None or pd.isna(atr):
        return None

    price = last["close"]

    if "LONG" in signal:
        side = "LONG"
    elif "SHORT" in signal:
        side = "SHORT"
    else:
        return None

    # Trend trade vs hồi kỹ thuật
    if "hồi kỹ thuật" in signal:
        SL_ATR = 0.7
        TP_RR = 1.2
    else:
        SL_ATR = 1.0
        TP_RR = 2.0

    entry = price  # có thể chỉnh về EMA nếu muốn

    if side == "LONG":
        sl = entry - SL_ATR * atr
        tp = entry + SL_ATR * TP_RR * atr
    else:  # SHORT
        sl = entry + SL_ATR * atr
        tp = entry - SL_ATR * TP_RR * atr

    return {
        "side": side,
        "entry": round(entry, 2),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "atr": round(atr, 2),
    }


def build_recommendation(signal: str, trend: str) -> str:
    if signal == "SHORT mạnh":
        return "Khuyến nghị: ⭐ SHORT mạnh theo xu hướng. TP xa, có thể giữ lệnh."
    if signal == "LONG mạnh":
        return "Khuyến nghị: ⭐ LONG mạnh theo xu hướng. TP xa, có thể giữ lệnh."

    if signal == "LONG hồi kỹ thuật":
        return "Khuyến nghị: LONG nhẹ (scalp). TP gần. Không giữ lâu vì ngược xu hướng."
    if signal == "SHORT hồi kỹ thuật":
        return "Khuyến nghị: SHORT nhẹ (scalp). TP gần. Không giữ lâu vì ngược xu hướng."

    if signal == "Chờ SHORT lại":
        return "Khuyến nghị: Nhịp hồi kỹ thuật trong Downtrend – chờ giá lên vùng cản rồi SHORT lại."
    if signal == "Chờ LONG lại":
        return "Khuyến nghị: Nhịp điều chỉnh trong Uptrend – chờ giá điều chỉnh xong rồi LONG lại."

    # fallback
    if trend == "DOWN":
        return "Khuyến nghị: Ưu tiên tìm điểm SHORT, hạn chế LONG dài."
    if trend == "UP":
        return "Khuyến nghị: Ưu tiên tìm điểm LONG, hạn chế SHORT dài."
    return "Khuyến nghị: Thị trường sideway, ưu tiên đứng ngoài."
    
def classify_atr(atr: float) -> str:
    if atr is None or pd.isna(atr):
        return "Không xác định"

    if atr < 150:
        return "Sideway nhẹ, dao động nhỏ"
    elif atr < 250:
        return "Biến động vừa"
    elif atr < 350:
        return "Thị trường bắt đầu mạnh"
    elif atr < 600:
        return "Trend mạnh, breakout mạnh"
    else:
        return "Biến động cực mạnh (thường khi tin tức)"


def build_retrace_zones(main_trend: str, signal: str,
                        df15: pd.DataFrame,
                        df30: pd.DataFrame,
                        c1h: dict,
                        atr: float):
    """
    Trả về dict:
      {
        "direction": "UP" or "DOWN",
        "zones": [(label, (low, high)), ...]
      }
    hoặc None nếu không phải sóng hồi / thiếu dữ liệu.
    """
    if atr is None or pd.isna(atr):
        return None

    # chỉ tính cho các trạng thái sóng hồi / chờ hồi
    is_down_retrace = (
        main_trend == "DOWN" and
        (signal in ["LONG hồi kỹ thuật", "Chờ SHORT lại"])
    )
    is_up_retrace = (
        main_trend == "UP" and
        (signal in ["SHORT hồi kỹ thuật", "Chờ LONG lại"])
    )

    if not (is_down_retrace or is_up_retrace):
        return None

    try:
        width = 0.4 * float(atr)  # biên mỗi vùng ~0.4 ATR

        if is_down_retrace:
            # Hồi lên trong Downtrend
            recent_high_15 = df15["high"].iloc[-10:-1].max()
            recent_high_30 = df30["high"].iloc[-6:-1].max()
            high_1h = float(c1h["high"])

            def z(center):
                return (round(center - width, 2), round(center + width, 2))

            zones = [
                ("Vùng 1", z(recent_high_15)),
                ("Vùng 2", z(recent_high_30)),
                ("Vùng 3 (thấp)", z(high_1h)),
            ]
            return {"direction": "UP", "zones": zones}

        if is_up_retrace:
            # Điều chỉnh xuống trong Uptrend
            recent_low_15 = df15["low"].iloc[-10:-1].min()
            recent_low_30 = df30["low"].iloc[-6:-1].min()
            low_1h = float(c1h["low"])

            def z(center):
                return (round(center - width, 2), round(center + width, 2))

            zones = [
                ("Vùng 1", z(recent_low_15)),
                ("Vùng 2", z(recent_low_30)),
                ("Vùng 3 (thấp)", z(low_1h)),
            ]
            return {"direction": "DOWN", "zones": zones}

    except Exception as e:
        print("Error build_retrace_zones:", repr(e))

    return None


# =========================
# MAIN ANALYSIS
# =========================

def analyze_and_build_message():
    ws = get_cache_sheet()

    # ---- Trend higher timeframe (1H,2H,4H) ----
    c1h = get_higher_tf_candle("1H", ws)
    c2h = get_higher_tf_candle("2H", ws)
    c4h = get_higher_tf_candle("4H", ws)

    t1h = detect_simple_trend_from_candle(c1h)
    t2h = detect_simple_trend_from_candle(c2h)
    t4h = detect_simple_trend_from_candle(c4h)

    main_trend = t4h
    if main_trend == "SIDEWAY":
        main_trend = t2h
    if main_trend == "SIDEWAY":
        main_trend = t1h

    # ---- TF trade chính: 15m ----
    df15 = get_lower_tf_df("15m", 200)
    df15["ema20"] = df15["close"].ewm(span=20).mean()
    df15["ema50"] = df15["close"].ewm(span=50).mean()
    df15["vol_ma20"] = df15["volume"].rolling(20).mean()

    df15["prev_close"] = df15["close"].shift(1)
    df15["tr1"] = df15["high"] - df15["low"]
    df15["tr2"] = (df15["high"] - df15["prev_close"]).abs()
    df15["tr3"] = (df15["low"] - df15["prev_close"]).abs()
    df15["tr"] = df15[["tr1", "tr2", "tr3"]].max(axis=1)
    df15["atr14"] = df15["tr"].rolling(14).mean()

    last = df15.iloc[-1]
    prev1 = df15.iloc[-2]
    prev2 = df15.iloc[-3]

    price = last["close"]
    atr = last["atr14"]
    atr_str = f"{atr:.2f}" if not pd.isna(atr) else "N/A"

    # lưu nến 15m cuối vào cache để theo dõi
    upsert_cache_row(
        ws,
        "15m",
        last["time"].isoformat(),
        last["open"],
        last["high"],
        last["low"],
        last["close"],
        last["volume"],
    )

    # ---- 30m để tham khảo xu hướng gần hơn ----
    df30 = get_lower_tf_df("30m", 200)
    df30["ema20"] = df30["close"].ewm(span=20).mean()
    df30["ema50"] = df30["close"].ewm(span=50).mean()
    trend_30m = detect_tf_trend(df30)

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

    # ---- Xác định tín hiệu trên 15m ----
    def is_bull(row):
        return row["close"] > row["open"]

    def is_bear(row):
        return row["close"] < row["open"]

    three_bull = (
        is_bull(last) and is_bull(prev1) and is_bull(prev2)
        and last["close"] > prev1["close"] > prev2["close"]
    )
    three_bear = (
        is_bear(last) and is_bear(prev1) and is_bear(prev2)
        and last["close"] < prev1["close"] < prev2["close"]
    )

    true_range = last["tr"]
    big_move = (not pd.isna(atr)) and (true_range > 1.0 * atr)
    moderate_move = (not pd.isna(atr)) and (true_range > 0.8 * atr)

    vol = last["volume"]
    vol_ma20 = last["vol_ma20"] if not pd.isna(last["vol_ma20"]) else 0.0
    vol_ok = (vol_ma20 == 0) or (vol > 1.1 * vol_ma20)

    force = "Trung lập"
    signal = "Không rõ"

    # DOWN trend logic
    if main_trend == "DOWN":
        if is_bear(last) and last["close"] < last["ema20"] < last["ema50"] and big_move and vol_ok:
            force = "Lực bán chiếm ưu thế trong Downtrend"
            signal = "SHORT mạnh"
        elif three_bull and last["close"] > last["ema20"] and moderate_move:
            force = "Nhịp hồi kỹ thuật trong Downtrend"
            signal = "LONG hồi kỹ thuật"
        else:
            force = "Nhịp hồi kỹ thuật trong Downtrend"
            signal = "Chờ SHORT lại"

    # UP trend logic
    elif main_trend == "UP":
        if is_bull(last) and last["close"] > last["ema20"] > last["ema50"] and big_move and vol_ok:
            force = "Lực mua chiếm ưu thế trong Uptrend"
            signal = "LONG mạnh"
        elif three_bear and last["close"] < last["ema20"] and moderate_move:
            force = "Nhịp điều chỉnh giảm trong Uptrend"
            signal = "SHORT hồi kỹ thuật"
        else:
            force = "Nhịp điều chỉnh giảm trong Uptrend"
            signal = "Chờ LONG lại"

    else:
        force = "Thị trường sideway"
        signal = "Sideway – ưu tiên đứng ngoài"

    # ---- Gợi ý lệnh & khuyến nghị ----
    recommendation = build_recommendation(signal, main_trend)
    trade = None
    if "LONG" in signal or "SHORT" in signal:
        trade = build_trade_suggestion(signal, last)

    # ---- Các vùng hồi / điều chỉnh ----
    retrace_info = build_retrace_zones(main_trend, signal, df15, df30, c1h, atr)

    # ---- Message ----
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    exness_price = to_exness_price(price)
    
    msg = f"""
    *✅✅✅BTC UPDATE (OKX: {OKX_SYMBOL})*
    Thời gian: `{now_str}`
    Giá EXNESS: `{exness_price:.2f}` (lệch {EXNESS_PRICE_OFFSET:+.2f})
    
    *Trend higher timeframe (cache):*
    - Trend 30m: `{trend_30m}`       
    - 1H: `{t1h}` (Close: {c1h['close']:.2f})
    - 2H: `{t2h}` (Close: {c2h['close']:.2f})
    - 4H: `{t4h}` (Close: {c4h['close']:.2f})
    → *Trend chính (ưu tiên 4H):* `{main_trend}`

    *Khung 15m (khung trade chính):*
    - {force}
    - Tín hiệu: *{signal}*
    - {recommendation}
    - ATR14 15m: `{atr_str}`
      → {classify_atr(atr)}

    if retrace_info:
        if retrace_info["direction"] == "UP":
            msg += "\n*Khả năng hồi lên các vùng (EXNESS):*"
        else:
            msg += "\n*Khả năng điều chỉnh về các vùng (EXNESS):*"
    
        for label, (z_low, z_high) in retrace_info["zones"]:
            ex_low = to_exness_price(z_low)
            ex_high = to_exness_price(z_high)
            msg += f"\n• {label}: {ex_low:.2f} – {ex_high:.2f}"
    
        msg += f"""
    """
    if trade:
        ex_entry = to_exness_price(trade["entry"])
        ex_tp = to_exness_price(trade["tp"])
        ex_sl = to_exness_price(trade["sl"])
        msg += f"""
    *🎯 Gợi ý lệnh (ATR-based 15m):*
    - Lệnh: **{trade['side']}**    
        - Entry: `{ex_entry}`
        - TP: `{ex_tp}`
        - SL: `{ex_sl}`
    """

    # ---- Signature chống spam ----
    trade_side = trade["side"] if trade else "NONE"
    price_band = int(price // 200)  # mỗi ~200$ gửi lại 1 lần dù trạng thái giống

    signature = "|".join([
        main_trend,
        signal,
        t1h,
        t2h,
        t4h,
        trend_30m,
        trade_side,
        str(price_band),
    ])

    return msg, signature


# =========================
# ENTRYPOINT
# =========================

def main():
    try:
        msg, sig = analyze_and_build_message()
        last_sig, state_ws, row_idx = get_last_signature()

        if last_sig == sig:
            print("No state change – skip Telegram.")
            return

        send_telegram(msg)
        update_last_signature(state_ws, row_idx, sig)
        print("Sent Telegram. New signature:", sig)

    except Exception as e:
        print("Error in main():", repr(e))


if __name__ == "__main__":
    main()
