import asyncio
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
)

# ================== КОНФИГ =====================

CONFIG = {
    "BYBIT_API_KEY": "#",
    "TELEGRAM_TOKEN": "#",
    "DEFAULT_TIMEFRAME": "60",      # по умолчанию 60m
    "LOOKBACK": 200,                # сколько свечей берем с биржи
    "BATCH_SIZE": 8,                # сколько монет сканируем параллельно
    "MAX_RETRIES": 3,
    "DELAY_BETWEEN_BATCHES": 1,
    "LEVEL_OFFSET_PCT": 0.0005,     # ~0.05% от цены как буфер для ENTRY/SL
    "NOTIFY_DELAY": 15,             # задержка между сигналами юзеру (сек)
}

TIMEFRAMES = ['15', '30', '60', '120', '240', 'D']

bot = Bot(token=CONFIG["TELEGRAM_TOKEN"])
dp = Dispatcher()
bybit = HTTP(api_key=CONFIG["BYBIT_API_KEY"])

active_users: set[int] = set()
user_timeframes: dict[int, str] = {}
last_notify_time: dict[int, float] = {}
last_symbol_signal_ts: dict[tuple[str, str], int] = {}  # (symbol, timeframe) -> ts последней свечи паттерна


# ================== ХЕЛПЕРЫ ДЛЯ СВЕЧЕЙ =====================

def candle_features(df: pd.DataFrame, i: int):
    row = df.iloc[i]
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    body = abs(c - o)
    rng = h - l
    upper = h - max(o, c)
    lower = min(o, c) - l
    return {
        "open": o, "high": h, "low": l, "close": c,
        "body": body, "range": rng,
        "upper": upper, "lower": lower,
        "bull": c > o,
        "bear": c < o,
    }


def is_up_trend(df: pd.DataFrame, i: int, lookback: int = 5) -> bool:
    if i - lookback + 1 < 0:
        return False
    closes = df["close"].iloc[i - lookback + 1: i + 1].values.astype(float)
    if len(closes) < 2:
        return False
    return closes[-1] > closes[0] * 1.01  # +1% за lookback свечей


def is_down_trend(df: pd.DataFrame, i: int, lookback: int = 5) -> bool:
    if i - lookback + 1 < 0:
        return False
    closes = df["close"].iloc[i - lookback + 1: i + 1].values.astype(float)
    if len(closes) < 2:
        return False
    return closes[-1] < closes[0] * 0.99  # -1% за lookback свечей


def add_atr_volume(df: pd.DataFrame, period_atr: int = 14, period_vol: int = 20):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period_atr).mean()

    df["vol_sma"] = df["volume"].rolling(period_vol).mean()


# ================== ФОРМИРОВАНИЕ СИГНАЛА =====================

def make_signal(df: pd.DataFrame, indices, direction: str,
                pattern_name: str, entry: float, sl: float):
    indices = list(indices)
    if direction == "long":
        risk = entry - sl
    else:
        risk = sl - entry

    if risk <= 0:
        return None

    price = entry
    # риск ~ 0.1–5% от цены
    if risk < 0.001 * price or risk > 0.05 * price:
        return None

    if direction == "long":
        tp1 = entry + risk
        tp2 = entry + 2 * risk
        tp3 = entry + 3 * risk
    else:
        tp1 = entry - risk
        tp2 = entry - 2 * risk
        tp3 = entry - 3 * risk

    ts_list = [int(df["timestamp"].iloc[i]) for i in indices]
    last_ts = ts_list[-1]

    return {
        "direction": direction,
        "pattern": pattern_name,
        "indices": indices,
        "ts_list": ts_list,
        "timestamp": last_ts,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
    }


# ================== ДЕТЕКТОРЫ ПАТТЕРНОВ =====================

def detect_hammer(df: pd.DataFrame):
    i = len(df) - 2
    if i < 5:
        return None

    c = candle_features(df, i)

    if not is_down_trend(df, i, lookback=5):
        return None
    if c["range"] <= 0:
        return None
    if c["lower"] < 2.5 * c["body"]:
        return None
    if c["lower"] < 0.7 * c["range"]:
        return None
    if c["upper"] > 0.2 * c["range"]:
        return None
    if c["close"] < c["low"] + 0.6 * c["range"]:
        return None

    if "atr" in df.columns and "vol_sma" in df.columns:
        atr = df["atr"].iloc[i]
        vol = df["volume"].iloc[i]
        vol_sma = df["vol_sma"].iloc[i]
        if not (np.isfinite(atr) and np.isfinite(vol_sma)):
            return None
        if c["range"] < 0.8 * atr:
            return None
        if vol < 1.2 * vol_sma:
            return None

    buf = c["close"] * CONFIG["LEVEL_OFFSET_PCT"]
    entry = c["high"] + buf
    sl = c["low"] - buf
    return make_signal(df, [i], "long", "Молот", entry, sl)


def detect_shooting_star(df: pd.DataFrame):
    i = len(df) - 2
    if i < 5:
        return None

    c = candle_features(df, i)

    if not is_up_trend(df, i, lookback=5):
        return None
    if c["range"] <= 0:
        return None
    if c["upper"] < 2.5 * c["body"]:
        return None
    if c["upper"] < 0.7 * c["range"]:
        return None
    if c["lower"] > 0.2 * c["range"]:
        return None
    if c["close"] > c["high"] - 0.6 * c["range"]:
        return None

    if "atr" in df.columns and "vol_sma" in df.columns:
        atr = df["atr"].iloc[i]
        vol = df["volume"].iloc[i]
        vol_sma = df["vol_sma"].iloc[i]
        if not (np.isfinite(atr) and np.isfinite(vol_sma)):
            return None
        if c["range"] < 0.8 * atr:
            return None
        if vol < 1.2 * vol_sma:
            return None

    buf = c["close"] * CONFIG["LEVEL_OFFSET_PCT"]
    entry = c["low"] - buf
    sl = c["high"] + buf
    return make_signal(df, [i], "short", "Падающая звезда", entry, sl)


def detect_bullish_engulfing(df: pd.DataFrame):
    i = len(df) - 2
    if i < 2:
        return None

    c1 = candle_features(df, i - 1)
    c2 = candle_features(df, i)

    if not is_down_trend(df, i - 1, lookback=5):
        return None
    if not c1["bear"] or not c2["bull"]:
        return None
    if c2["body"] < 1.3 * c1["body"]:
        return None
    if not (c2["open"] <= c1["close"] and c2["close"] >= c1["open"]):
        return None

    if "vol_sma" in df.columns:
        vol = df["volume"].iloc[i]
        vol_sma = df["vol_sma"].iloc[i]
        if np.isfinite(vol_sma) and vol < 1.2 * vol_sma:
            return None

    buf = c2["close"] * CONFIG["LEVEL_OFFSET_PCT"]
    entry = c2["high"] + buf
    sl = min(c1["low"], c2["low"]) - buf
    return make_signal(df, [i - 1, i], "long", "Бычье поглощение", entry, sl)


def detect_bearish_engulfing(df: pd.DataFrame):
    i = len(df) - 2
    if i < 2:
        return None

    c1 = candle_features(df, i - 1)
    c2 = candle_features(df, i)

    if not is_up_trend(df, i - 1, lookback=5):
        return None
    if not c1["bull"] or not c2["bear"]:
        return None
    if c2["body"] < 1.3 * c1["body"]:
        return None
    if not (c2["open"] >= c1["close"] and c2["close"] <= c1["open"]):
        return None

    if "vol_sma" in df.columns:
        vol = df["volume"].iloc[i]
        vol_sma = df["vol_sma"].iloc[i]
        if np.isfinite(vol_sma) and vol < 1.2 * vol_sma:
            return None

    buf = c2["close"] * CONFIG["LEVEL_OFFSET_PCT"]
    entry = c2["low"] - buf
    sl = max(c1["high"], c2["high"]) + buf
    return make_signal(df, [i - 1, i], "short", "Медвежье поглощение", entry, sl)


def detect_three_white_soldiers(df: pd.DataFrame):
    i = len(df) - 2
    if i < 3:
        return None

    c0 = candle_features(df, i - 2)
    c1 = candle_features(df, i - 1)
    c2 = candle_features(df, i)

    if not is_down_trend(df, i - 2, lookback=5):
        return None
    for c in (c0, c1, c2):
        if not c["bull"]:
            return None
    if not (c1["close"] > c0["close"] and c2["close"] > c1["close"]):
        return None

    bodies = [c0["body"], c1["body"], c2["body"]]
    if min(bodies) <= 0 or max(bodies) > min(bodies) * 2.5:
        return None

    buf = c2["close"] * CONFIG["LEVEL_OFFSET_PCT"]
    entry = c2["high"] + buf
    sl = min(c0["low"], c1["low"], c2["low"]) - buf
    return make_signal(df, [i - 2, i - 1, i], "long", "Три белых солдата", entry, sl)


def detect_three_black_crows(df: pd.DataFrame):
    i = len(df) - 2
    if i < 3:
        return None

    c0 = candle_features(df, i - 2)
    c1 = candle_features(df, i - 1)
    c2 = candle_features(df, i)

    if not is_up_trend(df, i - 2, lookback=5):
        return None
    for c in (c0, c1, c2):
        if not c["bear"]:
            return None
    if not (c1["close"] < c0["close"] and c2["close"] < c1["close"]):
        return None

    bodies = [c0["body"], c1["body"], c2["body"]]
    if min(bodies) <= 0 or max(bodies) > min(bodies) * 2.5:
        return None

    buf = c2["close"] * CONFIG["LEVEL_OFFSET_PCT"]
    entry = c2["low"] - buf
    sl = max(c0["high"], c1["high"], c2["high"]) + buf
    return make_signal(df, [i - 2, i - 1, i], "short", "Три чёрные вороны", entry, sl)


def detect_morning_star(df: pd.DataFrame):
    i = len(df) - 2
    if i < 3:
        return None

    c1 = candle_features(df, i - 2)
    c2 = candle_features(df, i - 1)
    c3 = candle_features(df, i)

    if not is_down_trend(df, i - 2, lookback=5):
        return None
    if not c1["bear"] or not c3["bull"]:
        return None
    if c2["body"] > min(c1["body"], c3["body"]) * 0.7:
        return None
    mid13 = (c1["close"] + c3["close"]) / 2.0
    if c3["close"] <= mid13:
        return None

    buf = c3["close"] * CONFIG["LEVEL_OFFSET_PCT"]
    entry = c3["high"] + buf
    sl = min(c1["low"], c2["low"], c3["low"]) - buf
    return make_signal(df, [i - 2, i - 1, i], "long", "Утренняя звезда", entry, sl)


def detect_evening_star(df: pd.DataFrame):
    i = len(df) - 2
    if i < 3:
        return None

    c1 = candle_features(df, i - 2)
    c2 = candle_features(df, i - 1)
    c3 = candle_features(df, i)

    if not is_up_trend(df, i - 2, lookback=5):
        return None
    if not c1["bull"] or not c3["bear"]:
        return None
    if c2["body"] > min(c1["body"], c3["body"]) * 0.7:
        return None
    mid13 = (c1["close"] + c3["close"]) / 2.0
    if c3["close"] >= mid13:
        return None

    buf = c3["close"] * CONFIG["LEVEL_OFFSET_PCT"]
    entry = c3["low"] - buf
    sl = max(c1["high"], c2["high"], c3["high"]) + buf
    return make_signal(df, [i - 2, i - 1, i], "short", "Вечерняя звезда", entry, sl)


def detect_any_pattern(df: pd.DataFrame):
    detectors = [
        detect_evening_star,
        detect_morning_star,
        detect_bearish_engulfing,
        detect_bullish_engulfing,
        detect_shooting_star,
        detect_hammer,
        detect_three_black_crows,
        detect_three_white_soldiers,
    ]
    for det in detectors:
        sig = det(df)
        if sig is not None:
            return sig
    return None


# ================== ГРАФИК =====================

async def generate_chart(df: pd.DataFrame, symbol: str, timeframe: str, signal: dict) -> BytesIO | None:
    try:
        df = df.copy()
        df["timestamp"] = df["timestamp"].astype(int)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("dt").reset_index(drop=True)

        # до 250 последних свечей
        N = 250
        start = max(0, len(df) - N)
        df_disp = df.iloc[start:].reset_index(drop=True)

        pattern_ts = signal.get("ts_list") or [signal["timestamp"]]
        pattern_ts = [int(t) for t in pattern_ts]
        mask_pat = df_disp["timestamp"].isin(pattern_ts)
        pat_local_idx = np.where(mask_pat)[0].tolist()
        if not pat_local_idx:
            pat_local_idx = [len(df_disp) - 2]

        pat_i_min = min(pat_local_idx)
        pat_i_max = max(pat_local_idx)

        pat_ymin = float(df_disp.loc[pat_i_min:pat_i_max, "low"].min())
        pat_ymax = float(df_disp.loc[pat_i_min:pat_i_max, "high"].max())

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_disp))

        # свечи
        for i, row in df_disp.iterrows():
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            color = "green" if c >= o else "red"

            ax.vlines(x[i], l, h, color=color, linewidth=1)
            body_bottom = min(o, c)
            body_height = abs(c - o)
            if body_height == 0:
                body_height = (h - l) * 0.02
            ax.add_patch(
                Rectangle(
                    (x[i] - 0.3, body_bottom),
                    0.6,
                    body_height,
                    edgecolor=color,
                    facecolor=color,
                    linewidth=1,
                )
            )

        # ===== ТРЕНД-ЛИНИИ ПО МАКСИМУМАМ/МИНИМУМАМ =====
        look_trend = min(150, len(df_disp))
        if look_trend >= 20:
            start_idx = len(df_disp) - look_trend
            end_idx = len(df_disp) - 1
            win = df_disp.iloc[start_idx:end_idx + 1]
            half = look_trend // 2

            lows = win["low"].values
            highs = win["high"].values

            # нижняя линия: минимум первой половины и минимум второй половины
            low_first_pos = np.argmin(lows[:half])
            low_second_pos = half + np.argmin(lows[half:])
            low_x1 = start_idx + low_first_pos
            low_x2 = start_idx + low_second_pos
            low_y1 = lows[low_first_pos]
            low_y2 = lows[low_second_pos]
            ax.plot([low_x1, low_x2], [low_y1, low_y2],
                    color="black", linewidth=1.6, alpha=0.9)

            # верхняя линия: максимум первой половины и максимум второй половины
            high_first_pos = np.argmax(highs[:half])
            high_second_pos = half + np.argmax(highs[half:])
            high_x1 = start_idx + high_first_pos
            high_x2 = start_idx + high_second_pos
            high_y1 = highs[high_first_pos]
            high_y2 = highs[high_second_pos]
            ax.plot([high_x1, high_x2], [high_y1, high_y2],
                    color="black", linewidth=1.6, alpha=0.9)
        # ===============================================

        dir_label = "лонг" if signal["direction"] == "long" else "шорт"
        ax.set_title(f"{symbol} | {timeframe} | {signal['pattern']} ({dir_label})")
        ax.set_xlabel("Свечи (последние)")
        ax.set_ylabel("Цена")

        # ===== ПОДСВЕТКА ПАТТЕРНА (без вертикальной линии) ==

        # Раньше была жёлтая вертикальная линия — убираем её:

        # ===== ПОДПИСЬ ПАТТЕРНА (БЕЗ ЖЁЛТОЙ ПОЛОСЫ) =====
        ax.text(
            pat_i_max,
            pat_ymax * 1.002,
            signal["pattern"],
            color="orange",
            fontsize=10,
            weight="bold",
            va="bottom",
            ha="center",
        )
        # ================================================

        entry = signal["entry"]
        sl = signal["sl"]
        tp1 = signal["tp1"]
        tp2 = signal["tp2"]
        tp3 = signal["tp3"]

        ax.axhline(entry, linestyle="--", linewidth=1.3, color="tab:blue")
        ax.axhline(sl,    linestyle="-",  linewidth=1.3, color="tab:red")
        ax.axhline(tp1,   linestyle="-.", linewidth=1.1, color="tab:green")
        ax.axhline(tp2,   linestyle="-.", linewidth=1.1, color="tab:green")
        ax.axhline(tp3,   linestyle="-.", linewidth=1.1, color="tab:green")

        # было: s=90
        ax.scatter(pat_i_max, entry, s=20, marker="o",
                   color="purple", edgecolor="black", zorder=3)

        # было: fontsize=9
        ax.text(
            pat_i_max,
            entry * (1.001 if signal["direction"] == "long" else 0.999),
            "ENTRY",
            color="purple",
            fontsize=5,  # меньше надпись
            ha="center",
            va="bottom" if signal["direction"] == "long" else "top",
            weight="bold",
            clip_on=True,
        )

        ax.set_xlim(-0.5, len(df_disp) - 0.5)
        y_candidates = [
            float(df_disp["low"].min()),
            float(df_disp["high"].max()),
            entry, sl, tp1, tp2, tp3,
            pat_ymin, pat_ymax,
        ]
        y_min = min(y_candidates)
        y_max = max(y_candidates)
        if y_max <= y_min:
            y_max = y_min * 1.001
        padding = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
        return buf

    except Exception as e:
        print(f"Ошибка генерации графика {symbol}: {e}")
        return None


# ================== РАБОТА С РЫНКОМ =====================

async def get_all_symbols():
    for _ in range(CONFIG["MAX_RETRIES"]):
        try:
            r = await asyncio.to_thread(bybit.get_tickers, category="linear")
            return [s["symbol"] for s in r["result"]["list"] if s["symbol"].endswith("USDT")]
        except Exception as e:
            print(f"Ошибка получения символов: {e}")
            await asyncio.sleep(2)
    return []


async def process_symbol(symbol: str, timeframe: str):
    for _ in range(CONFIG["MAX_RETRIES"]):
        try:
            raw = await asyncio.to_thread(
                bybit.get_kline,
                category="linear",
                symbol=symbol,
                interval=timeframe,
                limit=CONFIG["LOOKBACK"],
            )
            data = raw["result"]["list"]
            if not data or len(data) < 30:
                return

            df = pd.DataFrame(
                data,
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
            )

            # ВАЖНО: сначала приводим типы и сортируем по времени (Bybit даёт в обратном порядке)
            df["timestamp"] = df["timestamp"].astype(int)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df = df.sort_values("timestamp").reset_index(drop=True)

            add_atr_volume(df)

            signal = detect_any_pattern(df)
            if not signal:
                return

            key = (symbol, timeframe)
            last_ts = last_symbol_signal_ts.get(key)
            if last_ts is not None and signal["timestamp"] <= last_ts:
                return
            last_symbol_signal_ts[key] = signal["timestamp"]

            print(f"✅ {symbol}: найден паттерн {signal['pattern']} ({signal['direction']})")

            chart = await generate_chart(df, symbol, timeframe, signal)
            if not chart:
                return

            direction_text = "ЛОНГ" if signal["direction"] == "long" else "ШОРТ"
            tf_text = timeframe if timeframe != "D" else "1D"

            caption = (
                f"🟢 {direction_text} {signal['pattern']} на {symbol} ({tf_text})\n"
                f"🕒 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                f"ENTRY (синяя пунктирная, метка ENTRY): {signal['entry']:.6f}\n"
                f"SL   (красная линия): {signal['sl']:.6f}\n"
                f"TP1/TP2/TP3 (зелёные линии, 1R/2R/3R):\n"
                f" • TP1: {signal['tp1']:.6f}\n"
                f" • TP2: {signal['tp2']:.6f}\n"
                f" • TP3: {signal['tp3']:.6f}\n\n"
                f"⚠️ Это не финансовая рекомендация."
            )

            await notify_users_for_symbol(caption, chart, timeframe)
            return

        except Exception as e:
            print(f"Ошибка {symbol}: {e}")
            await asyncio.sleep(1)


def group_users_by_timeframe():
    res: dict[str, set[int]] = {}
    for uid in active_users:
        tf = user_timeframes.get(uid, CONFIG["DEFAULT_TIMEFRAME"])
        res.setdefault(tf, set()).add(uid)
    return res


async def notify_users_for_symbol(message: str, chart: BytesIO, timeframe: str):
    if not active_users:
        return
    try:
        photo = BufferedInputFile(chart.getvalue(), filename=f"chart_{datetime.now().timestamp()}.png")
        for uid in list(active_users):
            now_ts = datetime.now().timestamp()
            last_ts = last_notify_time.get(uid, 0)
            if now_ts - last_ts < CONFIG["NOTIFY_DELAY"]:
                continue
            last_notify_time[uid] = now_ts
            try:
                await bot.send_photo(chat_id=uid, photo=photo, caption=message)
                print(f"Отправлено уведомление пользователю {uid} (tf={timeframe}).")
            except Exception as e:
                print(f"Ошибка отправки пользователю {uid}: {e}")
    finally:
        chart.close()


async def market_monitor():
    while True:
        symbols = await get_all_symbols()
        if not symbols:
            await asyncio.sleep(30)
            continue

        grouped = group_users_by_timeframe()
        if not grouped:
            await asyncio.sleep(30)
            continue

        for timeframe, users_set in grouped.items():
            if not users_set:
                continue

            print(f"[SCAN] TF={timeframe}, users={len(users_set)}, symbols={len(symbols)}")
            for i in range(0, len(symbols), CONFIG["BATCH_SIZE"]):
                batch = symbols[i:i + CONFIG["BATCH_SIZE"]]
                await asyncio.gather(*(process_symbol(sym, timeframe) for sym in batch))
                await asyncio.sleep(CONFIG["DELAY_BETWEEN_BATCHES"])

        await asyncio.sleep(30)


# ================== TELEGRAM: КНОПКИ/КОМАНДЫ =====================

def reply_keyboard():
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="⏱ Таймфрейм")]],
        resize_keyboard=True,
    )
    return kb


def timeframe_selection_keyboard():
    buttons = [
        InlineKeyboardButton(text=f"{tf}m" if tf != "D" else "Daily", callback_data=f"set_tf:{tf}")
        for tf in TIMEFRAMES
    ]
    rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
    return InlineKeyboardMarkup(inline_keyboard=rows)


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    if user_id not in active_users:
        markup = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="✅ Подписаться на сигналы", callback_data="subscribe")]]
        )
        await message.answer(
            "👋 Привет! Бот ищет паттерны (молоты, поглощения, звёзды, солдаты/вороны) "
            "на фьючерсах Bybit и присылает уровни входа/SL/TP.\n\nПодписаться на уведомления?",
            reply_markup=markup,
        )
    else:
        await message.answer("Вы уже подписаны на уведомления.", reply_markup=reply_keyboard())


@dp.callback_query(F.data == "subscribe")
async def subscribe_user(callback: CallbackQuery):
    user_id = callback.from_user.id
    active_users.add(user_id)
    user_timeframes[user_id] = CONFIG["DEFAULT_TIMEFRAME"]
    await callback.answer("✅ Подписка оформлена.")
    await callback.message.edit_text(
        "Теперь бот будет отправлять вам сигналы.\n"
        f"Текущий таймфрейм: {CONFIG['DEFAULT_TIMEFRAME']}m.",
        reply_markup=None,
    )
    await bot.send_message(user_id, "Для смены таймфрейма жми «⏱ Таймфрейм»", reply_markup=reply_keyboard())


@dp.message(F.text == "⏱ Таймфрейм")
async def on_timeframe_button(message: types.Message):
    await message.answer(
        "Выбери таймфрейм для сигналов:",
        reply_markup=timeframe_selection_keyboard(),
    )


@dp.callback_query(F.data.startswith("set_tf:"))
async def on_timeframe_selected(callback: CallbackQuery):
    tf = callback.data.split(":")[1]
    user_timeframes[callback.from_user.id] = tf
    text = f"✅ Таймфрейм установлен на {tf} минут." if tf != "D" else "✅ Таймфрейм установлен на 1D."
    await callback.message.edit_text(text)
    await callback.answer("Таймфрейм изменён")


@dp.message(Command("stop"))
async def cmd_stop(message: types.Message):
    user_id = message.from_user.id
    active_users.discard(user_id)
    user_timeframes.pop(user_id, None)
    await message.answer("🚫 Уведомления выключены", reply_markup=types.ReplyKeyboardRemove())


# ================== MAIN =====================

async def main():
    monitor_task = asyncio.create_task(market_monitor())
    try:
        await dp.start_polling(bot)
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Остановлено пользователем.")

