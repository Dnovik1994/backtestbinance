"""
Скрипт 4: Полный бэктест и аналитика сигналов
Читает trades.csv → скачивает свечи Binance → симулирует сделки → считает все метрики

Установка:
    pip install python-binance pandas tabulate

Запуск:
    python3 4_analytics.py

Результат:
    - backtest_full.csv      — детали по каждой сделке
    - backtest_summary.txt   — полный отчёт со всеми метриками
"""

import csv
import os
import time
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
from binance.client import Client
from tabulate import tabulate

# ===========================
# Настройки
# ===========================
API_KEY    = ""
API_SECRET = ""

INPUT_FILE   = "trades.csv"
OUTPUT_CSV   = "backtest_full.csv"
OUTPUT_TXT   = "backtest_summary.txt"
INVALID_FILE = "backtest_invalid.csv"  # сделки с опечатками/аномалиями

POSITION_USDT   = 250.0   # базовый объём позиции
DCA_USDT        = 250.0   # объём при усреднении
LEVERAGE        = 20      # плечо

TP_WEIGHTS      = [0.5, 0.3, 0.2]   # доля позиции на TP1/TP2/TP3

# Горизонты для анализа вероятности направления (в минутах)
DIRECTION_HORIZONS = [5, 15, 30, 60]

# Варианты стопа после TP1 (% от точки входа)
SL_AFTER_TP1_VARIANTS = [-0.05, -0.10, -0.15]  # -5%, -10%, -15%

# Максимальный размер стопа от entry (защита от опечаток)
MAX_SL_DIST_PCT = 0.20   # 20%

# Максимум дней ожидания закрытия сделки (защита от бесконечного цикла)
# 7 дней × 24ч × 60мин = 10 080 минутных свечей
MAX_DAYS = 7
BATCH_CANDLES = 1500  # максимум свечей за один запрос к Binance
# ===========================

client = Client(API_KEY, API_SECRET)

CANDLES_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candles_cache")
os.makedirs(CANDLES_CACHE_DIR, exist_ok=True)


# ===========================
# Утилиты
# ===========================
def _cache_path(symbol: str, start_ts_ms: int) -> str:
    """Путь к кэш-файлу: candles_cache/{symbol}_{ts_ms}.csv.gz (ts округлён до минуты)."""
    ts_rounded = (start_ts_ms // 60_000) * 60_000
    return os.path.join(CANDLES_CACHE_DIR, f"{symbol}_{ts_rounded}.csv.gz")


def get_klines_batch(symbol: str, start_ts_ms: int, limit: int = BATCH_CANDLES) -> pd.DataFrame:
    """Загружает один батч минутных свечей (с кэшированием)."""
    cache_file = _cache_path(symbol, start_ts_ms)

    # Проверяем кэш
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, compression="gzip")
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            print(f"  [cache] {symbol} loaded from {os.path.basename(cache_file)}")
            return df
        except Exception as e:
            print(f"  [!] cache read error {cache_file}: {e}")

    # Скачиваем с Binance
    try:
        raw = client.futures_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            startTime=start_ts_ms,
            limit=limit,
        )
    except Exception as e:
        print(f"  [!] klines error {symbol}: {e}")
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    # Сохраняем в кэш
    try:
        df.to_csv(cache_file, index=False, compression="gzip")
        print(f"  [cache] {symbol} saved to {os.path.basename(cache_file)}")
    except Exception as e:
        print(f"  [!] cache write error: {e}")

    return df


def get_klines_until_close(symbol: str, start_ts_ms: int,
                            tp_prices: list, sl: float, side: str) -> pd.DataFrame:
    """
    Подгружает свечи батчами пока не сработает TP или SL.
    Максимум MAX_DAYS дней — защита от бесконечного цикла.
    """
    max_ts_ms = start_ts_ms + MAX_DAYS * 24 * 60 * 60 * 1000
    all_frames = []
    current_ts = start_ts_ms
    total_candles = 0

    while current_ts < max_ts_ms:
        df = get_klines_batch(symbol, current_ts, limit=BATCH_CANDLES)
        if df.empty:
            break

        all_frames.append(df)
        total_candles += len(df)

        # Проверяем — закрылась ли сделка в этом батче
        closed = False
        for _, row in df.iterrows():
            high, low = row["high"], row["low"]
            # Проверяем все TP
            for tp in tp_prices:
                if (side == "BUY" and high >= tp) or (side == "SELL" and low <= tp):
                    closed = True
                    break
            # Проверяем SL
            if (side == "BUY" and low <= sl) or (side == "SELL" and high >= sl):
                closed = True
                break
            if closed:
                break

        if closed:
            break

        # Если батч неполный — данных больше нет (будущее)
        if len(df) < BATCH_CANDLES:
            break

        # Следующий батч начинается после последней свечи
        last_ts = int(df.iloc[-1]["open_time"].timestamp() * 1000)
        current_ts = last_ts + 60_000  # +1 минута
        time.sleep(0.1)  # пауза чтобы не словить rate limit

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    if total_candles > MAX_DAYS * 24 * 60:
        print(f"    → загружено {total_candles} свечей ({total_candles//1440:.1f} дней)")
    return result


def pnl_usdt(entry: float, exit_price: float, side: str, usdt: float) -> float:
    """PnL в USDT с учётом плеча."""
    qty = (usdt * LEVERAGE) / entry
    if side == "BUY":
        return (exit_price - entry) * qty
    else:
        return (entry - exit_price) * qty


def roi_pct(pnl: float, usdt_invested: float) -> float:
    """ROI в % от вложенного капитала (без плеча)."""
    if usdt_invested == 0:
        return 0.0
    return (pnl / usdt_invested) * 100


# ===========================
# Симуляция одной сделки
# ===========================
def simulate(row: Dict, df: pd.DataFrame, force_no_dca: bool = False) -> Dict[str, Any]:
    """
    Симулирует сделку по минутным свечам.
    Возвращает подробный результат по всем сценариям.
    force_no_dca=True — всегда использует POSITION_USDT (без усреднения).
    """
    side        = row["side"]
    entry       = float(row["entry_price"])
    tp1         = float(row["tp1"]) if row["tp1"] else None
    tp2         = float(row["tp2"]) if row["tp2"] else None
    tp3         = float(row["tp3"]) if row["tp3"] else None
    sl          = float(row["sl"])  if row["sl"]  else None
    has_dca     = int(row["dca_updates_count"]) > 0 or bool(row["dca_price"])

    tp_prices   = [t for t in [tp1, tp2, tp3] if t is not None]
    n_tp        = len(tp_prices)

    # Объём с учётом усреднения
    if force_no_dca:
        total_usdt = POSITION_USDT
    else:
        total_usdt  = POSITION_USDT + (DCA_USDT if has_dca else 0)

    if df.empty or not tp_prices or sl is None:
        return _empty_result()

    # Фильтр опечаток: SL не может быть дальше MAX_SL_DIST_PCT от entry
    sl_dist_pct = abs(sl - entry) / entry
    if sl_dist_pct > MAX_SL_DIST_PCT:
        print(f"    → пропуск: SL слишком далеко ({sl_dist_pct*100:.1f}% от entry={entry}, sl={sl})")
        return _empty_result(reason="INVALID_SL")

    # ── Сценарий A: Базовый (без троллинга) ──────────────────
    result_base = _sim_base(df, side, entry, tp_prices, sl, total_usdt)

    # ── Сценарий B: Троллинг стопа (после TP1 → в б/у, после TP2 → TP1) ──
    result_trail = _sim_trailing(df, side, entry, tp_prices, sl, total_usdt)

    # ── Сценарий C: Варианты стопа после TP1 (-5/-10/-15%) ───
    variants_c = {}
    for pct in SL_AFTER_TP1_VARIANTS:
        sl_new = entry * (1 + pct) if side == "BUY" else entry * (1 - pct)
        variants_c[f"sl_after_tp1_{int(abs(pct)*100)}pct"] = _sim_trailing_custom(
            df, side, entry, tp_prices, sl, sl_new, total_usdt
        )

    # ── Сценарий D: Широкий стоп 50% от объёма ───────────────
    sl_wide_dist = (total_usdt * LEVERAGE * 0.5) / ((total_usdt * LEVERAGE) / entry)
    sl_wide = (entry - sl_wide_dist) if side == "BUY" else (entry + sl_wide_dist)
    result_wide_sl = _sim_base(df, side, entry, tp_prices, sl_wide, total_usdt)

    # ── Сценарий E: 48-часовое управление ─────────────────────
    result_48h = _sim_48h(df, side, entry, tp_prices, sl, total_usdt)

    # ── Блок вероятности направления ─────────────────────────
    direction_prob = _calc_direction_prob(df, side, DIRECTION_HORIZONS)

    return {
        "total_usdt":     total_usdt,
        "has_dca":        has_dca,
        "tp_prices":      tp_prices,
        "sl":             sl,

        # Сценарий A
        "base":           result_base,

        # Сценарий B
        "trail":          result_trail,

        # Сценарий C
        **variants_c,

        # Сценарий D
        "wide_sl":        result_wide_sl,

        # Сценарий E
        "48h":            result_48h,

        # Вероятность направления
        "direction":      direction_prob,
    }


def _sim_base(df, side, entry, tp_prices, sl, total_usdt):
    """Базовая симуляция: TP1-50%/TP2-30%/TP3-20%, SL без троллинга."""
    n = len(tp_prices)
    weights = _norm_weights(TP_WEIGHTS, n)

    realized    = 0.0
    remaining_w = 1.0
    tps_hit     = 0
    outcome     = "TIMEOUT"
    close_price = None
    candles     = 0

    for _, row in df.iterrows():
        candles += 1
        high, low = row["high"], row["low"]

        while tps_hit < n:
            tp = tp_prices[tps_hit]
            if (side == "BUY" and high >= tp) or (side == "SELL" and low <= tp):
                w = weights[tps_hit]
                realized    += pnl_usdt(entry, tp, side, total_usdt * w)
                remaining_w -= w
                tps_hit     += 1
            else:
                break

        if sl and ((side == "BUY" and low <= sl) or (side == "SELL" and high >= sl)):
            realized    += pnl_usdt(entry, sl, side, total_usdt * remaining_w)
            remaining_w  = 0.0
            outcome      = "SL"
            close_price  = sl
            break

        if tps_hit == n and remaining_w <= 0.001:
            outcome     = f"TP{n}"
            close_price = tp_prices[-1]
            break

    if outcome == "TIMEOUT" and remaining_w > 0.001:
        last = float(df.iloc[-1]["close"])
        realized   += pnl_usdt(entry, last, side, total_usdt * remaining_w)
        close_price = last
        if tps_hit > 0:
            outcome = f"TP{tps_hit}_TIMEOUT"

    return {
        "outcome":    outcome,
        "tps_hit":    tps_hit,
        "pnl":        round(realized, 4),
        "roi":        round(roi_pct(realized, total_usdt), 2),
        "candles":    candles,
        "close_price": close_price,
    }


def _sim_trailing(df, side, entry, tp_prices, sl_orig, total_usdt):
    """
    Троллинг стопа:
      После TP1 → SL на точку входа (б/у)
      После TP2 → SL на TP1
    """
    n       = len(tp_prices)
    weights = _norm_weights(TP_WEIGHTS, n)

    realized    = 0.0
    remaining_w = 1.0
    tps_hit     = 0
    current_sl  = sl_orig
    outcome     = "TIMEOUT"
    close_price = None
    candles     = 0

    for _, row in df.iterrows():
        candles += 1
        high, low = row["high"], row["low"]

        # Проверяем TP
        while tps_hit < n:
            tp = tp_prices[tps_hit]
            if (side == "BUY" and high >= tp) or (side == "SELL" and low <= tp):
                w = weights[tps_hit]
                realized    += pnl_usdt(entry, tp, side, total_usdt * w)
                remaining_w -= w
                tps_hit     += 1

                # Двигаем SL
                if tps_hit == 1:
                    current_sl = entry          # б/у
                elif tps_hit == 2 and len(tp_prices) >= 1:
                    current_sl = tp_prices[0]   # на TP1
            else:
                break

        # Проверяем SL
        if (side == "BUY" and low <= current_sl) or (side == "SELL" and high >= current_sl):
            realized    += pnl_usdt(entry, current_sl, side, total_usdt * remaining_w)
            remaining_w  = 0.0
            outcome      = f"SL_after_TP{tps_hit}" if tps_hit > 0 else "SL"
            close_price  = current_sl
            break

        if tps_hit == n and remaining_w <= 0.001:
            outcome     = f"TP{n}"
            close_price = tp_prices[-1]
            break

    if outcome == "TIMEOUT" and remaining_w > 0.001:
        last = float(df.iloc[-1]["close"])
        realized   += pnl_usdt(entry, last, side, total_usdt * remaining_w)
        close_price = last
        if tps_hit > 0:
            outcome = f"TP{tps_hit}_TIMEOUT"

    return {
        "outcome":    outcome,
        "tps_hit":    tps_hit,
        "pnl":        round(realized, 4),
        "roi":        round(roi_pct(realized, total_usdt), 2),
        "candles":    candles,
        "close_price": close_price,
    }


def _sim_trailing_custom(df, side, entry, tp_prices, sl_orig, sl_after_tp1, total_usdt):
    """
    Троллинг с кастомным SL после TP1 (например entry -5%).
    """
    n       = len(tp_prices)
    weights = _norm_weights(TP_WEIGHTS, n)

    realized    = 0.0
    remaining_w = 1.0
    tps_hit     = 0
    current_sl  = sl_orig
    outcome     = "TIMEOUT"
    close_price = None
    candles     = 0

    for _, row in df.iterrows():
        candles += 1
        high, low = row["high"], row["low"]

        while tps_hit < n:
            tp = tp_prices[tps_hit]
            if (side == "BUY" and high >= tp) or (side == "SELL" and low <= tp):
                w = weights[tps_hit]
                realized    += pnl_usdt(entry, tp, side, total_usdt * w)
                remaining_w -= w
                tps_hit     += 1

                if tps_hit == 1:
                    current_sl = sl_after_tp1
                elif tps_hit == 2 and len(tp_prices) >= 1:
                    current_sl = tp_prices[0]
            else:
                break

        if (side == "BUY" and low <= current_sl) or (side == "SELL" and high >= current_sl):
            realized    += pnl_usdt(entry, current_sl, side, total_usdt * remaining_w)
            remaining_w  = 0.0
            outcome      = f"SL_after_TP{tps_hit}" if tps_hit > 0 else "SL"
            close_price  = current_sl
            break

        if tps_hit == n and remaining_w <= 0.001:
            outcome     = f"TP{n}"
            close_price = tp_prices[-1]
            break

    if outcome == "TIMEOUT" and remaining_w > 0.001:
        last = float(df.iloc[-1]["close"])
        realized   += pnl_usdt(entry, last, side, total_usdt * remaining_w)
        close_price = last
        if tps_hit > 0:
            outcome = f"TP{tps_hit}_TIMEOUT"

    return {
        "outcome":    outcome,
        "tps_hit":    tps_hit,
        "pnl":        round(realized, 4),
        "roi":        round(roi_pct(realized, total_usdt), 2),
        "candles":    candles,
        "close_price": close_price,
    }


MAX_48H_CANDLES = 48 * 60  # 2880 минутных свечей = 48 часов


def _sim_48h(df, side, entry, tp_prices, sl_orig, total_usdt):
    """
    Сценарий E: 48-часовое управление с динамическим SL.
    Троллинг стопа (как сценарий B) + динамический SL по глубине убытка +
    временные правила (36ч/42ч/48ч) + отслеживание ликвидации.
    """
    n       = len(tp_prices)
    weights = _norm_weights(TP_WEIGHTS, n)

    realized    = 0.0
    remaining_w = 1.0
    tps_hit     = 0
    outcome     = "E_timeout_48h"
    close_price = None
    candles     = 0
    liquidation_at_candle = None
    sl_locked   = False  # SL фиксируется после TP1 (троллинг)

    # Порог ликвидации: цена, при которой убыток >= 100% от total_usdt
    qty = (total_usdt * LEVERAGE) / entry
    liq_dist = total_usdt / qty  # ценовое расстояние до ликвидации
    if side == "BUY":
        liq_price = entry - liq_dist
    else:
        liq_price = entry + liq_dist

    for _, row in df.iterrows():
        candles += 1
        if candles > MAX_48H_CANDLES:
            break
        high, low, close = row["high"], row["low"], float(row["close"])

        # --- Проверяем ликвидацию (фиксируем момент, но не останавливаем) ---
        if liquidation_at_candle is None:
            if (side == "BUY" and low <= liq_price) or (side == "SELL" and high >= liq_price):
                liquidation_at_candle = candles

        # --- Проверяем TP (троллинг как в сценарии B) ---
        while tps_hit < n:
            tp = tp_prices[tps_hit]
            if (side == "BUY" and high >= tp) or (side == "SELL" and low <= tp):
                w = weights[tps_hit]
                realized    += pnl_usdt(entry, tp, side, total_usdt * w)
                remaining_w -= w
                tps_hit     += 1
                sl_locked    = True

                # Двигаем SL (троллинг)
                if tps_hit == 1:
                    current_sl_trailing = entry          # б/у
                elif tps_hit == 2 and len(tp_prices) >= 1:
                    current_sl_trailing = tp_prices[0]   # на TP1
            else:
                break

        # Все TP взяты
        if tps_hit == n and remaining_w <= 0.001:
            outcome     = f"TP{n}"
            close_price = tp_prices[-1]
            break

        # --- Динамический SL (пока не зафиксирован троллингом) ---
        if not sl_locked:
            current_roi = roi_pct(pnl_usdt(entry, close, side, total_usdt), total_usdt)
            if current_roi < -40:
                # Не ставим SL, ждём 48ч
                current_sl_dynamic = None
            elif current_roi < -30:
                # SL = entry ± 41%
                if side == "BUY":
                    current_sl_dynamic = entry * (1 - 0.41)
                else:
                    current_sl_dynamic = entry * (1 + 0.41)
            elif current_roi < -20:
                # SL = entry ± 31%
                if side == "BUY":
                    current_sl_dynamic = entry * (1 - 0.31)
                else:
                    current_sl_dynamic = entry * (1 + 0.31)
            else:
                # ROI от 0% до -20% → SL = entry ± 21%
                if side == "BUY":
                    current_sl_dynamic = entry * (1 - 0.21)
                else:
                    current_sl_dynamic = entry * (1 + 0.21)

            active_sl = current_sl_dynamic
        else:
            active_sl = current_sl_trailing

        # --- Проверяем SL ---
        if active_sl is not None:
            if (side == "BUY" and low <= active_sl) or (side == "SELL" and high >= active_sl):
                realized    += pnl_usdt(entry, active_sl, side, total_usdt * remaining_w)
                remaining_w  = 0.0
                if sl_locked:
                    outcome = f"E_SL_after_TP{tps_hit}" if tps_hit > 0 else "E_SL"
                else:
                    outcome = "E_SL"
                close_price  = active_sl
                break

        # --- Временные правила (свеча >= 2160 = 36ч) ---
        if remaining_w > 0.001:
            current_roi_now = roi_pct(pnl_usdt(entry, close, side, total_usdt), total_usdt)

            if candles >= 2160:
                # ROI от 1% до 1.5% → закрыть
                if 1.0 <= current_roi_now <= 1.5:
                    realized    += pnl_usdt(entry, close, side, total_usdt * remaining_w)
                    remaining_w  = 0.0
                    outcome      = "E_close_36_48h"
                    close_price  = close
                    break
                # ROI отрицательный и <= -10% → закрыть немедленно
                elif current_roi_now <= -10:
                    realized    += pnl_usdt(entry, close, side, total_usdt * remaining_w)
                    remaining_w  = 0.0
                    outcome      = "E_close_minus10pct"
                    close_price  = close
                    break
                # ROI > 1.7% → продолжать до 48ч (ничего не делаем)

            if candles >= 2520:
                # 42ч: если ROI от -10% до -15%
                if -15 <= current_roi_now < -10:
                    # закрыть в -1% от entry
                    if side == "BUY":
                        close_at = entry * 0.99
                    else:
                        close_at = entry * 1.01
                    realized    += pnl_usdt(entry, close_at, side, total_usdt * remaining_w)
                    remaining_w  = 0.0
                    outcome      = "E_close_42h_minus1pct"
                    close_price  = close_at
                    break

    # Принудительное закрытие по 48ч лимиту
    if remaining_w > 0.001:
        if not df.empty:
            last = float(df.iloc[min(candles - 1, len(df) - 1)]["close"])
        else:
            last = entry
        realized   += pnl_usdt(entry, last, side, total_usdt * remaining_w)
        close_price = last
        outcome     = "E_timeout_48h"

    # Если была ликвидация — PnL = -total_usdt
    if liquidation_at_candle is not None and outcome not in ("TP1", "TP2", "TP3"):
        # Проверяем, не вышли ли мы по TP до ликвидации
        if outcome.startswith("E_SL") or outcome == "E_liquidation":
            realized = -total_usdt
            outcome  = "E_liquidation"

    return {
        "outcome":               outcome,
        "tps_hit":               tps_hit,
        "pnl":                   round(realized, 4),
        "roi":                   round(roi_pct(realized, total_usdt), 2),
        "candles":               candles,
        "close_price":           close_price,
        "liquidation_at_candle": liquidation_at_candle,
    }


def _calc_direction_prob(df: pd.DataFrame, side: str, horizons: List[int]) -> Dict[str, float]:
    """
    Для каждого горизонта считает % свечей где цена пошла в нужную сторону.
    BUY  → цена закрытия выше цены открытия первой свечи
    SELL → цена закрытия ниже цены открытия первой свечи
    """
    if df.empty:
        return {f"dir_{h}m": None for h in horizons}

    base_price = float(df.iloc[0]["open"])
    result = {}
    for h in horizons:
        sub = df.head(h)
        if sub.empty:
            result[f"dir_{h}m"] = None
            continue
        close = float(sub.iloc[-1]["close"])
        if side == "BUY":
            result[f"dir_{h}m"] = 1 if close > base_price else 0
        else:
            result[f"dir_{h}m"] = 1 if close < base_price else 0
    return result


def _norm_weights(weights, k):
    w = (weights[:k] if len(weights) >= k else weights + [0.0] * (k - len(weights)))
    s = sum(x for x in w if x > 0)
    if s <= 0:
        return [1.0 / k] * k
    return [max(0.0, x) / s for x in w]


def _empty_result(reason: str = "NO_DATA"):
    base = {"outcome": reason, "tps_hit": 0, "pnl": 0.0, "roi": 0.0,
            "candles": None, "close_price": None}
    base_48h = {**base, "liquidation_at_candle": None}
    result = {
        "total_usdt": POSITION_USDT, "has_dca": False,
        "tp_prices": [], "sl": None,
        "base": base, "trail": base,
        "wide_sl": base,
        "48h": base_48h,
        "direction": {f"dir_{h}m": None for h in DIRECTION_HORIZONS},
    }
    for pct in SL_AFTER_TP1_VARIANTS:
        result[f"sl_after_tp1_{int(abs(pct)*100)}pct"] = base
    return result


# ===========================
# Главный цикл
# ===========================
def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        trades = list(csv.DictReader(f))

    # Берём только сделки с закрытием (не still_open и не new_signal_opened без данных)
    valid = [t for t in trades if t["entry_price"]]
    print(f"Всего сделок: {len(trades)} | К анализу: {len(valid)}")

    results  = []
    invalids = []  # сделки с опечатками/аномалиями

    for i, trade in enumerate(valid):
        sym    = trade["symbol"]
        ts_str = trade["signal_ts"]
        side   = trade["side"]
        entry  = trade["entry_price"]

        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            ts_ms = int(dt.timestamp() * 1000)
        except Exception:
            continue

        print(f"[{i+1}/{len(valid)}] {sym} {side} @ {entry} ({ts_str})")

        # Загружаем свечи — батчами пока не закроется сделка
        try:
            tp_prices_for_fetch = [float(trade["tp1"]), float(trade["tp2"]) if trade["tp2"] else None,
                                    float(trade["tp3"]) if trade["tp3"] else None]
            tp_prices_for_fetch = [t for t in tp_prices_for_fetch if t]
            sl_for_fetch = float(trade["sl"]) if trade["sl"] else None
        except Exception:
            tp_prices_for_fetch = []
            sl_for_fetch = None

        if tp_prices_for_fetch and sl_for_fetch:
            df = get_klines_until_close(sym, ts_ms, tp_prices_for_fetch, sl_for_fetch, side)
        else:
            df = get_klines_batch(sym, ts_ms)
        sim = simulate(trade, df)
        sim_nd = simulate(trade, df, force_no_dca=True)

        # Если сделка невалидная (опечатка) — пишем в отдельный файл
        if sim["base"]["outcome"] == "INVALID_SL":
            sl_val  = float(trade["sl"]) if trade["sl"] else 0
            en_val  = float(entry) if entry else 1
            invalids.append({
                "msg_id":    trade["signal_msg_id"],
                "ts":        ts_str,
                "symbol":    sym,
                "side":      side,
                "entry":     entry,
                "tp1":       trade["tp1"],
                "tp2":       trade["tp2"],
                "tp3":       trade["tp3"],
                "sl":        trade["sl"],
                "sl_dist_pct": f"{abs(sl_val - en_val) / en_val * 100:.1f}%",
                "reason":    "SL дальше 20% от entry — возможна опечатка",
            })
            continue

        row = {
            "msg_id":       trade["signal_msg_id"],
            "ts":           ts_str,
            "symbol":       sym,
            "side":         side,
            "entry":        entry,
            "tp1":          trade["tp1"],
            "tp2":          trade["tp2"],
            "tp3":          trade["tp3"],
            "sl":           trade["sl"],
            "has_dca":      sim["has_dca"],
            "total_usdt":   sim["total_usdt"],

            # Сценарий A: Базовый
            "A_outcome":    sim["base"]["outcome"],
            "A_tps_hit":    sim["base"]["tps_hit"],
            "A_pnl":        sim["base"]["pnl"],
            "A_roi":        sim["base"]["roi"],
            "A_candles":    sim["base"]["candles"],

            # Сценарий B: Троллинг стопа
            "B_outcome":    sim["trail"]["outcome"],
            "B_tps_hit":    sim["trail"]["tps_hit"],
            "B_pnl":        sim["trail"]["pnl"],
            "B_roi":        sim["trail"]["roi"],

            # Сценарий C: SL после TP1 -5%/-10%/-15%
            "C5_outcome":   sim["sl_after_tp1_5pct"]["outcome"],
            "C5_pnl":       sim["sl_after_tp1_5pct"]["pnl"],
            "C5_roi":       sim["sl_after_tp1_5pct"]["roi"],

            "C10_outcome":  sim["sl_after_tp1_10pct"]["outcome"],
            "C10_pnl":      sim["sl_after_tp1_10pct"]["pnl"],
            "C10_roi":      sim["sl_after_tp1_10pct"]["roi"],

            "C15_outcome":  sim["sl_after_tp1_15pct"]["outcome"],
            "C15_pnl":      sim["sl_after_tp1_15pct"]["pnl"],
            "C15_roi":      sim["sl_after_tp1_15pct"]["roi"],

            # Сценарий D: Широкий стоп 50%
            "D_outcome":    sim["wide_sl"]["outcome"],
            "D_pnl":        sim["wide_sl"]["pnl"],
            "D_roi":        sim["wide_sl"]["roi"],

            # Сценарий E: 48-часовое управление
            "E_outcome":    sim["48h"]["outcome"],
            "E_pnl":        sim["48h"]["pnl"],
            "E_roi":        sim["48h"]["roi"],
            "E_liquidation_at_candle": sim["48h"].get("liquidation_at_candle"),

            # Вероятность направления
            "dir_5m":       sim["direction"].get("dir_5m"),
            "dir_15m":      sim["direction"].get("dir_15m"),
            "dir_30m":      sim["direction"].get("dir_30m"),
            "dir_60m":      sim["direction"].get("dir_60m"),

            # ND (No DCA) — пересчёт всех сценариев с позицией = POSITION_USDT
            "ND_A_outcome":    sim_nd["base"]["outcome"],
            "ND_A_tps_hit":    sim_nd["base"]["tps_hit"],
            "ND_A_pnl":        sim_nd["base"]["pnl"],
            "ND_A_roi":        sim_nd["base"]["roi"],
            "ND_A_candles":    sim_nd["base"]["candles"],

            "ND_B_outcome":    sim_nd["trail"]["outcome"],
            "ND_B_tps_hit":    sim_nd["trail"]["tps_hit"],
            "ND_B_pnl":        sim_nd["trail"]["pnl"],
            "ND_B_roi":        sim_nd["trail"]["roi"],

            "ND_C5_outcome":   sim_nd["sl_after_tp1_5pct"]["outcome"],
            "ND_C5_pnl":       sim_nd["sl_after_tp1_5pct"]["pnl"],
            "ND_C5_roi":       sim_nd["sl_after_tp1_5pct"]["roi"],

            "ND_C10_outcome":  sim_nd["sl_after_tp1_10pct"]["outcome"],
            "ND_C10_pnl":      sim_nd["sl_after_tp1_10pct"]["pnl"],
            "ND_C10_roi":      sim_nd["sl_after_tp1_10pct"]["roi"],

            "ND_C15_outcome":  sim_nd["sl_after_tp1_15pct"]["outcome"],
            "ND_C15_pnl":      sim_nd["sl_after_tp1_15pct"]["pnl"],
            "ND_C15_roi":      sim_nd["sl_after_tp1_15pct"]["roi"],

            "ND_D_outcome":    sim_nd["wide_sl"]["outcome"],
            "ND_D_pnl":        sim_nd["wide_sl"]["pnl"],
            "ND_D_roi":        sim_nd["wide_sl"]["roi"],

            "ND_E_outcome":    sim_nd["48h"]["outcome"],
            "ND_E_pnl":        sim_nd["48h"]["pnl"],
            "ND_E_roi":        sim_nd["48h"]["roi"],
            "ND_E_liquidation_at_candle": sim_nd["48h"].get("liquidation_at_candle"),
        }
        results.append(row)
        time.sleep(0.25)

    if not results:
        print("Нет результатов.")
        return

    df_res = pd.DataFrame(results)

    # Переносим только чистые TIMEOUT в invalid (закрыты вручную без сообщения в канале)
    # TP1_TIMEOUT и TP2_TIMEOUT — реальные частичные прибыли, их оставляем
    timeout_mask = df_res["A_outcome"] == "TIMEOUT"
    df_timeout = df_res[timeout_mask].copy()
    df_clean   = df_res[~timeout_mask].copy()

    for _, r in df_timeout.iterrows():
        invalids.append({
            "msg_id":    r["msg_id"],
            "ts":        r["ts"],
            "symbol":    r["symbol"],
            "side":      r["side"],
            "entry":     r["entry"],
            "tp1":       r["tp1"],
            "tp2":       r["tp2"],
            "tp3":       r["tp3"],
            "sl":        r["sl"],
            "sl_dist_pct": "—",
            "reason":    f"MANUAL_CLOSE — не закрылась за {MAX_DAYS} дней, вероятно закрыта вручную",
        })

    df_clean.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # Сохраняем невалидные сделки
    if invalids:
        df_inv = pd.DataFrame(invalids)
        df_inv.to_csv(INVALID_FILE, index=False, encoding="utf-8")
        invalid_sl      = sum(1 for x in invalids if "SL" in x["reason"] and "MANUAL" not in x["reason"])
        invalid_timeout = sum(1 for x in invalids if "MANUAL_CLOSE" in x["reason"])
        print(f"\n⚠️  Невалидных сделок: {len(invalids)} → {INVALID_FILE}")
        print(f"   Опечатки (SL >20%):  {invalid_sl}")
        print(f"   Ручное закрытие:     {invalid_timeout}")
    else:
        print("\n✅ Невалидных сделок не найдено")

    # ===========================
    # Формируем отчёт
    # ===========================
    report = []
    report.append("=" * 60)
    report.append("         ПОЛНЫЙ АНАЛИТИЧЕСКИЙ ОТЧЁТ ПО КАНАЛУ")
    report.append("=" * 60)
    report.append(f"Сделок проанализировано: {len(df_clean)}")
    report.append(f"  (исключено TIMEOUT/ручное закрытие: {len(df_timeout)})")
    report.append(f"Объём позиции: {POSITION_USDT} USDT (при усреднении +{DCA_USDT} USDT)")
    report.append(f"Плечо: x{LEVERAGE}")
    report.append("")

    def scenario_stats(col_prefix, label, data=None):
        src = data if data is not None else df_clean
        sub = src[src[f"{col_prefix}_outcome"] != "NO_DATA"]
        if sub.empty:
            return
        total   = len(sub)
        outcome_col = sub[f"{col_prefix}_outcome"]

        # Winrate: для E/ND_E — победа = НЕ E_SL* и НЕ E_liquidation
        if col_prefix in ("E", "ND_E"):
            wins   = sub[~outcome_col.str.startswith("E_SL") & (outcome_col != "E_liquidation")]
            losses = sub[outcome_col.str.startswith("E_SL") | (outcome_col == "E_liquidation")]
        else:
            wins   = sub[~outcome_col.str.startswith("SL")]
            losses = sub[outcome_col.str.startswith("SL")]

        winrate = len(wins) / total * 100 if total else 0
        total_pnl = sub[f"{col_prefix}_pnl"].sum()
        avg_pnl   = sub[f"{col_prefix}_pnl"].mean()
        avg_roi   = sub[f"{col_prefix}_roi"].mean()
        best      = sub[f"{col_prefix}_pnl"].max()
        worst     = sub[f"{col_prefix}_pnl"].min()

        report.append(f"── {label} {'─'*(50-len(label))}")
        stats = [
            ["Winrate",              f"{winrate:.1f}%"],
            ["Победных сделок",      f"{len(wins)} / {total}"],
            ["Убыточных (SL/ликв.)" if col_prefix in ("E", "ND_E") else "Убыточных (SL)",
                                     f"{len(losses)} / {total}"],
            ["Суммарный PnL",        f"{total_pnl:+.2f} USDT"],
            ["Средний PnL",          f"{avg_pnl:+.2f} USDT"],
            ["Средний ROI",          f"{avg_roi:+.2f}%"],
            ["Лучшая сделка",        f"{best:+.2f} USDT"],
            ["Худшая сделка",        f"{worst:+.2f} USDT"],
        ]
        report.append(tabulate(stats, tablefmt="simple"))

        # Разбивка по исходам
        outcomes = sub[f"{col_prefix}_outcome"].value_counts().to_dict()
        report.append("  Исходы: " + " | ".join(f"{k}: {v}" for k, v in sorted(outcomes.items())))
        report.append("")

    scenario_stats("A", "A) Базовый (без троллинга стопа)")
    scenario_stats("B", "B) Троллинг стопа (б/у после TP1, TP1 после TP2)")
    scenario_stats("C5",  "C) SL после TP1: -5% от входа")
    scenario_stats("C10", "C) SL после TP1: -10% от входа")
    scenario_stats("C15", "C) SL после TP1: -15% от входа")
    scenario_stats("D",  "D) Широкий стоп 50% от объёма позиции")
    scenario_stats("E",  "E) 48ч управление (троллинг + динамический SL)")

    # ── Статистика без усреднения (ND_* колонки) ──────────────
    report.append("═" * 60)
    report.append(f"     СТАТИСТИКА БЕЗ УСРЕДНЕНИЯ (все {len(df_clean)} сделки, позиция всегда {POSITION_USDT:.0f} USDT)")
    report.append("═" * 60)
    report.append("")

    scenario_stats("ND_A",   "A) Базовый (без троллинга стопа)")
    scenario_stats("ND_B",   "B) Троллинг стопа (б/у после TP1, TP1 после TP2)")
    scenario_stats("ND_C5",  "C) SL после TP1: -5% от входа")
    scenario_stats("ND_C10", "C) SL после TP1: -10% от входа")
    scenario_stats("ND_C15", "C) SL после TP1: -15% от входа")
    scenario_stats("ND_D",   "D) Широкий стоп 50% от объёма позиции")
    scenario_stats("ND_E",   "E) 48ч управление (троллинг + динамический SL)")

    # Зависимости
    report.append("── Зависимость: размер стопа vs результат ─────────────")
    df_clean["sl_dist_pct"] = df_clean.apply(lambda r: abs(
        (float(r["sl"]) - float(r["entry"])) / float(r["entry"]) * 100
    ) if r["sl"] and r["entry"] else None, axis=1)

    bins = [0, 1, 2, 3, 5, 10, 100]
    labels = ["<1%", "1-2%", "2-3%", "3-5%", "5-10%", ">10%"]
    df_clean["sl_bucket"] = pd.cut(df_clean["sl_dist_pct"], bins=bins, labels=labels)
    sl_analysis = df_clean.groupby("sl_bucket", observed=True).agg(
        count=("A_pnl", "count"),
        winrate=("A_outcome", lambda x: (x != "SL").mean() * 100),
        avg_pnl=("A_pnl", "mean"),
    ).round(2)
    report.append(tabulate(sl_analysis, headers="keys", tablefmt="simple"))
    report.append("")

    # Зависимость: дистанция до TP1 vs вероятность достижения
    report.append("── Зависимость: дистанция до TP1 vs вероятность достижения ──")
    df_clean["tp1_dist_pct"] = df_clean.apply(lambda r: abs(
        (float(r["tp1"]) - float(r["entry"])) / float(r["entry"]) * 100
    ) if r["tp1"] and r["entry"] else None, axis=1)
    df_clean["tp1_hit"] = df_clean["A_tps_hit"] >= 1

    tp_bins   = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 100]
    tp_labels = ["<0.5%", "0.5-1%", "1-1.5%", "1.5-2%", "2-3%", ">3%"]
    df_clean["tp1_bucket"] = pd.cut(df_clean["tp1_dist_pct"], bins=tp_bins, labels=tp_labels)
    tp_analysis = df_clean.groupby("tp1_bucket", observed=True).agg(
        count=("tp1_hit", "count"),
        tp1_hit_rate=("tp1_hit", lambda x: x.mean() * 100),
        avg_pnl=("A_pnl", "mean"),
    ).round(2)
    report.append(tabulate(tp_analysis, headers="keys", tablefmt="simple"))
    report.append("")

    # Вероятность направления
    report.append("── Вероятность движения цены в нужную сторону ─────────")
    dir_stats = []
    for h in DIRECTION_HORIZONS:
        col = f"dir_{h}m"
        sub = df_clean[df_clean[col].notna()]
        prob = sub[col].mean() * 100 if not sub.empty else 0
        dir_stats.append([f"{h} минут", f"{prob:.1f}%", f"{100-prob:.1f}%"])
    report.append(tabulate(dir_stats,
        headers=["Горизонт", "В нужную сторону", "Против"],
        tablefmt="simple"))
    report.append("")

    # Влияние DCA на результат
    report.append("── Влияние усреднения (DCA) на результат ───────────────")
    dca_yes = df_clean[df_clean["has_dca"] == True]
    dca_no  = df_clean[df_clean["has_dca"] == False]
    dca_table = [
        ["С усреднением",    len(dca_yes),
         f"{(dca_yes['A_outcome'] != 'SL').mean()*100:.1f}%" if len(dca_yes) else "-",
         f"{dca_yes['A_pnl'].mean():+.2f}" if len(dca_yes) else "-"],
        ["Без усреднения",   len(dca_no),
         f"{(dca_no['A_outcome'] != 'SL').mean()*100:.1f}%" if len(dca_no) else "-",
         f"{dca_no['A_pnl'].mean():+.2f}" if len(dca_no) else "-"],
    ]
    report.append(tabulate(dca_table,
        headers=["Тип", "Кол-во", "Winrate", "Средний PnL"],
        tablefmt="simple"))
    report.append("")

    report.append("=" * 60)
    report.append(f"Детальные данные → {OUTPUT_CSV}")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print("\n" + report_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n✅ Отчёт сохранён → {OUTPUT_TXT}")

    generate_equity_chart(df_clean)


# ── Equity curve ────────────────────────────────────────────────────────────

OUTPUT_HTML = "backtest_equity.html"
INITIAL_BALANCE = 4000.0

SCENARIOS = {
    "A":   {"label": "A · базовый",    "color": "#3b82f6", "width": 1.5},
    "B":   {"label": "B · троллинг",   "color": "#f59e0b", "width": 1.5},
    "C5":  {"label": "C5 · SL-5%",     "color": "#06b6d4", "width": 1.5},
    "C10": {"label": "C10 · SL-10%",   "color": "#f97316", "width": 1.5},
    "C15": {"label": "C15 · SL-15%",   "color": "#ef4444", "width": 1.5},
    "D":   {"label": "D · фикс.стоп",  "color": "#10b981", "width": 2.5},
    "E":   {"label": "E · 48ч",        "color": "#a855f7", "width": 2.0},
}

def _pnl_col(scen):
    return f"{scen}_pnl"

def _out_col(scen):
    return f"{scen}_outcome"


def generate_equity_chart(df: "pd.DataFrame") -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("⚠️  plotly не установлен — пропускаю equity chart (pip install plotly)")
        return

    import pandas as pd

    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)

    # --- накопленный баланс и просадка по каждому сценарию ---
    eq   = {s: [INITIAL_BALANCE] for s in SCENARIOS}
    peak = {s: INITIAL_BALANCE   for s in SCENARIOS}
    dd   = {s: [0.0]             for s in SCENARIOS}

    for _, row in df.iterrows():
        for s in SCENARIOS:
            col = _pnl_col(s)
            bal = eq[s][-1] + (row[col] if col in df.columns else 0)
            eq[s].append(bal)
            if bal > peak[s]:
                peak[s] = bal
            dd[s].append(-(peak[s] - bal) / peak[s] * 100)

    x_trades = list(range(n + 1))

    # hover-текст для каждой точки
    hover = ["Старт"] + [
        f"#{i+1} {row['ts'][:10]}<br>{row['symbol']} {row['side']}"
        for i, (_, row) in enumerate(df.iterrows())
    ]

    # --- итоговые метрики для карточек ---
    stats = {}
    for s in SCENARIOS:
        final  = eq[s][-1]
        profit = final - INITIAL_BALANCE
        pct    = profit / INITIAL_BALANCE * 100
        max_dd = min(dd[s])
        stats[s] = dict(final=final, profit=profit, pct=pct, max_dd=max_dd)

    # --- помесячный PnL ---
    df["_month"] = pd.to_datetime(df["ts"]).dt.to_period("M").astype(str)
    months = sorted(df["_month"].unique())
    monthly = {}
    for s in SCENARIOS:
        col = _pnl_col(s)
        if col in df.columns:
            monthly[s] = df.groupby("_month")[col].sum().reindex(months, fill_value=0)

    # ── Figure: 2 rows (equity + drawdown) ──────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
    )

    for s, cfg in SCENARIOS.items():
        visible = True  # все видимы по умолчанию
        # equity
        fig.add_trace(go.Scatter(
            x=x_trades, y=eq[s],
            name=cfg["label"],
            line=dict(color=cfg["color"], width=cfg["width"]),
            hovertemplate="%{text}<br><b>%{y:$,.0f}</b><extra>" + cfg["label"] + "</extra>",
            text=hover,
            legendgroup=s,
            showlegend=True,
            visible=visible,
        ), row=1, col=1)
        # drawdown
        fig.add_trace(go.Scatter(
            x=x_trades, y=dd[s],
            name=cfg["label"],
            line=dict(color=cfg["color"], width=1, dash="dot"),
            hovertemplate="%{text}<br><b>%{y:.1f}%</b><extra>DD · " + cfg["label"] + "</extra>",
            text=hover,
            legendgroup=s,
            showlegend=False,
            visible=visible,
        ), row=2, col=1)

    # опорная линия стартового баланса
    fig.add_hline(y=INITIAL_BALANCE, line_dash="dash", line_color="#334155",
                  line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#334155",
                  line_width=1, row=2, col=1)

    # --- аннотации итогов (правый край) ---
    for s, cfg in SCENARIOS.items():
        st = stats[s]
        sign = "+" if st["profit"] >= 0 else ""
        fig.add_annotation(
            x=n, y=eq[s][-1],
            text=f"  {sign}{st['pct']:.0f}%",
            font=dict(color=cfg["color"], size=10),
            showarrow=False, xanchor="left",
            row=1, col=1,
        )

    # ── Layout ──────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Equity curve · Старт ${INITIAL_BALANCE:,.0f} · {n} сделок",
            font=dict(size=16, color="#f1f5f9"),
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=100, t=80, b=40),
        height=650,
    )
    fig.update_xaxes(
        gridcolor="#1e3a5f", zeroline=False,
        title_text="Сделка №", row=2, col=1,
    )
    fig.update_yaxes(gridcolor="#1e3a5f", zeroline=False,
                     tickprefix="$", row=1, col=1)
    fig.update_yaxes(gridcolor="#1e3a5f", zeroline=False,
                     ticksuffix="%", title_text="Просадка %", row=2, col=1)

    # ── Карточки итогов (таблица под графиком) ───────────────────────────────
    card_rows = [
        ["Сценарий", "Итог $", "Прибыль $", "%", "Макс.просадка %"],
    ]
    for s, cfg in SCENARIOS.items():
        st = stats[s]
        sign = "+" if st["profit"] >= 0 else ""
        card_rows.append([
            cfg["label"],
            f"${st['final']:,.0f}",
            f"{sign}${st['profit']:,.0f}",
            f"{sign}{st['pct']:.1f}%",
            f"{st['max_dd']:.1f}%",
        ])

    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=card_rows[0],
            fill_color="#1e293b",
            font=dict(color="#94a3b8", size=11),
            align="left",
            line_color="#334155",
        ),
        cells=dict(
            values=list(zip(*card_rows[1:])),
            fill_color=[
                ["#0f172a"] * len(SCENARIOS),
                [
                    "#166534" if stats[s]["profit"] >= 0 else "#7f1d1d"
                    for s in SCENARIOS
                ],
                [
                    "#166534" if stats[s]["profit"] >= 0 else "#7f1d1d"
                    for s in SCENARIOS
                ],
                [
                    "#166534" if stats[s]["profit"] >= 0 else "#7f1d1d"
                    for s in SCENARIOS
                ],
                ["#1e293b"] * len(SCENARIOS),
            ],
            font=dict(color="#e2e8f0", size=11),
            align="left",
            line_color="#334155",
        ),
    )])
    fig_table.update_layout(
        paper_bgcolor="#0f172a",
        margin=dict(l=0, r=0, t=0, b=0),
        height=260,
    )

    # ── Помесячный PnL (bar chart) ───────────────────────────────────────────
    fig_monthly = go.Figure()
    for s, cfg in SCENARIOS.items():
        if s not in monthly:
            continue
        vals = monthly[s].values
        colors = ["#166534" if v >= 0 else "#7f1d1d" for v in vals]
        fig_monthly.add_trace(go.Bar(
            x=months, y=vals,
            name=cfg["label"],
            marker_color=cfg["color"],
            opacity=0.8,
            legendgroup=s,
            showlegend=True,
            hovertemplate="%{x}<br><b>%{y:+,.0f} USDT</b><extra>" + cfg["label"] + "</extra>",
        ))
    fig_monthly.update_layout(
        title=dict(text="Помесячный PnL по сценариям",
                   font=dict(size=13, color="#f1f5f9")),
        barmode="group",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=40),
        height=320,
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", ticksuffix=" $"),
    )
    fig_monthly.add_hline(y=0, line_color="#334155", line_width=1)

    # ── Собираем HTML ────────────────────────────────────────────────────────
    html_equity  = fig.to_html(full_html=False, include_plotlyjs="cdn")
    html_table   = fig_table.to_html(full_html=False, include_plotlyjs=False)
    html_monthly = fig_monthly.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Backtest Equity</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; margin:0; padding:20px; }}
  h1   {{ font-size:18px; color:#f1f5f9; margin-bottom:4px; }}
  p    {{ color:#475569; font-size:12px; margin:0 0 16px; }}
  .section {{ margin-bottom:24px; }}
</style>
</head>
<body>
<h1>Backtest · Симуляция баланса</h1>
<p>Позиция {POSITION_USDT:.0f} USDT фиксированная · x20 · старт ${INITIAL_BALANCE:,.0f}</p>
<div class="section">{html_equity}</div>
<div class="section">{html_table}</div>
<div class="section">{html_monthly}</div>
</body>
</html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Equity chart сохранён → {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
