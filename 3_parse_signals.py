"""
Скрипт 3: Парсер сигналов из signals.csv
Читает signals.csv → выделяет сделки → склеивает с усреднениями
Результат: trades.csv

Изменения vs оригинал:
  - Убрана логика закрытия (new_signal_opened, full_close, close, still_open)
  - Каждый сигнал = отдельная строка, независимо от символа
  - DCA/усреднения привязываются к ближайшему предшествующему сигналу по символу
  - Закрытие сделок полностью отдано бэктесту (4_analytics.py по свечам Binance)

Установка:
    pip install pandas

Запуск:
    python3 3_parse_signals.py
"""

import csv
import re
import json
from typing import Optional, List, Dict, Any

# ===========================
# Настройки
# ===========================
INPUT_FILE  = "signals.csv"
OUTPUT_FILE = "trades.csv"
SKIPPED_LOG = "parse_skipped.log"
# ===========================

NUM_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?")

def float_from_str(s: str) -> float:
    s = s.strip().replace(",", ".").replace("\xa0", "").replace("\u202f", "")
    return float(s)

def all_floats(line: str) -> List[float]:
    return [float_from_str(x) for x in NUM_RE.findall(line)]

def first_float(line: str) -> Optional[float]:
    m = NUM_RE.search(line)
    return float_from_str(m.group(0)) if m else None


# ===========================
# Классификатор сообщений
# ===========================
SIGNAL_SIDE_RE = re.compile(
    r"^([A-Za-z0-9]+)\s+(long|short|buy|sell)",
    re.IGNORECASE
)

DCA_UPDATE_KW = ["усредн", "твх", "докупил", "добрал", "актуальные данные"]
TP_HIT_KW     = ["первый тейк", "первый есть", "второй тейк", "третий тейк",
                  "два тейка", "тейк профит", "тейк с двойной", "тейк 🔥", "тейк ✅"]
BU_KW         = ["стоп в б/у", "стоп в безубыток", "б/у"]


def classify(text: str) -> str:
    t   = text.strip()
    low = t.lower()

    first_line = t.splitlines()[0].strip() if t else ""
    if SIGNAL_SIDE_RE.match(first_line):
        if "вход" in low or "тейк" in low:
            return "signal"

    if any(k in low for k in TP_HIT_KW):
        return "tp_hit"
    if any(k in low for k in BU_KW):
        return "move_sl_bu"
    if any(k in low for k in DCA_UPDATE_KW):
        return "dca_update"

    return "other"


# ===========================
# Парсер нового сигнала
# ===========================
ENTRY_PREFIXES = ("вход", "вхож", "entry", "вхождение", "точка входа")
TP_PREFIXES    = ("тейки", "тейк", "tp", "take", "цели", "цель")
SL_PREFIXES    = ("стоп", "sl", "stop")
DCA_PREFIXES   = ("усреднение", "dca")


def clean_text(text: str) -> str:
    text = re.sub(r"\*\*|__|\\*|_", "", text)
    text = re.sub(r"\u200b|\ufeff|\u202f|\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-лосс", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_signal_text(text: str) -> Optional[Dict[str, Any]]:
    text  = clean_text(text)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    if len(lines) == 1:
        line = lines[0]
        line = re.sub(
            r'\s+(вход|тейки|тейк|цели|цель|стоп|усреднение)',
            r'\n\1', line, flags=re.IGNORECASE
        )
        lines = [l.strip() for l in line.splitlines() if l.strip()]

    if not lines:
        return None

    m = SIGNAL_SIDE_RE.match(lines[0])
    if not m:
        return None
    symbol = m.group(1).upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    side_raw = m.group(2).lower()
    side = "BUY" if side_raw in ("long", "buy") else "SELL"

    entry_price = tp_prices = sl_price = dca_price = None

    for ln in lines[1:]:
        low = ln.lower()
        if entry_price is None and low.startswith(ENTRY_PREFIXES):
            entry_price = first_float(ln)
        elif tp_prices is None and low.startswith(TP_PREFIXES):
            tp_prices = all_floats(ln)
        elif sl_price is None and low.startswith(SL_PREFIXES):
            sl_price = first_float(ln)
        elif dca_price is None and low.startswith(DCA_PREFIXES):
            dca_price = first_float(ln)

    if not entry_price or not tp_prices or not sl_price:
        return None

    tp_prices = _fix_prices(entry_price, tp_prices, sl_price, side)
    if tp_prices is None:
        return None

    return {
        "symbol":      symbol,
        "side":        side,
        "entry_price": entry_price,
        "tp1":         tp_prices[0] if len(tp_prices) > 0 else None,
        "tp2":         tp_prices[1] if len(tp_prices) > 1 else None,
        "tp3":         tp_prices[2] if len(tp_prices) > 2 else None,
        "sl":          sl_price,
        "dca_price":   dca_price,
    }


def _fix_prices(entry: float, tps: List[float], sl: float, side: str) -> Optional[List[float]]:
    fixed = []
    for tp in tps:
        corrected = _try_fix_price(entry, tp, side, expect_profit=True)
        if corrected is None:
            return None
        fixed.append(corrected)
    return fixed


def _try_fix_price(entry: float, price: float, side: str, expect_profit: bool) -> Optional[float]:
    if _price_ok(entry, price, side, expect_profit):
        return price
    for factor in [0.1, 0.01, 10.0, 100.0, 0.001, 1000.0]:
        candidate = price * factor
        if _price_ok(entry, candidate, side, expect_profit):
            return candidate
    return None


def _price_ok(entry: float, price: float, side: str, expect_profit: bool) -> bool:
    EPS = 1e-9
    if abs(price - entry) / entry > 0.5:
        return False
    if side == "BUY":
        return price > entry + EPS if expect_profit else price < entry - EPS
    else:
        return price < entry - EPS if expect_profit else price > entry + EPS


# ===========================
# Парсер DCA-обновления
# ===========================
def parse_dca_update(text: str) -> Optional[Dict[str, Any]]:
    lines    = [l.strip() for l in text.strip().splitlines() if l.strip()]
    low_full = text.lower()

    new_entry = new_tps = new_sl = None

    for ln in lines:
        low = ln.lower()
        if low.startswith(("твх", "т.вх", "точка входа", "вход")):
            new_entry = first_float(ln)
        elif low.startswith(("тейки", "тейк", "цели")):
            nums = all_floats(ln)
            if nums:
                new_tps = nums
        elif low.startswith(("стоп", "sl")):
            new_sl = first_float(ln)

    if new_sl is None:
        m = re.search(
            r"стоп\s+(?:сюда\s+поставил|подвинул|передвинул)[^\d]*([0-9]+(?:[.,][0-9]+)?)",
            low_full
        )
        if m:
            new_sl = float_from_str(m.group(1))

    if new_entry is None and new_tps is None and new_sl is None:
        return None

    return {"new_entry": new_entry, "new_tps": new_tps, "new_sl": new_sl}


# ===========================
# Утилиты поиска
# ===========================
def _extract_symbol(text: str) -> Optional[str]:
    if not text:
        return None
    first_line = text.strip().splitlines()[0].strip()
    m = SIGNAL_SIDE_RE.match(first_line)
    if m:
        sym = m.group(1).upper()
        return sym if sym.endswith("USDT") else sym + "USDT"
    return None


def _extract_symbol_from_dca(text: str) -> Optional[str]:
    m = re.search(r"(?:докупил|добрал|усредн\w+)\s+([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        sym = m.group(1).upper()
        return sym if sym.endswith("USDT") else sym + "USDT"
    return None


def _find_latest_signal(sym: str, before_msg_id: str,
                        signals_by_sym: Dict[str, List[Dict]]) -> Optional[Dict]:
    """
    Находит последний сигнал по символу с msg_id < before_msg_id.
    Сигналы в списке отсортированы по возрастанию msg_id.
    """
    candidates = signals_by_sym.get(sym, [])
    result = None
    for s in candidates:
        if int(s["signal_msg_id"]) < int(before_msg_id):
            result = s
        else:
            break
    return result


# ===========================
# Основной цикл
# ===========================
def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Загружено {len(rows)} сообщений")

    msg_index: Dict[str, Dict] = {r["message_id"]: r for r in rows}

    # Проход 1: собираем все сигналы
    trades  = []
    skipped = []

    for row in rows:
        msg_id = row["message_id"]
        ts     = row["timestamp"]
        ts_unix = row["timestamp_unix"]
        text   = row["text"].strip()

        if not text:
            continue

        kind = classify(text)

        if kind != "signal":
            continue

        parsed = parse_signal_text(text)
        if not parsed:
            skipped.append({
                "msg_id": msg_id, "ts": ts,
                "reason": "parse_failed", "text": text[:120]
            })
            continue

        trade = {
            "signal_msg_id":  msg_id,
            "signal_ts":      ts,
            "signal_ts_unix": ts_unix,
            "symbol":         parsed["symbol"],
            "side":           parsed["side"],
            "entry_price":    parsed["entry_price"],
            "tp1":            parsed["tp1"],
            "tp2":            parsed["tp2"],
            "tp3":            parsed["tp3"],
            "sl":             parsed["sl"],
            "dca_price":      parsed["dca_price"],
            "dca_updates":    [],
            "tp_hits":        [],
            "moved_sl_bu":    False,
        }
        trades.append(trade)

    print(f"Сигналов распознано: {len(trades)}")

    # Индекс сигналов по символу (отсортированы по msg_id)
    signals_by_sym: Dict[str, List[Dict]] = {}
    for t in sorted(trades, key=lambda x: int(x["signal_msg_id"])):
        signals_by_sym.setdefault(t["symbol"], []).append(t)

    # Проход 2: привязываем DCA / TP hit / BU к сигналам
    for row in rows:
        msg_id   = row["message_id"]
        ts       = row["timestamp"]
        reply_to = row["reply_to_id"]
        text     = row["text"].strip()

        if not text:
            continue

        kind = classify(text)

        if kind not in ("dca_update", "tp_hit", "move_sl_bu"):
            continue

        # Ищем целевой сигнал: сначала через reply, потом по символу
        target = None

        # 1) Через reply_to → символ родителя
        if reply_to and reply_to in msg_index:
            parent = msg_index[reply_to]
            sym = _extract_symbol(parent["text"])
            if sym:
                target = _find_latest_signal(sym, msg_id, signals_by_sym)

            # Поднимаемся на уровень выше если не нашли
            if target is None:
                gp_id = parent.get("reply_to_id")
                if gp_id and gp_id in msg_index:
                    gp  = msg_index[gp_id]
                    sym = _extract_symbol(gp["text"])
                    if sym:
                        target = _find_latest_signal(sym, msg_id, signals_by_sym)

        # 2) Символ упомянут в тексте DCA
        if target is None and kind == "dca_update":
            sym = _extract_symbol_from_dca(text)
            if sym:
                target = _find_latest_signal(sym, msg_id, signals_by_sym)

        if target is None:
            continue

        # Применяем обновление
        if kind == "dca_update":
            upd = parse_dca_update(text)
            if not upd:
                continue

            upd["ts"] = ts
            target["dca_updates"].append(upd)

            new_entry = upd["new_entry"] or target["entry_price"]
            side      = target["side"]

            if upd["new_entry"]:
                target["entry_price"] = upd["new_entry"]

            if upd["new_tps"]:
                fixed = _fix_prices(float(new_entry), upd["new_tps"],
                                    float(target["sl"] or 0), side)
                if fixed:
                    target["tp1"] = fixed[0] if len(fixed) > 0 else target["tp1"]
                    target["tp2"] = fixed[1] if len(fixed) > 1 else target["tp2"]
                    target["tp3"] = fixed[2] if len(fixed) > 2 else target["tp3"]

            if upd["new_sl"]:
                sl_ok = _price_ok(float(new_entry), float(upd["new_sl"]), side, expect_profit=False)
                if sl_ok:
                    target["sl"] = upd["new_sl"]

        elif kind == "tp_hit":
            low = text.lower()
            if "первый" in low or "tp1" in low:
                target["tp_hits"].append("tp1")
            elif "второй" in low or "tp2" in low:
                target["tp_hits"].append("tp2")
            elif "третий" in low or "tp3" in low:
                target["tp_hits"].append("tp3")
            elif "два тейка" in low:
                if "tp1" not in target["tp_hits"]: target["tp_hits"].append("tp1")
                if "tp2" not in target["tp_hits"]: target["tp_hits"].append("tp2")
            else:
                target["tp_hits"].append("tp_unknown")

        elif kind == "move_sl_bu":
            target["moved_sl_bu"] = True

    # ===========================
    # Сохраняем результат
    # ===========================
    fieldnames = [
        "signal_msg_id", "signal_ts", "symbol", "side",
        "entry_price", "tp1", "tp2", "tp3", "sl", "dca_price",
        "dca_updates_count", "dca_updates_detail",
        "tp_hits", "moved_sl_bu",
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "signal_msg_id":      t["signal_msg_id"],
                "signal_ts":          t["signal_ts"],
                "symbol":             t["symbol"],
                "side":               t["side"],
                "entry_price":        t["entry_price"],
                "tp1":                t["tp1"] or "",
                "tp2":                t["tp2"] or "",
                "tp3":                t["tp3"] or "",
                "sl":                 t["sl"] or "",
                "dca_price":          t["dca_price"] or "",
                "dca_updates_count":  len(t["dca_updates"]),
                "dca_updates_detail": json.dumps(t["dca_updates"], ensure_ascii=False),
                "tp_hits":            ", ".join(t["tp_hits"]),
                "moved_sl_bu":        t["moved_sl_bu"],
            })

    # Лог пропущенных
    with open(SKIPPED_LOG, "w", encoding="utf-8") as f:
        f.write("msg_id,ts,reason,text\n")
        for s in skipped:
            f.write(f"{s['msg_id']},{s['ts']},{s['reason']},\"{s['text'].replace(chr(10),' ')}\"\n")

    # Статистика
    dca_count = sum(1 for t in trades if t["dca_updates"])
    bu_count  = sum(1 for t in trades if t["moved_sl_bu"])
    print(f"\n{'='*50}")
    print(f"  Итого сигналов:      {len(trades)}")
    print(f"  Пропущено:           {len(skipped)}")
    print(f"  С усреднениями:      {dca_count}")
    print(f"  Со стопом в б/у:     {bu_count}")
    print(f"{'='*50}")
    print(f"\n  Результат → {OUTPUT_FILE}")
    print(f"  Пропущенные → {SKIPPED_LOG}")


if __name__ == "__main__":
    main()
