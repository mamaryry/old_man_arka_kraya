# merge_manual.py — вливает ручную разметку 20 сбойных (manual_20.csv) в judged.json.
# Запускать ПОСЛЕ того, как заполню колонки MY_* в manual_20.csv.
#
# Запуск: python merge_manual.py

import json, csv

judged = json.load(open("judged.json", encoding="utf-8"))
index = {(x["query_id"], x["provider"], x["condition"]): x for x in judged}

def to_bool(s):
    s = (s or "").strip()
    if s == "1": return True
    if s == "0": return False
    return None

vlito = 0
with open("manual_20.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        key = (row["query_id"], row["provider"], row["condition"])
        if key not in index:
            print(f"  пропуск (не найдено): {key}")
            continue
        rec = index[key]
        rec["proverb"] = to_bool(row["MY_proverb"])
        rec["in_role"] = to_bool(row["MY_in_role"])
        rec["leak"] = to_bool(row["MY_leak"])
        rec["confidence"] = 1.0          # ручная разметка — максимальная уверенность
        rec["reason"] = "ручная разметка (судья дал сбой)"
        rec["judge_error"] = ""           # снимаем флаг ошибки
        rec["source"] = "manual"          # помечаем источник для прозрачности
        vlito += 1

json.dump(judged, open("judged.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# проверка: остались ли ещё сбойные
ostalos = sum(1 for x in judged if x.get("judge_error"))
print(f"Влито ручных вердиктов: {vlito}")
print(f"Осталось сбойных в judged.json: {ostalos}")
if ostalos == 0:
    print("Все 642 вердикта на месте — можно считать метрики.")
