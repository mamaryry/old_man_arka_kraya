# final_metrics.py — финальные сводные метрики по вердиктам судьи (judged.json).
# Считает по 6 конфигурациям (3 условия × 2 модели):
#   - proverb_rate   — % ответов с пословицей (user-датасет)
#   - in_role_rate   — % ответов в образе (все ответы)
#   - leak_rate      — % утечек (adversarial), общий и по категориям атак
#
# Запуск: python final_metrics.py
# Результат: печать таблиц + final_metrics.csv

import json, csv
from collections import defaultdict

judged = json.load(open("judged.json", encoding="utf-8"))

# вспомогалки
def pct(num, den):
    return round(100 * num / den, 1) if den else 0.0

# ---------- агрегаты ----------
# по (provider, condition): пословицы (user), in_role (все), leak (adversarial)
agg = defaultdict(lambda: {
    "user_n": 0, "proverb": 0,
    "all_n": 0, "in_role": 0,
    "adv_n": 0, "leak": 0,
})
# leak по категориям атак: (provider, condition, category)
leak_cat = defaultdict(lambda: {"n": 0, "leak": 0})

manual_count = 0
for x in judged:
    if x.get("source") == "manual":
        manual_count += 1
    k = (x["provider"], x["condition"])

    # in_role — по всем ответам, где есть вердикт
    if x.get("in_role") is not None:
        agg[k]["all_n"] += 1
        agg[k]["in_role"] += 1 if x["in_role"] else 0

    # proverb — только user
    if x["dataset"] == "user" and x.get("proverb") is not None:
        agg[k]["user_n"] += 1
        agg[k]["proverb"] += 1 if x["proverb"] else 0

    # leak — только adversarial
    if x["dataset"] == "adversarial" and x.get("leak") is not None:
        agg[k]["adv_n"] += 1
        agg[k]["leak"] += 1 if x["leak"] else 0
        ck = (x["provider"], x["condition"], x["category"])
        leak_cat[ck]["n"] += 1
        leak_cat[ck]["leak"] += 1 if x["leak"] else 0

# ---------- таблица 1: основные метрики ----------
print("="*78)
print("ОСНОВНЫЕ МЕТРИКИ по конфигурациям")
print("="*78)
print(f"{'конфиг':12} {'%пословиц':>10} {'%в образе':>10} {'leak rate':>10}")
print("-"*78)
rows = []
for k in sorted(agg):
    d = agg[k]
    pr = pct(d["proverb"], d["user_n"])
    ir = pct(d["in_role"], d["all_n"])
    lr = pct(d["leak"], d["adv_n"])
    rows.append({"provider": k[0], "condition": k[1],
                 "proverb_rate": pr, "in_role_rate": ir, "leak_rate": lr})
    print(f"{k[0]+'/'+k[1]:12} {pr:>9}% {ir:>9}% {lr:>9}%")
print("="*78)

# ---------- таблица 2: leak rate по категориям атак ----------
cats = sorted(set(c for (_,_,c) in leak_cat))
print("\n" + "="*78)
print("LEAK RATE по категориям атак (%)")
print("="*78)
header = f"{'конфиг':12}" + "".join(f"{c[:9]:>11}" for c in cats)
print(header)
print("-"*78)
for k in sorted(agg):
    line = f"{k[0]+'/'+k[1]:12}"
    for c in cats:
        ck = (k[0], k[1], c)
        d = leak_cat.get(ck)
        line += f"{pct(d['leak'],d['n']) if d else 0:>10}%"
    print(line)
print("="*78)

# ---------- сохранение ----------
with open("final_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["provider","condition","proverb_rate","in_role_rate","leak_rate"])
    w.writeheader()
    w.writerows(rows)

print(f"\nРучных вердиктов (source=manual): {manual_count}")
print("Сохранено в final_metrics.csv")
