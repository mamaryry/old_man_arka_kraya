# stat_significance.py — статистическая значимость различий между условиями.
# Точный тест Фишера на осмысленных контрастах (под гипотезы) + поправка Холма.
#
# Запуск: python stat_significance.py
# Требует: scipy (pip install scipy)
# Результат: печать таблицы + stat_results.csv

import json, csv, os
from scipy.stats import fisher_exact

# ---------- загрузка вердиктов ----------
allj = []
for fn in ["judged.json", "judged_D1.json", "judged_D2.json"]:
    if os.path.exists(fn):
        allj.extend(json.load(open(fn, encoding="utf-8")))

def key(x):
    c = x["condition"]; p = x.get("provider","")
    return f"{p}/{c}" if c in ("A","B","C") else c

# ---------- собираем счётчики по конфигурации ----------
# для leak: (утечки, всего) на adversarial; для proverb: (есть, всего) на user
from collections import defaultdict
leak = defaultdict(lambda: [0,0])     # [число утечек, всего]
prov = defaultdict(lambda: [0,0])     # [с пословицей, всего]
role = defaultdict(lambda: [0,0])     # [в образе, всего]

for x in allj:
    k = key(x)
    if x["dataset"]=="adversarial" and x.get("leak") is not None:
        leak[k][1]+=1; leak[k][0]+=1 if x["leak"] else 0
    if x["dataset"]=="user" and x.get("proverb") is not None:
        prov[k][1]+=1; prov[k][0]+=1 if x["proverb"] else 0
    if x.get("in_role") is not None:
        role[k][1]+=1; role[k][0]+=1 if x["in_role"] else 0

def fisher(counts, a, b):
    """Тест Фишера: доля 'успехов' в a против b.
    counts[k] = [успехи, всего]. Таблица 2x2: [[усп_a, неусп_a],[усп_b, неусп_b]]."""
    sa, na = counts[a]; sb, nb = counts[b]
    table = [[sa, na-sa],[sb, nb-sb]]
    odds, p = fisher_exact(table)
    return p, (sa,na), (sb,nb)

# ---------- контрасты под гипотезы ----------
# (метрика, конфиг1, конфиг2, гипотеза/смысл)
contrasts = [
    ("leak","ollama/B","ollama/A","H3: промпт↓утечки (Qwen3)"),
    ("leak","yandex/B","yandex/A","H3: промпт↓утечки (Яндекс)"),
    ("leak","D1","ollama/B","H2: дообучение ломает границы"),
    ("leak","D2","D1","накачка усугубляет границы"),
    ("leak","ollama/C","ollama/B","RAG vs промпт (границы)"),
    ("leak","ollama/B","yandex/B","RQ3: открытая vs проприетарная"),
    ("proverb","yandex/B","yandex/A","промпт↑пословицы (Яндекс)"),
    ("proverb","D2","D1","накачка↑пословицы"),
]

metric_map = {"leak":leak, "proverb":prov, "role":role}

results = []
for metric, a, b, hyp in contrasts:
    cm = metric_map[metric]
    if a not in cm or b not in cm:
        continue
    p, (sa,na), (sb,nb) = fisher(cm, a, b)
    results.append({
        "metric": metric, "a": a, "b": b, "hypothesis": hyp,
        "rate_a": f"{100*sa/na:.1f}%", "rate_b": f"{100*sb/nb:.1f}%",
        "p_raw": p,
    })

# ---------- поправка Холма ----------
# сортируем по p, корректируем
results_sorted = sorted(results, key=lambda r: r["p_raw"])
m = len(results_sorted)
prev = 0
for idx, r in enumerate(results_sorted):
    holm = r["p_raw"] * (m - idx)        # Холм: умножаем на (m - ранг)
    holm = min(1.0, max(holm, prev))      # монотонность
    prev = holm
    r["p_holm"] = holm
    r["significant"] = "ДА" if holm < 0.05 else "нет"

# ---------- вывод ----------
print("="*100)
print("СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ (точный тест Фишера + поправка Холма)")
print("="*100)
print(f"{'метрика':8} {'сравнение':22} {'доли':16} {'p_raw':>9} {'p_holm':>9} {'знач.':>6}  гипотеза")
print("-"*100)
for r in results_sorted:
    print(f"{r['metric']:8} {r['a']+' vs '+r['b']:22} "
          f"{r['rate_a']+' / '+r['rate_b']:16} "
          f"{r['p_raw']:>9.4f} {r['p_holm']:>9.4f} {r['significant']:>6}  {r['hypothesis']}")
print("="*100)
print("Значимо при p_holm < 0.05. 'доли' = доля у конфига A / у конфига B.")

# сохранить
with open("stat_results.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["metric","a","b","hypothesis","rate_a","rate_b","p_raw","p_holm","significant"])
    w.writeheader()
    for r in results_sorted: w.writerow(r)
print("\nСохранено в stat_results.csv")
