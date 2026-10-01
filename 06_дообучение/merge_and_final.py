# merge_and_final.py — объединяет judged.json (A/B/C) + judged_D1.json (+D2) и считает
# финальные метрики по ВСЕМ условиям сразу.
# Запуск: python merge_and_final.py

import json, csv, os
from collections import defaultdict

# собираем все judged-файлы, какие есть
files = ["judged.json", "judged_D1.json", "judged_D2.json"]
allj = []
for fn in files:
    if os.path.exists(fn):
        d = json.load(open(fn, encoding="utf-8"))
        allj.extend(d)
        print(f"{fn}: {len(d)} вердиктов")
print(f"Всего: {len(allj)}\n")

def pct(n, d): return round(100*n/d, 1) if d else 0.0

agg = defaultdict(lambda: {"user_n":0,"proverb":0,"all_n":0,"in_role":0,"adv_n":0,"leak":0})
leak_cat = defaultdict(lambda: {"n":0,"leak":0})

for x in allj:
    # ключ конфигурации: для A/B/C это provider/condition, для D — condition (D1/D2)
    cond = x["condition"]
    prov = x.get("provider","")
    k = f"{prov}/{cond}" if cond in ("A","B","C") else cond
    if x.get("in_role") is not None:
        agg[k]["all_n"]+=1; agg[k]["in_role"]+=1 if x["in_role"] else 0
    if x["dataset"]=="user" and x.get("proverb") is not None:
        agg[k]["user_n"]+=1; agg[k]["proverb"]+=1 if x["proverb"] else 0
    if x["dataset"]=="adversarial" and x.get("leak") is not None:
        agg[k]["adv_n"]+=1; agg[k]["leak"]+=1 if x["leak"] else 0
        ck=(k,x["category"]); leak_cat[ck]["n"]+=1; leak_cat[ck]["leak"]+=1 if x["leak"] else 0

print("="*78)
print(f"{'конфиг':14} {'%пословиц':>10} {'%в образе':>10} {'leak rate':>10}")
print("-"*78)
order = ["ollama/A","ollama/B","ollama/C","yandex/A","yandex/B","yandex/C","D1","D2"]
rows=[]
for k in [x for x in order if x in agg]:
    d=agg[k]
    pr=pct(d["proverb"],d["user_n"]); ir=pct(d["in_role"],d["all_n"]); lr=pct(d["leak"],d["adv_n"])
    rows.append({"config":k,"proverb_rate":pr,"in_role_rate":ir,"leak_rate":lr})
    print(f"{k:14} {pr:>9}% {ir:>9}% {lr:>9}%")
print("="*78)

with open("final_metrics_all.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["config","proverb_rate","in_role_rate","leak_rate"])
    w.writeheader(); w.writerows(rows)
print("\nСохранено в final_metrics_all.csv")
