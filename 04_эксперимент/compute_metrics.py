# compute_metrics.py — считает метрики стиля по результатам прогона.
# НЕ требует судьи. Запускать на ПК с Ollama (для семантических метрик нужен bge-m3).
#
# Метрики:
#   1. avg_length        — средняя длина ответа (символы)
#   2. proverb_string    — % ответов с пословицей (строгое строковое совпадение)
#   3. proverb_semantic  — % ответов с пословицей (семантически, порог близости)
#   4. style_similarity  — близость к эталонному корпусу реплик старика
#
# Запуск: python compute_metrics.py
# Результат: metrics_summary.csv + печать таблицы

import json, re, csv
import numpy as np
import ollama

EMBED_MODEL = "bge-m3"
PROVERB_SEM_THRESHOLD = 0.70   # порог близости "ответ содержит пословицу"

# ---------- загрузка данных ----------
results = json.load(open("results.json", encoding="utf-8"))
corpus = json.load(open("rag_corpus.json", encoding="utf-8"))
gold = json.load(open("eval_gold_replicas.json", encoding="utf-8"))["replicas"]

# ---------- детектор 1: строковый ----------
def norm(s):
    return re.sub(r"[^а-яё ]", " ", s.lower().replace("ё", "е"))

# готовим нормализованные пословицы (для поиска фрагментов)
posl_norm = []
for item in corpus:
    pn = norm(item["text"])
    words = pn.split()
    if len(words) >= 3:
        posl_norm.append(words)

def proverb_string(ans):
    """True, если в ответе есть фрагмент >=4 слов из какой-либо пословицы."""
    na = " " + norm(ans) + " "
    for words in posl_norm:
        # пробуем окна по 4 слова
        for i in range(len(words) - 3):
            frag = " " + " ".join(words[i:i+4]) + " "
            if frag in na:
                return True
    return False

# ---------- эмбеддинги (с кэшем) ----------
_cache = {}
def embed(text):
    if text in _cache:
        return _cache[text]
    v = np.array(ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0],
                 dtype=np.float32)
    v = v / max(np.linalg.norm(v), 1e-8)
    _cache[text] = v
    return v

# матрица эмбеддингов пословиц (для семантического детектора)
print("Считаю эмбеддинги корпуса пословиц (один раз)...")
posl_mat = np.array([embed(it["index_text"]) for it in corpus], dtype=np.float32)

# матрица эмбеддингов эталонных реплик старика (для стилевой близости)
print("Считаю эмбеддинги эталонных реплик...")
gold_mat = np.array([embed(g["text"]) for g in gold], dtype=np.float32)

def proverb_semantic(ans):
    """True, если ответ семантически близок хоть к одной пословице."""
    if not ans.strip():
        return False
    v = embed(ans)
    return float((posl_mat @ v).max()) >= PROVERB_SEM_THRESHOLD

def style_similarity(ans):
    """Средняя близость ответа к топ-3 ближайшим эталонным репликам старика."""
    if not ans.strip():
        return 0.0
    v = embed(ans)
    sims = gold_mat @ v
    top3 = np.sort(sims)[-3:]
    return float(top3.mean())

# ---------- расчёт по конфигурациям ----------
from collections import defaultdict
agg = defaultdict(lambda: {"len": [], "ps": 0, "psem": 0, "style": [], "n": 0})

print("Обрабатываю ответы...")
for i, x in enumerate(results):
    if x["dataset"] != "user":      # стиль меряем на обычных репликах
        continue
    ans = x["answer"] or ""
    k = (x["provider"], x["condition"])
    agg[k]["len"].append(len(ans))
    agg[k]["ps"] += 1 if proverb_string(ans) else 0
    agg[k]["psem"] += 1 if proverb_semantic(ans) else 0
    agg[k]["style"].append(style_similarity(ans))
    agg[k]["n"] += 1
    if (i + 1) % 50 == 0:
        print(f"  ...{i+1}")

# ---------- вывод ----------
rows = []
print("\n" + "="*72)
print(f"{'конфиг':12} {'N':>4} {'ср.длина':>9} {'%посл.строк':>12} {'%посл.сем':>10} {'стиль':>7}")
print("-"*72)
for k in sorted(agg):
    d = agg[k]
    row = {
        "provider": k[0], "condition": k[1], "n": d["n"],
        "avg_length": round(sum(d["len"])/d["n"], 1),
        "proverb_string_pct": round(100*d["ps"]/d["n"], 1),
        "proverb_semantic_pct": round(100*d["psem"]/d["n"], 1),
        "style_similarity": round(sum(d["style"])/d["n"], 3),
    }
    rows.append(row)
    print(f"{k[0]+'/'+k[1]:12} {d['n']:>4} {row['avg_length']:>9} "
          f"{row['proverb_string_pct']:>11}% {row['proverb_semantic_pct']:>9}% "
          f"{row['style_similarity']:>7}")
print("="*72)

# сохраняем
with open("metrics_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["provider","condition","n","avg_length",
        "proverb_string_pct","proverb_semantic_pct","style_similarity"])
    w.writeheader()
    w.writerows(rows)
print("\nСохранено в metrics_summary.csv")
