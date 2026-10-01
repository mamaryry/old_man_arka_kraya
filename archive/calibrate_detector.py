# calibrate_detector.py — калибрует детектор пословиц по твоей ручной разметке.
# Подбирает оптимальный порог семантической близости и проверяет лексический фильтр.
#
# Перед запуском: заполнить колонку has_proverb в proverb_labeling.csv (1/0).
# Запуск: python calibrate_detector.py

import csv, json, re
import numpy as np
import ollama

EMBED_MODEL = "bge-m3"

# ---------- загрузка размеченных данных ----------
labeled = []
with open("proverb_labeling.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        hp = (row["has_proverb"] or "").strip()
        if hp not in ("0", "1"):
            continue   # пропускаем неразмеченные
        labeled.append({"answer": row["answer"], "label": int(hp)})

print(f"Размеченных ответов: {len(labeled)}")
pos = sum(x["label"] for x in labeled)
print(f"  с пословицей (1): {pos}")
print(f"  без пословицы (0): {len(labeled)-pos}")
if len(labeled) < 10 or pos == 0 or pos == len(labeled):
    print("\nМало данных или все метки одинаковые — размечай разнообразнее.")
    raise SystemExit

# ---------- корпус пословиц ----------
corpus = json.load(open("rag_corpus.json", encoding="utf-8"))

def norm(s):
    return re.sub(r"[^а-яё ]", " ", s.lower().replace("ё", "е"))

# редкие слова каждой пословицы (для лексического фильтра)
import collections
all_words = collections.Counter()
posl_wordsets = []
for it in corpus:
    ws = set(norm(it["text"]).split())
    posl_wordsets.append(ws)
    all_words.update(ws)
# "содержательные" слова — не самые частые предлоги/союзы
stop = {"и","а","в","на","не","что","то","с","да","как","ты","по","за","о","у",
        "же","бы","ли","из","от","до","но","я","мне","тебе","его","это","так"}

def embed(text):
    v = np.array(ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0], dtype=np.float32)
    return v / max(np.linalg.norm(v), 1e-8)

print("Считаю эмбеддинги корпуса...")
posl_mat = np.array([embed(it["index_text"]) for it in corpus], dtype=np.float32)

# для каждого размеченного ответа: макс. близость + есть ли лексическое пересечение
print("Считаю близости размеченных ответов...")
for x in labeled:
    if not x["answer"].strip():
        x["sim"] = 0.0; x["lex"] = False; continue
    v = embed(x["answer"])
    sims = posl_mat @ v
    best = int(np.argmax(sims))
    x["sim"] = float(sims[best])
    # лексический фильтр: делит ли ответ хоть одно содержательное слово
    # с какой-либо из топ-5 ближайших пословиц
    top5 = np.argsort(-sims)[:5]
    ans_words = set(norm(x["answer"]).split()) - stop
    x["lex"] = any(len((posl_wordsets[i]-stop) & ans_words) >= 1 for i in top5)

# ---------- подбор порога ----------
def metrics_at(threshold, use_lex):
    tp=fp=tn=fn=0
    for x in labeled:
        pred = (x["sim"] >= threshold) and (x["lex"] if use_lex else True)
        if pred and x["label"]==1: tp+=1
        elif pred and x["label"]==0: fp+=1
        elif not pred and x["label"]==0: tn+=1
        else: fn+=1
    acc = (tp+tn)/len(labeled)
    prec = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    return acc, prec, rec, f1

print("\n" + "="*70)
print("ПОДБОР ПОРОГА (без лексического фильтра / с фильтром):")
print(f"{'порог':>6} | {'acc':>5} {'prec':>5} {'rec':>5} {'F1':>5} | "
      f"{'acc+L':>5} {'prec+L':>6} {'rec+L':>5} {'F1+L':>5}")
print("-"*70)
best = (0, 0, False)   # f1, threshold, use_lex
for thr in [round(t,2) for t in np.arange(0.55, 0.86, 0.05)]:
    a0,p0,r0,f0 = metrics_at(thr, False)
    a1,p1,r1,f1 = metrics_at(thr, True)
    print(f"{thr:>6} | {a0:.2f} {p0:.2f} {r0:.2f} {f0:.2f} | "
          f"{a1:.2f}  {p1:.2f}  {r1:.2f} {f1:.2f}")
    for f1v, uselex in [(f0, False), (f1, True)]:
        if f1v > best[0]:
            best = (f1v, thr, uselex)

print("="*70)
print(f"\nЛУЧШИЙ ВАРИАНТ: порог={best[1]}, лексический фильтр={'ДА' if best[2] else 'НЕТ'}, F1={best[0]:.2f}")
print(f"\nВставь в compute_metrics.py:")
print(f"  PROVERB_SEM_THRESHOLD = {best[1]}")
print(f"  использовать лексический фильтр: {best[2]}")
