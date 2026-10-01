# prepare_enriched_labeling.py — готовит ОБОГАЩЁННУЮ выборку для дозаметки.
# Переиспользует уже размеченные 50 (proverb_labeling.csv) и добавляет ~60 новых
# ответов со СМЕЩЕНИЕМ в сторону кандидатов с высокой близостью к пословицам,
# чтобы среди размеченного было больше реальных паремий (решает редкий класс).
#
# Нужен Ollama + bge-m3 (считает близости).
# Запуск: python prepare_enriched_labeling.py
# Результат: proverb_labeling_v2.csv — дозаметь НОВЫЕ строки (has_proverb пустое)

import json, csv, re, random
import numpy as np
import ollama

EMBED_MODEL = "bge-m3"
NEW_TO_ADD = 60          # сколько новых добавить к старым 50
ENRICH_RATIO = 0.6       # доля новых из "зоны высокой близости"

random.seed(43)

results = json.load(open("results.json", encoding="utf-8"))
corpus = json.load(open("rag_corpus.json", encoding="utf-8"))

# --- старая разметка: какие answer уже размечены ---
already = {}
try:
    with open("proverb_labeling.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            already[row["answer"]] = (row.get("has_proverb") or "").strip()
except FileNotFoundError:
    pass
print(f"Старых размеченных строк: {sum(1 for v in already.values() if v in ('0','1'))}")

# --- кандидаты: user-ответы, ещё не размеченные ---
user_ans = [x["answer"] for x in results
            if x["dataset"] == "user" and (x["answer"] or "").strip()]
user_ans = list(dict.fromkeys(user_ans))           # уникальные
kandidaty = [a for a in user_ans if a not in already]

# --- считаем близость каждого кандидата к корпусу пословиц ---
def embed(t):
    v = np.array(ollama.embed(model=EMBED_MODEL, input=t)["embeddings"][0], dtype=np.float32)
    return v / max(np.linalg.norm(v), 1e-8)

print("Считаю эмбеддинги корпуса...")
posl_mat = np.array([embed(it["index_text"]) for it in corpus], dtype=np.float32)

print("Оцениваю близость кандидатов...")
scored = []
for a in kandidaty:
    v = embed(a)
    scored.append((float((posl_mat @ v).max()), a))
scored.sort(reverse=True)   # по убыванию близости

# --- обогащённый отбор ---
n_high = int(NEW_TO_ADD * ENRICH_RATIO)            # из зоны высокой близости
n_rest = NEW_TO_ADD - n_high                        # из остальных (для отриц. класса)
high_zone = [a for _, a in scored[:max(n_high*3, 30)]]   # верхушка по близости
rest_zone = [a for _, a in scored[max(n_high*3, 30):]]

novye = random.sample(high_zone, min(n_high, len(high_zone)))
novye += random.sample(rest_zone, min(n_rest, len(rest_zone)))
random.shuffle(novye)

# --- пишем объединённый файл: старые (с метками) + новые (пустые) ---
with open("proverb_labeling_v2.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row_id", "query", "answer", "has_proverb", "is_new"])
    rid = 0
    # старые размеченные
    for ans, lab in already.items():
        if lab in ("0", "1"):
            w.writerow([rid, "", ans, lab, 0]); rid += 1
    # новые на дозаметку
    for ans in novye:
        w.writerow([rid, "", ans, "", 1]); rid += 1

vsego_staryh = sum(1 for v in already.values() if v in ('0','1'))
print(f"\nГотово! proverb_labeling_v2.csv")
print(f"  старых (уже размечено): {vsego_staryh}")
print(f"  новых на дозаметку:     {len(novye)}")
print(f"  итого после дозаметки:  {vsego_staryh + len(novye)}")
print(f"\nДозаметь строки с is_new=1 (has_proverb пустое): 1=есть пословица, 0=нет.")
print(f"Старые (is_new=0) НЕ трогай — они уже размечены.")
print(f"Потом запусти calibrate_detector_v2.py")
