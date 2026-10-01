# build_rag_index.py — считает эмбеддинги пословиц и сохраняет индекс на диск.
# Запускается ОДИН РАЗ (и повторно — только если поменялся rag_corpus.json).
#
# Перед запуском: ollama pull bge-m3
#
# Результат: файл rag_index.npz с векторами всех пословиц.

import json
import numpy as np
import ollama

EMBED_MODEL = "bge-m3"          # многоязычная модель эмбеддингов (русский ок)

# --- Загружаем отфильтрованный корпус ---
with open("rag_corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Пословиц для индексации: {len(corpus)}")
print(f"Модель эмбеддингов: {EMBED_MODEL}")
print("Считаю эмбеддинги... (это займёт несколько минут, идёт по одной)\n")

vectors = []
ids = []
for i, item in enumerate(corpus):
    # Считаем эмбеддинг индексной строки (пословица + толкование)
    resp = ollama.embed(model=EMBED_MODEL, input=item["index_text"])
    vec = resp["embeddings"][0]      # вектор-список чисел
    vectors.append(vec)
    ids.append(item["id"])

    if (i + 1) % 200 == 0:
        print(f"  обработано {i + 1} / {len(corpus)}")

# Превращаем в массив numpy и нормализуем (для косинусной близости)
mat = np.array(vectors, dtype=np.float32)
# нормализация: делим каждый вектор на его длину
norms = np.linalg.norm(mat, axis=1, keepdims=True)
mat = mat / np.clip(norms, 1e-8, None)

# Сохраняем индекс: матрицу векторов и порядок id
np.savez("rag_index.npz", vectors=mat, ids=np.array(ids))
print(f"\nГотово! Индекс сохранён в rag_index.npz")
print(f"Размер матрицы: {mat.shape} (пословиц × размерность вектора)")
