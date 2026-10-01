# rag_poslovicy.py — поиск пословиц по смыслу (RAG) для условия C.
# Загружает индекс один раз, затем быстро ищет под каждую реплику.

import json
import numpy as np
import ollama

EMBED_MODEL = "bge-m3"

# --- Загружаем корпус и индекс при импорте модуля ---
with open("rag_corpus.json", encoding="utf-8") as f:
    _corpus = json.load(f)
_by_id = {item["id"]: item for item in _corpus}

_data = np.load("rag_index.npz", allow_pickle=True)
_vectors = _data["vectors"]        # матрица нормализованных векторов
_ids = list(_data["ids"])          # порядок id, совпадает со строками матрицы


def naydi_poslovicy(zapros, top_n=3):
    """Возвращает top_n наиболее подходящих по смыслу пословиц к тексту zapros.
    Каждая пословица — словарь с полями text, theme, explanation."""
    # 1) эмбеддинг запроса
    resp = ollama.embed(model=EMBED_MODEL, input=zapros)
    q = np.array(resp["embeddings"][0], dtype=np.float32)
    q = q / max(np.linalg.norm(q), 1e-8)      # нормализуем

    # 2) косинусная близость = скалярное произведение (всё нормализовано)
    scores = _vectors @ q                      # вектор близостей ко всем пословицам

    # 3) берём индексы top_n самых близких
    top_idx = np.argsort(-scores)[:top_n]

    # 4) собираем результат
    rezultat = []
    for idx in top_idx:
        pid = _ids[idx]
        item = _by_id[pid]
        rezultat.append({
            "text": item["text"],
            "theme": item.get("theme", ""),
            "explanation": item.get("explanation", ""),
            "score": float(scores[idx]),
        })
    return rezultat


# Быстрый тест при прямом запуске: python rag_poslovicy.py
if __name__ == "__main__":
    for zapros in ["расскажи про дружбу", "я боюсь идти дальше",
                   "научи меня рыбачить", "стоит ли торопиться"]:
        print(f"\nЗапрос: {zapros}")
        for p in naydi_poslovicy(zapros, top_n=3):
            print(f"  [{p['score']:.3f}] {p['text']}")
