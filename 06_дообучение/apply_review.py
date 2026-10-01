# apply_review.py — убирает из корпуса пары, помеченные на удаление в finetune_review.csv.
# Запуск ПОСЛЕ вычитки: python apply_review.py
import csv, json

delete_idx = set()
with open("finetune_review.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if (row.get("DELETE_if_bad") or "").strip():
            delete_idx.add(int(row["i"]))

corpus = [json.loads(l) for l in open("finetune_corpus_v2.jsonl", encoding="utf-8")]
kept = [d for i,d in enumerate(corpus) if i not in delete_idx]

with open("finetune_final.jsonl","w",encoding="utf-8") as f:
    for d in kept:
        f.write(json.dumps(d,ensure_ascii=False)+"\n")

print(f"Было: {len(corpus)}, удалено: {len(delete_idx)}, осталось: {len(kept)}")
print("Финальный корпус: finetune_final.jsonl — готов для дообучения.")
