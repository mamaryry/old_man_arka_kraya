# build_D1_D2.py — собирает два обучающих корпуса для сравнения:
#   D1 = только диалоги (finetune_final.jsonl)
#   D2 = диалоги + пословичные пары (после вычитки proverb_pairs_review.csv)
#
# Запуск: python build_D1_D2.py

import json, csv

# --- база: 745 диалогов после вычитки ---
base = [json.loads(l) for l in open("finetune_final.jsonl", encoding="utf-8")]
print(f"База диалогов (D1): {len(base)}")

# D1 — просто база, переименуем поля чисто (instruction/output)
with open("finetune_D1.jsonl","w",encoding="utf-8") as f:
    for d in base:
        f.write(json.dumps({"instruction":d["instruction"],"output":d["output"]},
                           ensure_ascii=False)+"\n")

# --- пословичные пары после вычитки ---
delete = set()
try:
    with open("proverb_pairs_review.csv", encoding="utf-8-sig") as cf:
        for row in csv.DictReader(cf):
            if (row.get("DELETE_if_bad") or "").strip():
                delete.add(int(row["i"]))
except FileNotFoundError:
    print("proverb_pairs_review.csv не найден — D2 не собран.")
    raise SystemExit

pairs = [json.loads(l) for l in open("proverb_pairs.jsonl", encoding="utf-8")]
pairs_clean = [{"instruction":d["instruction"],"output":d["output"]}
               for i,d in enumerate(pairs) if i not in delete]
print(f"Пословичных пар после вычитки: {len(pairs_clean)} (удалено {len(delete)})")

# D2 = база + пословичные пары
import random
random.seed(21)
d2 = base[:] + pairs_clean
# приводим к чистым полям и перемешиваем
d2 = [{"instruction":d["instruction"],"output":d["output"]} for d in d2]
random.shuffle(d2)
with open("finetune_D2.jsonl","w",encoding="utf-8") as f:
    for d in d2:
        f.write(json.dumps(d,ensure_ascii=False)+"\n")

print(f"\nГотово:")
print(f"  finetune_D1.jsonl — {len(base)} пар (только диалоги)")
print(f"  finetune_D2.jsonl — {len(d2)} пар (диалоги + пословицы)")
print("Оба грузи в Colab по очереди, дообучай, получишь две LoRA для сравнения.")
