# prepare_proverb_labeling.py — готовит выборку ответов для ручной разметки.
# Цель: 50 разнообразных ответов (из разных конфигураций), чтобы откалибровать
# детектор пословиц. Нужно разметить колонку has_proverb (1/0) вручную.
#
# Запуск: python prepare_proverb_labeling.py
# Результат: proverb_labeling.csv — открыть в Excel, проставить 1/0 в has_proverb

import json, csv, random

random.seed(42)   # воспроизводимость выборки

results = json.load(open("results.json", encoding="utf-8"))

# Берём только user-ответы (там меряем стиль/пословицы)
user_answers = [x for x in results if x["dataset"] == "user" and (x["answer"] or "").strip()]

# Сбалансированно по конфигурациям: примерно поровну из каждой из 6
from collections import defaultdict
by_conf = defaultdict(list)
for x in user_answers:
    by_conf[(x["provider"], x["condition"])].append(x)

vyborka = []
per_conf = 50 // len(by_conf)   # ~8 из каждой конфигурации
for conf, items in by_conf.items():
    vyborka.extend(random.sample(items, min(per_conf, len(items))))

# добиваем до 50, если не хватило
ostatok = [x for x in user_answers if x not in vyborka]
while len(vyborka) < 50 and ostatok:
    pick = random.choice(ostatok)
    vyborka.append(pick)
    ostatok.remove(pick)

random.shuffle(vyborka)   # перемешиваем, чтобы разметчик не видел паттерн конфигураций

# Пишем для разметки. Колонку has_proverb оставляем ПУСТОЙ — её заполнить руками.
with open("proverb_labeling.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row_id", "query", "answer", "has_proverb"])
    for i, x in enumerate(vyborka):
        w.writerow([i, x["query"], x["answer"], ""])

print(f"Готово! {len(vyborka)} ответов в proverb_labeling.csv")
print()
print("ЧТО ДЕЛАТЬ ДАЛЬШЕ:")
print("1. Открой proverb_labeling.csv в Excel.")
print("2. В колонке has_proverb поставь:")
print("   1 — если в ответе ЕСТЬ пословица/поговорка (даже перефразированная),")
print("   0 — если пословицы НЕТ (обычная речь, даже образная).")
print("3. Сохрани файл (оставь формат CSV).")
print("4. Запусти calibrate_detector.py — он подберёт порог по твоей разметке.")
print()
print("Подсказка по разметке: пословица — это устойчивое народное изречение")
print("с обобщённым смыслом ('без труда не выловишь рыбку'). Просто красивая")
print("или мудрая фраза, придуманная на ходу — это НЕ пословица (ставь 0).")
