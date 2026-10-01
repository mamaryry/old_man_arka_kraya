# run_experiment.py — ПРОГОНЩИК эксперимента.
# Гоняет тестовые датасеты через все конфигурации (условие × модель),
# сохраняет ответы для последующей оценки метриками.
#
# Каждый запрос прогоняется НЕЗАВИСИМО (без памяти) — это требование
# честной оценки: одинаковые условия для всех, воспроизводимость.
#
# Запуск: python run_experiment.py
# Результат: results.json (для метрик) + results.csv (смотреть глазами)

import json, csv, time, os
from starik_core import otvet_starika

# ===================================================================
#  КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
CONDITIONS = ["A", "B", "C"]            # методы внедрения персоны
PROVIDERS = ["ollama", "yandex"]       # модели (6 конфигураций = 3×2)

# Какие датасеты прогонять
DATASETS = {
    "user": "eval_user_prompts.json",          # обычные реплики (стиль)
    "adversarial": "eval_adversarial.json",    # провокации (удержание роли)
}

OUT_JSON = "results.json"
OUT_CSV = "results.csv"
# ===================================================================


def zagruzit_zaprosy():
    """Собирает все тестовые запросы из датасетов в единый список."""
    zaprosy = []
    # пользовательские
    with open(DATASETS["user"], encoding="utf-8") as f:
        for p in json.load(f)["prompts"]:
            zaprosy.append({
                "id": p["id"], "text": p["text"],
                "dataset": "user", "category": p["category"],
            })
    # адверсариальные
    with open(DATASETS["adversarial"], encoding="utf-8") as f:
        for a in json.load(f)["items"]:
            zaprosy.append({
                "id": a["id"], "text": a["text"],
                "dataset": "adversarial", "category": a["category"],
                "expected": a.get("expected", ""), "fail_if": a.get("fail_if", ""),
            })
    return zaprosy


def main():
    zaprosy = zagruzit_zaprosy()
    vsego = len(zaprosy) * len(CONDITIONS) * len(PROVIDERS)
    print(f"Запросов: {len(zaprosy)}, конфигураций: {len(CONDITIONS)*len(PROVIDERS)}")
    print(f"Всего генераций: {vsego}\n")

    # --- Возобновление: если файл уже есть, подхватываем сделанное ---
    rezultaty = []
    sdelano = set()
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            rezultaty = json.load(f)
        sdelano = {(r["query_id"], r["condition"], r["provider"]) for r in rezultaty}
        print(f"Найдено {len(rezultaty)} готовых ответов — продолжаю с места остановки.\n")

    schetchik = len(rezultaty)
    for provider in PROVIDERS:
        for condition in CONDITIONS:
            for z in zaprosy:
                kluch = (z["id"], condition, provider)
                if kluch in sdelano:
                    continue   # уже сделано в прошлый раз

                schetchik += 1
                t0 = time.time()
                try:
                    otvet = otvet_starika(z["text"], condition=condition,
                                          provider=provider, istoria=None)
                    oshibka = ""
                except Exception as e:
                    otvet = ""
                    oshibka = str(e)

                dt = time.time() - t0
                rezultaty.append({
                    "query_id": z["id"],
                    "dataset": z["dataset"],
                    "category": z["category"],
                    "query": z["text"],
                    "condition": condition,
                    "provider": provider,
                    "answer": otvet,
                    "expected": z.get("expected", ""),
                    "fail_if": z.get("fail_if", ""),
                    "seconds": round(dt, 2),
                    "error": oshibka,
                })

                # прогресс
                status = "OK" if not oshibka else f"ОШИБКА: {oshibka[:40]}"
                print(f"[{schetchik}/{vsego}] {provider}/{condition} {z['id']}: {status} ({dt:.1f}с)")

                # сохраняем КАЖДЫЙ раз — чтобы не потерять при сбое
                with open(OUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(rezultaty, f, ensure_ascii=False, indent=1)

    # --- Финальный экспорт в CSV ---
    polya = ["query_id", "dataset", "category", "condition", "provider",
             "query", "answer", "expected", "fail_if", "seconds", "error"]
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=polya, extrasaction="ignore")
        w.writeheader()
        for r in rezultaty:
            w.writerow(r)

    print(f"\nГотово! {len(rezultaty)} ответов.")
    print(f"  {OUT_JSON} — для метрик")
    print(f"  {OUT_CSV} — смотреть глазами (Excel)")


if __name__ == "__main__":
    main()
