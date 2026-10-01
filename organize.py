# organize.py — наводит порядок в папке проекта: создаёт подпапки и раскладывает файлы.
# Запуск: python organize.py
# Безопасно: если файла нет — просто пропускает. Запускать ИЗ папки oldman_bot.

import os, shutil

# структура: папка -> список файлов, которые в неё переедут
PLAN = {
    "01_персонаж": [
        "prompt_starik.txt", "proverbs.txt", "zagadki.json",
        "lore.txt", "starik_core.py",
    ],
    "02_данные": [
        "proverbs_v1_fixed.csv", "rag_corpus.json",
        "eval_gold_replicas.json", "eval_user_prompts.json", "eval_adversarial.json",
        "poslovicy_for_pairs.json",
    ],
    "03_RAG": [
        "build_rag_corpus.py", "build_rag_index.py", "rag_poslovicy.py", "rag_index.npz",
    ],
    "04_эксперимент": [
        "run_experiment.py", "compute_metrics.py",
        "results.json", "results.csv", "metrics_summary.csv",
    ],
    "05_судья": [
        "judge.py", "judged.json", "final_metrics.py", "final_metrics.csv",
        "merge_manual.py", "manual_20.csv",
    ],
    "06_дообучение": [
        "gen_finetune_corpus.py", "gen_finetune_corpus_v2.py", "apply_review.py",
        "gen_proverb_pairs.py", "build_D1_D2.py",
        "finetune_colab.ipynb", "finetune_colab.py", "colab_run_D.py",
        "judge_D1.py", "judge_D2.py", "merge_and_final.py",
        "results_D1.json", "results_D2.json", "judged_D1.json", "judged_D2.json",
        "finetune_final.jsonl", "finetune_D1.jsonl", "finetune_D2.jsonl",
        "starik_lora_D1.jsonl", "starik_lora_D2.jsonl", "proverb_pairs.jsonl",
    ],
    "07_статистика": [
        "stat_significance.py", "stat_results.csv", "final_metrics_all.csv",
    ],
    "08_текст": [
        "Введение_ВКР.docx", "ЖУРНАЛ.md", "README.md",
    ],
    "archive": [
        # старые версии бота и рабочие файлы калибровки
        "starik_v1.py", "starik_v2.py", "starik_v3.py", "starik_v4.py", "starik_ya.py",
        "test_ollama.py", "proverbs_starik_full.txt", "poslovicy_starik.txt",
        "proverb_labeling.csv", "proverb_labeling_v2.csv",
        "prepare_proverb_labeling.py", "prepare_enriched_labeling.py",
        "calibrate_detector.py", "calibrate_detector_v2.py",
        "proverb_pairs_review.csv", "finetune_review.csv", "judge_review.csv",
        "judge_review_D1.csv", "judge_review_D2.csv", "finetune_raw.jsonl",
        "build_finetune.py", "reference.docx",
    ],
}

print("Навожу порядок в папке проекта...\n")
moved = 0
missing = []
for folder, files in PLAN.items():
    os.makedirs(folder, exist_ok=True)
    for fname in files:
        if os.path.exists(fname) and os.path.isfile(fname):
            try:
                shutil.move(fname, os.path.join(folder, fname))
                moved += 1
            except Exception as e:
                print(f"  ! не удалось переместить {fname}: {e}")
        else:
            missing.append(fname)

print(f"Перемещено файлов: {moved}")
print(f"Создано папок: {len(PLAN)}")

# что осталось в корне (не вошло в план) — покажем, чтобы ты решила сама
ostatok = [f for f in os.listdir(".")
           if os.path.isfile(f) and f != "organize.py" and not f.startswith(".")]
if ostatok:
    print(f"\nОсталось в корне (не разложено — посмотри сама): {len(ostatok)}")
    for f in sorted(ostatok):
        print(f"   {f}")
else:
    print("\nВ корне чисто (кроме самого organize.py).")

print("\nГотово! Структура наведена. README.md лежит в 08_текст.")
print("Можно удалить organize.py после проверки.")
