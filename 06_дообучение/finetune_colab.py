# =====================================================================
#  ДООБУЧЕНИЕ СТАРИКА (условие D) — QLoRA для Qwen3-8B через Unsloth
#  Запускать в Google Colab (Runtime -> Change runtime type -> GPU -> T4)
#
#  Этот файл — ноутбук в виде ячеек (разделены # %% ). В Colab:
#  либо загрузить .ipynb-версию, либо копировать блоки между # %% в ячейки.
#  Каждая ячейка прокомментирована. Запускать ПО ПОРЯДКУ сверху вниз.
# =====================================================================


# %% [ЯЧЕЙКА 1] Проверка GPU
# Убедимся, что Colab дал видеокарту. Должна показаться Tesla T4 (~15 ГБ).
# Если пишет "command not found" или нет GPU — Runtime -> Change runtime type -> T4 GPU.
!nvidia-smi


# %% [ЯЧЕЙКА 2] Установка Unsloth
# Unsloth — библиотека для быстрого дообучения. Ставит и transformers, peft, trl, bitsandbytes.
# Занимает 1-2 минуты.
!pip install unsloth
# на всякий случай свежая версия (Unsloth часто обновляется)
!pip install --upgrade --no-cache-dir unsloth


# %% [ЯЧЕЙКА 3] Подключаем Google Drive (туда сохраним веса)
# Откроется окно авторизации Google — разрешить доступ.
from google.colab import drive
drive.mount('/content/drive')

# создадим папку для результата
import os
SAVE_DIR = "/content/drive/MyDrive/starik_lora"
os.makedirs(SAVE_DIR, exist_ok=True)
print("Веса сохраним в:", SAVE_DIR)


# %% [ЯЧЕЙКА 4] Загружаем обучающий корпус
# Загрузить finetune_final.jsonl в Colab:
#   слева иконка папки -> кнопка "Upload" -> выбрать finetune_final.jsonl
# (или положить его на Google Drive и поменять путь ниже)
import json

CORPUS_PATH = "/content/finetune_final.jsonl"   # если грузилось через Upload
# CORPUS_PATH = "/content/drive/MyDrive/finetune_final.jsonl"  # если с Drive

data = [json.loads(l) for l in open(CORPUS_PATH, encoding="utf-8")]
print(f"Загружено пар: {len(data)}")
print("Пример:", data[0])


# %% [ЯЧЕЙКА 5] Загружаем модель Qwen3-8B в 4-битах (QLoRA)
from unsloth import FastModel

MODEL_NAME = "unsloth/Qwen3-8B"   # та же Qwen3-8B, что в условиях A/B/C
MAX_SEQ = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ,
    load_in_4bit = True,        # 4-битная квантизация (QLoRA) — экономит память
    full_finetuning = False,
)
print("Модель загружена.")


# %% [ЯЧЕЙКА 6] Добавляем LoRA-адаптеры (обучаться будут только они)
model = FastModel.get_peft_model(
    model,
    r = 16,                     # ранг LoRA (16 — разумно для стилевого дообучения)
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",   # экономия памяти
    random_state = 3407,        # фиксируем для воспроизводимости
)
print("LoRA-адаптеры добавлены.")


# %% [ЯЧЕЙКА 7] Форматируем данные БЕЗ системного промпта
# Решение (см. журнал): обучаем без system prompt — стиль "вшивается" в веса.
# Формат — чат-шаблон Qwen3: user (реплика игрока) -> assistant (ответ старика).
def format_example(ex):
    messages = [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,      # отключаем "размышления" Qwen3
    )
    return {"text": text}

from datasets import Dataset
dataset = Dataset.from_list(data).map(format_example)
print("Пример отформатированного текста:\n")
print(dataset[0]["text"][:500])


# %% [ЯЧЕЙКА 8] Настройка тренировки
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        max_seq_length = MAX_SEQ,
        per_device_train_batch_size = 2,     # на T4 безопасно; если OOM — поставить 1
        gradient_accumulation_steps = 4,     # эффективный батч = 2*4 = 8
        warmup_steps = 5,
        num_train_epochs = 3,                # 3 прохода по корпусу — норм для стиля
        learning_rate = 2e-4,                # стандарт для LoRA
        logging_steps = 10,
        optim = "adamw_8bit",                # экономный оптимизатор
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)
print("Тренер готов. Запускай следующую ячейку для обучения.")


# %% [ЯЧЕЙКА 9] ОБУЧЕНИЕ (самый долгий шаг)
# На T4 с ~800 примерами и 3 эпохами — порядка 20-40 минут.
# Следить за колонкой training loss: она должна СНИЖАТЬСЯ. Если скачет вверх —
# модель не учится (тогда уменьшить learning_rate). Если loss падает до ~0.5-1.0 —
# хорошо. Слишком низкий loss (<0.3) может значить переобучение.
trainer_stats = trainer.train()
print("Обучение завершено!")
print(trainer_stats)


# %% [ЯЧЕЙКА 10] Быстрая проверка дообученной модели
# Спросим старика БЕЗ системного промпта — стиль должен проявиться сам.
from unsloth import FastModel
FastModel.for_inference(model)   # режим инференса

def ask(text):
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt"
    ).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=150, temperature=0.7)
    return tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=True)

for q in ["Здравствуй, старик!", "Перевези меня на тот берег",
          "Расскажи про профессора", "Дай покурить"]:
    print(f"\n[{q}]")
    print(ask(q))


# %% [ЯЧЕЙКА 11] Сохраняем LoRA-адаптеры на Google Drive
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"LoRA-адаптеры сохранены в {SAVE_DIR}")


# %% [ЯЧЕЙКА 12] (ОПЦИЯ) Экспорт в формат для Ollama (GGUF)
# Чтобы условие D работало через Ollama, как A/B/C — экспортируем в GGUF.
# Это создаст файл, который можно подключить к Ollama локально.
# ВНИМАНИЕ: занимает время и место. Делать, если надо гонять D через свой
# существующий конвейер (starik_core.py с provider="ollama").
model.save_pretrained_gguf(
    SAVE_DIR + "_gguf",
    tokenizer,
    quantization_method = "q4_k_m",   # та же квантизация, что у qwen3:8b
)
print("GGUF сохранён. Скачай его и подключи к Ollama (инструкция отдельно).")
