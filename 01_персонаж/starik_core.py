# starik_core.py — ЯДРО персоны старика для экспериментов.
# Не интерактивный чат, а чистая функция: дай запрос + настройки -> получи ответ.
# Поддерживает условия A/B/C и две модели (ollama/yandex).
#
# Использование из другого скрипта:
#   from starik_core import otvet_starika
#   text = otvet_starika("Здравствуй!", condition="B", provider="ollama")

import os, re, json
from dotenv import load_dotenv

load_dotenv()

# ===================================================================
#  НАСТРОЙКИ ПО УМОЛЧАНИЮ
OLLAMA_MODEL = "qwen3:8b"
YANDEX_MODEL = "yandexgpt"
TEMPERATURE = 0.7
MAX_TOKENS = 400
RAG_TOP_N = 3            # сколько пословиц подмешивать в условии C
# ===================================================================

# Ленивая инициализация провайдеров (чтобы не грузить лишнее)
_ya_model = None
_ollama = None

def _get_ollama():
    global _ollama
    if _ollama is None:
        import ollama
        _ollama = ollama
    return _ollama

def _get_yandex():
    global _ya_model
    if _ya_model is None:
        from yandex_cloud_ml_sdk import YCloudML
        sdk = YCloudML(
            folder_id=os.environ["YANDEX_FOLDER_ID"],
            auth=os.environ["YANDEX_API_KEY"],
        )
        _ya_model = sdk.models.completions(YANDEX_MODEL).configure(
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS
        )
    return _ya_model

# Загрузка промптов и данных персоны
with open("prompt_starik.txt", encoding="utf-8") as f:
    _PROMPT_FULL = f.read()
with open("proverbs.txt", encoding="utf-8") as f:
    _POSLOVICY_STATIC = f.read()

# Условие B: полный промпт со статическими пословицами
_PROMPT_B = _PROMPT_FULL.replace("{POSLOVICY}", _POSLOVICY_STATIC)

# Условие A (baseline): минимальная инструкция, без примеров и ограничений
_PROMPT_A = (
    "Ты — старый рыбак, сидишь в лодке у реки. "
    "Отвечай на реплики собеседника."
)

# Условие C: тот же каркас, что B, но БЕЗ статических пословиц —
# вместо них подставим найденные RAG (заполним в рантайме).
# Берём промпт B и заменяем список пословиц на плейсхолдер для RAG.
_PROMPT_C_TEMPLATE = _PROMPT_FULL  # содержит {POSLOVICY}, заполним динамически


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _generate(messages, provider):
    """Низкоуровневый вызов модели."""
    if provider == "yandex":
        result = _get_yandex().run(messages)
        return result[0].text
    elif provider == "ollama":
        oll = _get_ollama()
        oll_msgs = [{"role": m["role"], "content": m["text"]} for m in messages]
        resp = oll.chat(
            model=OLLAMA_MODEL,
            messages=oll_msgs,
            think=False,
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        )
        return _strip_think(resp["message"]["content"])
    else:
        raise ValueError(f"Неизвестный provider: {provider}")


def _build_system_prompt(zapros, condition):
    """Собирает системный промпт под условие A/B/C."""
    if condition == "A":
        return _PROMPT_A
    elif condition == "B":
        return _PROMPT_B
    elif condition == "C":
        # RAG: находим уместные пословицы под реплику игрока
        from rag_poslovicy import naydi_poslovicy
        nayden = naydi_poslovicy(zapros, top_n=RAG_TOP_N)
        spisok = "\n".join(p["text"] for p in nayden)
        return _PROMPT_C_TEMPLATE.replace("{POSLOVICY}", spisok)
    else:
        raise ValueError(f"Неизвестное условие: {condition}")


def otvet_starika(zapros, condition="B", provider="ollama", istoria=None):
    """ГЛАВНАЯ ФУНКЦИЯ. Возвращает ответ старика.

    zapros    — реплика игрока (строка)
    condition — "A" / "B" / "C"
    provider  — "ollama" / "yandex"
    istoria   — список предыдущих реплик [{"role":"user"/"assistant","text":...}]
                или None (без памяти, для одиночных прогонов в эксперименте)
    """
    system_prompt = _build_system_prompt(zapros, condition)
    messages = [{"role": "system", "text": system_prompt}]
    if istoria:
        messages.extend(istoria)
    messages.append({"role": "user", "text": zapros})
    return _generate(messages, provider)


# Быстрый тест: python starik_core.py
if __name__ == "__main__":
    test_zapros = "Перевези меня на тот берег"
    for cond in ["A", "B", "C"]:
        print(f"\n=== Условие {cond} (ollama) ===")
        try:
            print(otvet_starika(test_zapros, condition=cond, provider="ollama"))
        except Exception as e:
            print(f"[ошибка: {e}]")
