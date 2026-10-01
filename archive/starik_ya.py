# starik_ya.py — старик-рыбак на YandexGPT (для России, без VPN)

import os
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

# ===================================================================
#  ПЕРЕКЛЮЧАТЕЛЬ МОДЕЛИ — меняем ТОЛЬКО эту строку, если нужно сменить модель.
#  "yandexgpt-lite" — дешевле и быстрее, но грубее.
#  "yandexgpt"      — умнее (Pro): лучше держит характер и пословицы.
MODEL = "yandexgpt"
# ===================================================================

sdk = YCloudML(
    folder_id=os.environ["YANDEX_FOLDER_ID"],
    auth=os.environ["YANDEX_API_KEY"],
)

# temperature — "живость" ответов: 0.6-0.8 в самый раз.
# max_tokens — потолок длины ответа. Ставим небольшой, чтобы старик не уходил
#              в длинные абзацы (примерно 200 токенов ≈ несколько коротких фраз).
model = sdk.models.completions(MODEL)
model = model.configure(temperature=0.7, max_tokens=200)

# --- Читаем характер и пословицы ---
with open("prompt_starik.txt", encoding="utf-8") as f:
    system_prompt = f.read()
with open("proverbs.txt", encoding="utf-8") as f:
    poslovicy = f.read()
system_prompt = system_prompt.replace("{POSLOVICY}", poslovicy)

print(f"Старый рыбак сидит в лодке у реки. (модель: {MODEL})")
print("Заговори с ним (или напиши 'выход').\n")

while True:
    soobshenie = input("Ты: ")
    if soobshenie.strip().lower() == "выход":
        print("Рыбак машет рукой: бывай, малец!")
        break

    result = model.run([
        {"role": "system", "text": system_prompt},
        {"role": "user", "text": soobshenie},
    ])

    print("Рыбак:", result[0].text, "\n")