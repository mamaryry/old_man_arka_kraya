# starik_v2.py — старик-рыбак на YandexGPT С ПАМЯТЬЮ о разговоре

import os
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

MODEL = "yandexgpt"   # "yandexgpt-lite" — дешевле; "yandexgpt" — умнее (Pro)

sdk = YCloudML(
    folder_id=os.environ["YANDEX_FOLDER_ID"],
    auth=os.environ["YANDEX_API_KEY"],
)
model = sdk.models.completions(MODEL)
model = model.configure(temperature=0.7, max_tokens=200)

with open("prompt_starik.txt", encoding="utf-8") as f:
    system_prompt = f.read()
with open("proverbs.txt", encoding="utf-8") as f:
    poslovicy = f.read()
system_prompt = system_prompt.replace("{POSLOVICY}", poslovicy)

print(f"Старый рыбак сидит в лодке у реки. (модель: {MODEL})")
print("Заговори с ним (или напиши 'выход').\n")

# Тут: список истории разговора. Каждый элемент — словарь с ролью и текстом.
# Системную инструкцию (характер) кладём первой и навсегда.
istoria = [{"role": "system", "text": system_prompt}]

# Тут: сколько последних ОБМЕНОВ помнить (1 обмен = реплика игрока + ответ старика).
# Больше — лучше память, но дороже и медленнее запрос. 6 — разумно.
PAMYAT_OBMENOV = 6

while True:
    soobshenie = input("Ты: ")
    if soobshenie.strip().lower() == "выход":
        print("Рыбак машет рукой: бывай, малец!")
        break

    # НОВОЕ: добавляем реплику игрока в историю
    istoria.append({"role": "user", "text": soobshenie})

    # НОВОЕ: обрезаем историю, чтобы не разрасталась.
    # Оставляем system (1 шт.) + последние PAMYAT_OBMENOV*2 сообщений.
    if len(istoria) > 1 + PAMYAT_OBMENOV * 2:
        istoria = [istoria[0]] + istoria[-(PAMYAT_OBMENOV * 2):]

    # НОВОЕ: передаём модели всю историю целиком (поэтому старик помнит контекст)
    result = model.run(istoria)
    otvet = result[0].text

    # НОВОЕ: сохраняем ответ старика в историю (роль "assistant")
    istoria.append({"role": "assistant", "text": otvet})

    print("Рыбак:", otvet, "\n")
