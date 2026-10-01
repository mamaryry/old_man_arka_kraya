# starik_v3.py — старик-рыбак: память + движок загадок из zagadki.json

import os, json, re, random
from difflib import SequenceMatcher
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

# --- Читаем файлы ---
with open("prompt_starik.txt", encoding="utf-8") as f:
    system_prompt = f.read()
with open("proverbs.txt", encoding="utf-8") as f:
    poslovicy = f.read()
system_prompt = system_prompt.replace("{POSLOVICY}", poslovicy)

with open("zagadki.json", encoding="utf-8") as f:
    zagadki = json.load(f)

# ===================================================================
#  ФУНКЦИЯ МЯГКОГО СРАВНЕНИЯ ОТВЕТА (прощает окончания и опечатки)
# ===================================================================
def normalize(s):
    s = s.lower().strip().replace("ё", "е")
    s = re.sub(r"[^а-яa-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def common_prefix_len(a, b):
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n

def words_match(uw, tw):
    if uw == tw:
        return True
    cpl = common_prefix_len(uw, tw)
    shorter = min(len(uw), len(tw))
    if shorter >= 3 and cpl >= shorter - 2 and cpl >= 2:
        return True
    if max(len(uw), len(tw)) >= 4 and SequenceMatcher(None, uw, tw).ratio() >= 0.8:
        return True
    return False

def is_correct(answer_user, answer_true):
    u = normalize(answer_user)
    t = normalize(answer_true)
    if not u:
        return False
    if u == t:
        return True
    for tw in t.split():
        for uw in u.split():
            if words_match(uw, tw):
                return True
    return False

# ===================================================================
#  ГЕНЕРАЦИЯ ОТВЕТА СТАРИКА (модель только озвучивает, в характере)
# ===================================================================
istoria = [{"role": "system", "text": system_prompt}]
PAMYAT_OBMENOV = 6

def sprosit_starika(zadanie):
    """Передаёт модели историю + спец-задание и возвращает ответ в характере."""
    soobsh = list(istoria) + [{"role": "user", "text": zadanie}]
    # обрезка памяти
    if len(soobsh) > 1 + PAMYAT_OBMENOV * 2 + 1:
        soobsh = [soobsh[0]] + soobsh[-(PAMYAT_OBMENOV * 2 + 1):]
    result = model.run(soobsh)
    return result[0].text

# ===================================================================
#  СОСТОЯНИЕ ЗАГАДОК
# ===================================================================
tekushaya = None        # текущая загаданная загадка (словарь) или None
popytok = 0             # сколько раз игрок ответил неверно на текущую
otgadannye = set()      # id уже отгаданных загадок

def vzyat_zagadku():
    """Берёт случайную ещё не отгаданную загадку, или None если кончились."""
    dostupnye = [z for z in zagadki if z["id"] not in otgadannye]
    if not dostupnye:
        return None
    return random.choice(dostupnye)

def prosyat_zagadku(text):
    t = normalize(text)
    return any(k in t for k in ["загадк", "загадай", "загани", "еще загадку", "ещё загадку"])

# ===================================================================
#  ГЛАВНЫЙ ЦИКЛ
# ===================================================================
print(f"Старый рыбак сидит в лодке у реки. (модель: {MODEL})")
print("Заговори с ним. Попроси загадку, если хочешь. ('выход' — закончить)\n")

while True:
    soobshenie = input("Ты: ")
    nizh = soobshenie.strip().lower()
    if nizh == "выход":
        print("Рыбак машет рукой: бывай, малец!")
        break

    # --- Игрок ответил на текущую загадку? ---
    if tekushaya is not None and not prosyat_zagadku(soobshenie):
        if is_correct(soobshenie, tekushaya["otvet"]):
            # ВЕРНО
            otgadannye.add(tekushaya["id"])
            verno_fraza = tekushaya.get("verno", "Верно, малец!")
            zagadka_resh = tekushaya
            tekushaya = None
            popytok = 0
            zadanie = (f"Игрок ПРАВИЛЬНО отгадал загадку (ответ: «{zagadka_resh['otvet']}»). "
                       f"Похвали его коротко, в своём ворчливо-добром стиле. "
                       f"Можешь использовать фразу: «{verno_fraza}»")
            istoria.append({"role": "user", "text": soobshenie})
            otvet = sprosit_starika(zadanie)
            istoria.append({"role": "assistant", "text": otvet})
            print("Рыбак:", otvet, "\n")
            continue
        else:
            # НЕВЕРНО
            popytok += 1
            nevernye = tekushaya.get("nevernye", ["Неверно, малец."])
            podkolka = random.choice(nevernye)
            if popytok >= 3:
                # даём подсказку
                podskazka = tekushaya.get("podskazka", "")
                zadanie = (f"Игрок снова не отгадал загадку «{tekushaya['vopros']}». "
                           f"Поддень его коротко («{podkolka}»), потом всё же дай подсказку, "
                           f"не называя ответа прямо: {podskazka}")
            else:
                zadanie = (f"Игрок ответил на загадку НЕВЕРНО. "
                           f"Поддень его коротко и в характере, например: «{podkolka}» "
                           f"Предложи попробовать ещё раз. Не называй правильный ответ.")
            istoria.append({"role": "user", "text": soobshenie})
            otvet = sprosit_starika(zadanie)
            istoria.append({"role": "assistant", "text": otvet})
            print("Рыбак:", otvet, "\n")
            continue

    # --- Игрок просит загадку ---
    if prosyat_zagadku(soobshenie):
        z = vzyat_zagadku()
        if z is None:
            istoria.append({"role": "user", "text": soobshenie})
            otvet = sprosit_starika("Все твои загадки игрок уже отгадал. "
                                    "Скажи об этом коротко, в характере, похвали смекалку.")
            istoria.append({"role": "assistant", "text": otvet})
            print("Рыбак:", otvet, "\n")
            continue
        tekushaya = z
        popytok = 0
        zadanie = (f"Загадай игроку ровно эту загадку, слово в слово, в своём стиле, "
                   f"коротко, без пояснений и без ответа: «{z['vopros']}»")
        istoria.append({"role": "user", "text": soobshenie})
        otvet = sprosit_starika(zadanie)
        istoria.append({"role": "assistant", "text": otvet})
        print("Рыбак:", otvet, "\n")
        continue

    # --- Обычный разговор ---
    istoria.append({"role": "user", "text": soobshenie})
    if len(istoria) > 1 + PAMYAT_OBMENOV * 2:
        istoria = [istoria[0]] + istoria[-(PAMYAT_OBMENOV * 2):]
    result = model.run(istoria)
    otvet = result[0].text
    istoria.append({"role": "assistant", "text": otvet})
    print("Рыбак:", otvet, "\n")
