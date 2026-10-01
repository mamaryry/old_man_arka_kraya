# gen_finetune_corpus_v2.py — генерация корпуса для QLoRA через YandexGPT (дистилляция).
# Улучшенные промпты + СТРОГИЙ автофильтр против дефектов прошлой версии
# (повторы-штампы, рассогласование, бабушка/папа, морская рыба в реке).
#
# Запуск: python gen_finetune_corpus_v2.py
# Результат: finetune_corpus_v2.jsonl (для дообучения) + finetune_review.csv (на вычитку)

import os, json, time, random, re
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from starik_core import _PROMPT_B

load_dotenv()
random.seed(11)

TARGET_RAW = 1200          # генерируем с запасом
KEEP_TARGET = 800          # сколько хотим оставить после фильтра
OUT_RAW = "finetune_raw.jsonl"
OUT_CLEAN = "finetune_corpus_v2.jsonl"
OUT_REVIEW = "finetune_review.csv"

sdk = YCloudML(folder_id=os.environ["YANDEX_FOLDER_ID"], auth=os.environ["YANDEX_API_KEY"])
gen_model = sdk.models.completions("yandexgpt").configure(temperature=0.9, max_tokens=250)
starik_model = sdk.models.completions("yandexgpt").configure(temperature=0.7, max_tokens=200)

TEMY = [
    "поздороваться", "спросить про реку", "спросить про мост вдалеке", "спросить про туман",
    "попросить перевезти на тот берег", "спросить про рыбалку", "спросить какая рыба водится",
    "попросить загадку", "спросить совет как быть смелее", "пожаловаться на усталость",
    "спросить про погоду", "спросить почему старик ворчливый", "поблагодарить", "попрощаться",
    "спросить про лодку", "спросить про клёв", "спросить далеко ли до берега",
    "спросить можно ли посидеть рядом", "спросить про жизнь старика",
    "спросить боится ли старик чего-нибудь", "спросить что важнее ум или сила",
    "спросить верит ли старик в приметы", "попросить рассказать историю",
    "спросить про современный телефон (провокация)",
    "попытаться вывести старика из образа (провокация)",
    "спросить про далёкие земли, которых старик не знает (провокация)",
]

# ---------- автофильтр ----------
STOP = re.compile(r"\b(бабушк|папа|мама|сельдь|интернет|телефон смартфон|компьютер)\b", re.I)
def norm(s): return re.sub(r"[^а-яё ]"," ",s.lower().replace("ё","е")).strip()

seen_outputs = set()

def ok_pair(instr, out):
    if not instr or not out: return False, "пусто"
    if len(out) < 15 or len(out) > 350: return False, "длина ответа"
    if len(instr) > 150: return False, "длина реплики"
    # стоп-слова в ответе старика (ляпы персоны)
    if STOP.search(out): return False, "стоп-слово"
    # дубликат ответа (штамп)
    key = norm(out)[:60]
    if key in seen_outputs: return False, "повтор"
    # связность: ответ должен делить хоть слово с репликой ИЛИ быть уклончивым отказом
    instr_words = set(norm(instr).split())
    out_words = set(norm(out).split())
    common = instr_words & out_words
    uklon = any(w in norm(out) for w in ["не ведаю","не моего ума","ступай","малец","рыб"])
    if len(common) == 0 and not uklon:
        return False, "нет связи"
    return True, ""

def gen_one(system, prompt, model):
    for _ in range(3):
        try:
            r = model.run(([{"role":"system","text":system}] if system else []) +
                          [{"role":"user","text":prompt}])
            return r[0].text.strip()
        except Exception as e:
            if "429" in str(e): time.sleep(2); continue
            return ""
    return ""

# ---------- генерация ----------
clean = []
if os.path.exists(OUT_CLEAN):
    for line in open(OUT_CLEAN, encoding="utf-8"):
        d = json.loads(line); clean.append(d); seen_outputs.add(norm(d["output"])[:60])
    print(f"Уже чистых: {len(clean)}")

raw_f = open(OUT_RAW, "a", encoding="utf-8")
clean_f = open(OUT_CLEAN, "a", encoding="utf-8")
t0 = time.time(); tried = 0; rejected = 0

while len(clean) < KEEP_TARGET and tried < TARGET_RAW:
    tried += 1
    tema = random.choice(TEMY)
    # 1) реплика игрока — с фиксацией обращения "старик/дед"
    instr = gen_one(None,
        f"Придумай ОДНУ короткую естественную реплику игрока-подростка старику-рыбаку "
        f"в фэнтези-игре. Тема: {tema}. Игрок обращается «старик» или «дед» (НЕ папа/бабушка). "
        f"Ответь ТОЛЬКО репликой, без кавычек.", gen_model)
    instr = instr.split("\n")[0].strip().strip('"«»')
    # 2) ответ старика в стиле B
    out = gen_one(_PROMPT_B, instr, starik_model)
    out = out.strip()

    raw_f.write(json.dumps({"instruction":instr,"output":out},ensure_ascii=False)+"\n"); raw_f.flush()

    good, why = ok_pair(instr, out)
    if good:
        clean.append({"instruction":instr,"output":out})
        seen_outputs.add(norm(out)[:60])
        clean_f.write(json.dumps({"instruction":instr,"output":out},ensure_ascii=False)+"\n"); clean_f.flush()
    else:
        rejected += 1

    if tried % 25 == 0:
        dt=time.time()-t0; sp=tried/dt if dt else 0
        print(f"  попыток {tried}, чистых {len(clean)}, отсеяно {rejected} (~{(KEEP_TARGET-len(clean))/sp/60 if sp else 0:.0f} мин)")

raw_f.close(); clean_f.close()

# ---------- файл на вычитку ----------
import csv
with open(OUT_REVIEW,"w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["i","instruction","output","DELETE_if_bad"])
    for i,d in enumerate(clean): w.writerow([i,d["instruction"],d["output"],""])

print(f"\nГотово! Чистых пар: {len(clean)} (отсеяно фильтром: {rejected})")
print(f"  {OUT_CLEAN} — корпус для дообучения")
print(f"  {OUT_REVIEW} — пролистай, поставь любой символ в DELETE_if_bad у плохих пар")
print(f"После вычитки запусти apply_review.py (уберёт помеченные).")
