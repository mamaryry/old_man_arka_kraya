# gen_proverb_pairs.py — генерирует диалоговые пары под конкретные пословицы.
# Для каждой пословицы: реплика игрока + ответ старика, где пословица УПОТРЕБЛЕНА к месту.
# Это учит модель вставлять паремии уместно (а не выдавать списком).
#
# Запуск: python gen_proverb_pairs.py
# Результат: proverb_pairs.jsonl + proverb_pairs_review.csv (на вычитку)

import os, json, time, re, csv
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from starik_core import _PROMPT_B

load_dotenv()
sdk = YCloudML(folder_id=os.environ["YANDEX_FOLDER_ID"], auth=os.environ["YANDEX_API_KEY"])
# для генерации реплики игрока
ctx_model = sdk.models.completions("yandexgpt").configure(temperature=0.8, max_tokens=120)
# для ответа старика — стиль B, но с требованием вставить конкретную пословицу
starik_model = sdk.models.completions("yandexgpt").configure(temperature=0.7, max_tokens=200)

poslovicy = json.load(open("poslovicy_for_pairs.json", encoding="utf-8"))
OUT = "proverb_pairs.jsonl"

def gen(system, prompt, model):
    for _ in range(3):
        try:
            r = model.run(([{"role":"system","text":system}] if system else []) +
                          [{"role":"user","text":prompt}])
            return r[0].text.strip()
        except Exception as e:
            if "429" in str(e): time.sleep(2); continue
            return ""
    return ""

done = 0
if os.path.exists(OUT):
    done = sum(1 for _ in open(OUT, encoding="utf-8"))
    print(f"Уже готово: {done}")

f = open(OUT, "a", encoding="utf-8")
t0 = time.time()
for idx, p in enumerate(poslovicy):
    if idx < done: continue
    posl = p["text"]
    # 1) придумываем реплику игрока, к которой эта пословица подошла бы ответом
    instr = gen(None,
        f"Старик-рыбак хочет ответить собеседнику пословицей «{posl}». "
        f"Придумай ОДНУ короткую реплику игрока-подростка, на которую такой ответ "
        f"был бы уместен. Игрок обращается «старик/дед». Ответь ТОЛЬКО репликой, без кавычек.",
        ctx_model)
    instr = instr.split("\n")[0].strip().strip('"«»')
    if not instr or len(instr) > 150:
        continue
    # 2) ответ старика, обязательно вплетающий пословицу
    out = gen(_PROMPT_B,
        instr + f"\n\n(Ответь в образе, ОБЯЗАТЕЛЬНО естественно вплети пословицу «{posl}».)",
        starik_model)
    out = out.strip()
    if not out or posl.split()[0].lower() not in out.lower():
        # грубая проверка, что пословица реально попала в ответ
        pass  # не отбрасываем жёстко — вычитка поймает
    f.write(json.dumps({"instruction": instr, "output": out, "proverb": posl},
                       ensure_ascii=False) + "\n")
    f.flush()
    if (idx+1) % 10 == 0:
        dt=time.time()-t0; sp=(idx+1-done)/dt if dt else 0
        print(f"  [{idx+1}/{len(poslovicy)}] ~{(len(poslovicy)-idx-1)/sp/60 if sp else 0:.0f} мин осталось")

f.close()

# файл на вычитку
pairs = [json.loads(l) for l in open(OUT, encoding="utf-8")]
with open("proverb_pairs_review.csv","w",encoding="utf-8-sig",newline="") as cf:
    w=csv.writer(cf); w.writerow(["i","proverb","instruction","output","DELETE_if_bad"])
    for i,d in enumerate(pairs):
        w.writerow([i,d.get("proverb",""),d["instruction"],d["output"],""])

print(f"\nГотово! {len(pairs)} пословичных пар в {OUT}")
print("Пролистай proverb_pairs_review.csv, помечай плохие (пословица не к месту/искажена).")
print("Потом запусти build_D1_D2.py — соберёт оба обучающих корпуса.")
