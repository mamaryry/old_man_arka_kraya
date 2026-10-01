# build_rag_corpus.py — фильтрует корпус для RAG условия C
import csv, re, json

rows = list(csv.DictReader(open("proverbs_v1_fixed.csv", encoding="utf-8-sig")))
print("Исходно:", len(rows))

# --- Фильтры безопасности для NPC 12+ ---
bad_themes = {"stereotypes","stereotype","ingroup","religion","faith","clergy",
    "violence","crime","death","mortality","drunkenness","alcohol","drinking",
    "gender","gender_roles","patriarchy","masculinity","marriage","sex",
    "deception","corruption","war_culture"}

stop = re.compile(r"\b(жид|хохол|цыган|татар|нехрист|барин|барч|поп|поповн|попов|"
    r"чёрт|черт|бес|дьявол|сатан|ад|адом|грех|вино|водк|пьян|хмел|кабак|блуд|"
    r"девк|вдов|смерт|умр|помр|могил|покойник|виселиц|кнут|плеть|икон|молитв|"
    r"аминь|божь|бог|господ|муж|жен|баб)\b", re.I)

def safe(r):
    if "не для бота" in (r.get("notes","") or "").lower():
        return False
    if (r.get("theme","") or "").strip().lower() in bad_themes:
        return False
    if stop.search(r["text"]):
        return False
    if len(r["text"]) > 90:   # слишком длинные диалоговые — мимо
        return False
    return True

kept = [r for r in rows if safe(r)]
print("После фильтра безопасности:", len(kept))

# Формируем записи для индексации: текст + толкование
corpus = []
for r in kept:
    text = r["text"].strip()
    expl = (r.get("explanation","") or "").strip()
    theme = (r.get("theme","") or "").strip()
    # строка для эмбеддинга: пословица + толкование (если есть)
    index_text = text
    if expl and expl != "—":
        index_text = index_text.rstrip(".!?") + ". " + expl
    corpus.append({
        "id": r["id"],
        "text": text,                    # сама пословица (её вставит старик)
        "index_text": index_text,        # по этому считаем эмбеддинг
        "theme": theme,
        "explanation": expl,
    })

json.dump(corpus, open("rag_corpus.json","w",encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("Сохранено в rag_corpus.json:", len(corpus), "пословиц")
print("\nПример записи:")
print(json.dumps(corpus[0], ensure_ascii=False, indent=1))
