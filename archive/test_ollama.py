# test_ollama.py — проверка связи с локальной моделью
import ollama

otvet = ollama.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Ответь одним предложением: ты работаешь локально?"}],
)
print(otvet["message"]["content"])