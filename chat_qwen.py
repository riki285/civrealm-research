import requests
import sys

API_URL = "http://localhost:8001/v1/chat/completions"
MODEL   = "Lorbus/Qwen3.6-27B-int4-AutoRound"


def call(messages, max_tokens: int = 1024) -> str:
    resp = requests.post(
        API_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    system = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    history = []
    if system:
        history.append({"role": "system", "content": system})
        print(f"[system] {system}\n")

    print("Qwen chat  (Ctrl-C or empty line to quit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_input:
            break

        history.append({"role": "user", "content": user_input})
        try:
            reply = call(history)
        except Exception as e:
            print(f"[error] {e}")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        print(f"Qwen: {reply}\n")


if __name__ == "__main__":
    main()
