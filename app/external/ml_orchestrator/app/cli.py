from app.orchestrator import process_ticket

def main():
    print("=== CLI обращалка к оркестратору ===")
    while True:
        try:
            q = input("> Введите обращение (или 'exit'): ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Пока.")
                break
            label, conf, answer = process_ticket(q)
            print(f"[AI] Класс: {label} (p={conf:.2f})")
            print(f"[AI] Ответ:\n{answer}\n")
        except KeyboardInterrupt:
            print("\nПока.")
            break

if __name__ == "__main__":
    main()
