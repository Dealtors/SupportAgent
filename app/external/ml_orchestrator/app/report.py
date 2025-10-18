import ujson as json
from collections import Counter
from datetime import datetime
from app.config import Cfg

def load_logs(path: str):
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return items

def generate_report():
    items = load_logs(Cfg.LOGS_PATH)
    if not items:
        print("Логи пусты. Гордиться нечем.")
        return

    by_label = Counter(i.get("label", "Unknown") for i in items)
    avg_conf = sum(i.get("confidence", 0.0) for i in items) / len(items)

    print("# Отчёт по обращениям")
    print(f"Всего кейсов: {len(items)}")
    print(f"Средняя уверенность: {avg_conf:.3f}")
    print("\nТоп классов:")
    for lbl, cnt in by_label.most_common():
        print(f"  - {lbl}: {cnt}")

    print("\nПоследние 5 кейсов:")
    for i in items[-5:]:
        ts = datetime.fromtimestamp(i["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"- [{ts}] {i['label']} (p={i['confidence']:.2f}) | {i['query'][:80]}...")

if __name__ == "__main__":
    generate_report()
