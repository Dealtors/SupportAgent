import os
import csv
import json
from datetime import datetime
from bs4 import BeautifulSoup
import docx
from pdfminer.high_level import extract_text as extract_pdf_text
import pytesseract
from pdf2image import convert_from_path

LOG_PATH = "../data/logs_parser.jsonl"


def log_event(event_type: str, filename: str, message: str = "", extra: dict = None):
    """Запись события в лог-файл (jsonl)."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "file": filename,
        "message": message
    }
    if extra:
        record.update(extra)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            text = extract_pdf(file_path)
        elif ext == ".docx":
            text = extract_docx(file_path)
        elif ext == ".html":
            text = extract_html(file_path)
        elif ext in [".md", ".txt"]:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            log_event("warning", file_path, f"Неподдерживаемый формат: {ext}")
            text = ""
    except Exception as e:
        log_event("error", file_path, "Ошибка при чтении файла", {"exception": str(e)})
        text = ""
    return normalize_text(text)


def extract_pdf(file_path: str) -> str:
    try:
        text = extract_pdf_text(file_path)
    except Exception as e:
        log_event("error", file_path, "Ошибка pdfminer", {"exception": str(e)})
        text = ""

    if len(text.strip()) < 50:
        log_event("warning", file_path, "Мало текста, запускаю OCR")
        text = extract_pdf_ocr(file_path)
    return text


def extract_pdf_ocr(file_path: str, max_pages: int = 20) -> str:
    text_blocks = []
    try:
        images = convert_from_path(file_path, dpi=200)
        for i, img in enumerate(images):
            if i >= max_pages:
                log_event("warning", file_path, f"Достигнут лимит OCR {max_pages} страниц")
                break
            try:
                ocr_text = pytesseract.image_to_string(img, lang="rus+eng")
                text_blocks.append(ocr_text)
            except MemoryError:
                log_event("error", file_path, f"MemoryError на странице {i+1}, продолжаю")
                continue
    except Exception as e:
        log_event("error", file_path, "Ошибка OCR", {"exception": str(e)})
    return "\n".join(text_blocks)


def extract_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        log_event("error", file_path, "Ошибка чтения DOCX", {"exception": str(e)})
        return ""


def extract_html(file_path: str) -> str:
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        log_event("error", file_path, "Ошибка чтения HTML", {"exception": str(e)})
        return ""


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_csv(input_dir: str, output_csv: str):
    rows = []
    file_id = 1

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".pdf", ".docx", ".html", ".md", ".txt"]:
                continue

            log_event("info", filename, "Начало обработки")
            content = extract_text(file_path)

            if not content:
                log_event("warning", filename, "Пустой текст после парсинга")
                continue

            first_line = content.split(".")[0][:100] or filename
            rows.append({
                "id": file_id,
                "source": filename,
                "title": first_line,
                "content": content,
                "cluster": "unknown"
            })

            log_event("processed", filename, "Файл успешно обработан", {
                "chars_extracted": len(content)
            })
            file_id += 1

    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "source", "title", "content", "cluster"])
        writer.writeheader()
        writer.writerows(rows)

    log_event("complete", output_csv, f"CSV сформирован, записей: {len(rows)}")
    print(f"[DONE] CSV файл создан: {output_csv}")


if __name__ == "__main__":
    knowledge_dir = "../data/knowledge_base"
    output_file = "../data/knowledge_base.csv"
    build_csv(knowledge_dir, output_file)
