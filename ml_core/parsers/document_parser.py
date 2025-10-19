import PyPDF2
import pdfplumber
import pytesseract
from typing import Dict
from pathlib import Path
from typing import List, Optional

try:
    from pdf2image import convert_from_path
except ImportError:
    print("⚠️  pdf2image не установлен. Установите: pip install pdf2image")

class DocumentParser:
    """Универсальный парсер документов с поддержкой PDF и OCR"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Args:
            tesseract_path: Путь к tesseract.exe (для Windows)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.supported_extensions = {'.pdf', '.txt'}
    
    def parse_document(self, file_path: str, use_ocr: bool = True) -> List[Dict[str, any]]:
        """Парсит документ и возвращает текст с метаданными"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if file_path.suffix.lower() == '.pdf':
            return self._parse_pdf(file_path, use_ocr)
        elif file_path.suffix.lower() == '.txt':
            return self._parse_txt(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {file_path.suffix}")
    
    def _parse_pdf(self, pdf_path: Path, use_ocr: bool = True) -> List[Dict[str, any]]:
        """Парсит PDF файл"""
        pages_data = []
        
        # Попытка 1: Извлечение текста напрямую
        direct_text = self._extract_text_directly(pdf_path)
        
        if direct_text and any(len(page['text'].strip()) > 50 for page in direct_text):
            print("✅ Текст извлечен напрямую из PDF")
            return direct_text
        
        # Попытка 2: Использование pdfplumber
        pdfplumber_text = self._extract_with_pdfplumber(pdf_path)
        if pdfplumber_text and any(len(page['text'].strip()) > 50 for page in pdfplumber_text):
            print("✅ Текст извлечен с помощью pdfplumber")
            return pdfplumber_text
        
        # Попытка 3: OCR если разрешено и предыдущие методы не сработали
        if use_ocr:
            print("🔄 Пытаемся извлечь текст с помощью OCR...")
            ocr_text = self._extract_with_ocr(pdf_path)
            if ocr_text:
                print("✅ Текст извлечен с помощью OCR")
                return ocr_text
        
        print("❌ Не удалось извлечь текст из PDF")
        return pages_data
    
    def _extract_text_directly(self, pdf_path: Path) -> List[Dict[str, any]]:
        """Извлекает текст напрямую с помощью PyPDF2"""
        pages_data = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'method': 'direct_extraction',
                        'char_count': len(text)
                    })
        except Exception as e:
            print(f"⚠️ Ошибка прямого извлечения: {e}")
        
        return pages_data
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Dict[str, any]]:
        """Извлекает текст с помощью pdfplumber"""
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    # Если текста мало, пытаемся извлечь таблицы
                    if len(text.strip()) < 100:
                        tables = page.extract_tables()
                        table_text = self._process_tables(tables)
                        text += "\n" + table_text
                    
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'method': 'pdfplumber',
                        'char_count': len(text)
                    })
        except Exception as e:
            print(f"⚠️ Ошибка pdfplumber: {e}")
        
        return pages_data
    
    def _extract_with_ocr(self, pdf_path: Path) -> List[Dict[str, any]]:
        """Извлекает текст с помощью OCR"""
        pages_data = []
        
        try:
            # Конвертируем PDF в изображения
            images = convert_from_path(pdf_path, dpi=300)
            
            for page_num, image in enumerate(images):
                # Используем tesseract для OCR
                text = pytesseract.image_to_string(image, lang='rus+eng')
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'method': 'ocr',
                    'char_count': len(text)
                })
                
        except Exception as e:
            print(f"⚠️ Ошибка OCR: {e}")
        
        return pages_data
    
    def _parse_txt(self, txt_path: Path) -> List[Dict[str, any]]:
        """Парсит текстовый файл"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return [{
                'page_number': 1,
                'text': text,
                'method': 'direct',
                'char_count': len(text)
            }]
        except Exception as e:
            print(f"❌ Ошибка чтения txt файла: {e}")
            return []
    
    def _process_tables(self, tables: List) -> str:
        """Обрабатывает таблицы и преобразует в текст"""
        table_text = ""
        
        for table in tables:
            if table:
                for row in table:
                    row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                    table_text += row_text + "\n"
                table_text += "\n"
        
        return table_text
    
    def batch_parse(self, folder_path: str, use_ocr: bool = True) -> Dict[str, List[Dict]]:
        """Парсит все документы в папке"""
        folder = Path(folder_path)
        results = {}
        
        for file_path in folder.iterdir():
            if file_path.suffix.lower() in self.supported_extensions:
                print(f"📄 Обрабатываем: {file_path.name}")
                try:
                    results[file_path.name] = self.parse_document(file_path, use_ocr)
                except Exception as e:
                    print(f"❌ Ошибка обработки {file_path.name}: {e}")
                    results[file_path.name] = []
        
        return results

# Пример использования
if __name__ == "__main__":
    parser = DocumentParser()
    
    # Парсим один документ
    try:
        result = parser.parse_document("example.pdf", use_ocr=True)
        for page in result:
            print(f"\n📄 Страница {page['page_number']} ({page['method']}):")
            print(f"Текст: {page['text'][:200]}...")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Парсим все документы в папке
    # results = parser.batch_parse("./documents", use_ocr=True)
    