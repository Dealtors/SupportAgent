from enhanced_classifier import EnhancedClassifierAgent
from document_parser import DocumentParser
import json
from typing import Dict
from pathlib import Path

class DocumentProcessingSystem:
    """Интеграционная система для обработки документов"""
    
    def __init__(self):
        self.classifier = EnhancedClassifierAgent()
        self.parser = DocumentParser()
        self.processed_documents = []
    
    def process_document(self, file_path: str, use_ocr: bool = True):
        """Обрабатывает документ: парсит, анализирует токсичность, классифицирует"""
        print(f"🚀 Начинаем обработку: {file_path}")
        
        # Парсим документ
        parsed_pages = self.parser.parse_document(file_path, use_ocr)
        
        results = []
        for page in parsed_pages:
            if page['text'].strip():  # Если есть текст на странице
                page_result = self._analyze_page(page, file_path)
                results.append(page_result)
                
                # Добавляем в хранилище эмбеддингов
                if 'classification' in page_result and 'error' not in page_result['classification']:
                    self.classifier.add_to_embedding_storage(
                        page['text'],
                        f"{file_path}_page_{page['page_number']}",
                        {
                            'file_path': file_path,
                            'page_number': page['page_number'],
                            'parsing_method': page['method']
                        }
                    )
        
        self.processed_documents.append({
            'file_path': file_path,
            'pages': results
        })
        
        return results
    
    def _analyze_page(self, page: Dict, file_path: str) -> Dict:
        """Анализирует одну страницу"""
        text = page['text']
        
        # Анализ токсичности и классификация
        analysis_result = self.classifier.predict_with_toxicity(text)
        
        return {
            'page_number': page['page_number'],
            'parsing_method': page['method'],
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'analysis': analysis_result
        }
    
    def search_in_documents(self, query: str, top_k: int = 5):
        """Ищет похожие тексты во всех обработанных документах"""
        return self.classifier.search_similar_texts(query, top_k)
    
    def save_results(self, output_file: str = "processing_results.json"):
        """Сохраняет результаты обработки"""
        data = {
            'processed_documents': self.processed_documents,
            'timestamp': str(Path(output_file).stat().st_mtime if Path(output_file).exists() else None)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Сохраняем эмбеддинги
        self.classifier.save_embeddings("document_embeddings.json")
        
        print(f"✅ Результаты сохранены в {output_file}")

# Пример использования всей системы
if __name__ == "__main__":
    system = DocumentProcessingSystem()
    
    # Обрабатываем документ
    results = system.process_document("example.pdf", use_ocr=True)
    
    # Выводим результаты
    for result in results:
        print(f"\n📄 Страница {result['page_number']}:")
        print(f"Метод парсинга: {result['parsing_method']}")
        print(f"Текст: {result['text_preview']}")
        
        analysis = result['analysis']
        print(f"🔞 Токсичность: {analysis['toxicity']}")
        print(f"🎯 Классификация: {analysis['classification']}")
    
    # Ищем похожие тексты
    print("\n🔍 Поиск похожих текстов:")
    similar = system.search_in_documents("техническая проблема")
    for doc, score in similar:
        print(f"  Сходство: {score:.3f}")
        print(f"  Источник: {doc.source}")
        print(f"  Текст: {doc.text[:100]}...")
        print()
    
    # Сохраняем результаты
    system.save_results()