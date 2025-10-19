import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import warnings
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

# ML импорты
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

# Для токсичности
import torch
from transformers import pipeline

warnings.filterwarnings('ignore')

@dataclass
class Document:
    """Класс для хранения информации о документе"""
    text: str
    source: str
    embedding: Optional[np.ndarray] = None
    toxicity_score: Optional[float] = None
    metadata: Optional[Dict] = None

class ToxicityChecker:
    """Класс для проверки токсичности текста"""
    
    def __init__(self, model_name="sismetanin/rubert-toxic-detection"):
        try:
            self.toxicity_classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Модель токсичности загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели токсичности: {e}")
            self.toxicity_classifier = None
    
    def check_toxicity(self, text: str) -> Dict[str, Any]:
        """Проверяет токсичность текста"""
        if not self.toxicity_classifier:
            return {"error": "Модель токсичности не загружена"}
        
        try:
            result = self.toxicity_classifier(text[:512])  # Ограничиваем длину
            toxicity_score = result[0]['score'] if result[0]['label'] == 'toxic' else 1 - result[0]['score']
            
            return {
                'is_toxic': toxicity_score > 0.7,
                'toxicity_score': float(toxicity_score),
                'label': result[0]['label'],
                'confidence': float(result[0]['score'])
            }
        except Exception as e:
            return {"error": f"Ошибка анализа токсичности: {e}"}

class EmbeddingStorage:
    """Класс для работы с эмбеддингами"""
    
    def __init__(self, embedding_model_name='cointegrated/rubert-tiny2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents: List[Document] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
    
    def add_document(self, text: str, source: str, metadata: Dict = None) -> str:
        """Добавляет документ и вычисляет его эмбеддинг"""
        doc_id = f"doc_{len(self.documents)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Создаем эмбеддинг
        embedding = self.embedding_model.encode(text)
        
        document = Document(
            text=text,
            source=source,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self.documents.append(document)
        self._update_embeddings_matrix()
        
        return doc_id
    
    def _update_embeddings_matrix(self):
        """Обновляет матрицу эмбеддингов"""
        if self.documents:
            self.embeddings_matrix = np.array([doc.embedding for doc in self.documents])
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Ищет похожие документы по запросу"""
        if not self.documents:
            return []
        
        # Эмбеддинг запроса
        query_embedding = self.embedding_model.encode(query)
        
        # Косинусное сходство
        similarities = np.dot(self.embeddings_matrix, query_embedding) / (
            np.linalg.norm(self.embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Сортируем по сходству
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Только положительное сходство
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def save_embeddings(self, filepath: str):
        """Сохраняет эмбеддинги в файл"""
        data = {
            'documents': [
                {
                    'text': doc.text,
                    'source': doc.source,
                    'embedding': doc.embedding.tolist() if doc.embedding is not None else None,
                    'metadata': doc.metadata
                }
                for doc in self.documents
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_embeddings(self, filepath: str):
        """Загружает эмбеддинги из файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = []
        for doc_data in data['documents']:
            document = Document(
                text=doc_data['text'],
                source=doc_data['source'],
                embedding=np.array(doc_data['embedding']) if doc_data['embedding'] else None,
                metadata=doc_data.get('metadata', {})
            )
            self.documents.append(document)
        
        self._update_embeddings_matrix()

class SentenceTransformerEmbedder(BaseEstimator, TransformerMixin):
    """Враппер для SentenceTransformer для использования в sklearn pipeline"""
    
    def __init__(self, model_name='cointegrated/rubert-tiny2'):
        self.model_name = model_name
        self.model = None
        
    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name)
        return self
        
    def transform(self, X):
        return self.model.encode(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class EnhancedClassifierAgent:
    """Улучшенный классификатор с токсичностью и эмбеддингами"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.toxicity_checker = ToxicityChecker()
        self.embedding_storage = EmbeddingStorage()
        
        # Демо-данные (ваш существующий код)
        self.demo_texts = [...]  # Ваши demo_texts
        self.demo_labels = [...]  # Ваши demo_labels
        self.class_descriptions = {...}  # Ваши class_descriptions
        
        if model_path:
            self.load_model(model_path)
        else:
            self._create_demo_model()
    
    def _create_demo_model(self):
        """Создает демо-модель"""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        
        self.model = Pipeline([
            ('embedder', SentenceTransformerEmbedder()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        self.model.fit(self.demo_texts, self.demo_labels)
        print("✅ Демо-модель создана")
    
    def predict_with_toxicity(self, text: str) -> Dict[str, Any]:
        """Предсказывает категорию с проверкой токсичности"""
        # Проверяем токсичность
        toxicity_result = self.toxicity_checker.check_toxicity(text)
        
        # Предсказываем категорию
        if self.model:
            try:
                category, confidence = self.predict(text)
                category_result = {
                    'category': category,
                    'confidence': confidence,
                    'description': self.class_descriptions.get(category, '')
                }
            except Exception as e:
                category_result = {'error': f'Ошибка классификации: {e}'}
        else:
            category_result = {'error': 'Модель не загружена'}
        
        return {
            'text': text,
            'toxicity': toxicity_result,
            'classification': category_result
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Базовое предсказание категории"""
        if not self.model:
            raise ValueError("Модель не загружена")
        
        probabilities = self.model.predict_proba([text])[0]
        max_prob_idx = np.argmax(probabilities)
        confidence = probabilities[max_prob_idx]
        predicted_class = self.model.classes_[max_prob_idx]
        
        return predicted_class, float(confidence)
    
    def add_to_embedding_storage(self, text: str, source: str, metadata: Dict = None):
        """Добавляет текст в хранилище эмбеддингов"""
        return self.embedding_storage.add_document(text, source, metadata)
    
    def search_similar_texts(self, query: str, top_k: int = 5):
        """Ищет похожие тексты"""
        return self.embedding_storage.search_similar(query, top_k)
    
    def save_model(self, model_path: str):
        """Сохраняет модель"""
        if self.model:
            joblib.dump(self.model, model_path)
            print(f"✅ Модель сохранена в {model_path}")
    
    def load_model(self, model_path: str):
        """Загружает модель"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Модель загружена из {model_path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self._create_demo_model()
    
    def save_embeddings(self, filepath: str):
        """Сохраняет эмбеддинги"""
        self.embedding_storage.save_embeddings(filepath)
        print(f"✅ Эмбеддинги сохранены в {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Загружает эмбеддинги"""
        self.embedding_storage.load_embeddings(filepath)
        print(f"✅ Эмбеддинги загружены из {filepath}")

# Демонстрация работы
if __name__ == "__main__":
    agent = EnhancedClassifierAgent()
    
    # Тест токсичности и классификации
    test_texts = [
        "не могу войти в аккаунт, эта дурацкая система опять глючит",
        "приложение вылетает при запуске, помогите пожалуйста",
        "вы все идиоты, ничего не работает!",
        "как поменять email в настройках профиля?"
    ]
    
    for text in test_texts:
        result = agent.predict_with_toxicity(text)
        print(f"\n📝 Текст: '{text}'")
        print(f"🔞 Токсичность: {result['toxicity']}")
        print(f"🎯 Классификация: {result['classification']}")
        
        # Добавляем в хранилище эмбеддингов
        if 'error' not in result['classification']:
            agent.add_to_embedding_storage(text, "test_input")
    
    # Поиск похожих текстов
    print("\n🔍 ПОИСК ПОХОЖИХ ТЕКСТОВ:")
    similar = agent.search_similar_texts("проблема с входом в систему")
    for doc, score in similar:
        print(f"  Сходство: {score:.3f} - '{doc.text[:50]}...'")
    
    # Сохраняем эмбеддинги
    agent.save_embeddings("text_embeddings.json")