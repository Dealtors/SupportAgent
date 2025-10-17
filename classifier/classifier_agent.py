# classifier_agent.py
import joblib
import numpy as np
from typing import Tuple, Dict, Any

class ClassifierAgent:
    def __init__(self, model_path: str = "\trained_classifier.joblib"):
        """
        Инициализация агента классификатора
        
        Args:
            model_path: Путь к сохраненной модели. Если None, создается демо-модель
        """
        self.model = None
        self.class_descriptions = {
            "login_issue": "Проблемы с входом в аккаунт, аутентификацией, восстановлением пароля",
            "technical_issue": "Технические проблемы с приложением: вылеты, ошибки, баги", 
            "account_update": "Изменение данных профиля, настроек аккаунта, персональной информации",
            "billing_issue": "Вопросы оплаты, подписок, возвратов, платежных методов",
            "feature_request": "Предложения по новым функциям, улучшению интерфейса, дополнительным возможностям"
        }
        
        if model_path:
            self.load_model(model_path)
        else:
            self._create_demo_model()
    
    def load_model(self, model_path: str) -> bool:
        """
        Загружает модель из файла
        
        Args:
            model_path: Путь к файлу модели
            
        Returns:
            bool: Успешно ли загружена модель
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Модель загружена из {model_path}")
            return True
        except FileNotFoundError:
            print(f"Файл {model_path} не найден!")
            return False
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Предсказывает категорию для текста
        
        Args:
            text: Текст для классификации
            
        Returns:
            Tuple[str, float]: (категория, уверенность)
        """
        if not self.model:
            raise ValueError("Модель не загружена!")
        
        probabilities = self.model.predict_proba([text])[0]
        max_prob_idx = np.argmax(probabilities)
        confidence = probabilities[max_prob_idx]
        predicted_class = self.model.classes_[max_prob_idx]
        
        return predicted_class, float(confidence)
    
    def predict_detailed(self, text: str) -> Dict[str, Any]:
        """
        Расширенное предсказание с детальной информацией
        
        Args:
            text: Текст для классификации
            
        Returns:
            Dict: Детальная информация о предсказаниях
        """
        if not self.model:
            raise ValueError("Модель не загружена!")
        
        probabilities = self.model.predict_proba([text])[0]
        
        result = {
            'text': text,
            'predictions': [],
            'top_prediction': None
        }
        
        # Собираем все предсказания
        for i, class_name in enumerate(self.model.classes_):
            prediction = {
                'class': class_name,
                'confidence': float(probabilities[i]),
                'description': self.class_descriptions.get(class_name, '')
            }
            result['predictions'].append(prediction)
        
        # Сортируем по уверенности
        result['predictions'].sort(key=lambda x: x['confidence'], reverse=True)
        result['top_prediction'] = result['predictions'][0]
        
        return result
    
    def batch_predict(self, texts: list) -> list:
        """
        Пакетная классификация нескольких текстов
        
        Args:
            texts: Список текстов для классификации
            
        Returns:
            list: Список результатов для каждого текста
        """
        return [self.predict_detailed(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели
        
        Returns:
            Dict: Информация о модели
        """
        if not self.model:
            return {"error": "Модель не загружена"}
        
        info = {
            "model_type": type(self.model).__name__,
            "classes": list(self.model.classes_) if hasattr(self.model, 'classes_') else [],
            "class_descriptions": self.class_descriptions
        }
        
        return info

# Фабричная функция для удобного создания агента
def create_classifier_agent(model_path: str = "trained_classifier.joblib"):
    """
    Создает и возвращает готовый к работе классификатор
    
    Args:
        model_path: Путь к модели (по умолчанию ищет trained_classifier.joblib)
        
    Returns:
        ClassifierAgent: Готовый агент классификатора
    """
    return ClassifierAgent(model_path)

# Демонстрация использования
if __name__ == "__main__":
    # Создаем агента (автоматически загрузит trained_classifier.joblib если есть)
    agent = create_classifier_agent()
    
    # Тестируем
    test_texts = [
        "Не работает SSO-авторизация на внутреннем портале: перенаправляет на страницу логина домена, но после ввода паро ля возвращает обратно.",
        "Слетела настройка электронной подписи в ДБО, при входе «Неверный сертификат». Сертификат действителен до 12.2026.",
        "Нужно добавить меня в грурпк AD «Finance-RW» для работы с отчётами. Логин: smirnovdv."
    ]
    
    for text in test_texts:
        category, confidence = agent.predict(text)
        print(f"📝 '{text}' -> {category} ({confidence:.3f})")