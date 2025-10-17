# train_model.py
import joblib
import numpy as np
from classifier_agent_mod import (
    optimize_parameters_with_metrics, 
    demo_texts, 
    demo_labels,
)

def train_and_save_model():
    """Обучает модель и сохраняет её в файл"""
    print("Начинаем обучение модели...")
    
    # Обучаем модель с полной оценкой метрик
    best_model, test_metrics, cv_metrics = optimize_parameters_with_metrics(
        demo_texts, demo_labels, test_size=0.2
    )
    
    # Тестируем на примерах
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА ПРИМЕРАХ")
    print("=" * 50)
    
    test_examples = [
        "не могу войти в аккаунт, пишет ошибка пароля",
        "приложение вылетает при открытии", 
        "как поменять email в настройках",
        "не прошел платеж за подписку",
        "хочу чтобы добавили новую функцию"
    ]
    
    for example in test_examples:
        prediction = best_model.predict([example])[0]
        probability = np.max(best_model.predict_proba([example])[0])
        print(f"'{example}'")
        print(f"   → {prediction} (уверенность: {probability:.3f})")
        print()
    
    # Сохраняем модель
    model_path = "trained_classifier.joblib"
    joblib.dump(best_model, model_path)
    print(f"Модель сохранена в {model_path}")
    
    return model_path, test_metrics, cv_metrics

if __name__ == "__main__":
    model_path, test_metrics, cv_metrics = train_and_save_model()
    
    # Финальная сводка
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ СВОДКА ОБУЧЕНИЯ")
    print("=" * 60)
    print(f"Модель обучена и сохранена: {model_path}")
    print(f"Точность на тесте: {test_metrics['accuracy']:.4f}")
    print(f"F1-Score: {test_metrics['f1_macro']:.4f}")