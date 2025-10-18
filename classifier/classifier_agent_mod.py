from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from scipy.stats import uniform
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class SentenceTransformerEmbedder(BaseEstimator, TransformerMixin):
    """Враппер для SentenceTransformer для использования в sklearn pipeline"""
    def __init__(self, model_name='cointegrated/rubert-tiny2'):
        self.model_name = model_name
        self.model = None
        
    def fit(self, X, y=None):
        # Загружаем модель (уже дообученную)
        self.model = SentenceTransformer(self.model_name)
        return self
        
    def transform(self, X):
        return self.model.encode(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# Расширенные демо-данные для обучения
demo_texts = [
    # ========== ПРОБЛЕМЫ С ВХОДОМ (login_issue) ==========
    "не могу войти в аккаунт",
    "забыл пароль от учетной записи",
    "проблема с входом в систему",
    "восстановить доступ к аккаунту",
    "сбросить пароль для входа",
    "аккаунт заблокирован после нескольких попыток",
    "не принимает правильный пароль",
    "ошибка аутентификации при входе",
    "требуется двухфакторная аутентификация но не приходит код",
    "сессия постоянно разрывается",
    "не запоминает логин на устройстве",
    "проблемы с входом через социальные сети",
    "ошибка 401 при авторизации",
    "не могу войти с нового устройства",
    "автоматический выход из системы",
    "учетная запись не активирована",
    "истек срок действия пароля",
    "неправильный логин или пароль",
    "заблокирован вход в аккаунт",
    "проблема с входом после обновления",
    
    # ========== ТЕХНИЧЕСКИЕ ПРОБЛЕМЫ (technical_issue) ==========
    "приложение вылетает при запуске",
    "программа зависает на старте",
    "ошибка при открытии приложения",
    "приложение не запускается на телефоне",
    "тормозит интерфейс при использовании",
    "глюки в программе во время работы",
    "вылетает при переходе между разделами",
    "не работает поиск в приложении",
    "медленно загружаются данные",
    "ошибка соединения с сервером",
    "приложение крашится при отправке сообщения",
    "не обновляется контент в ленте",
    "пропал звук в приложении",
    "не работает уведомления",
    "ошибка 500 внутренняя ошибка сервера",
    "проблемы с загрузкой файлов",
    "не скачиваются вложения",
    "не синхронизируются данные между устройствами",
    "баг с отображением изображений",
    "неправильно работает прокрутка",
    
    # ========== ОБНОВЛЕНИЕ АККАУНТА (account_update) ==========
    "как поменять email в профиле",
    "хочу сменить телефон в учетной записи",
    "обновление личных данных",
    "сменить номер телефона для восстановления",
    "добавить резервный email",
    "изменить страну в профиле",
    "обновить фото профиля",
    "поменять имя пользователя",
    "изменить настройки приватности",
    "настроить уведомления в аккаунте",
    "добавить способ оплаты",
    "сменить язык интерфейса",
    "обновить информацию о компании",
    "изменить дату рождения в профиле",
    "добавить дополнительные контакты",
    "сменить пароль для безопасности",
    "обновить подписку и тарифный план",
    "изменить настройки рассылки",
    "добавить администратора аккаунта",
    "настроить автоматические ответы",
    
    # ========== ПЛАТЕЖИ И БИЛЛИНГ (billing_issue) ==========
    "не прошел платеж за подписку",
    "проблема с оплатой картой",
    "не пришел чек после оплаты",
    "ошибка при списании средств",
    "как отменить автоплатеж",
    "не работает промокод",
    "вернуть деньги за подписку",
    "проблема с налоговыми данными",
    "не отображается история платежей",
    "ошибка в сумме счета",
    "не принимается карта для оплаты",
    "срок действия карты истек",
    "двойное списание средств",
    "не приходит инвойс на email",
    "проблема с валютой платежа",
    "не активируется премиум доступ",
    "ошибка при обновлении подписки",
    "не работает возврат средств",
    "проблема с корпоративной оплатой",
    "не отображается баланс счета",
    
    # ========== ФУНКЦИОНАЛЬНОСТЬ (feature_request) ==========
    "хочу новую функцию в приложении",
    "как сделать экспорт данных",
    "добавить возможность сортировки",
    "предложение по улучшению интерфейса",
    "нужна темная тема для приложения",
    "хочу интеграцию с другими сервисами",
    "добавить горячие клавиши",
    "предлагаю улучшить поиск",
    "нужна оффлайн работа приложения",
    "добавить дополнительные фильтры",
    "хочу кастомизацию dashboard",
    "предложение по новым функциям",
    "нужны расширенные отчеты",
    "добавить возможность комментирования",
    "хочу больше шаблонов для документов",
    "предлагаю улучшить мобильную версию",
    "нужна функция напоминаний",
    "добавить совместную работу",
    "хочу API для разработчиков",
    "предложение по улучшению производительности"
]

demo_labels = [
    # Проблемы с входом (20 примеров)
    "login_issue", "login_issue", "login_issue", "login_issue", "login_issue",
    "login_issue", "login_issue", "login_issue", "login_issue", "login_issue",
    "login_issue", "login_issue", "login_issue", "login_issue", "login_issue",
    "login_issue", "login_issue", "login_issue", "login_issue", "login_issue",
    
    # Технические проблемы (20 примеров)
    "technical_issue", "technical_issue", "technical_issue", "technical_issue", "technical_issue",
    "technical_issue", "technical_issue", "technical_issue", "technical_issue", "technical_issue",
    "technical_issue", "technical_issue", "technical_issue", "technical_issue", "technical_issue",
    "technical_issue", "technical_issue", "technical_issue", "technical_issue", "technical_issue",
    
    # Обновление аккаунта (20 примеров)
    "account_update", "account_update", "account_update", "account_update", "account_update",
    "account_update", "account_update", "account_update", "account_update", "account_update",
    "account_update", "account_update", "account_update", "account_update", "account_update",
    "account_update", "account_update", "account_update", "account_update", "account_update",
    
    # Платежи и биллинг (20 примеров)
    "billing_issue", "billing_issue", "billing_issue", "billing_issue", "billing_issue",
    "billing_issue", "billing_issue", "billing_issue", "billing_issue", "billing_issue",
    "billing_issue", "billing_issue", "billing_issue", "billing_issue", "billing_issue",
    "billing_issue", "billing_issue", "billing_issue", "billing_issue", "billing_issue",
    
    # Функциональность (20 примеров)
    "feature_request", "feature_request", "feature_request", "feature_request", "feature_request",
    "feature_request", "feature_request", "feature_request", "feature_request", "feature_request",
    "feature_request", "feature_request", "feature_request", "feature_request", "feature_request",
    "feature_request", "feature_request", "feature_request", "feature_request", "feature_request"
]

# Дополнительная информация о классах
class_descriptions = {
    "login_issue": "Проблемы с входом в аккаунт, аутентификацией, восстановлением пароля",
    "technical_issue": "Технические проблемы с приложением: вылеты, ошибки, баги", 
    "account_update": "Изменение данных профиля, настроек аккаунта, персональной информации",
    "billing_issue": "Вопросы оплаты, подписок, возвратов, платежных методов",
    "feature_request": "Предложения по новым функциям, улучшению интерфейса, дополнительным возможностям"
}

def evaluate_model(model, X_test, y_test):
    """Комплексная оценка модели с метриками"""
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("=" * 60)
    print("КОМПЛЕКСНАЯ ОЦЕНКА МОДЕЛИ")
    print("=" * 60)
    
    # 1. Основные метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\nОСНОВНЫЕ МЕТРИКИ:")
    print(f"Accuracy (Точность):           {accuracy:.4f}")
    print(f"Precision (Точность макро):    {precision_macro:.4f}")
    print(f"Recall (Полнота макро):        {recall_macro:.4f}")
    print(f"F1-Score (F1-мера макро):      {f1_macro:.4f}")
    
    # 2. Метрики для вероятностей
    try:
        log_loss_value = log_loss(y_test, y_pred_proba)
        # ROC-AUC для многоклассовой классификации
        y_test_bin = label_binarize(y_test, classes=model.classes_)
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
        print(f"Log Loss:                      {log_loss_value:.4f}")
        print(f"ROC-AUC (One-vs-Rest):         {roc_auc:.4f}")
    except Exception as e:
        print(f"Метрики на вероятностях не вычислены: {e}")
        log_loss_value = None
        roc_auc = None
    
    # 3. Детальный отчет по классам
    print("\nДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
    print(classification_report(y_test, y_pred, target_names=model.classes_, zero_division=0))
    
    # 4. Матрица ошибок
    print("МАТРИЦА ОШИБОК:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 5. Метрики по каждому классу
    print("\nМЕТРИКИ ПО КЛАССАМ:")
    classes = model.classes_
    for i, class_name in enumerate(classes):
        precision = precision_score(y_test, y_pred, labels=[class_name], average='micro', zero_division=0)
        recall = recall_score(y_test, y_pred, labels=[class_name], average='micro', zero_division=0)
        f1 = f1_score(y_test, y_pred, labels=[class_name], average='micro', zero_division=0)
        
        print(f"  {class_name:15}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'log_loss': log_loss_value,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def cross_validation_metrics(model, texts, labels, cv=5):
    """Метрики кросс-валидации"""
    
    print("\n" + "=" * 50)
    print("МЕТРИКИ КРОСС-ВАЛИДАЦИИ")
    print("=" * 50)
    
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro', 
        'f1_macro': 'f1_macro'
    }
    
    cv_results = {}
    
    for metric_name, scoring in scoring_metrics.items():
        scores = cross_val_score(model, texts, labels, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'all_scores': scores
        }
        
        print(f"{metric_name:20}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results

def optimize_parameters_randomized(texts, labels, model_path=None):
    """Оптимизация параметров с использованием дообученного SentenceTransformer"""
    
    # Создаем pipeline с SentenceTransformer
    pipeline = Pipeline([
        ('embedder', SentenceTransformerEmbedder(
            model_name=model_path or 'cointegrated/rubert-tiny2'
        )),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Параметры ТОЛЬКО для LogisticRegression
    parameters = {
        'clf__C': uniform(0.001, 100),        # регуляризация
        'clf__penalty': ['l1', 'l2'],         # тип регуляризации
        'clf__solver': ['liblinear', 'saga'], # решатели, поддерживающие L1/L2
        'clf__class_weight': [None, 'balanced'] # балансировка классов
    }
    
    # RandomizedSearch
    random_search = RandomizedSearchCV(
        pipeline, 
        parameters, 
        n_iter=20,           
        cv=3, 
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("🔍 Запуск оптимизации с дообученным SentenceTransformer...")
    random_search.fit(texts, labels)
    
    print("\nОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print("Лучшие параметры:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"Лучшая точность (CV): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def optimize_parameters_with_metrics(texts, labels, test_size=0.2, model_path=None):
    """Оптимизация параметров с полной оценкой метрик"""
    
    # Подавляем конкретные предупреждения
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=42, 
        stratify=labels
    )
    
    print("НАЧИНАЕМ ОПТИМИЗАЦИЮ ПАРАМЕТРОВ")
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки:  {len(X_test)}")
    print(f"Используемая модель: {model_path or 'cointegrated/rubert-tiny2'}")
    
    # Обучаем модель с оптимизацией параметров
    best_model = optimize_parameters_randomized(X_train, y_train, model_path=model_path)
    
    # Оцениваем на тестовых данных
    test_metrics = evaluate_model(best_model, X_test, y_test)
    
    # Кросс-валидационные метрики на всех данных
    cv_metrics = cross_validation_metrics(best_model, texts, labels, cv=5)
    
    # Сводка результатов
    print("\n" + "=" * 60)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    print(f"Лучшая точность (CV):          {cv_metrics['accuracy']['mean']:.4f}")
    print(f"Точность на тесте:             {test_metrics['accuracy']:.4f}")
    print(f"F1-Score макро на тесте:       {test_metrics['f1_macro']:.4f}")
    
    return best_model, test_metrics, cv_metrics

if __name__ == "__main__":
    print("Этот файл содержит функции для обучения модели")
    print("Используйте train_model.py для обучения")