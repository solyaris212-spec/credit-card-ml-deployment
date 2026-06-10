
# Сервис прогнозирования дефолта по кредитным картам

## Описание

ML-сервис для прогнозирования вероятности дефолта по кредитным картам на основе данных UCI Default of Credit Card Clients.

**Раздел:** финансы / кредитный скоринг  
**Целевая переменная:** `default.payment.next.month` (1 = дефолт в следующем месяце, 0 = нет)

## Структура проекта

```text
credit-card-ml-deployment/
├── app/                    # Flask-приложение
│   ├── __init__.py
│   ├── api.py             # API эндпоинты с JSON-логированием
│   └── model_handler.py   # Загрузка и инференс модели
├── models/                 # Обученные модели
│   └── pipeline_v1.pkl    # Пайплайн предобработки + классификатор
├── logs/                   # Логи API (создается автоматически)
│   └── api_requests.log
├── docker/                 # Конфигурация контейнеров
│   ├── Dockerfile
│   └── nginx.conf
├── tests/                  # Тесты API
├── data/                   # Данные
├── notebooks/              # Jupyter ноутбуки
├── ARCHITECTURE.md         # Описание архитектуры
├── ab_test_plan.md         # План A/B-тестирования
├── ab_test_analysis.py     # Скрипт анализа A/B-теста
├── docker-compose.yml      # Оркестрация сервисов
├── requirements.txt        # Зависимости Python
├── .dockerignore
├── .gitignore
└── README.md
```

## Быстрый старт

### Локальный запуск (без Docker)

1. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   ```
2. Активируйте окружение:
   ```bash
   # Linux/Mac
   source venv/bin/activate  
   # Windows
   .\venv\Scripts\activate   
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Запустите сервис:
   ```bash
   python -m app.api
   ```
Сервер запустится на `http://localhost:5000`

### Запуск в Docker (один контейнер)

1. Скачайте образ из Docker Hub:
   ```bash
   docker pull solyaris212/credit-card-ml:v1
   ```
2. Запустите контейнер:
   ```bash
   docker run -d -p 5000:5000 --name credit-ml solyaris212/credit-card-ml:v1
   ```

### Запуск через Docker Compose (сервис + сбор логов)

```bash
docker-compose up -d --build
```

Посмотреть логи (имитация дашборда): `http://localhost:8080/logs/`  
Остановить: 
```bash
docker-compose down
```

## API Документация

### POST /predict

Предсказание дефолта по кредитной карте.

**Формат запроса (JSON):**
```json
{
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -1,
    "PAY_6": -1,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 689,
    "PAY_AMT2": 0,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
}
```

**Формат ответа (JSON):**
```json
{"model_version":"v2_pipeline","prediction":1,"probability":0.8546}
```

**Пример curl-запроса (для Windows PowerShell):**
```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "`@test_request.json"
```

### Проверка работоспособности сервиса.

В адресной строке введите: http://127.0.0.1:5000/health

Вы должны увидеть: {"status":"healthy"}.

## Логирование

Все запросы к `/predict` логируются в формате JSON в файл `logs/api_requests.log`.

**Пример записи:**
```json
{"timestamp": "2026-06-10T10:43:38.069250", "level": "INFO", "message": "Prediction made", "endpoint": "/predict", "client_ip": "127.0.0.1", "status": "success", "prediction": 1, "probability": 0.8546, "model_version": "v2_pipeline", "response_time_ms": 16}
```

## A/B-тестирование

Полный план A/B-теста описан в файле [ab_test_plan.md](ab_test_plan.md).

### Краткое описание

Сравниваются две версии модели:
- **Контрольная группа (A)**: текущая модель v1 (LogisticRegression).
- **Тестовая группа (B)**: новая модель v2 (GradientBoosting).

**Разделение трафика:** случайное 50/50 на уровне запросов.  
**Продолжительность:** 14 дней (минимум), 30 дней (максимум).

### Метрики

- **Основная:** F1-score для класса дефолта (анализ через bootstrap).
- **Дополнительная:** Recall (обоснование: стоимость пропуска дефолта в 2–6 раз выше стоимости ложного отказа).

### Статистический анализ

- **F1-score:** bootstrap-тест (10000 итераций), 95% CI через percentile method.
- **Recall/Precision:** двухвыборочный z-test для пропорций.
- **Критерий успешности:** p-value < 0.05, относительный lift F1 >= 3%, Recall не ухудшился.

### Запуск анализа

```bash
python ab_test_analysis.py
```

## Архитектура

Подробное описание архитектурных решений см. в файле [ARCHITECTURE.md](ARCHITECTURE.md):
- Обоснование выбора монолитной архитектуры
- Концепт брокеров сообщений (RabbitMQ)
- Стратегия логирования и мониторинга
- MLOps инструменты (DVC, MLflow)
- Бизнес-метрики

## Метрики модели

### Технические метрики (на тестовой выборке):
- Accuracy: ~0.69
- Precision (класс 1): ~0.48
- Recall (класс 1): ~0.68
- F1-score (класс 1): ~0.56

### Бизнес-метрики:
- **Expected Loss** = PD × LGD × EAD — ожидаемые финансовые потери по клиенту
- **Approval Rate at Risk Level** — доля одобренных заявок при заданном пороге риска
- **Lift over Random Approval** — прирост качества по сравнению со случайным одобрением

## Ссылки

- [Датасет UCI Credit Card](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [[Docker Hub Image](https://hub.docker.com/r/solyaris212/credit-card-ml)
