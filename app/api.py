from flask import Flask, request, jsonify
import sys
import os
import pandas as pd
import logging
import json
import time
from datetime import datetime

# Добавляем путь к папке app для импорта model_handler
sys.path.append(os.path.dirname(__file__))
from model_handler import ModelHandler

app = Flask(__name__)

# === Настройка JSON-логирования ===
# Создаем отдельный логгер для API-запросов
api_logger = logging.getLogger('api_logger')
api_logger.setLevel(logging.INFO)

# Обработчик для записи логов в файл logs/api_requests.log
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/api_requests.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Формат: каждая запись - одна строка JSON
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }
        # Если есть дополнительные поля, добавляем их
        if hasattr(record, 'extra_data'):
            log_record.update(record.extra_data)
        return json.dumps(log_record, ensure_ascii=False)

file_handler.setFormatter(JsonFormatter())
api_logger.addHandler(file_handler)

# Также выводим логи в консоль для удобства отладки
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(JsonFormatter())
api_logger.addHandler(console_handler)

# === Загрузка модели ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_handler = ModelHandler(
    pipeline_path=os.path.join(BASE_DIR, "models", "pipeline_v1.pkl")
)
model_handler.load_model()


def validate_input(data):
    """Валидация входных данных клиента."""
    # 1. Проверка числовых признаков
    numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                        'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    for feature in numeric_features:
        if feature not in data:
            return False, f"Отсутствует признак: {feature}"
        if not isinstance(data[feature], (int, float)):
            return False, f"Признак {feature} должен быть числом"
        if data[feature] < 0:
            return False, f"Признак {feature} не может быть отрицательным"

    # 2. Проверка возраста
    if data['AGE'] < 18 or data['AGE'] > 100:
        return False, "AGE должен быть в диапазоне от 18 до 100"

    # 3. Проверка категориальных признаков
    if data.get('SEX') not in [1, 2]:
        return False, "SEX должен быть 1 (male) или 2 (female)"

    if data.get('EDUCATION') not in [1, 2, 3, 4, 5, 6]:
        return False, "EDUCATION должен быть от 1 до 6"

    if data.get('MARRIAGE') not in [1, 2, 3]:
        return False, "MARRIAGE должен быть 1, 2 или 3"

    pay_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for feature in pay_features:
        if feature not in data:
            return False, f"Отсутствует признак: {feature}"
        if data[feature] not in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return False, f"Признак {feature} должен быть от -1 до 9"

    return True, "OK"


@app.route('/predict', methods=['POST'])
def predict():
    """Эндпоинт предсказания дефолта с JSON-логированием."""
    start_time = time.time()
    client_ip = request.remote_addr

    try:
        data = request.get_json()
        if data is None:
            # Логируем некорректный запрос
            log_extra = {
                "endpoint": "/predict",
                "client_ip": client_ip,
                "status": "error",
                "error": "Invalid JSON"
            }
            api_logger.info("Invalid request", extra={'extra_data': log_extra})
            return jsonify({"error": "Неверный формат данных, ожидается JSON"}), 400

        # Валидация
        is_valid, message = validate_input(data)
        if not is_valid:
            log_extra = {
                "endpoint": "/predict",
                "client_ip": client_ip,
                "status": "validation_failed",
                "error": message
            }
            api_logger.info("Validation failed", extra={'extra_data': log_extra})
            return jsonify({"error": message}), 400

        # Предсказание
        features_df = pd.DataFrame([data])
        prediction, probability = model_handler.predict(features_df)

        # Расчет времени отклика
        response_time_ms = int((time.time() - start_time) * 1000)

        # Успешное логирование
        log_extra = {
            "endpoint": "/predict",
            "client_ip": client_ip,
            "status": "success",
            "prediction": prediction,
            "probability": round(probability, 4),
            "model_version": "v2_pipeline",
            "response_time_ms": response_time_ms
        }
        api_logger.info("Prediction made", extra={'extra_data': log_extra})

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4),
            "model_version": "v2_pipeline"
        }), 200

    except Exception as e:
        # Логирование внутренней ошибки
        log_extra = {
            "endpoint": "/predict",
            "client_ip": client_ip,
            "status": "internal_error",
            "error": str(e)
        }
        api_logger.error("Internal error", extra={'extra_data': log_extra})
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Эндпоинт проверки работоспособности."""
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)