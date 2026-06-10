
import pytest
import sys
import os
import json

# Добавляем путь к корню проекта, чтобы корректно импортировать app
# Это необходимо для корректной работы путей при запуске pytest из корня репозитория
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.api import app

@pytest.fixture
def client():
    """Фикстура pytest для создания тестового клиента Flask."""
    app.config['TESTING'] = True
    # Отключаем JSON-логирование в файл во время тестов, чтобы не засорять диск
    # (в реальном проекте можно настроить отдельный логгер для тестов)
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Тест 1: Проверка эндпоинта /health."""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_predict_success(client):
    """Тест 2: Успешное предсказание с валидными данными."""
    test_data = {
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 689, "PAY_AMT2": 0,
        "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }
    
    response = client.post('/predict', json=test_data)
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability' in data
    assert data['model_version'] == 'v2_pipeline'
    assert isinstance(data['prediction'], int)
    assert 0.0 <= data['probability'] <= 1.0

def test_predict_validation_error_age(client):
    """Тест 3: Ошибка валидации (некорректный возраст)."""
    test_data = {
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 150, # Невалидный возраст
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 689, "PAY_AMT2": 0,
        "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }
    
    response = client.post('/predict', json=test_data)
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'AGE' in data['error']

def test_predict_validation_error_sex(client):
    """Тест 4: Ошибка валидации (некорректный пол)."""
    test_data = {
        "LIMIT_BAL": 20000, "SEX": 3, # Невалидный пол
        "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 689, "PAY_AMT2": 0,
        "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }
    
    response = client.post('/predict', json=test_data)
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'SEX' in data['error']
