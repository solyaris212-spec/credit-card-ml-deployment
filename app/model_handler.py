
import joblib
import os
import pandas as pd

class ModelHandler:
    def __init__(self, pipeline_path):
        self.pipeline_path = pipeline_path
        self.pipeline = None

    def load_model(self):
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Файл пайплайна не найден: {self.pipeline_path}")
        self.pipeline = joblib.load(self.pipeline_path)

    def predict(self, features_df):
        if self.pipeline is None:
            raise RuntimeError("Пайплайн не загружен. Вызовите load_model()")

        # Передаем DataFrame напрямую в пайплайн.
        # ColumnTransformer сам обработает числовые и категориальные признаки.
        prediction = self.pipeline.predict(features_df)[0]
        probability = self.pipeline.predict_proba(features_df)[0][1]

        return int(prediction), float(probability)
