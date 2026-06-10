"""
Скрипт для анализа результатов A/B-теста моделей кредитного скоринга.

Проводит статистическое сравнение двух моделей по метрикам:
- F1-score (основная метрика, анализ через bootstrap)
- Recall (дополнительная метрика, анализ через z-test для пропорций)
- Precision (контрольная метрика)

Использует симуляцию данных для демонстрации работы.
В production данные загружаются из логов API.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import norm
from typing import Dict, Tuple


def simulate_ab_data(
    n_samples: int = 3500,
    recall_a: float = 0.68,
    recall_b: float = 0.75,
    precision_a: float = 0.48,
    precision_b: float = 0.50,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Симулирует данные A/B-теста для демонстрации.

    Args:
        n_samples: количество заявок в каждой группе.
        recall_a: ожидаемый Recall модели A.
        recall_b: ожидаемый Recall модели B.
        precision_a: ожидаемый Precision модели A.
        precision_b: ожидаемый Precision модели B.
        random_state: seed для воспроизводимости.

    Returns:
        DataFrame с колонками: group, actual, predicted.
    """
    np.random.seed(random_state)

    # Генерация фактических исходов (22% дефолтов, как в исходном датасете)
    default_rate = 0.22
    actual_a = np.random.binomial(1, default_rate, n_samples)
    actual_b = np.random.binomial(1, default_rate, n_samples)

    # Генерация предсказаний с заданными Recall и Precision
    def generate_predictions(actual, recall, precision):
        n = len(actual)
        predicted = np.zeros(n, dtype=int)
        n_defaults = actual.sum()
        n_non_defaults = n - n_defaults

        # TP = recall * n_defaults
        tp = int(recall * n_defaults)
        # FN = n_defaults - TP
        fn = n_defaults - tp
        # FP = TP * (1 - precision) / precision
        if precision > 0:
            fp = int(tp * (1 - precision) / precision)
        else:
            fp = 0
        # TN = n_non_defaults - FP
        tn = max(0, n_non_defaults - fp)

        # Заполняем массив предсказаний
        idx_defaults = np.where(actual == 1)[0]
        idx_non_defaults = np.where(actual == 0)[0]

        # TP: предсказали дефолт, и он реальный
        predicted[np.random.choice(idx_defaults, tp, replace=False)] = 1
        # FP: предсказали дефолт, но его нет
        if fp > 0 and len(idx_non_defaults) > 0:
            fp_actual = min(fp, len(idx_non_defaults))
            predicted[np.random.choice(idx_non_defaults, fp_actual, replace=False)] = 1

        return predicted

    predicted_a = generate_predictions(actual_a, recall_a, precision_a)
    predicted_b = generate_predictions(actual_b, recall_b, precision_b)

    df = pd.DataFrame({
        'group': ['A'] * n_samples + ['B'] * n_samples,
        'actual': np.concatenate([actual_a, actual_b]),
        'predicted': np.concatenate([predicted_a, predicted_b])
    })
    return df


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Рассчитывает метрики для бинарной классификации.

    Args:
        actual: фактические метки (0 или 1).
        predicted: предсказанные метки (0 или 1).

    Returns:
        Словарь с метриками: f1, precision, recall, accuracy.
    """
    return {
        'f1': f1_score(actual, predicted, pos_label=1),
        'precision': precision_score(actual, predicted, pos_label=1, zero_division=0),
        'recall': recall_score(actual, predicted, pos_label=1, zero_division=0),
        'accuracy': np.mean(actual == predicted)
    }


def z_test_proportions(
    count_b: int, n_b: int,
    count_a: int, n_a: int
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Двухвыборочный z-test для сравнения пропорций.

    Args:
        count_b, n_b: число "успехов" и размер выборки в группе B.
        count_a, n_a: число "успехов" и размер выборки в группе A.

    Returns:
        z_stat: z-статистика.
        p_value: двустороннее p-value.
        ci_95: 95% доверительный интервал для разницы (p_B - p_A).
    """
    p_a = count_a / n_a
    p_b = count_b / n_b
    p_pool = (count_a + count_b) / (n_a + n_b)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return 0.0, 1.0, (0.0, 0.0)

    z_stat = (p_b - p_a) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_lower = (p_b - p_a) - 1.96 * se_diff
    ci_upper = (p_b - p_a) + 1.96 * se_diff

    return z_stat, p_value, (ci_lower, ci_upper)


def bootstrap_f1_test(
    actual_a: np.ndarray, predicted_a: np.ndarray,
    actual_b: np.ndarray, predicted_b: np.ndarray,
    n_iterations: int = 10000,
    random_state: int = 42
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Bootstrap-тест для сравнения F1-score двух моделей.

    Args:
        actual_a, predicted_a: данные группы A.
        actual_b, predicted_b: данные группы B.
        n_iterations: количество bootstrap-итераций.
        random_state: seed для воспроизводимости.

    Returns:
        f1_diff: наблюдаемая разница F1_B - F1_A.
        p_value: доля bootstrap-разниц <= 0 (для одностороннего теста "B > A").
        ci_95: 95% доверительный интервал разницы (percentile method).
    """
    np.random.seed(random_state)
    n_a = len(actual_a)
    n_b = len(actual_b)

    f1_a_observed = f1_score(actual_a, predicted_a, pos_label=1)
    f1_b_observed = f1_score(actual_b, predicted_b, pos_label=1)
    f1_diff = f1_b_observed - f1_a_observed

    bootstrap_diffs = np.zeros(n_iterations)
    for i in range(n_iterations):
        idx_a = np.random.choice(n_a, n_a, replace=True)
        idx_b = np.random.choice(n_b, n_b, replace=True)
        f1_a_boot = f1_score(actual_a[idx_a], predicted_a[idx_a], pos_label=1)
        f1_b_boot = f1_score(actual_b[idx_b], predicted_b[idx_b], pos_label=1)
        bootstrap_diffs[i] = f1_b_boot - f1_a_boot

    # p-value: доля бутстрап-разниц <= 0 (тест "B лучше A")
    p_value = np.mean(bootstrap_diffs <= 0)

    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return f1_diff, p_value, (ci_lower, ci_upper)


def run_ab_test_analysis(df: pd.DataFrame) -> Dict:
    """
    Полный анализ A/B-теста.

    Args:
        df: DataFrame с колонками group, actual, predicted.

    Returns:
        Словарь с результатами анализа.
    """
    df_a = df[df['group'] == 'A']
    df_b = df[df['group'] == 'B']

    actual_a = df_a['actual'].values
    predicted_a = df_a['predicted'].values
    actual_b = df_b['actual'].values
    predicted_b = df_b['predicted'].values

    # 1. Метрики по группам
    metrics_a = calculate_metrics(actual_a, predicted_a)
    metrics_b = calculate_metrics(actual_b, predicted_b)

    # 2. Анализ F1-score (bootstrap)
    f1_diff, f1_pvalue, f1_ci = bootstrap_f1_test(
        actual_a, predicted_a, actual_b, predicted_b
    )

    # 3. Анализ Recall (z-test)
    tp_a = ((actual_a == 1) & (predicted_a == 1)).sum()
    fn_a = ((actual_a == 1) & (predicted_a == 0)).sum()
    tp_b = ((actual_b == 1) & (predicted_b == 1)).sum()
    fn_b = ((actual_b == 1) & (predicted_b == 0)).sum()

    recall_z, recall_pvalue, recall_ci = z_test_proportions(
        tp_b, tp_b + fn_b, tp_a, tp_a + fn_a
    )

    # 4. Анализ Precision (z-test)
    fp_a = ((actual_a == 0) & (predicted_a == 1)).sum()
    fp_b = ((actual_b == 0) & (predicted_b == 1)).sum()

    precision_z, precision_pvalue, precision_ci = z_test_proportions(
        tp_b, tp_b + fp_b, tp_a, tp_a + fp_a
    )

    # 5. Критерий успешности
    relative_lift = f1_diff / metrics_a['f1'] if metrics_a['f1'] > 0 else 0
    success = (
        f1_pvalue < 0.05 and
        f1_diff > 0 and
        relative_lift >= 0.03 and
        metrics_b['recall'] >= metrics_a['recall']
    )

    return {
        'sample_sizes': {'A': len(df_a), 'B': len(df_b)},
        'metrics_A': metrics_a,
        'metrics_B': metrics_b,
        'f1_analysis': {
            'difference': f1_diff,
            'relative_lift': relative_lift,
            'p_value': f1_pvalue,
            'ci_95': f1_ci
        },
        'recall_analysis': {
            'difference': metrics_b['recall'] - metrics_a['recall'],
            'z_stat': recall_z,
            'p_value': recall_pvalue,
            'ci_95': recall_ci
        },
        'precision_analysis': {
            'difference': metrics_b['precision'] - metrics_a['precision'],
            'z_stat': precision_z,
            'p_value': precision_pvalue,
            'ci_95': precision_ci
        },
        'success': success,
        'recommendation': 'ВНЕДРИТЬ v2' if success else 'ОСТАВИТЬ v1'
    }


def print_report(results: Dict) -> None:
    """Выводит форматированный отчёт по результатам A/B-теста."""
    print("=" * 70)
    print("ОТЧЁТ ПО РЕЗУЛЬТАТАМ A/B-ТЕСТА МОДЕЛЕЙ КРЕДИТНОГО СКОРИНГА")
    print("=" * 70)

    print(f"\nРазмеры выборок: A = {results['sample_sizes']['A']}, "
          f"B = {results['sample_sizes']['B']}")

    print("\n--- МЕТРИКИ ПО ГРУППАМ ---")
    print(f"{'Метрика':<15} {'Модель A (v1)':<15} {'Модель B (v2)':<15} {'Разница':<10}")
    print("-" * 55)
    for metric in ['f1', 'precision', 'recall', 'accuracy']:
        val_a = results['metrics_A'][metric]
        val_b = results['metrics_B'][metric]
        diff = val_b - val_a
        print(f"{metric:<15} {val_a:<15.4f} {val_b:<15.4f} {diff:+.4f}")

    print("\n--- АНАЛИЗ F1-SCORE (bootstrap, 10000 итераций) ---")
    f1 = results['f1_analysis']
    print(f"Разница (B - A):           {f1['difference']:+.4f}")
    print(f"Относительный lift:        {f1['relative_lift']*100:+.2f}%")
    print(f"p-value:                   {f1['p_value']:.4f}")
    print(f"95% CI для разницы:        [{f1['ci_95'][0]:+.4f}, {f1['ci_95'][1]:+.4f}]")
    print(f"Статистически значимо:     {'ДА' if f1['p_value'] < 0.05 else 'НЕТ'}")

    print("\n--- АНАЛИЗ RECALL (z-test для пропорций) ---")
    rec = results['recall_analysis']
    print(f"Разница (B - A):           {rec['difference']:+.4f}")
    print(f"z-статистика:              {rec['z_stat']:.4f}")
    print(f"p-value:                   {rec['p_value']:.4f}")
    print(f"95% CI для разницы:        [{rec['ci_95'][0]:+.4f}, {rec['ci_95'][1]:+.4f}]")

    print("\n--- АНАЛИЗ PRECISION (z-test для пропорций) ---")
    prec = results['precision_analysis']
    print(f"Разница (B - A):           {prec['difference']:+.4f}")
    print(f"z-статистика:              {prec['z_stat']:.4f}")
    print(f"p-value:                   {prec['p_value']:.4f}")

    print("\n" + "=" * 70)
    print(f"РЕКОМЕНДАЦИЯ: {results['recommendation']}")
    print("=" * 70)

    if results['success']:
        print("\nВсе критерии успешности выполнены:")
        print("  [✓] F1-score статистически значимо выше (p < 0.05)")
        print("  [✓] Направление эффекта положительное")
        print(f"  [✓] Относительный lift >= 3% ({f1['relative_lift']*100:.2f}%)")
        print("  [✓] Recall не ухудшился")
    else:
        print("\nКритерии успешности НЕ выполнены:")
        if f1['p_value'] >= 0.05:
            print("  [✗] F1-score не статистически значим (p >= 0.05)")
        if f1['difference'] <= 0:
            print("  [✗] Разница F1-score отрицательная или нулевая")
        if f1['relative_lift'] < 0.03:
            print(f"  [✗] Относительный lift < 3% ({f1['relative_lift']*100:.2f}%)")
        if results['metrics_B']['recall'] < results['metrics_A']['recall']:
            print("  [✗] Recall ухудшился")


def main():
    """Точка входа: симуляция данных и запуск анализа."""
    print("Генерация симулированных данных A/B-теста...")
    print("(В production данные загружаются из логов API)\n")

    # Симулируем ситуацию, где v2 лучше v1
    df = simulate_ab_data(
        n_samples=3500,
        recall_a=0.68,
        recall_b=0.75,
        precision_a=0.48,
        precision_b=0.50,
        random_state=42
    )

    results = run_ab_test_analysis(df)
    print_report(results)


if __name__ == '__main__':
    main()