import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, precision_score

def apply_consecutive_threshold(series, threshold=5):
    anomaly_groups = (series.diff() != 0).cumsum()
    group_sizes = series.groupby(anomaly_groups).transform('size')

    filtered_series = series.copy()
    filtered_series[(series == -1) & (group_sizes < threshold)] = 1
    return filtered_series

# 1. Генерация данных
np.random.seed(42)
time = pd.date_range(start="2023-01-01", periods=1000, freq="s")
current = np.sin(np.arange(1000)/10) + np.random.normal(0, 0.05, 1000)
temperature = 70 + 0.002*np.arange(1000) + np.random.normal(0, 0.3, 1000)

# 2. Добавляем три типа аномалий
current[200:220] = 0.1  
temperature[600:620] += np.linspace(0, 20, 20)  
current[400:430] += np.linspace(0, 0.8, 30)  
temperature[400:430] += np.linspace(0, 25, 30)

# 3. Создаем DataFrame
df = pd.DataFrame({
    "time": time,
    "current": current,
    "temperature": temperature
})

# 4. Добавим фичей

df["delta_current"] = df["current"].diff().fillna(0)
df["rolling_mean"] = df["current"].rolling(window=10).mean()
df["rolling_std"] = df["current"].rolling(window=10).std().fillna(0)

df["true_anomaly"] = 1
df.loc[200:220, "true_anomaly"] = -1  
df.loc[400:430, "true_anomaly"] = -1  
df.loc[600:620, "true_anomaly"] = -1  

# 5. Детекция аномалий с помощью отдельных моделей

# Модель для тока
current_features = ["current", "rolling_mean", "rolling_std"]
df = df.dropna(subset=current_features)
current_model = IsolationForest(contamination=0.04, random_state=42)
df["current_anomaly_raw"] = current_model.fit_predict(df[current_features])

# Модель для температуры
temp_features = ["temperature"]
temp_model = IsolationForest(contamination=0.05, random_state=42)
df["temp_anomaly_raw"] = temp_model.fit_predict(df[temp_features])

# Введем фичу, что только в случае трех подряд идущих аномалий они будут ими являться

df["current_anomaly"] = apply_consecutive_threshold(df["current_anomaly_raw"], threshold=3)
df["temp_anomaly"] = apply_consecutive_threshold(df["temp_anomaly_raw"], threshold=3)

# Объеденим предсказания
df["combined_anomaly"] = np.where((df["current_anomaly"] == -1) | (df["temp_anomaly"] == -1), -1, 1)

# 6. Визуализация
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
sns.lineplot(data=df, x="time", y="current", color="blue", label="Ток (А)")
sns.scatterplot(data=df[df["current_anomaly"] == -1], x="time", y="current",
                color="red", s=40, label="Аномалии тока")
plt.axvspan(df["time"][200], df["time"][220], color="red", alpha=0.1, label="Обрыв цепи")
plt.axvspan(df["time"][400], df["time"][430], color="purple", alpha=0.1, label="Перегрузка")
plt.title("Ток на обмотках двигателя с обнаруженными аномалиями (порог: 3 точки)")
plt.ylabel("Ток (А)")
plt.legend()

plt.subplot(3, 1, 2)
sns.lineplot(data=df, x="time", y="temperature", color="green", label="Температура (°C)")
sns.scatterplot(data=df[df["temp_anomaly"] == -1], x="time", y="temperature",
                color="red", s=40, label="Аномалии температуры")
plt.axvspan(df["time"][400], df["time"][430], color="purple", alpha=0.1, label="Перегрузка")
plt.axvspan(df["time"][600], df["time"][620], color="orange", alpha=0.1, label="Отказ охлаждения")
plt.title("Температура обмоток с обнаруженными аномалиями (порог: 3 точки)")
plt.ylabel("Температура (°C)")
plt.legend()

plt.tight_layout()
plt.show()

# 7. Проверка эффективности
print("=== Эффективность детекции для тока ===")
print(f"Обрыв цепи (200-220): {df[(df['current_anomaly'] == -1) & (df.index >= 200) & (df.index < 220)].shape[0]}/20")
print(f"Перегрузка (400-430): {df[(df['current_anomaly'] == -1) & (df.index >= 400) & (df.index < 430)].shape[0]}/30")

print("\n=== Эффективность детекции для температуры ===")
print(f"Перегрузка (400-430): {df[(df['temp_anomaly'] == -1) & (df.index >= 400) & (df.index < 430)].shape[0]}/30")
print(f"Отказ охлаждения (600-620): {df[(df['temp_anomaly'] == -1) & (df.index >= 600) & (df.index < 620)].shape[0]}/20")

print("\n=== Эффективность детекции ===")
print(f"Обрыв цепи (200-220): {df[(df['combined_anomaly'] == -1) & (df.index >= 200) & (df.index < 220)].shape[0]}/20")
print(f"Перегрузка (400-430): {df[(df['combined_anomaly'] == -1) & (df.index >= 400) & (df.index < 430)].shape[0]}/30")
print(f"Отказ охлаждения (600-620): {df[(df['combined_anomaly'] == -1) & (df.index >= 600) & (df.index < 620)].shape[0]}/20")

# 7. Расчет метрик precision
print("\nСчитаем точность:")
# Для тока
y_true_current = df["true_anomaly"].replace({1: 0, -1: 1})  # 1 - аномалия, 0 - норма
y_pred_current = df["current_anomaly"].replace({1: 0, -1: 1})
current_precision = precision_score(y_true_current, y_pred_current)
print(f"Точность для тока: {current_precision:.2f}")

# Для температуры
y_pred_temp = df["temp_anomaly"].replace({1: 0, -1: 1})
temp_precision = precision_score(y_true_current, y_pred_temp)
print(f"Точность для температуры: {temp_precision:.2f}")

# Для комбинированной модели
y_pred_combined = df["combined_anomaly"].replace({1: 0, -1: 1})
combined_precision = precision_score(y_true_current, y_pred_combined)
print(f"Объединенная точность: {combined_precision:.2f}")