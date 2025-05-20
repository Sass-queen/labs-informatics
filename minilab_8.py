import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#  Генерация исходной функции (нелинейной)
def f(x):
    return np.sin(x) + 0.1 * x**2

#  Генерация данных с шумом
np.random.seed(17)
x = np.linspace(-5, 5, 100)
y = f(x) + np.random.uniform(-0.5, 0.5, size=len(x))
X = x.reshape(-1, 1)  # Преобразуем в 2D-массив для sklearn

#  Инициализация моделей
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

#  Обучение и предсказание
results = {}
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    results[name] = {"y_pred": y_pred, "mse": mse}

#  Визуализация
plt.figure(figsize=(17, 7))
for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 3, i)
    plt.scatter(x, y, color='blue', label='Данные', alpha=0.5)
    plt.plot(x, f(x), color='green', label='Исходная функция', linewidth=2)
    plt.plot(x, result["y_pred"], color='red', label=f'{name}', linewidth=2)
    plt.title(f"{name}\nMSE: {result['mse']:.4f}")
    plt.legend()

plt.tight_layout()
plt.show()

# Вывод MSE
print("\nСреднеквадратичные ошибки (MSE):")
for name, result in results.items():
    print(f"{name}: {result['mse']:.4f}")