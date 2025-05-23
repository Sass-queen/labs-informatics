import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random


# 1. Исходная показательная функция y = a*b^x + c
def true_function(x, a, b, c):
    return a * (b ** x) + c


# 2. Генерация данных с шумом
x_min = 0
x_max = 5
points = 20
noise_scale = 0.5

# Истинные параметры
true_a = 1.5
true_b = 2.0
true_c = 0.5

x = np.linspace(x_min, x_max, points)
y = true_function(x, true_a, true_b, true_c) + np.array(
    [random.uniform(-noise_scale, noise_scale) for _ in range(points)])


# 3. Функции для вычисления градиентов
def get_da(x, y, a, b, c):
    return (2 / len(x)) * np.sum((a * (b ** x) + c - y) * (b ** x))


def get_db(x, y, a, b, c):
    return (2 / len(x)) * np.sum((a * (b ** x) + c - y) * (a * x * (b ** (x - 1))))


def get_dc(x, y, a, b, c):
    return (2 / len(x)) * np.sum(a * (b ** x) + c - y)


# 4. Реализация градиентного спуска
def fit(x, y, speed, epochs, a0, b0, c0):
    a = a0
    b = b0
    c = c0
    history = {'a': [a], 'b': [b], 'c': [c], 'mse': [np.mean((y - true_function(x, a, b, c)) ** 2)]}

    for _ in range(epochs):
        da = get_da(x, y, a, b, c)
        db = get_db(x, y, a, b, c)
        dc = get_dc(x, y, a, b, c)

        a -= speed * da
        b = max(0.1, b - speed * db)  # Ограничение b > 0
        c -= speed * dc

        history['a'].append(a)
        history['b'].append(b)
        history['c'].append(c)
        history['mse'].append(np.mean((y - true_function(x, a, b, c)) ** 2))

    return history


# 5. Параметры обучения
learning_rate = 0.001
epochs = 500
initial_a = random.uniform(0.5, 3)
initial_b = random.uniform(1, 3)
initial_c = random.uniform(-1, 1)

# Обучаем модель
history = fit(x, y, learning_rate, epochs, initial_a, initial_b, initial_c)

# 6. Визуализация с одним графиком
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)

# График данных и кривой регрессии
scatter = ax.scatter(x, y, color='red', label='Исходные данные')
curve, = ax.plot(x, true_function(x, history['a'][0], history['b'][0], history['c'][0]),
                 'b-', label='Регрессия', linewidth=2)
ax.set_xlim(x_min, x_max)
ax.set_ylim(min(y) - 1, max(y) + 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Показательная регрессия (a={history["a"][0]:.2f}, b={history["b"][0]:.2f}, c={history["c"][0]:.2f})')
ax.legend()
ax.grid(True)

# Ползунок для выбора эпохи
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, 'Эпоха', 0, epochs, valinit=0, valstep=1)


# Функция обновления графика
def update(val):
    epoch = int(slider.val)
    current_a = history['a'][epoch]
    current_b = history['b'][epoch]
    current_c = history['c'][epoch]
    current_mse = history['mse'][epoch]

    curve.set_ydata(true_function(x, current_a, current_b, current_c))
    ax.set_title(
        f'Показательная регрессия (a={current_a:.2f}, b={current_b:.2f}, c={current_c:.2f})\nMSE: {current_mse:.4f}')
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()