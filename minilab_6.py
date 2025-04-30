import numpy as np
import matplotlib.pyplot as plt


# Минимум: около x = 0.35 (найден численно)

def gradientDescend(func=lambda x: x ** 2, diffFunc=lambda x: 2 * x,
                    x0=3, speed=0.01, epochs=100):
    xList = []
    yList = []
    x = x0

    for i in range(epochs):
        xList.append(x)
        yList.append(func(x))
        x = x - speed * diffFunc(x)

    return xList, yList

func = lambda x: x ** 2 + np.exp(-x)
diffFunc = lambda x: 2 * x - np.exp(-x)

xList, yList = gradientDescend(func, diffFunc, x0=3, speed=0.1, epochs=50)

def plot_results(xList, yList, func):
    x_vals = np.linspace(-5, 5, 400)
    y_vals = func(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='Функция: $f(x) = x^2 + \exp(-x)$', color='blue')
    plt.scatter(xList, yList, color='red', label='Точки градиентного спуска')
    plt.scatter(xList[-1], yList[-1], color='green', s=100, label='Найденный минимум')



    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Градиентный спуск')
    plt.legend()
    plt.grid(True)
    plt.show()






plot_results(xList, yList, func)
print(f"Найденный минимум: x = {xList[-1]:.4f}, f(x) = {yList[-1]:.4f}")


def find_critical_speed(func, diffFunc, x0=3, epochs=100, tol=1e-2):
    low = 0.0
    high = 1.0
    target_min = -0.45

    for i in range(20):
        mid = (low + high) / 2
        xList, _ = gradientDescend(func, diffFunc, x0, mid, epochs)
        final_x = xList[-1]

        if abs(final_x - target_min) < tol:
            high = mid
        else:
            low = mid

    return (low + high) / 2


critical_speed = find_critical_speed(func, diffFunc)
print(f"Граничное значение (speed): {critical_speed:.4f}")