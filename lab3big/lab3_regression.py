import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
from collections import Counter


# 1. Генерация исходных данных
def generate_data(points_count1=50, points_count2=50):
    # Границы для первого класса
    xMin1, xMax1 = 1, 3
    yMin1, yMax1 = 1, 3

    # Границы для второго класса
    xMin2, xMax2 = 3, 5
    yMin2, yMax2 = 3, 5

    # Генерация точек для первого класса
    class1 = [
        [random.uniform(xMin1, xMax1), random.uniform(yMin1, yMax1)]
        for _ in range(points_count1)
    ]

    # Генерация точек для второго класса
    class2 = [
        [random.uniform(xMin2, xMax2), random.uniform(yMin2, yMax2)]
        for _ in range(points_count2)
    ]

    # Объединение данных
    x = class1 + class2
    y = [0] * points_count1 + [1] * points_count2

    return x, y


# 2. Разделение на обучающую и тестовую выборки
def train_test_split(x, y, p=0.8):
    data = list(zip(x, y))
    random.shuffle(data)
    split_idx = int(len(data) * p)

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    x_train = [point for point, label in train_data]
    y_train = [label for point, label in train_data]
    x_test = [point for point, label in test_data]
    y_test = [label for point, label in test_data]

    return x_train, x_test, y_train, y_test


# 3. Реализация KNN
def euclidean_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def fit(x_train, y_train, x_test, k=3):
    y_predict = []
    for test_point in x_test:
        # Вычисление расстояний до всех точек обучающей выборки
        distances = [
            (euclidean_distance(test_point, train_point), label)
            for train_point, label in zip(x_train, y_train)
        ]
        # Сортировка по расстоянию и выбор k ближайших
        k_nearest = sorted(distances)[:k]
        k_nearest_labels = [label for _, label in k_nearest]
        # Голосование большинством
        most_common = Counter(k_nearest_labels).most_common(1)
        y_predict.append(most_common[0][0])
    return y_predict


# 4. Метрика accuracy
def compute_accuracy(y_test, y_predict):
    correct = sum(1 for true, pred in zip(y_test, y_predict) if true == pred)
    return correct / len(y_test)


# 5. Визуализация
def plot_results(x_train, y_train, x_test, y_test, y_predict):
    plt.figure(figsize=(12, 5))

    # Обучающие данные
    plt.subplot(1, 2, 1)
    class0_train = [point for point, label in zip(x_train, y_train) if label == 0]
    class1_train = [point for point, label in zip(x_train, y_train) if label == 1]

    plt.scatter([p[0] for p in class0_train], [p[1] for p in class0_train],
                color='blue', label='Class 0 (train)')
    plt.scatter([p[0] for p in class1_train], [p[1] for p in class1_train],
                color='red', label='Class 1 (train)')
    plt.title('Training Data')
    plt.legend()

    # Тестовые данные с предсказаниями
    plt.subplot(1, 2, 2)
    class0_test = [point for point, label in zip(x_test, y_test) if label == 0]
    class1_test = [point for point, label in zip(x_test, y_test) if label == 1]
    class0_pred = [point for point, label in zip(x_test, y_predict) if label == 0]
    class1_pred = [point for point, label in zip(x_test, y_predict) if label == 1]

    plt.scatter([p[0] for p in class0_test], [p[1] for p in class0_test],
                color='blue', marker='o', label='Class 0 (true)')
    plt.scatter([p[0] for p in class1_test], [p[1] for p in class1_test],
                color='red', marker='o', label='Class 1 (true)')
    plt.scatter([p[0] for p in class0_pred], [p[1] for p in class0_pred],
                color='cyan', marker='x', s=100, label='Class 0 (pred)')
    plt.scatter([p[0] for p in class1_pred], [p[1] for p in class1_pred],
                color='orange', marker='x', s=100, label='Class 1 (pred)')
    plt.title('Test Data with Predictions')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Основная программа
if __name__ == "__main__":
    # Генерация данных
    x, y = generate_data()

    # Разделение на train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Обучение и предсказание
    y_predict = fit(x_train, y_train, x_test, k=3)

    # Оценка точности
    accuracy = compute_accuracy(y_test, y_predict)
    print(f"Accuracy: {accuracy:.2f}")

    # Визуализация
    plot_results(x_train, y_train, x_test, y_test, y_predict)
