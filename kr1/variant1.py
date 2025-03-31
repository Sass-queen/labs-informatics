import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Загрузка датасета Iris
iris = load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=45)
# Создание модели KNN с K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Обучение модели на обучающей выборке
knn.fit(x_train, y_train)
# Предсказание классов на тестовой выборке
y_pred = knn.predict(x_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")


# Визуализация
def visualizer(x_train, y_train, x_test, y_test, y_predict):
    plt.figure(figsize=(13, 8))

    # Обучающие точки
    for i, point in enumerate(x_train):
        if y_train[i] == 0:
            plt.scatter(point[0], point[1], color='lightblue', marker='o', label='Класс 0' if i == 0 else "")
        else:
            plt.scatter(point[0], point[1], color='lightblue', marker='x', label='Класс 1' if i == 0 else "")

    # Тестовые точки
    for i, point in enumerate(x_test):
        if y_test[i] == y_predict[i]:  # Верно классифицированные
            if y_test[i] == 0:
                plt.scatter(point[0], point[1], color='darkgreen', marker='o',label='Класс o верно' if i == 0 else "")
            else:
                plt.scatter(point[0], point[1], color='darkgreen', marker='x', label='Класс 1 верно' if i == 0 else "")
        else:  # Неверно классифицированные
            if y_test[i] == 0:
                plt.scatter(point[0], point[1],  color='red', marker='o',label='Класс o неверно' if i == 0 else "")
            else:
                plt.scatter(point[0], point[1],  color='red', marker='x' ,label='Класс 1 неверно' if i == 0 else "")

    plt.title('Классификация k-точек')
    plt.xlabel('X-ось')
    plt.ylabel('Y-ось')
    plt.legend()
    plt.grid()
    plt.show()


visualizer(x_train, y_train, x_test, y_test, y_pred)







