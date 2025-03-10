import matplotlib.pyplot as plt
import random
#инициализация исхданных
x = []
y = []
countpoints1 = 50
countpoints2 = 50
class2_p = []
for i in range(50):
    x.append([random.uniform(3, 10), random.uniform(3, 10)])
    y.append(0)
class1_points = x
for i in range(50):
    class2_p.append([random.uniform(7, 13), random.uniform(7, 13)])
    y.append(1)
x = x + class2_p

#разбивка на обучающие и тестовые выборки
def split_datatrainandtest(x, y, p=0.8):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    k = []
    for i in range(80):
        index = random.randint(0, len(x) - 1)
        if index not in k:
            x_train.append(x[index])
            y_train.append(y[index])
            k.append(index)
            x.remove(x[index])
            y.remove(y[index])
    x_test = x
    y_test = y
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split_datatrainandtest(x, y)


# Реализация метода k ближайших соседей
def fit(x_train, y_train, x_test, n=3):
    y_predict = []
    for test_point in x_test:
        distances = []
        i = 0
        for train_point in x_train:
            distance = ((train_point[0] - test_point[0]) ** 2 + (train_point[1] - test_point[1]) ** 2) ** (0.5)
            distances.append((distance, y_train[i]))
            i += 1

        # Сортировка по расстоянию
        distances.sort(key=lambda x: x[0])
        # Получение меток классов ближайших соседей
        neighbors = [distances[i][1] for i in range(n)]
        # Определение наиболее частого класса
        y_predict.append(max(set(neighbors), key=neighbors.count))

    return y_predict


#  accuracy
def compaccuracy(y_test, y_predict):
    correct = sum(1 for yt, yp in zip(y_test, y_predict) if yt == yp)
    accuracy = correct / len(y_test)
    return accuracy


# Классификация точек из тестовой выборки
k = 3
y_predict = fit(x_train, y_train, x_test, k)

# Оценка точности работы алгоритма
accuracy = compaccuracy(y_test, y_predict)
print(f'Accuracy: {accuracy * 100:.2f}%')

# визуализация
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


visualizer(x_train, y_train, x_test, y_test, y_predict)