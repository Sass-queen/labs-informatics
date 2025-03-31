array = [3, -1, -4, 0, -2, 5, -3]
def sort_negatives(arr):
    # Извлекаем отрицательные элементы
    negatives = [x for x in arr if x < 0]

    # Сортируем их по убыванию
    negatives.sort(reverse=True)

    # Индекс для отслеживания текущего отрицательного элемента
    neg_index = 0

    # Создаем новый массив для результата
    result = []

    for x in arr:
        if x < 0:
            result.append(negatives[neg_index])
            neg_index += 1
        else:
            result.append(x)

    return result

sorted_array = sort_negatives(array)
print(sorted_array)
