with open('star.txt', 'r') as file:
    randomnumbers = list(map(int, file.readline().strip().split()))


max_element = max(randomnumbers)
min_element = min(randomnumbers)
max_index = randomnumbers.index(max_element)
min_index = randomnumbers.index(min_element)

if max_index < min_index:

    positive_sum = 0
    positive_count = 0
    for num in randomnumbers:
        if num > 0:
            positive_sum += num
            positive_count += 1
    if positive_count > 0:
        average = positive_sum / positive_count
    else:
        average = 0
else:

    negative_sum = 0
    negative_count = 0
    for num in randomnumbers:
        if num < 0:
            negative_sum += num
            negative_count += 1
    if negative_count > 0:
        average = negative_sum / negative_count
    else:
        average = 0


with open('result.txt', 'w') as file:
    file.write(f"Результат: {average}\n")

