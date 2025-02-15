with open('star.txt', 'r') as file:

    numbers = list(map(int, file.readline().strip().split()))


max_value = max(numbers)
max_index = numbers.index(max_value)

min_value = min(numbers[max_index + 1:]) if max_index + 1 < len(numbers) else None

if min_value is not None:
    min_index = numbers.index(min_value, max_index + 1)

    numbers[max_index], numbers[min_index] = numbers[min_index], numbers[max_index]

with open('result.txt', 'w') as file:

    file.write(' '.join(map(str, numbers)))

