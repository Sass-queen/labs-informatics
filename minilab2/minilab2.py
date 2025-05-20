def process_array(input_file, output_file):

    with open(input_file, 'r') as f:
        data = f.read().strip()


    try:
        arr = list(map(int, data.split()))
    except ValueError:
        print("Ошибка: файл должен содержать только числа, разделенные пробелами")
        return


    negatives = [(i, x) for i, x in enumerate(arr) if x < 0]


    negatives_sorted = sorted(negatives, key=lambda item: item[1], reverse=True)


    for original_pos, (i, x) in zip([item[0] for item in negatives], negatives_sorted):
        arr[original_pos] = x

    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, arr)))
    print(f"Результат записан в файл {output_file}")



input_filename = 'input.txt'  # Файл с исходными данными
output_filename = 'output.txt'  # Файл для записи результата

with open(input_filename, 'w') as f:
    f.write("5 -3 2 -8 0 -187 7 4 -5")

process_array(input_filename, output_filename)