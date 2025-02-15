input_string = "Васильчиков, Загогулька, Небаба, Смирнов, Чистяков, Тараканчиков, Дубинка"
surnames = [surname.strip() for surname in input_string.split(',')]
surnames.sort()
sorted_surnames = ', '.join(surnames)
print("Отсортированные фамилии:", sorted_surnames)