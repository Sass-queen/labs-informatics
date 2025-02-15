import string
with open('input.txt', 'r', encoding='utf-8') as file:
    sentence = file.readline().strip()
words = sentence.split()
word_count = len(words)
punctuation_count = sum(1 for char in sentence if char in string.punctuation)

complexity = word_count + punctuation_count


with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(f'Количество слов: {word_count}\n')
    file.write(f'Количество знаков препинания: {punctuation_count}\n')
    file.write(f'Сложность предложения: {complexity}\n')

print("Сложность предложения определена и записана в output.txt.")
