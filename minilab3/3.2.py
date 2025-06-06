import matplotlib.pyplot as plt
import numpy as np

file = open('данные для графика 2.txt', 'r')
xy = []
for i in file.readlines():
    xy.append(i.split())
file.close()
x = list(map(float, xy[0]))
y = list(map(float, xy[-1]))
colors = np.random.uniform(15, 80, len(x))
plt.scatter(x, y, s=5, c=colors, vmin=0, vmax=100)
plt.show()
