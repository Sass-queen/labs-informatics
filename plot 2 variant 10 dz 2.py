import matplotlib.pyplot as plt

with open('2.txt') as f:
    lines = f.readlines()
    x = [line.split()[0] for line in lines]
    y = [line.split()[1] for line in lines]

plt.plot(x, y, '-k', marker='o', markersize=3)
plt.show()

