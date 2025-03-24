
import matplotlib.pyplot as plt
import numpy as np

a = 2
b = 1
c = 1
x = np.linspace(0, 2 * np.pi, 200)
y = a+np.cos(x) * b*x*x + c*x*np.sin(x)
plt.plot(x, y)
plt.show()
