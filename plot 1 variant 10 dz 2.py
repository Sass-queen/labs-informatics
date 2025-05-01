import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
plt.axis([0,5,0,100])
plt.title('y=a+b*(x**2)+c*(x**(1*2))')
plt.xlabel('x')
plt.ylabel('y')
a=1
b=2
c=4
y=a+b*(x**2)+c*(x**(1*2))
plt.plot(x, y, '--r', marker='o', markersize=3)

plt.show()




