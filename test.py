import numpy as np
import matplotlib.pyplot as plt

grid = np.load("runs\weight_0.01_slippery_False_\grid.npy")

plt.imshow(grid)
x = [0, 0]
y = [0, 4]
plt.plot(x, y, color='r', linewidth=10)

x = [0, 2]
y = [4, 4]
plt.plot(x, y, color='r', linewidth=10)

x = [2, 2]
y = [4, 3]
plt.plot(x, y, color='r', linewidth=10)

x = [2, 4]
y = [3, 3]
plt.plot(x, y, color='r', linewidth=10)

x = [4, 4]
y = [3, 5]
plt.plot(x, y, color='r', linewidth=10)

x = [4, 5]
y = [5, 5]
plt.plot(x, y, color='r', linewidth=10)

x = [5, 5]
y = [5, 7]
plt.plot(x, y, color='r', linewidth=10)

x = [5, 7]
y = [7, 7]
plt.plot(x, y, color='r', linewidth=10)

plt.savefig("without.png")
plt.show()

plt.imshow(grid)
x = [0, 0]
y = [0, 3]
plt.plot(x, y, color='r', linewidth=10)

x = [0, 4]
y = [3, 3]
plt.plot(x, y, color='r', linewidth=10)

x = [2, 4]
y = [3, 3]
plt.plot(x, y, color='r', linewidth=10)

x = [4, 4]
y = [3, 5]
plt.plot(x, y, color='r', linewidth=10)

x = [4, 5]
y = [5, 5]
plt.plot(x, y, color='r', linewidth=10)

x = [5, 5]
y = [5, 7]
plt.plot(x, y, color='r', linewidth=10)

x = [5, 7]
y = [7, 7]
plt.plot(x, y, color='r', linewidth=10)

plt.savefig("with.png")
plt.show()