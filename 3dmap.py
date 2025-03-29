import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(x,y)
def f(x, y):
    return x**2 + 100 * y**2  # a simple quadratic function

# Define the domain for x and y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)  # create a grid of (x,y) pairs

# Compute Z for each (x,y)
Z = f(X, Y)

# 3D Surface Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.title('3D Surface Plot of f(x,y) = x^2 + 100 y^2')
plt.show()

# Contour Plot
plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot of f(x,y) = x^2 + 100 y^2')
plt.colorbar()
plt.show()
