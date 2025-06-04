import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,1)
y = np.arange(-5,5,1)
X, Y = np.meshgrid(x,y)

dy = X - Y
dx = X + 2*Y
norm = np.sqrt(X**2 + Y**2)
dyu = dy/norm
dxu = dx/norm

plt.quiver(X,Y,dxu,dyu,color="purple")
plt.show()