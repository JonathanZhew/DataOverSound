import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

y = np.array([0,2,4,5,7,78,6,4,4,4])
x = np.arange(len(y))
f = interpolate.interp1d(x, y)

xnew = np.arange(0, len(y)-1, 0.25)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, 'x')
plt.show()
