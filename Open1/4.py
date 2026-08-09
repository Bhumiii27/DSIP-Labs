import numpy as np
import matplotlib.pyplot as plt 
n4 = np.arange(-2, 11)
u = lambda n: np.where(n >= 0, 1, 0) 
x4 = 4*u(n4) - u(n4-3) - 5*u(n4-7) 
plt.figure(figsize=(5,3))
plt.stem(n4, x4) 
plt.title("Signal 4") 
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()