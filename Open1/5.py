import numpy as np
import matplotlib.pyplot as plt 
n5 = np.arange(-3, 4)
delta = lambda n: np.where(n == 0, 1, 0)
x5 = delta(n5) + 3*delta(n5-1) + 5*delta(n5+1) 
plt.figure(figsize=(5,3))
plt.stem(n5, x5) 
plt.title("Signal 5") 
plt.xlabel("n")
plt.ylabel("x[n]") 
plt.grid(True) 
plt.show()
