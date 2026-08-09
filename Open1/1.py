import numpy as np
import matplotlib.pyplot as plt 
n = np.arange(-3, 7)
x1 = np.where(n % 2 == 0, 2, 3)
plt.figure(figsize=(5,3)) 
plt.stem(n, x1)
plt.title("Signal 1") 
plt.xlabel("n")
plt.ylabel("x[n]") 
plt.grid(True)
plt.show()
