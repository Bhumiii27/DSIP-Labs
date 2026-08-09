import numpy as np
import matplotlib.pyplot as plt 
n2 = np.arange(0, 6)
x2 = [1, 2, 3, 3, 2, 1]
plt.figure(figsize=(5,3)) 
plt.step(n2, x2, where='post') 
plt.title("Signal 2") 
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()
