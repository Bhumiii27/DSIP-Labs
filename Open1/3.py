import numpy as np
import matplotlib.pyplot as plt 
n3 = np.linspace(-2, 2, 400)
x3 = np.piecewise( n3,[n3 < -1,(n3 >= -1) & (n3 <= 1),
    (n3 > 1) & (n3 <= 2)],[-2,lambda n: 2*n, 2])
plt.figure(figsize=(5,3)) 
plt.plot(n3, x3, linewidth=2) 
plt.axhline(0, color='black') 
plt.axvline(0, color='black') 
plt.title("Signal 3") 
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()