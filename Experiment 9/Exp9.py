import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = cv2.imread('E:\DSIP\VS Py\Experiment 9\image.png', 0)
#plt.imshow(img)
#Fig size
plt.figure(figsize=(12, 5))
#plt.show()
plt.imshow(img)
img.shape

weighted_kernel=np.asarray([
    [1,2,1], 
    [1,2,1], 
    [1,2,1]], 
    dtype=np.float32)

weighted_kernel=weighted_kernel/10
weighted_mean=cv2.filter2D(img, -1, weighted_kernel)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(weighted_mean, cv2.COLOR_BGR2RGB))
plt.title('Weighted Mean Filter')
plt.axis('off')
plt.show()
