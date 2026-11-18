
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image in grayscale
img = cv2.imread('image.png', 0)
# Check if image is loaded
if img is None:
    print("Error: Image not found. Check the image path.")
    exit()

# 2. Apply Fourier Transform
imageF = np.fft.fftshift(np.fft.fft2(img))
M, N = img.shape

# 3. Create frequency distance matrix
u, v = np.meshgrid(np.arange(N) - N//2,
                   np.arange(M) - M//2)

D = np.sqrt(u**2 + v**2)

# 4. Create Low-Pass and High-Pass Filters
LPF = D < 30
HPF = D >= 30

# 5. Apply Low-Pass Filter (Smoothening)
smooth = np.abs(
    np.fft.ifft2(np.fft.ifftshift(imageF * LPF))
)

# 6. Apply High-Pass Filter (Edge Enhancement)
high_pass = np.abs(
    np.fft.ifft2(np.fft.ifftshift(imageF * HPF))
)

# 7. Sharpen the original image
sharp = np.clip(
    img.astype(float) + high_pass, 0, 255
)

# 8. Display the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(smooth, cmap='gray')
plt.title('LPF - Smoothening')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharp, cmap='gray')
plt.title('HPF - Sharpening')
plt.axis('off')

plt.tight_layout()
plt.show()
