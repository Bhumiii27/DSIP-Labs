import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_median_filter(image, kernel_size):
# Apply median filter to remove noise
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image

def apply_bilateral_filter(image, d, sigma_color, sigma_space):
# Apply bilateral filter to remove noise
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image

def apply_min_filter(image, kernel_size):
    # Apply minimum filter
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_image = cv2.erode(image, kernel)
    return filtered_image

def apply_max_filter(image, kernel_size):
    # Apply maximum filter
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_image = cv2.dilate(image, kernel)
    return filtered_image

def apply_erosion(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel)

def apply_dilation(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel)

# Load the input image
image_path = 'E:\DSIP\VS Py\Experiment 10\image.png'
input_image = cv2.imread(image_path)

# Apply median filter
median_filtered_image = apply_median_filter(input_image, kernel_size=7)
#Median removes noise and boundaries
# Apply bilateral filter
bilateral_filtered_image = apply_bilateral_filter(input_image, d=9, sigma_color=75, sigma_space=75)
# bilateral preserves boundaries while removing noise
# Apply minimum filter
min_filtered_image = apply_min_filter(input_image, kernel_size=7)
# minimum filter removes salt noise
# Apply maximum filter
max_filtered_image = apply_max_filter(input_image, kernel_size=7)
# maximum filter removes pepper noise
# Apply erosion
erosion_image = apply_erosion(input_image, 7)
# Apply dilation
dilation_image = apply_dilation(input_image, 7)

# Display all images together
plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.imshow(input_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(median_filtered_image)
plt.title("Median Filtered")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(bilateral_filtered_image)
plt.title("Bilateral Filtered")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(min_filtered_image)
plt.title("Minimum Filtered")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(max_filtered_image)
plt.title("Maximum Filtered")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(erosion_image)
plt.title("Erosion")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(dilation_image)
plt.title("Dilation")
plt.axis("off")

plt.tight_layout()
plt.show()