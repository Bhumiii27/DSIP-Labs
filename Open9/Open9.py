import cv2
import numpy as np
import matplotlib.pyplot as plt

# Change the image path here
image_path = "E:\\DSIP\\VS Py\\Open9\\image5.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    # Convert original image for display
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Gaussian Smoothing
    gaussian_kernel = cv2.getGaussianKernel(5, 1.5)
    gaussian_kernel = np.outer(gaussian_kernel, gaussian_kernel)
    gaussian = cv2.filter2D(image, -1, gaussian_kernel)

    # 2. Mean Filter
    mean_kernel = np.ones((5, 5), dtype=np.float32) / 25
    mean = cv2.filter2D(image, -1, mean_kernel)

    # 3. Median Filter
    median = cv2.medianBlur(image, 5)

    # 4. Sharpening Filter
    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(image, -1, sharpening_kernel)

    # 5. High-Pass Sharpening
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    high_pass_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    high_pass = cv2.filter2D(blurred, -1, high_pass_kernel)

    # Convert all results to RGB
    images = [
        rgb,
        cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(mean, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(median, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(high_pass, cv2.COLOR_BGR2RGB)
    ]

    titles = [
        "Original",
        "Gaussian Smoothing",
        "Mean Filter",
        "Median Filter",
        "Sharpening",
        "High-Pass Sharpening"
    ]

    # Display all results
    plt.figure(figsize=(15, 10))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()