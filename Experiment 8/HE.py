import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Load image
path = r"E:\DSIP\VS Py\Experiment 8\beach.png"
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def histogram(image):
    return cv2.calcHist([image], [0], None, [256], [0, 256])

intensity = np.arange(256)

# Original image and histogram
hist = histogram(gray)

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.bar(intensity, hist.ravel())
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 255])
plt.show()


# Normalization
normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
normalized_hist = histogram(normalized)

plt.subplot(1, 2, 1)
plt.imshow(normalized, cmap="gray")
plt.title("Normalized Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.bar(intensity, normalized_hist.ravel())
plt.title("Normalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 255])
plt.show()


# Histogram equalization
equalized = cv2.equalizeHist(gray)
equalized_hist = histogram(equalized)

plt.subplot(1, 2, 1)
plt.imshow(equalized, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.bar(intensity, equalized_hist.ravel())
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 255])
plt.show()


# Histogram matching
source = gray
reference = gray

matched_image = match_histograms(source, reference)
matched_image = np.uint8(matched_image)

source_hist = histogram(source)
reference_hist = histogram(reference)
matched_hist = histogram(matched_image)

# Display images
plt.subplot(2, 3, 1)
plt.imshow(source, cmap="gray")
plt.title("Source Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(reference, cmap="gray")
plt.title("Reference Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(matched_image, cmap="gray")
plt.title("Matched Image")
plt.axis("off")

# Display histograms
plt.subplot(2, 3, 4)
plt.bar(intensity, source_hist.ravel())
plt.title("Source Histogram")

plt.subplot(2, 3, 5)
plt.bar(intensity, reference_hist.ravel())
plt.title("Reference Histogram")

plt.subplot(2, 3, 6)
plt.bar(intensity, matched_hist.ravel())
plt.title("Matched Histogram")

plt.tight_layout()
plt.show()