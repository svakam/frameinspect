#%% Imports
import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

print("OpenCV:", cv.__version__)
print("NumPy:", np.__version__)
print("Imported")
 
#%% Load and inspect frame
img = cv.imread("./img/ai_baby_color.png") # load as default BGR uint8
if img is None:
    sys.exit("Could not read the image.")

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # for mpl
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

print("Shape (color):", img.shape)
print("Shape (gray):", gray.shape)
print("Dtype:", gray.dtype)
print("Min max pixels:", gray.min(), gray.max())

# plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(img_rgb); plt.title("Color")
plt.subplot(1, 2, 2); plt.imshow(gray, cmap="gray"); plt.title("Grayscale")
plt.show()
# cv.imshow("Display window", img)
# k = cv.waitKey(10000)


# %% Thresholding and edge detection
_, binary_mask = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("Grayscale")
plt.subplot(1, 3, 2); plt.imshow(binary_mask, cmap="gray"); plt.title("Threshold mask")
# plt.subplot(1, 3, 3); plt.imshow(edges, cmap="gray"); plt.title("Canny edges")

# %%
