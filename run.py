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
img = cv.imread("./img/ai_baby_color_straight.png") # load as default BGR uint8
if img is None:
    sys.exit("Could not read the image.")

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # for mpl
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

print("Shape (color):", img.shape)
print("Shape (gray):", gray.shape)
print("Dtype:", gray.dtype)
print("Min max pixels:", gray.min(), gray.max())
print("Blue @ (100,100):", img[100,100,0])
print("Green @ (100,100):", img[100,100,1])
print("Red @ (100,100):", img[100,100,2])

# plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(img_rgb); plt.title("Color")
plt.subplot(1, 2, 2); plt.imshow(gray, cmap="gray"); plt.title("Grayscale")
plt.show()
# cv.imshow("Display window", img)
# k = cv.waitKey(10000)


# %% Thresholding vs. edge detection
_, binary_mask = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)
edges = cv.Canny(gray, threshold1=50, threshold2=100)

plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("Grayscale")
plt.subplot(1, 3, 2); plt.imshow(binary_mask, cmap="gray"); plt.title("Threshold mask")
plt.subplot(1, 3, 3); plt.imshow(edges, cmap="gray"); plt.title("Canny edges")
plt.show()

# %%
