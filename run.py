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
img = cv.imread("./img/ai_baby_color_emojis.png") # load as default BGR uint8
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


#%% Global thresholding vs. adaptive mean vs. Gaussian vs. Otsu vs. Canny
_, binary_mask = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)
adaptive_mean_mask = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY, 11, 2)
adaptive_gaussian_mask = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY, 11, 2)
# blur = cv.GaussianBlur(gray, (5,5), 0)
_, otsu_gaussian_mask = cv.threshold(gray, 0, 255, \
                                  cv.THRESH_BINARY + cv.THRESH_OTSU)
edges = cv.Canny(adaptive_gaussian_mask, threshold1=125, threshold2=1100)


# plot all
titles = ["Grayscale", "Threshold Mask", "Otsu Thresholding", \
          "Adaptive Mean Thresholding", "Adaptive Gaussian Thresholding",\
              "Canny"]
imgs = [gray, binary_mask, otsu_gaussian_mask, \
        adaptive_mean_mask, adaptive_gaussian_mask, edges]

plt.figure(figsize=(12, 9))

for i in range(len(titles)):
    plt.subplot(2, 3, i + 1); plt.imshow(imgs[i], "gray")
    plt.title(titles[i])

plt.show()

#%% Contour detection and overlaying on template image
# set up contours, passed in Canny object
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contours. Displaying first 10")

# copy rgb image for setting up overlay and draw over
img_contours = img_rgb.copy()
cv.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

for c in contours[:10]:
    area = cv.contourArea(c)
    x, y, w, h = cv.boundingRect(c)
    print(f"Area: {area:.1f}  Bounding box: x={x}, y={y}, w={w}, h={h}")

plt.imshow(img_contours)
plt.title(f'{len(contours)} contours; pre-morph')
plt.show()

#%% Applying morphological updates
# set up kernel typed to uint8
kernel = np.ones((1,1), np.uint8)
kernel_2 = np.ones((2,2), np.uint8)
kernel_3 = np.ones((3,3), np.uint8)
kernel_4 = np.ones((4,4), np.uint8)

# kernels
kernels = [kernel, kernel_2, kernel_3, kernel_4]
num_kernels = len(kernels)

plt.figure(figsize=(16,12)) 

# plot: for each morph. transf., try it with a different kernel
for i in range(num_kernels):

    titles = [f"Open morph with {i}x{i} kernel", f"Closed morph with {i}x{i} kernel", f"Gradient morph with {i}x{i} kernel"]
    num_titles = len(titles)

    for j in range(num_titles):
        # titles
        k = j + 1
        print(i, j)
        
        opening_morph = cv.morphologyEx(edges, cv.MORPH_OPEN, kernels[i])
        closed_morph = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernels[i])
        gradient_morph = cv.morphologyEx(edges, cv.MORPH_GRADIENT, kernels[i])

        plt.subplot(num_kernels, num_titles, i + j + 1)
        plt.imshow(opening_morph)
    
    plt.title(titles[i])

plt.show()


# %%
