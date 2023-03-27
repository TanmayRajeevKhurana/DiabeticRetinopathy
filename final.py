# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:05:37 2023

@author: tanma
"""

import numpy as np
import cv2
from skimage import measure

def adjust_gamma(image, gamma=1.0):
    # Create a lookup table with gamma correction applied to each pixel value
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Apply the lookup table to the image using cv2 LUT function
    return cv2.LUT(image, table)

def extract_ma(image):
    # Split the image into its red, green, and blue channels
    r, g, b = cv2.split(image)
    
    # Merge blue and green channels
    merged_img = cv2.merge((b, g, g))
    
    cv2.imwrite("Green Channel.jpg", g)
    cv2.imwrite("Blue Channel.jpg", b)
    cv2.imwrite("Red Channel.jpg", r)
    
    cv2.imwrite("Merge.jpg", merged_img)
    
    # Invert the green/BLUE channel
    comp = 255 - b
    cv2.imwrite("Inverted GC.jpg", comp)
    # Apply contrast limited adaptive histogram equalization (CLAHE) to the inverted green channel
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    histe = clahe.apply(comp)
    cv2.imwrite("CLAHE.jpg", histe)
    # Adjust the gamma of the CLAHE image to enhance the contrast of the microaneurysms
    adjustImage = adjust_gamma(histe, gamma=3)
    cv2.imwrite("Adjusted Image.jpg", adjustImage)
    # Invert the adjusted image
    comp = 255 - adjustImage
    cv2.imwrite("Invert Adjusted Image.jpg", comp)
    # Adjust the gamma of the inverted image
    J = adjust_gamma(comp, gamma=4)
    # Invert the gamma adjusted image
    J = 255 - J
    # Adjust the gamma of the inverted gamma adjusted image
    J = adjust_gamma(J, gamma=4)
    cv2.imwrite("Invert Adjusted Gamma Image.jpg", J)

    # Define a 11x11 kernel for filtering
    K = np.ones((11,11), np.float32)
    # Apply a 2D filter to the gamma adjusted image using the defined kernel
    L = cv2.filter2D(J, -1, K)

    # Apply Otsu's thresholding to the filtered image
    ret3, thresh2 = cv2.threshold(L, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("Otsu Thresh.jpg", thresh2)

    # Define a 9x9 kernel for morphological transformation
    kernel2 = np.ones((9,9), np.uint8)
    # Apply top hat morphological transformation to the thresholded image using the defined kernel
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    cv2.imwrite("Top Hat.jpg", tophat)
    # Define a 7x7 kernel for morphological transformation
    kernel3 = np.ones((7,7), np.uint8)
    # Apply opening morphological transformation to the top hat transformed image using the defined kernel
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    cv2.imwrite("Opening MT.jpg", opening)

    # Return the resulting image
    return opening

# Load the image
image = cv2.imread("02_test.jpg")
#image = cv2.resize(image, (512, 512))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract the microaneurysms from the image
ma = extract_ma(image)

# Label the connected components in the extracted microaneurysm image
labels = measure.label(ma, connectivity=2)

# Print the number of detected microaneurysms
print("Number of microaneurysms detected: ", labels.max())

# Show the original image and the extracted microaneurysm image side by side
cv2.imwrite("Original Image.jpg", gray)
cv2.imwrite("Extracted Microaneurysm Image.jpg", ma)
cv2.waitKey(0)
