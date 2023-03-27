# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:34:33 2023

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
    # Invert the green channel
    comp = 255 - g
    # Apply contrast limited adaptive histogram equalization (CLAHE) to the inverted green channel
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    histe = clahe.apply(comp)
    # Adjust the gamma of the CLAHE image to enhance the contrast of the microaneurysms
    adjustImage = adjust_gamma(histe, gamma=3)
    # Invert the adjusted image
    comp = 255 - adjustImage
    # Adjust the gamma of the inverted image
    J = adjust_gamma(comp, gamma=4)
    # Invert the gamma adjusted image
    J = 255 - J
    # Adjust the gamma of the inverted gamma adjusted image
    J = adjust_gamma(J, gamma=4)

    # Define a 21x21 kernel for filtering
    K = np.ones((21,21), np.float32)
    # Apply a 2D filter to the gamma adjusted image using the defined kernel
    L = cv2.filter2D(J, -1, K)

    # Apply Otsu's thresholding to the filtered image
    ret3, thresh2 = cv2.threshold(L, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Define a 15x15 kernel for morphological transformation
    kernel2 = np.ones((15,15), np.uint8)
    # Apply top hat morphological transformation to the thresholded image using the defined kernel
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    # Define a 11x11 kernel for morphological transformation
    kernel3 = np.ones((11,11), np.uint8)
    # Apply opening morphological transformation to the top hat transformed image using the defined kernel
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)

    # Return the resulting image
    return opening

# Load the image
image = cv2.imread("01_test.jpeg")
image = cv2.resize(image, (512, 512))

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
cv2.imshow("Image",ma)
cv2.imwrite("Extracted Microaneurysm Image.jpg", ma)
cv2.waitKey(0)
