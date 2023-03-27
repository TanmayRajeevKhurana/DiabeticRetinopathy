# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 00:01:20 2023

@author: tanma
"""

import cv2
from matplotlib import pyplot as plt

# Load image in grayscale mode
img = cv2.imread('T:\Btech\Sem VI\dip\proo/green channel.jpg', 0)
comp = 255 - img

# Calculate histogram before applying CLAHE
hist_before = cv2.calcHist([comp], [0], None, [256], [0, 256])

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img)

# Calculate histogram after applying CLAHE
hist_after = cv2.calcHist([img_clahe], [0], None, [256], [0, 256])

# Plot histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.hist(img.ravel(), 256, [0,256], color='gray')
ax1.set_title('Before CLAHE')
ax1.set_xlabel('Intensity')
ax1.set_ylabel('Frequency')

ax2.hist(img_clahe.ravel(), 256, [0,256], color='gray')
ax2.set_title('After CLAHE')
ax2.set_xlabel('Intensity')
ax2.set_ylabel('Frequency')

plt.show()
