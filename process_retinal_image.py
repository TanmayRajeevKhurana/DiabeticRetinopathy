# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:32:54 2023

@author: tanma
"""

from skimage import measure
import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def extract_ma(image):
    r, g, b = cv2.split(image)
    comp = 255 - g
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    histe = clahe.apply(comp)
    adjustImage = adjust_gamma(histe, gamma=3)
    comp = 255 - adjustImage
    J = adjust_gamma(comp, gamma=4)
    J = 255 - J
    J = adjust_gamma(J, gamma=4)

    K = np.ones((11,11), np.float32)
    L = cv2.filter2D(J, -1, K)

    ret3, thresh2 = cv2.threshold(L, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel2 = np.ones((9,9), np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    kernel3 = np.ones((7,7), np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    

    return opening

def process_retinal_image(filename):
    fundus = cv2.imread(filename)
    bloodvessel = extract_ma(fundus)
    return bloodvessel

processed_image = process_retinal_image("15_right.jpeg")
cv2.imwrite("output_image.jpg", processed_image)

