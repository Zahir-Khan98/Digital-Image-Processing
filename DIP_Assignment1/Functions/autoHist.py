#Author: ZAHIR KHAN, 112202010

import cv2
import numpy as np

def autoHist(rawImage):
    # doing Contrast Stretching

    m_intnsty = 0 #minimum intensity
    M_intnsty = 255 #maximum intensity
    strtchdIMG = (rawImage - m_intnsty) / (M_intnsty - m_intnsty)  # Normalize pixel values
    strtchdIMG = np.clip(strtchdIMG, 0, 1)  # Clip values to the range [0, 1]
    strtchdIMG = (strtchdIMG * 255).astype(np.uint8)  # Scale back to 8-bit range

    # doing Histogram Equalization

    # Calculating histogram and cumulative histogram
    h = cv2.calcHist([strtchdIMG], [0], None, [256], [0, 256])
    h = h / (strtchdIMG.shape[0] * strtchdIMG.shape[1])  # Normalized histogram
    C = np.cumsum(h)  # cumulative histogram
    # Mapping pixel values to their equalized values
    equalized_image = C[strtchdIMG]  
    equalized_image = np.uint8(255 * equalized_image)  # Scaling to 8-bit range

    return equalized_image


