#Author: ZAHIR KHAN, 112202010

import cv2
import numpy as np


def grain(rawImage):
    #creating digital negative of original image
    digital_negative = np.max(rawImage) - rawImage

    # Defining the weight (alpha)
    alpha = 0.3

    ImgDiff=rawImage + alpha*digital_negative #doing image addition/subtraction with the weight alpha of digital negative
    ImgDiff[ImgDiff<0]=0 #converting all intensity to zero which are less than zero
    

    return ImgDiff
    


