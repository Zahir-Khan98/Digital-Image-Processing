'''
# Name- Zahir Khan (112202010)
# DIP Assignment2, Question-2 for Block images
'''
# Importing necessary libraries
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import convolve2d as conv2

#Reading the Block image
Img=cv2.imread("Blocks.jpg")  

# Checking dimension of the images for grayscale conversion
if(Img.ndim==3): #checking color image or not
    Img= cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY) #grayscale conversion
    
#converting to double to prevent quantization issues
Img = cv2.resize(Img,(256,256))
Img_norm=cv2.normalize(Img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#Defining the Sobel filter kernel (spatial fiter)
hy=[[-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]]
hy=np.array(hy)
hx=[[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
hx=np.array(hx)

# Filtering in all direction
gx=conv2(Img_norm,hx,mode='same')
gy=conv2(Img_norm,hy,mode='same')

#Diplaying resultant filtered image
G=np.sqrt(gx**2+gy**2)
plt.imshow(G,cmap='gray')
plt.title('Edges')
plt.show()

# Normalizing the magnitude values to the range [0, 255], without doing this image will be saved black in directory
G_normalized = ((G - G.min()) / (G.max() - G.min()) * 255).astype(np.uint8)
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q2 Edge of Block Image.JPG", G_normalized)