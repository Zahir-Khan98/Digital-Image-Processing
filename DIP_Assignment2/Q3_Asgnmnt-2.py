'''
#Name- Zahir Khan (112202010)
#DIP Assignment2, Question-3
'''
#Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import convolve2d as conv2


#Reading two Images
Im1=cv2.imread("blur1.JPG") 
Im2=cv2.imread("blur2.JPG")

# Checking dimension of the images for grayscale conversion
if(Im1.ndim==3): #checking color image or not
    Im1= cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY) #grayscale conversion
if(Im2.ndim==3):
    Im2= cv2.cvtColor(Im2, cv2.COLOR_BGR2GRAY)

#Resizing the image
Im1 = cv2.resize(Im1,(256,256))
Im2 = cv2.resize(Im2,(256,256))

#Taking 2D Fourier Transform of Images
F1=np.fft.fftshift(np.fft.fft2(Im1))
F2=np.fft.fftshift(np.fft.fft2(Im2))

#Taking means of original image's DFT magnitude
mean_total1 = np.mean(np.abs(F1))
mean_total2 = np.mean(np.abs(F2))

#creating 60x60 Zero square center in the Frequency domain of images (DFT images)
size=30
F1[128 - size:128 + size, 128 - size:128 + size] = 0
F2[128 - size:128 + size, 128 - size:128 + size] = 0

#Taking means of high frequency DFT image's magnitude
mean_high1 = np.sum(np.abs(F1))/(256*256-60*60)
mean_high2 = np.sum(np.abs(F2))/(256*256-60*60)

#Result
m1=(1-(mean_high1/mean_total1))*100 #blur percentage values
m2=(1-(mean_high2/mean_total2))*100

print("Blur percentage of image1=",m1,'%')          
print("Blur percentage of image2=",m2,'%')

#Diplaying results and saving the images in local directory
epsilon = 1e-10  # Small constant to avoid taking the logarithm of zero
Mag1=np.log(abs(F1) + epsilon) 
Mag2=np.log(abs(F2) + epsilon) 
plt.imshow(Mag1,cmap='gray')
plt.title('Low frequency removed (central zero square) DFT of first image')
plt.show()
plt.imshow(Mag2,cmap='gray')
plt.title('Low frequency removed (central zero square) DFT of second image')
plt.show()
# Normalizing the log magnitude values to the range [0, 255], without doing this image will be saved black in directory
Mag1_normalized = ((Mag1 - Mag1.min()) / (Mag1.max() - Mag1.min()) * 255).astype(np.uint8)
Mag2_normalized = ((Mag2 - Mag2.min()) / (Mag2.max() - Mag2.min()) * 255).astype(np.uint8)
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q3 DFT of Blur1.JPG", Mag1_normalized)
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q3 DFT of Blur2.JPG", Mag2_normalized)

