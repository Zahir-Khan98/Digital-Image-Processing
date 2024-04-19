'''
# Name- Zahir Khan (112202010)
# DIP Assignment2, Question 1.(b)
'''
# Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d as conv2


#Reading two Images
Im1=cv2.imread("image_1.JPG") 
Im2=cv2.imread("image_2.JPG")

# Checking dimension of the images for grayscale conversion
if(Im1.ndim==3): #checking color image or not
    Im1= cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY) #grayscale conversion
if(Im2.ndim==3):
    Im2= cv2.cvtColor(Im2, cv2.COLOR_BGR2GRAY)

#Taking Fourier Transform of Images and shifting 0 frequencies to center and taking log magnitudes
I1=np.fft.fft2(Im1) #performing 2 dim FFT
I2=np.fft.fft2(Im2)
Imf1=np.fft.fftshift(I1) # Shifting 0 frequencies at the center
Imf2=np.fft.fftshift(I2)
Mag1=np.log(abs(Imf1)) # taking log of magnitude values of FFT
Mag2=np.log(abs(Imf2))

#Displaying the DFT images and saving the local directory 'Results'
plt.imshow(Mag1,cmap='gray')
plt.title('DFT of image_1')
plt.show()

# Normalize the log magnitude values to the range [0, 255]
Mag1_normalized = ((Mag1 - Mag1.min()) / (Mag1.max() - Mag1.min()) * 255).astype(np.uint8)
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q1_(b)_DFT_image_1.JPG", Mag1_normalized )

plt.imshow(Mag2,cmap='gray')
plt.title('DFT of image_2')
plt.show()

# Normalizing the log magnitude values to the range [0, 255], without doing this image will be saved black in directory
Mag2_normalized = ((Mag2 - Mag2.min()) / (Mag2.max() - Mag2.min()) * 255).astype(np.uint8)
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q1_(b)_DFT_image_2.JPG", Mag2_normalized)
