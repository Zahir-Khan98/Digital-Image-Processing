''''
#Name- Zahir Khan (112202010)
#DIP Assignment2, Question 1.(c)
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

#Computing the 2D FT for image 1 and 2  and shifting the zero frequencies to the center
FIm1=np.fft.fftshift(np.fft.fft2(Im1))
FIm2=np.fft.fftshift(np.fft.fft2(Im2))


#Constructing Notch filter for image_1
# creatin mask for first image, Ring shaped square with value 0, and remaining all 1
rows, cols = Im1.shape #counting no. of rows and cols in image
R1,C1 = rows//2 , cols//2
mask1 = np.ones((rows,cols),np.uint8)
mask1[R1-60:R1+60, C1-60:C1-30] = 0
mask1[R1-60:R1+60, C1+30:C1+60] = 0
mask1[R1-60:R1-30, C1-60:C1+60] = 0
mask1[R1+30:R1+60, C1-60:C1+60] = 0

# applying mask1 and inverse DFT on first image
FT_mask_conv_img1 = FIm1*mask1
Shift_FT_mask_conv_img1 = np.fft.ifftshift(FT_mask_conv_img1) #transfering zero frequencies to the center
FT_to_Spatial1 = np.fft.ifft2(Shift_FT_mask_conv_img1) #reconstructing the image by inverse_FT
notch_filt_img1=np.abs(FT_to_Spatial1) #taking absolute values

#Constructing Notch filter for image_2
# creatin mask for second image, square shaped with value 1, and remaining all 0
rows, cols = Im2.shape #counting no. of rows and cols in image
R2, C2= rows//2 , cols//2
mask2 = np.zeros((rows,cols),np.uint8)
mask2[R2-60:R2+60, C2-60:C2+60] = 1

# apply mask2 and inverse DFT on second image
FT_mask_conv_img2 = FIm2*mask2
Shift_FT_mask_conv_img2 = np.fft.ifftshift(FT_mask_conv_img2)  #transfering zero frequencies to the center
FT_to_Spatial2 = np.fft.ifft2(Shift_FT_mask_conv_img2) #reconstructing the image by inverse_FT
notch_filt_img2=np.abs(FT_to_Spatial2) #taking absolute values

# Diplaying results for image1 and saving them in local directory
I1 = np.log(np.abs(FT_mask_conv_img1))  #log absolute values of DFT masked image
epsilon = 1e-10  # Small constant to avoid taking the logarithm of zero
I1 = np.log(np.abs(FT_mask_conv_img1) + epsilon)
plt.imshow(I1,cmap='gray')
plt.title('Filtered DFT Magnitude Image1')
plt.show()
# Normalizing the magnitude values to the range [0, 255], without doing this image will be saved black in directory
I1_normalized = ((I1 - I1.min()) / (I1.max() - I1.min()) * 255).astype(np.uint8)

cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q1_(c)_Filtered_DFT_Magnitude_Image1.JPG", I1_normalized)
plt.imshow(notch_filt_img1,cmap='gray')
plt.title('Notch Filtered Image1')
plt.axis('off')  # Turning off axis values
plt.show()
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q1_(c)_Notch_Filtered_Image1.JPG", notch_filt_img1)

# Diplaying results for image2 and saving them in local directory 
I2 = np.log(np.abs(FT_mask_conv_img2))  #log absolute values of DFT masked image
epsilon = 1e-10  # Small constant to avoid taking the logarithm of zero
I2 = np.log(np.abs(FT_mask_conv_img2) + epsilon)
plt.imshow(I2,cmap='gray')
plt.title('Filtered DFT Magnitude Image2')
plt.show()
# Normalizing the magnitude values to the range [0, 255], without doing this image will be saved black in directory
I2_normalized = ((I2 - I2.min()) / (I2.max() - I2.min()) * 255).astype(np.uint8)
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q1_(c)_Filtered_DFT_Magnitude_Image2.JPG", I2_normalized)
plt.imshow(notch_filt_img2,cmap='gray')
plt.title('Notch filtered Image2')
plt.axis('off')  # Turning off axis values
plt.show()
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q1_(c)_Notch_Filtered_Image2.JPG", notch_filt_img2)