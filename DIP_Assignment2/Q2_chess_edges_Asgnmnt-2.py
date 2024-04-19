'''
#Name- Zahir Khan (112202010)
#DIP Assignment2, Question-2 Chess board image
'''
# Importing necessary libraries
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import convolve2d as conv2

#Reading the Chess board image
Img=cv2.imread("Chess.PNG")  

# Checking dimension of the images for grayscale conversion
if(Img.ndim==3): #checking color image or not
    Img= cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY) #grayscale conversion
    
#converting to double to prevent quantization issues
Img = cv2.resize(Img,(256,256))
Img_norm=cv2.normalize(Img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
filt=np.ones((256,256))

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

#Extracting the blocks by zero padding on the area of the chess image (filtered) except the chess board part
#For this work we have found the approximate equation of the chess board (for four sides) in the image
G=np.sqrt(gx**2+gy**2)
for x in range(256):
  for y in range(256):
    if( 70*x-130*y>6000) or (x+2*y < 170) or (x-y < -100) or (x+y > 400): 
      filt[y,x] = 0
G=filt*G 

#Diplaying resultant filtered image
plt.imshow(G,cmap='gray')
plt.title('Edges of chess board')
plt.show()

# Normalizing the magnitude values to the range [0, 255], without doing this image will be saved black in directory
G_normalized = ((G - G.min()) / (G.max() - G.min()) * 255).astype(np.uint8)
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q2 Edge of Chess Image.JPG", G_normalized)