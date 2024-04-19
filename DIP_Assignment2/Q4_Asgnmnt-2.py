'''
#Name- Zahir Khan (112202010)
#DIP Assignment2, Question-4
'''
# Importing necessary libraries
import cv2 
import matplotlib.pyplot as plt
import numpy as np

#Reading the input image
input_image=cv2.imread("ContrastSample.TIFF")  

# Checking dimension of the images for grayscale conversion
if(input_image.ndim==3): #checking color image or not
    input_image= cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) #grayscale conversion
    

# Resizing the image to a standard size (256x256 in this case)
resized_image = cv2.resize(input_image, (256, 256))

# Generating Gaussian pyramid for the input image
gaussian_pyramid = [resized_image.copy()]
for i in range(6):
    resized_image = cv2.pyrDown(resized_image)
    gaussian_pyramid.append(resized_image)

# Generating Laplacian Pyramid for the input image
laplacian_pyramid = [gaussian_pyramid[5]]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
    laplacian_pyramid.append(laplacian)

# Applying gamma correction to each level of the Laplacian Pyramid
gamma_value = 0.4
gamma_corrected_pyramid = [np.power(level, gamma_value) for level in laplacian_pyramid]

# Reconstructing the enhanced image from the gamma-corrected Laplacian Pyramid
enhanced_image = gamma_corrected_pyramid[0]
for i in range(1, 6):
    enhanced_image = cv2.pyrUp(enhanced_image)
    enhanced_image = cv2.add(enhanced_image, gamma_corrected_pyramid[i])

# Displaying the enhanced image and saving to local directory
plt.imshow(enhanced_image, cmap='gray')
plt.title('Contrast Enhanced Image')
plt.show()
# Normalizing the magnitude values to the range [0, 255], without doing this image will be saved black in directory
enhanced_image_normalized = ((enhanced_image - enhanced_image.min()) / (enhanced_image.max() - enhanced_image.min()) * 255).astype(np.uint8)
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q4 Contrast Enhanced Image.JPG", enhanced_image_normalized)