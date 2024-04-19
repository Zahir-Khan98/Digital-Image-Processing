'''
# Name- Zahir Khan (112202010)
# DIP Assignment2, Question 1.(a)
'''
#Importing necessary libraries
import cv2
import matplotlib.pyplot as plt

#Reading two Images
Im1=cv2.imread("image_1.JPG") 
Im2=cv2.imread("image_2.JPG")

# Checking dimension of the images for grayscale conversion
if(Im1.ndim==3): #checking color image or not
    Im1= cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY) #grayscale conversion
if(Im2.ndim==3):
    Im2= cv2.cvtColor(Im2, cv2.COLOR_BGR2GRAY)

# Median Filtering
Filtered_image_1=cv2.medianBlur(Im1,15) #using 15*15 kernal
Filtered_image_2=cv2.medianBlur(Im2,13) #using 13*13 kernal

#Displaying the filtered image and saving the local directory 'Results'
plt.imshow(Filtered_image_1,cmap='gray')
plt.title('Median filtered image_1')
plt.show()
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q1_(a)_Filtered_image_1.JPG", Filtered_image_1)

plt.imshow(Filtered_image_2,cmap='gray')
plt.title('Median filtered image_2')
plt.show()
cv2.imwrite(r"C:\Users\ZAHIR\OneDrive\Desktop\DIP\Assignment2\Results\Q1_(a)_Filtered_image_2.JPG", Filtered_image_2)
 