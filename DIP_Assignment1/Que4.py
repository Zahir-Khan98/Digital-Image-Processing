import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#%% Importing the relevant function and setting directories
from Functions import affineTransform
from Functions import invaffineTransform

os.chdir(os.path.dirname(os.path.realpath(__file__)))

functionName = os.getcwd()+os.sep+'Functions'+os.sep+'affineTransform.py'

exec(open(functionName).read())

functionName = os.getcwd()+os.sep+'Functions'+os.sep+'invaffineTransform.py'

exec(open(functionName).read())

#%% Reading Image and initial conversions
rawImage = cv2.imread(r'Data'+os.sep+'grain.png') #Image to be improved

if(rawImage.ndim==3):
    rawImage= cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY) #grayscale conversion

# rawImage = cv2.normalize(rawImage.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #Converting to avoid quantization issues

#%% Affine Transform Matrix Parameters
a = 0.3
b= 0.1
c = 0.5
d = 1.9
tx = 0
ty = 0
T = [[a, b, 0, tx],[c, d, 0, ty],[0, 0, 1, 0]]
    
#%% The main functionality
interpType= cv2.INTER_LINEAR # cv2.INTER_NEAREST or cv2.INTER_LINEAR as needed
transImage = affineTransform(rawImage,T,interpType) 
invtransImage = invaffineTransform(transImage,T,interpType)


#%% Displaying Result and Writing the same
transImage = np.uint8(255.0*transImage)
invtransImage = np.uint8(255.0*invtransImage)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(rawImage,cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(transImage,cmap='gray')
axs[1].set_title('Affine Transformed Image')
axs[2].imshow(invtransImage,cmap='gray')
axs[2].set_title('Inverse Affine Transformed Image')

cv2.imwrite('Results'+os.sep+'Que4_1.png',transImage)
cv2.imwrite('Results'+os.sep+'Que4_2.png',invtransImage)

plt.savefig('Results'+os.sep+'Que4_sub.png')
plt.show()