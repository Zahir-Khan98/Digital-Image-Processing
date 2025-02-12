import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Functions.autoHist import autoHist

#%% Importing the relevant function and setting directories
os.chdir(os.path.dirname(os.path.realpath(__file__)))

functionName = os.getcwd()+os.sep+'Functions'+os.sep+'autoHist.py'

exec(open(functionName).read())

#%% Reading Image and initial conversions
rawImage = cv2.imread(r'Data'+os.sep+'Q1_myRoomWindow.jpg') #Image to be improved

if(rawImage.ndim==3):
    rawImage= cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY) #grayscale conversion

rawImage = cv2.normalize(rawImage, None, 0, 255, cv2.NORM_MINMAX) #Converting to avoid quantization issues
    
#%% The main functionality
procImage = autoHist(rawImage) 

#%% Displaying Result and Writing the same
procImage = np.uint8(255 * procImage)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(rawImage,cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(procImage,cmap='gray')
axs[1].set_title('autoHist Image')
cv2.imwrite('Results'+os.sep+'Que1.png',procImage)