import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Functions.grain import grain

#%% Importing the relevant function and setting directories
os.chdir(os.path.dirname(os.path.realpath(__file__)))

functionName = os.getcwd()+os.sep+'Functions'+os.sep+'grain.py'

exec(open(functionName).read())

#%% Reading Image and initial conversions
rawImage = cv2.imread(r'Data'+os.sep+'grain.png') #Image to be improved

if(rawImage.ndim==3):
    rawImage= cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY) #grayscale conversion

rawImage = cv2.normalize(rawImage.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #Converting to avoid quantization issues
    
#%% The main functionality
procImage = grain(rawImage)   

#%% Displaying Result and Writing the same
procImage = np.uint8(255.0*procImage)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(rawImage,cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(procImage,cmap='gray')
axs[1].set_title('Improved Image')
cv2.imwrite('Results'+os.sep+'Que3.png',procImage)
# %%
