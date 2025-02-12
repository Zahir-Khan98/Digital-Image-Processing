#Changing to grayscale image.Changes done for that

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Functions.motionSegmentation import motionSegmentation

#%% Importing the relevant function and setting directories
os.chdir(os.path.dirname(os.path.realpath(__file__)))

functionName = os.getcwd()+os.sep+'Functions'+os.sep+'motionSegmentation.py'

exec(open(functionName).read())

#%% Reading video

video_file = r'Data'+os.sep+'motionSegmentation.mp4' #%Put your video file inside data folder and change the name of the file and format if needed

video=[]
cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()
#reading all frames 
while ret:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #adding the additional line of code for black and white conversion
    video.append(frame)
    ret, frame = cap.read() 
    
cap.release()
video = np.array(video)
video=np.moveaxis(video, 0, -1)
#%% The main functionality
[backgroundImage,foregroundVideo] = motionSegmentation(video)


#%% Displaying Result and Writing the same
plt.imshow(backgroundImage,cmap='gray'); plt.title('Background Image')
cv2.imwrite('Results'+os.sep+'Que2.png',backgroundImage)

size= np.shape(foregroundVideo)
result = cv2.VideoWriter('Results'+os.sep+'Que2.mp4',  
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          30, (size[1],size[0]),0)
for ii in range(size[2]):              
    result.write(foregroundVideo[:,:,ii])   # subsequent changes done
result.release()