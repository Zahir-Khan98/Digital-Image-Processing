#The video is changed to black and white. Changes for that are made
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Functions.motionEstimation import motionEstimation
#%% Importing the relevant function and setting directories
os.chdir(os.path.dirname(os.path.realpath(__file__)))

functionName = os.getcwd()+os.sep+'Functions'+os.sep+'motionEstimation.py'

exec(open(functionName).read())

#%% Reading video

video_file = r'Data'+os.sep+'motionEstimation.mp4' #%Put your video file inside data folder and change the name of the file and format if needed

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
[StabilizedVideo, sigEst] = motionEstimation(video)


#%% Displaying Result and Writing the same
fig, axs = plt.subplots(1, 2)
axs[0].plot(sigEst[0,:])
axs[0].set_title('Y direction motion')
axs[1].plot(sigEst[1,:])
axs[1].set_title('X direction motion')
fig.savefig('Results'+os.sep+'Que5.png')

size= np.shape(StabilizedVideo)

result = cv2.VideoWriter('Results'+os.sep+'Que5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (size[1], size[0]))

for ii in range(size[2]):
    result.write(StabilizedVideo[:,:,ii])        #Changes for grayscale
result.release()