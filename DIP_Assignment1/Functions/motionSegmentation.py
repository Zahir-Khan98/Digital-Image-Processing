#Author: ZAHIR KHAN, 112202010

import numpy as np

#image normalisation function
def imgNormalize(img):  
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm

def motionSegmentation(video):
    frames=video.shape[2]    #finding number of frames

    # initializing new video file and finding backgroundimage by finding mean of all the frames
    foregroundVideo=np.zeros((video.shape[0],video.shape[1], frames-1))       
    backgroundImage=np.mean(video,axis=2)                                     
    backgroundImage=np.uint8(255*imgNormalize(backgroundImage))

    #extracting the moving foreground by subtraction of frames and converting intensity value zero which are less than zero
    foregroundVideo=np.diff(video, axis=2)                                    
    foregroundVideo[foregroundVideo<0]=0                                  
    return backgroundImage,foregroundVideo





