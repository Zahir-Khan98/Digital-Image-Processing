#Author: ZAHIR KHAN , 112202010


import cv2
import numpy as np

def motionEstimation(video):
    # Defining parameters
    max_keypoints = 500  # Maximum number of keypoints to detect
    min_matches = 10    # Minimum number of matches for estimating motion
    motion_scale = 10   # Scaling factor for motion estimation

    # Initializing variables
    prev_frame = video[:, :, 0]
    StabilizedVideo = np.zeros_like(video)
    sigEst = np.zeros((2, video.shape[2]))

    # Creating ORB detector (ORB is a fusion of FAST keypoint detector and BRIEF descriptor with some added features to improve the performance)
    orb = cv2.ORB_create(nfeatures=max_keypoints)

    for i in range(1, video.shape[2]):
        curr_frame = video[:, :, i] #current frame

        # Finding  ORB keypoints and descriptors for the first frame
        if i == 1:
            kp1, des1 = orb.detectAndCompute(prev_frame, None)

        # Finding  ORB keypoints and descriptors for the current frame
        kp2, des2 = orb.detectAndCompute(curr_frame, None)

        # Creating a Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Matching descriptors from the first frame and current frame
        matches = bf.match(des1, des2)

        # Sorting matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Taking  the best matches
        matches = matches[:min(min_matches, len(matches))]

        # Extracting  matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimating affine transformation
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts)

        if M is not None:
            # Applying the estimated motion to the current frame
            StabilizedVideo[:, :, i] = cv2.warpAffine(curr_frame, M, (curr_frame.shape[1], curr_frame.shape[0]))

            # Estimating the motion and store it in sigEst
            motion = motion_scale * np.linalg.norm(M[:2, 2])
            sigEst[0, i] = motion  # Y direction motion
            sigEst[1, i] = motion  # X direction motion
        else:
            StabilizedVideo[:, :, i] = curr_frame
            sigEst[:, i] = 0  # No motion

        # Setting  the current frame as the previous frame for the next iteration
        prev_frame = StabilizedVideo[:, :, i]

    return StabilizedVideo, sigEst
