#Author: ZAHIR KHAN, 112202010

import cv2
import numpy as np
def invaffineTransform(transImage, T, interpType):
    # Define the 2D affine transformation matrix
    # transformation_matrix = np.array([[T[0][0], T[0][1], T[0][2]],
    #                                   [T[1][0], T[1][1], T[1][2]]])


    #finding inverse transformation matrix
    inverse_transformation_matrix = np.array([[T[0][0], T[0][1]],[T[1][0], T[1][1]]])
    inv_T = np.linalg.inv(inverse_transformation_matrix)
    transformation_matrix = np.array([[inv_T[0][0], inv_T[0][1],0],[inv_T[1][0], inv_T[1][1],0]])
    #  Getting height and width of the input image
    img_height, img_width = transImage.shape[:2]

    # Defining corner points of the image
    corner_points = np.array([[0, 0, 1], [img_width - 1, 0, 1], [0, img_height - 1, 1], [img_width - 1, img_height - 1, 1]])

    # Apply the transformation to the corners
    transformed_corners = np.dot(transformation_matrix, corner_points.T).T

    #Finding the maximum and minimum x, y values in the transformed coordinates
    max_x = max(transformed_corners[:, 0])
    min_x = min(transformed_corners[:, 0])
    max_y = max(transformed_corners[:, 1])
    min_y = min(transformed_corners[:, 1])

    # Calculating the dimensions of the canvas
    canvas_width = int(np.ceil(max_x - min_x))
    canvas_height = int(np.ceil(max_y - min_y))

    # Creating transformation matrix that translates to ensure positive coordinates
    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # Combining the translation matrix and the original transformation matrix
    final_transform = np.dot(transformation_matrix, translation_matrix)

    # Creating a blank canvas (white background)
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Applying the combined transformation to the image
    transformed_image = cv2.warpAffine(transImage, final_transform, (canvas_width, canvas_height), flags=interpType)

    # putting the transformed image on the canvas
    canvas[:canvas_height, :canvas_width] = transformed_image

    invtransImage = canvas

    # return transImage

    return invtransImage