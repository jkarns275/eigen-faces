import cv2
import numpy as np

import matplotlib.pyplot as plt

def load_att_faces(att_faces_folder_path):
    height, width = att_faces_dims()
    faces = np.zeros((400, height, width))
    
    i =  0

    for subject in range(1, 41):
        for j in range(1, 11):
            i = i + 1
            impath = f"{att_faces_folder_path}/s{subject}/{j}.pgm"
            image = cv2.imread(impath)
            # Image is in BGR (blue green red) format by default but we want grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Map 2d index of (subject, j) to 1d index between 0 and 399 (400 faces!)
            index = (subject - 1) * 10 + (j - 1)
            faces[index, :, :] = image
    
    assert i == 400 # Should be exactly 400 images
    
    # Normalize between 0 and 1
    faces = faces / 255
    faces = faces.reshape(400, height, width, 1)
    return faces

def att_faces_dims():
    return (112, 92)

def n_latent_vars():
    return 3
