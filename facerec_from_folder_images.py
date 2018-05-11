import sys
import os
import dlib
import glob
import numpy as np
from functools import reduce
import math
from skimage import io
import shutil
import cv2
import json
from common import utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to image folder")
ap.add_argument("-m", "--model", required=True,
	help="trained model")
args = vars(ap.parse_args())

images = args['file']
model_path = args['model']

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

model = json.load(open(model_path))

win = dlib.image_window()
        
for f in glob.glob(os.path.join(images, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    
    resized = cv2.resize(img, (0, 0), fx = 1/RATIO, fy = 1/RATIO)
    win.clear_overlay()
    win.set_image(resized)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    dets = detector(gray, 1)
    
    for k, d in enumerate(dets):
        shape = sp(gray, d)
        face_descriptor = facerec.compute_face_descriptor(resized, shape, 5)
        descriptor = list(face_descriptor)
        
        face = utils.predictBest(model, descriptor, 0.6)
        if len(face) > 0:
            name = face[1]
        else:
            distance = 0
            name = "Other"
        
        if name != "Other":
            file = os.path.join(output_folder_path, name)
            path = os.listdir(file)[0]
            if len(os.listdir(file)) > 0:
                photo = io.imread(os.path.join(file, path))
                photo_resize = cv2.resize(photo, (250, 250))
                actual = path.split('_')[0]
                cv2.line(photo_resize,(0,240),(250,240),(0,0,0),20)        
                cv2.putText(photo_resize, "{}".format("Actual: " + actual), (5, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
                face = dlib.get_face_chip(resized, shape, 250, 0.25)
                cv2.line(face,(0,240),(250,240),(0,0,0),20)
                cv2.putText(face, "{}".format("Guess: " + name), (5, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
                vis = np.concatenate((face, photo_resize), axis=1)
                cv2.imshow('Image', vis)
        
        win.add_overlay(d)
        win.add_overlay(shape)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
