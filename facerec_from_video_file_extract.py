import argparse
import cv2
import json
import dlib
from common import utils
import time
import pickle
import numpy as np
import pathlib
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
	help="path to input video")
ap.add_argument("-m", "--model", required=True,
	help="trained model")
args = vars(ap.parse_args())

#Read video path from argument
video_path = args["file"]

#Load trained data
#model = json.load(open(args["model"]))

with open(args['model'], 'rb') as f:
    (le, clf) = pickle.load(f)
    
#Load video from video_path
cap = cv2.VideoCapture(video_path)

dirpath = pathlib.Path(__file__).parent

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(dirpath, 'models/shape_predictor_68_face_landmarks.dat'))
facerec = dlib.face_recognition_model_v1(os.path.join(dirpath, 'models/dlib_face_recognition_resnet_model_v1.dat'))

RATIO = 2
locations = {}

while True:
    ret, frame = cap.read()

    start = time.time()
    
    if (type(frame) == type(None)):
        break

    frame_displayed = cv2.resize(frame, (0,0), fx=1/2, fy=1/2)
    frame_resized = cv2.resize(frame_displayed, (0,0), fx=1/RATIO, fy=1/RATIO) 

    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    face_names = []
    face_confidences = []
        
    face_locations = detector(frame_gray, 1)
    face_descriptors = [np.asarray(list(facerec.compute_face_descriptor(frame_resized, shape))) for shape in [predictor(frame_gray, d) for d in face_locations]]
    
    for descriptor in face_descriptors:
        predictions = clf.predict_proba(descriptor.reshape(1, -1)).ravel()
        maxI = np.argmax(predictions)
        name = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        face_confidences.append(confidence)
            
        if confidence < 0.5:
            face_names.append("unknown")
        else:
            face_names.append(name)
   
    for d, name, confidence in zip(face_locations, face_names, face_confidences):
        top = d.top() * RATIO
        left = d.left() * RATIO
        bottom = d.bottom() * RATIO
        right = d.right() * RATIO
        cv2.rectangle(frame_displayed, (left, top),(right, bottom), (0, 0, 255), 1)
        cv2.rectangle(frame_displayed, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_displayed, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        print("{}".format(name) + " (" + "{0:.2f})".format(confidence))
        print("Recognition face took {} seconds.".format(time.time() - start))
        utils.save_face_image(frame_resized, shape, name, os.path.join(os.path.abspath(dirpath), "test", "results"), confidence)
    
    cv2.imshow('Face Recognition', frame_displayed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

