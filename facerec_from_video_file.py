import argparse
import cv2
import json
import dlib
from common import utils

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
model = json.load(open(args["model"]))

#Load video from video_path
cap = cv2.VideoCapture(video_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

RATIO = 4

while True:
    ret, frame = cap.read()
    
    if (type(frame) == type(None)):
        break

    frame_displayed = cv2.resize(frame, (0,0), fx=1/2, fy=1/2)
    frame_resized = cv2.resize(frame_displayed, (0,0), fx=1/RATIO, fy=1/RATIO) 

    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    rects = detector(frame_gray, 1)

    for k, d in enumerate(rects):
        shape = predictor(frame_gray, d)
        cv2.rectangle(frame_displayed, (d.left() * RATIO, d.top() * RATIO),(d.right() * RATIO, d.bottom() * RATIO), (0, 0, 255), 1)
        
        face_descriptor = facerec.compute_face_descriptor(frame_resized, shape)
        descriptor = list(face_descriptor)
        face = utils.predictBest(model, descriptor, 0.6)
        
        if len(face) > 0:
            name = face[1]
        else:
            name = "Unknown"

        # Draw a label with a name below the face
        cv2.rectangle(frame_displayed, (d.left() * RATIO, d.bottom() * RATIO - 25), (d.right() * RATIO, d.bottom() * RATIO), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_displayed, name, (d.left() * RATIO + 6, d.bottom() * RATIO - 6), font, 0.5, (255, 255, 255), 1)
        
    cv2.imshow('Face Recognition', frame_displayed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

