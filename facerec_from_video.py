import cv2
import dlib
import os

import threading
import time

import pathlib

import pickle
import numpy as np

from common import utils, config
import random
import glob
import math
import argparse
from PIL import Image, ImageDraw, ImageFont
import json

import pafy

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=False,
	help="path to input video")
ap.add_argument("-e", "--extract", required=False,default="",
	help="path to extract video")
ap.add_argument("-r", "--ratio", required=False,default=1,
	help="path to extract video")

args = vars(ap.parse_args())

#Read video path from argument
video_path = args["file"]
RATIO = args["ratio"]

if video_path.find('youtube.com') != -1:
    youtube_url = args["file"]
    video = pafy.new(youtube_url)
    best = video.getbest()
    video_path = best.url

if video_path:
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

frameCounter = 0
currentFaceID = 0

faceTrackers = {}
faceNames = {}
face_locations = {}
    
def doRecognizePerson(img, sp, fid, location):
    time.sleep(2)

    face_descriptor = config.facerec.compute_face_descriptor(img, sp)
    descriptor = np.asarray(list(face_descriptor))
    predictions = config.clf.predict_proba(descriptor.reshape(1, -1)).ravel()
    maxI = np.argmax(predictions)
    name = config.le.inverse_transform(maxI)
    confidence = int(math.ceil(predictions[maxI]*100))
        
    if confidence < config.threshold_recognition:
        name = config.unknown_name

    output_path = args["extract"]
    if output_path != "":
        face = img[location.top() * RATIO : location.top() * RATIO + location.height() * RATIO, location.left() * RATIO : location.left() * RATIO + location.width() * RATIO]        
        utils.save_face_image(face, name, output_path, confidence)
                
    faceNames[fid] = "{0} ({1}%)".format(name, confidence)
    
while True:
    ret, frame = cap.read()
    
    if (type(frame) == type(None)):
        break

    baseImage = cv2.resize(frame, (0,0), fx = 1/RATIO, fy = 1/RATIO)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    resultImage = frame.copy()
    image = baseImage.copy()
    
    frameCounter += 1
    
    fidsToDelete = []
        
    for fid in list(faceTrackers):
        trackingQuality = faceTrackers[fid].update(baseImage)
        
        #If the tracking quality is good enough, we must delete this tracker
        if trackingQuality < 7:
            fidsToDelete.append(fid)

        for fid in fidsToDelete:
            faceTrackers.pop(fid, None)

    if (frameCounter % 10) == 0:
         gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
         face_locations = config.detector(gray, 0)
             
         for d in face_locations:
            shape = config.predictor(gray, d)
            #landmarks = utils.raw_land_marks(shape)

            #for i in landmarks:
                #cv2.line(resultImage,(i[0], i[1]),(i[2], i[3]), (0,255,0), 1)
        
            x = d.left()
            y = d.top()
            w = d.width()
            h = d.height()

            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            matchedFid = None

            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                #calculate the centerpoint
                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                #check if the centerpoint of the face is within the 
                #rectangleof a tracker region. Also, the centerpoint
                #of the tracker region must be within the region 
                #detected as a face. If both of these conditions hold
                #we have a match
                if ((t_x <= x_bar <= (t_x + t_w)) and 
                    (t_y <= y_bar <= (t_y + t_h)) and 
                    (x <= t_x_bar <= (x + w)) and 
                    (y <= t_y_bar <= (y + h))):
                    matchedFid = fid

            if matchedFid is None:

                #Create and store the tracker 
                tracker = dlib.correlation_tracker()
                tracker.start_track(baseImage, dlib.rectangle(x - 10, y - 20, x + w + 10, y + h + 20))

                faceTrackers[currentFaceID] = tracker

                t = threading.Thread(target = doRecognizePerson, args=(baseImage, shape, currentFaceID, d))
                t.start()

                #Increase the currentFaceID counter
                currentFaceID += 1
            
    for fid in faceTrackers.keys():
        tracked_position =  faceTrackers[fid].get_position()

        t_x = int(tracked_position.left() * RATIO)
        t_y = int(tracked_position.top() * RATIO)
        t_w = int(tracked_position.width() * RATIO)
        t_h = int(tracked_position.height() * RATIO)

        cv2.rectangle(resultImage, (t_x, t_y),(t_x + t_w, t_y + t_h), (0, 0, 255), 1)

        left = t_x + t_w
        
        if fid in faceNames.keys():
            name = faceNames[fid]                
            resultImage = utils.draw_face_name(resultImage, tracked_position, name, RATIO)
        else:
            resultImage = utils.draw_face_name(resultImage, tracked_position, 'Detecting...', RATIO)
        
    #Finally, we want to show the images on the screen
    cv2.imshow("Face Recognition", resultImage)
         
cap.release()
cv2.destroyAllWindows()

