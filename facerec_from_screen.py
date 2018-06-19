import time
import threading

import cv2
import dlib
import mss
import numpy
import math

from common import config, utils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--extract", required=False,default="",
	help="path to extract video")
args = vars(ap.parse_args())

frameCounter = 0
currentFaceID = 0

faceTrackers = {}
faceNames = {}
face_locations = {}

def doRecognizePerson(img, sp, fid, location):
    time.sleep(2)

    face_descriptor = config.facerec.compute_face_descriptor(img, sp)
    descriptor = numpy.asarray(list(face_descriptor))
    predictions = config.clf.predict_proba(descriptor.reshape(1, -1)).ravel()
    maxI = numpy.argmax(predictions)
    name = config.le.inverse_transform(maxI)
    confidence = int(math.ceil(predictions[maxI]*100))
        
    if confidence < config.threshold_recognition:
        name = config.unknown_name
                
    faceNames[fid] = "{0} ({1}%)".format(name, confidence)

def extract_face(img, face_locations):
    if cv2.waitKey(1) & 0xFF == ord('s'):
        output_path = args["extract"]
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if output_path != "":
            print('Extract images to {}'.format(output_path))
            for d in face_locations:
                face = img[d.top(): d.top() + d.height(), d.left(): d.left() + d.width()]
                utils.save_face_image(face, "", output_path, 0)
        else:
            print('No extract folder to save file')
    
with mss.mss() as sct:
    monitor = { 'top': 0, 'left': 0, 'width': 800, 'height': 800 }
    while 'Screen Capturing':
        img = numpy.array(sct.grab(monitor))

        isSaved = False
        baseImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        resultImage = img.copy()
        frameCounter += 1
        fidsToDelete = []

        for fid in list(faceTrackers):
            trackingQuality = faceTrackers[fid].update(baseImage)
            if trackingQuality < 7:
                fidsToDelete.append(fid)

            for fid in fidsToDelete:
                faceTrackers.pop(fid, None)

        if (frameCounter % 10) == 0:
            gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
            face_locations = config.detector(gray, 1)
                
            for d in face_locations:
                shape = config.predictor(gray, d)

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

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and 
                    (t_y <= y_bar <= (t_y + t_h)) and 
                    (x <= t_x_bar <= (x + w)) and 
                    (y <= t_y_bar <= (y + h))):
                        matchedFid = fid

                if matchedFid is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(gray, dlib.rectangle(x - 10, y - 20, x + w + 10, y + h + 20))

                    faceTrackers[currentFaceID] = tracker

                    t = threading.Thread(target = doRecognizePerson, args=(baseImage, shape, currentFaceID, d))
                    t.start()

                    #Increase the currentFaceID counter
                    currentFaceID += 1

        extract_face(resultImage, face_locations)
            
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

        cv2.imshow("Recognition", resultImage)
