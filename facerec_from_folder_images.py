import numpy as np
import cv2
import argparse
from common import utils, config, data
from random import randint
import os
from skimage import io
from PIL import Image, ImageFont, ImageDraw
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to image folder")
ap.add_argument("-e", "--extract", required=False, default="",
	help="path to extract folder")
args = vars(ap.parse_args())

images = args['folder']
extracted = args['extract']

window_name = 'Face Recognition'
window_width = 1024
window_height = 700
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_width, window_height)
cv2.moveWindow(window_name, 0,0)
    
for f in data.iterImgs(images):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print('')
    print("Processing file: {}".format(f.path))

    img = f.getRGB()
    actual = Image.fromarray(img)

    w_r = actual.width / window_width
    h_r = actual.height / window_height
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resize_image = cv2.resize(img, (window_width, window_height))
    resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    cv2.imshow(window_name, resize_image)
    cv2.waitKey(1)

    start_time = time.time()
    face_locations = config.detector(gray, 0)
    end_time = time.time()
    print("\tFound {0} face(s) in {1:.2f} miliseconds".format(len(face_locations), end_time - start_time))

    start_time = time.time()
    face_descriptors = [np.asarray(list(config.facerec.compute_face_descriptor(img, shape))) for shape in [config.predictor(gray, d) for d in face_locations]]
    recognitions = utils.face_recognition(face_descriptors)
    end_time = time.time()
    print('\tTook {0:.2f} miliseconds to recognize {1} faces'.format(end_time - start_time, len(face_locations)))
    img_actual_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for index, item in enumerate(recognitions):
        name = item[0]
        confidence = item[1]

        if confidence < config.threshold_recognition:
            name = config.unknown_name

        print("\tFound {} with ({}%) confidence".format(name, confidence))
        
        top = face_locations[index].top()
        bottom = face_locations[index].bottom()
        left = face_locations[index].left()
        right = face_locations[index].right()
        width = face_locations[index].width()
        height = face_locations[index].height()


        t_x = int((window_width * left) / actual.width)
        t_y = int((window_height * top) / actual.height)
        
        t_w = int(t_x + (width/w_r))
        t_h = int(t_y + (height/h_r))
        
        cv2.rectangle(resize_image, (t_x, t_y),(t_w, t_h), (0, 0, 255), 2)
        cv2.imshow(window_name, resize_image)
        cv2.waitKey(1)
        
        if extracted != "":
            face_img = img_actual_color[top : top + height, left : left + width]        
            utils.save_face_image(face_img, name, extracted, confidence)
        else:
            try:
                face = img[top : top + height, left : left + width]
                tmp = Image.fromarray(face)
                if tmp.width > 50 and tmp.height > 50:
                    face_resize = cv2.resize(face, (250, 250))
                            
                    cv2.line(face_resize,(0,240),(250,240),(255,0,0),20)
                    array = Image.fromarray(face_resize)
                    draw = ImageDraw.Draw(array)
                    font = ImageFont.truetype("arial", 18)
                    draw.text((5, 230), "{} - ({}%)".format(name, confidence), fill="white", font=font)
                    result = np.array(array)
                                
                    vis = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    cv2.imshow('Result', vis)
                    
                    cv2.waitKey(300)
            except:
                pass
 
cv2.destroyAllWindows()
