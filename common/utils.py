import cv2
import numpy as np
from functools import reduce
import math
import time

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def computeMeanDistance(descriptors, inputDescriptor):
    result = 1
    distances = list(map(lambda x : np.linalg.norm(np.array(x) - np.array(inputDescriptor)), descriptors))
    
    if len(distances) > 0:
        mean = reduce((lambda x, y: (x + y)/len(distances)), distances)
        f = math.pow(10, 2)
        result = round(mean * f) / f

    return result

def predictBest(model, descriptor, unknownThreshold = 0):
    means = list(map(lambda item : (computeMeanDistance(item['faceDescriptors'], descriptor), item['className']), model))
    
    array = sorted(means, key=lambda x: x[0])
    distance = array[0][0]
    
    if distance >= unknownThreshold:
        return []
    else:
        return array[0]
    
def save_image(frame, shape, name):
    if IS_SAVED == True:
        file = str(uuid.uuid4())
        print('Saving file: ' + file)
        if name != 'Unknown':
            dlib.save_face_chip(frame, shape, 'data/faces/' + name + '/' + file, 150, 0.2)
        else:
            dlib.save_face_chip(frame, shape, 'data/images/' + file, 150, 0.2)
    return result

