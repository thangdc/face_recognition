import cv2
import numpy as np
from functools import reduce
import math
import time
import os.path
import uuid
import dlib
from common import config
from PIL import Image, ImageFont, ImageDraw

INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]
    
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

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
    
def save_face_image(frame, name, folder, confidence):
    file = str(uuid.uuid4()) + ".jpg"

    try:
        height, width = frame.shape[:2]
        if width > 100 and height > 100:
            if name == "":
                filename = os.path.join(folder, "{}".format(file))
            else:
                filename = os.path.join(folder, "{}_{}_{}".format(name, confidence, file))
            print('\tSaving file: ' + filename)
            
            image = Image.fromarray(frame).convert('RGB')
            image.save(filename)
    except Exception as error:
        print('\t' + str(error))

def align_face(rgbImg, points, imgDim, landmarkIndices=OUTER_EYES_AND_NOSE):
    landmarks = list(map(lambda p: (p.x, p.y), points.parts()))
    npLandmarks = np.float32(landmarks)
    npLandmarkIndices = np.array(landmarkIndices)
    H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                   imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
        
    thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
    
    return thumbnail

def raw_land_marks(landmark):
    landmarks = []
    # Around Chin. Ear to Ear
    i = 1
    while i <= 16:
        landmarks.append([landmark.part(i).x, landmark.part(i).y, landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1

    #Line on top of nose
    i = 28
    while i <= 30:
        landmarks.append([landmark.part(i).x, landmark.part(i).y,landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1

    #left eyebrow
    i = 18
    while i <= 21:
        landmarks.append([landmark.part(i).x, landmark.part(i).y,landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1

    #Right eyebrow
    i = 23
    while i <= 26:
        landmarks.append([landmark.part(i).x, landmark.part(i).y,landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1

    #Bottom part of the nose
    i = 31
    while i <= 35:
        landmarks.append([landmark.part(i).x, landmark.part(i).y,landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1

    #Line from the nose to the bottom part above
    landmarks.append([landmark.part(30).x, landmark.part(30).y,landmark.part(35).x, landmark.part(35).y])

    #Left eye
    i = 37
    while i <= 41:
        landmarks.append([landmark.part(i).x, landmark.part(i).y,landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1
    landmarks.append([landmark.part(36).x, landmark.part(36).y,landmark.part(41).x, landmark.part(41).y])

    #Right eye
    i = 43
    while i <= 47:
        landmarks.append([landmark.part(i).x, landmark.part(i).y,landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1
    landmarks.append([landmark.part(42).x, landmark.part(42).y,landmark.part(47).x, landmark.part(47).y])

    #Lips outer part
    i = 49
    while i <= 59:
        landmarks.append([landmark.part(i).x, landmark.part(i).y,landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1
    landmarks.append([landmark.part(48).x, landmark.part(48).y,landmark.part(59).x, landmark.part(59).y])

    #Lips inside part
    i = 61
    while i <= 67:
        landmarks.append([landmark.part(i).x, landmark.part(i).y,landmark.part(i - 1).x, landmark.part(i - 1).y])
        i += 1
    landmarks.append([landmark.part(60).x, landmark.part(60).y,landmark.part(67).x, landmark.part(67).y])

    return landmarks

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords

def face_recognition(face_descriptors):
    result = []
    for index, descriptor in enumerate(face_descriptors):
        predictions = config.clf.predict_proba(descriptor.reshape(1, -1)).ravel()
        maxI = np.argmax(predictions)
        name = config.le.inverse_transform(maxI)
        confidence = int(math.ceil(predictions[maxI]*100))
        result.append([name, confidence])
    
    return result

def draw_face_name(img, face_location, name, ratio):

    top = int(face_location.top()) * ratio
    left = int(face_location.left()) * ratio
    bottom = int(face_location.bottom()) * ratio
    right = int(face_location.right()) * ratio
    width = int(face_location.width()) * ratio
    height = int(face_location.height()) * ratio
    
    #cv2.rectangle(img, (left, top),(right, bottom), (0, 0, 255), 1)

    #baseline = 0
    #textSize = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    #rect_left = left + textSize[0][0] + 10

    font = ImageFont.truetype("arial", 18)
    text_size = font.getsize(name)
    rect_left = left + text_size[0] + 10
    
    if rect_left < left + width:
        rect_left = left + width

    cv2.rectangle(img, (left, top + height - 25), (rect_left, top + height), (0, 0, 255), cv2.FILLED)
    #cv2.putText(img, name, (left + 6, top + height - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    array = Image.fromarray(img)
    draw = ImageDraw.Draw(array)
    draw.text((left + 6, top + height - 22), name, fill="white", font=font)

    return np.array(array)
