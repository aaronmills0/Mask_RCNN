from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import cv2 as cv
import numpy as np
from locate_object import Locator
import matplotlib.pyplot as plt

# Resize the provided image to have a height of 600 and a width to maintain original proportions.
def resize(img):
    ratio = img.shape[1]/img.shape[0] # obtain ratio of width/height

    dim = (int(ratio*600), 600) # tuple of two integers: (width, height).

    # resize the image and return.

    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    return img

# Obtain a binarized image using the provided image and masked.
def apply_mask(img, mask):
    blank = np.copy(img) # deep copy the original image.

    # For each colour channel if mask==1 set the value to 255 (When all 3 colour channels are 255 we have a white pixel).
    for c in range(3):
        blank[:,:,c] = np.where(mask==1, 255, 0)

    return blank # Return the binarized image.

# Given a query image and an roi from the train image determine if the object in the reference image is in the roi (returns a boolean).
def is_object(query, train, roi, query_med, masked, deviation=35, num=10, thresh=40, orb=None, bf=None):
    img = train[roi[0]:roi[2], roi[1]:roi[3]] # obtain the roi

    # If we have not passed an orb detector create one.
    if orb == None: 
        orb = cv.ORB_create()
     
    # Obtain key points and descriptors for both the reference image and the roi.
    kp1, des1 = orb.detectAndCompute(query, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    # If we have not passed a bf matcher create one.
    if bf == None:
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
     
    # Obtain matches.
    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)

    print("===============================")
    for i in range(len(matches)):
        print(f"{matches[i].distance} ", end="")
    print("\n===============================")
     

    # Compute the average distance of the best num matches (lower is better).
    s = 0
    for j in matches[:num]:
        s += j.distance
    s = s/num

    # If we consider the average to be sufficiently small.
    if s <= thresh:
        blank = np.copy(train) # deep copy of the train image.
        b_vals = []
        g_vals = []
        r_vals = []
        # Obtain the section of the train image that the mask considers to be the object. Set the rest of the image to be black.
        blank = np.where(masked>0, blank, 0)
    
        cv.imshow('blank', blank)

        cv.waitKey(0)

        # Iterate through each pixel.
        for i in range(blank.shape[0]):
            for j in range(blank.shape[1]):
                # If the pixel is not black (according to the mask this should be part of the object) save its b, g, and r values.
                if blank[i,j].any() > 0:
                    b_vals.append(blank[i,j,0])
                    g_vals.append(blank[i,j,1])
                    r_vals.append(blank[i,j,2])

        # Find the median blue, green, and red values.

        b_vals = sorted(b_vals)
        g_vals = sorted(g_vals)
        r_vals = sorted(r_vals)

        b_med = b_vals[int(len(b_vals)//2)]
        g_med = g_vals[int(len(g_vals)//2)]
        r_med = r_vals[int(len(r_vals)//2)]

        # Perform BM-Normalization

        diff = query_med[0]-b_med

        b_med += int(diff)

        g_med += int(diff)

        r_med +=  int(diff)

        print(f"Train -> Blue: {b_med}, Green: {g_med}, Red: {r_med}")

        # For each colour channel check if the median colour value of the object in the roi is close to that of the query image.

        if not ((abs(int(query_med[1])-int(g_med)) < deviation and abs(int(query_med[2])-int(r_med)) < deviation)):
            return False

        return True

    return False
     
def main():
    # define 81 classes that the coco model knowns about
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']
     
    # define the test configuration
    class TestConfig(Config):
        NAME = "test"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80
    
    # Prompt the user for information
    file_query = input('Filename of the query image: ')

    query = cv.imread(f'images/{file_query}')

    if query is None:
        print(f'Failed to find \'{file_query}\'.')
        exit(1)
    
    # Obtain a version of the query image with a black background

    copy = np.copy(query)
    copy = np.where(copy[:,:,:] == query[0,0,:], 0, copy)

    b_vals = []
    g_vals = []
    r_vals = []

    # For each colour channel find the median value (excluding black pixels).

    for i in range(copy.shape[0]):
        for j in range(copy.shape[1]):
            if copy[i,j].any() > 0:
                b_vals.append(copy[i,j,0])
                g_vals.append(copy[i,j,1])
                r_vals.append(copy[i,j,2])

    b_vals = sorted(b_vals)
    g_vals = sorted(g_vals)
    r_vals = sorted(r_vals)

    b_med = b_vals[int(len(b_vals)//2)]
    g_med = g_vals[int(len(g_vals)//2)]
    r_med = r_vals[int(len(r_vals)//2)]

    query_med = (b_med, g_med, r_med)
    
    file_train = input('Filename of the train image: ')

    train = cv.imread(f'images/{file_train}')
    
    if train is None:
        print(f'Failed to find \'{file_train}\'.')
        exit(1)
    
    train = resize(train)

    # Allow the user to select which of 81 categories the object belongs to (can be more than one category).

    count = 0
    for k in range(len(class_names)-1):
        print(f"{class_names[k]}, ", end="")
    print(class_names[-1])
    print('Enter the categories that your query may belong to. Press \'enter\' to continue. If no categories are specified all categories will be considered.')
    categories = []
    all_types = False
    while True:
        cat = input(f'type {len(categories)+1}: ')
        if cat == "":
            break
        else:
            if class_names.count(cat) == 0:
                print('Invalid type name.')
                continue
            elif categories.count(cat) > 0:
                print('Duplicate type.')
            else:
                categories.append(cat)
    if len(categories) == 0:
        all_types = True
     
    # define the model
    rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
    # load coco model weights
    rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
    # load photograph
    img = load_img(f'images/{file_train}')

    img = img.resize((train.shape[1], train.shape[0]))

    if img is None:
        print(f'Failed to find \'{file_train}\'.')
        exit(1)

    img = img_to_array(img)
    # make prediction
    results = rcnn.detect([img], verbose=0)
    # get dictionary for first prediction
    r = results[0]

    locator = Locator()
     
    # show photo with bounding boxes, masks, class labels and scores
    objects = []
    indices = []

    # Find indices that are labelled as the categories provided by the user. If the user did not provide any categories then all categories will be considered.
    if all_types:
        indices = list(range(len(r['class_ids'])))
    else:
        for i in range(len(r['class_ids'])):
            if categories.count(class_names[r['class_ids'][i]]) > 0:
                # Save the objects roi and its index.
                objects.append(r['rois'][i])
                indices.append(i)
    
    # Apply the mask for objects
    objs = []
    found = []

    # Create an orb detector and a bf matcher. 

    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    print(f"Blue: {b_med}, Green: {g_med}, Red: {r_med}")

    # Iterate through each object that Mask rcnn considers to be in the provided categories.
    for j in indices:
        y0, x0, y1, x1 = r['rois'][j]
        cv.imshow('Portion', train[y0:y1, x0:x1])
        cv.waitKey(0)
        roi = (y0, x0, y1, x1)
        mask = r['masks'][:,:,j] # Obtain the mask.
        masked = apply_mask(train, mask) # Obtain a binarized image according to the mask.
        cv.imshow('mask', masked)
        cv.waitKey(0)
        
        # Check if the found object matches our query object.
        if is_object(query, train, roi, query_med, masked, deviation=35, num=10, thresh=40, orb=orb, bf=bf):
            # Save the binarized image of all objects indentified as the query object.
            objs.append(masked)
            # Save indices of objects identified as the query object.
            found.append(j)
    all_objs = []

    # Obtain a binarized images of every object detected.
    for k in range(len(r['rois'])):
        # Ignore dining tables as they tend to lead to false positives.
        if class_names[r['class_ids'][k]] != 'dining table':
            y0, x0, y1, x1 = r['rois'][k]
            roi = (y0, x0, y1, x1)
            mask = r['masks'][:,:,k]
            masked = apply_mask(train, mask)
            all_objs.append(masked)

    # Combine the binarized images of all of the objects that match the query object.
    if len(objs) > 0:
        comb = objs[0]
    else:
        print('No instances of the object have been found...')
        exit(1)
    for i in range(1, len(objs)):
        comb = cv.bitwise_or(comb, objs[i])
    comb = cv.cvtColor(comb, cv.COLOR_BGR2GRAY)

    cont = np.zeros((comb.shape[0], comb.shape[1], 3), dtype=np.uint8)

    contours, hierarchy = cv.findContours(comb, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # Obtain the contours of the combined binarized image.

    # For all the objects indetified as a match to the query object.
    for i in found:
        blank = np.zeros((train.shape[0], train.shape[1], 3), dtype=np.uint8) # Black image of the same dimensions as the train image.
        y0, x0, y1, x1 = r['rois'][i]
        blank[y0:y1, x0:x1] = train[y0:y1, x0:x1] # Set the roi pixels of the blank image to those in the train image.
        mask = r['masks'][:,:,i] # Obtain the mask.
        masked = apply_mask(train, mask) # Obtain the binarized image.
        # Iterate through the array containing the binarized images of all objects.
        for j in range(len(all_objs)):
            # If it is not our match, check if it occludes the roi.
            if not np.array_equal(masked, all_objs[j]):
                white = np.where(all_objs[j]==255)
                blank[white[0],white[1],:] = [255, 0, 0]
                train[y0:y1, x0:x1] = blank[y0:y1, x0:x1]
        cv.imshow('blank', blank)

    res = cv.drawContours(train, contours, -1, (0, 0, 255), 10) # Draw the contours.

    plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))

    display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    cv.waitKey(0)

# Python main check
if __name__ == '__main__':
    main()