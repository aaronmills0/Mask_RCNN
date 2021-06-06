from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import cv2 as cv
import numpy as np
from locate_object import Locator
import matplotlib.pyplot as plt

def apply_mask(img, mask):
    blank = np.copy(img)
    for c in range(3):
        blank[:,:,c] = np.where(mask==1, 255, 0)
    return blank

def is_object(query, train, roi, num=10, thresh=40, orb=None, bf=None):
     img = train[roi[0]:roi[2], roi[1]:roi[3]]
     if orb == None:
          orb = cv.ORB_create()
     
     kp1, des1 = orb.detectAndCompute(query, None)
     kp2, des2 = orb.detectAndCompute(img, None)

     if bf == None:
          bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
     
     matches = bf.match(des1, des2)

     matches = sorted(matches, key = lambda x:x.distance)

     print("===============================")
     for i in range(len(matches)):
          print(f"{matches[i].distance} ", end="")
     print("\n===============================")
     
     s = 0
     for j in matches[:num]:
          s += j.distance

     s = s/num

     if s <= thresh:
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
    file_train = input('Filename of the train image: ')
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
    img = img_to_array(img)
    # make prediction
    results = rcnn.detect([img], verbose=0)
    # get dictionary for first prediction
    r = results[0]

    query = cv.imread(f'images/{file_query}')
    train = cv.imread(f'images/{file_train}')

    locator = Locator()
     
    # show photo with bounding boxes, masks, class labels and scores
    objects = []
    indices = []

    # Find indices that are labelled as bottles
    if all_types:
        indices = list(range(len(r['class_ids'])))
    else:
        for i in range(len(r['class_ids'])):
            if categories.count(class_names[r['class_ids'][i]]) > 0:
                objects.append(r['rois'][i])
                indices.append(i)
    
    # Apply the mask for objects
    objs = []
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    for j in indices:
        y0, x0, y1, x1 = r['rois'][j]
        cv.imshow('Portion', train[y0:y1, x0:x1])
        cv.waitKey(0)
        roi = (y0, x0, y1, x1)
        if is_object(query, train, roi, num=10, thresh=40, orb=orb, bf=bf):
            mask = r['masks'][:,:,j]
            masked = apply_mask(train, mask)
            objs.append(masked)
        """
        mask = r['masks'][:,:,j]
        masked = apply_mask(train, mask)
        """
    if len(objs) > 0:
        comb = objs[0]
    else:
        print('No instances of the object have been found...')
        exit(1)
    for i in range(1, len(objs)):
        comb = cv.bitwise_or(comb, objs[i])
    comb = cv.cvtColor(comb, cv.COLOR_BGR2GRAY)

    cont = np.zeros((comb.shape[0], comb.shape[1], 3), dtype=np.uint8)

    contours, heirarchy = cv.findContours(comb, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    res = cv.drawContours(train, contours, -1, (0, 0, 255), 10)

    plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))

    display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    cv.waitKey(0)

if __name__ == '__main__':
    main()