import cv2 as cv
import numpy as np

"""
Locator Class: Used to locate a provided reference object (iamge) in another image.
Locate.find() takes a query image and a train image and returns the (x,y) coordinate indicating where the query was found in the train.
Benji Barnes, Aaron Mills
06/01/2021
"""

class Locator:
    def find(self, query, train, num=10, minMatches=50):
        img1 = cv.cvtColor(query, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(train, cv.COLOR_BGR2GRAY)

        orb = cv.ORB_create()

        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1,des2)

        matches = sorted(matches, key = lambda x:x.distance)

        if len(matches) < minMatches:
            print(f'Less than {minMatches} matches were found: {len(matches)}')
            return None

        x = 0
        y = 0

        for match in matches[:num]:
            idx = match.trainIdx
            x0, y0 = kp2[idx].pt
            x += x0
            y += y0
        
        x = int(x//num)
        y = int(y//num)

        return x, y


if __name__ == '__main__':
    query_file_name = input("Enter the file name (and extension) of the query image: ")
    train_file_name = input("Enter the file name (and extension) of the train image: ")
    query = cv.imread(f'images/{query_file_name}')
    train = cv.imread(f'images/{train_file_name}')
    if train.shape[0] > 1000 or train.shape[1] > 1000:
        dim = (int(train.shape[1]/2), int(train.shape[0]/2))
        train = cv.resize(train, dim, interpolation=cv.INTER_AREA)
    cv.imshow('query', query)
    cv.imshow('train', train)
    locator = Locator()
    x, y = locator.find(query, train)

    cv.circle(train, (x, y), 10, (0, 0, 255), -1)

    cv.imshow('result', train)

    cv.waitKey(0)
    