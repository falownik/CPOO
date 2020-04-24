import cv2
import numpy as np
import argparse
from math import *
import os

class fruit_detection:
    def __init__(self, filename):
        self.filename = filename
        self.img_original = cv2.imread(self.filename)
        height, width, depth = self.img_original.shape

        scale =  height / 200

        self.img_original = cv2.resize(self.img_original,(int(width/scale),int(height/scale)))
        
    def edge_detection(self):
        
        #self.dim_x, self.dim_y = self.img_original.shape
        self.img_edge = self.img_original.copy()
        self.threshold1 = 100
        self.threshold2 = 500
        self.img_edge = cv2.Canny(self.img_edge, self.threshold1, self.threshold2)
    
    def show(self):
        cv2.imshow('image', self.img_original)
        cv2.waitKey(0)
    
    def approx_line(self, points):

        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0

        points = np.array(points)

        n = len(points)

        sum_x = sum(points[:,0])
        sum_y = sum(points[:,[1]])
        sum_xy = sum([points[i,0]*points[i,1] for i in range(n)])
        sum_x2 = sum([points[i,0]**2 for i in range(n)])

        a = (float)((n*sum_xy - sum_x*sum_y)/(n*sum_x2 - sum_x**2))

        b = (float)((sum_y - a*sum_x)/n)
  
    def find_contours_orange(self):
        
        img_original = self.img_original
        img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

        mask_orange = cv2.inRange(img_hsv, (0,0,0), (40,255,255))
        mask_white_neg = cv2.inRange(img_hsv, (0,0,0), (255,80,255) )

        mask_white_neg = cv2.bitwise_not(mask_white_neg)

        mask = cv2.bitwise_and(mask_white_neg, mask_orange)
        target = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        img_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        

        contours, _ = cv2.findContours(img_gray.copy(), cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)


        return contours

    def find_contours_banana(self):
        img_original = self.img_original
        img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(img_hsv, (10,0,0), (40,255,255))
        mask_white_neg = cv2.inRange(img_hsv, (0,0,0), (255,80,255) )

        mask_white_neg = cv2.bitwise_not(mask_white_neg)

        mask = cv2.bitwise_and(mask_white_neg, mask_yellow)
        target = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        img_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        

        contours, _ = cv2.findContours(img_gray.copy(), cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)


        return contours

    def find_contours_strawberry(self):

        img_original = self.img_original
        img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

        mask_orange = cv2.inRange(img_hsv, (0,0,0), (10,255,255))
        mask_white_neg = cv2.inRange(img_hsv, (0,0,0), (255,80,255) )     
        mask_white_neg = cv2.bitwise_not(mask_white_neg)

        mask = cv2.bitwise_and(mask_white_neg, mask_orange)
        target = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        img_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        

        contours, _ = cv2.findContours(img_gray.copy(), cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE)


        return contours   


    def detect_circle(self, contour):

        (x,y), r = cv2.minEnclosingCircle(contour)

        return(int(x), int(y)), int(r)


    def detect_fruit(self):
        img_original = self.img_original

        orange_contour = max(self.find_contours_orange(), key = cv2.contourArea)
        banana_contour = max(self.find_contours_banana(), key = cv2.contourArea)
        strawberry_contour = max(self.find_contours_strawberry(), key = cv2.contourArea)

        (x_o, y_o), r_o = self.detect_circle(orange_contour)
        (x_b, y_b), r_b = self.detect_circle(banana_contour)
        (x_s, y_s), r_s = self.detect_circle(strawberry_contour)

        cntArea_orange = cv2.contourArea(orange_contour)
        cntArea_banana = cv2.contourArea(banana_contour)
        cntArea_strawberry = cv2.contourArea(strawberry_contour)

        if r_o != 0:
            cirArea_orange = pi*(r_o**2)
            ratio_orange = cntArea_orange/cirArea_orange
        else:
            ratio_orange = 0
        if r_b != 0:
            cirArea_banana = pi*(r_b**2)
            ratio_banana = cntArea_banana/cirArea_banana
        else:
            ratio_banana = 0
        if r_s != 0:
            cirArea_strawberry = pi*(r_s**2)
            ratio_strawberry = cntArea_strawberry/cirArea_strawberry
        else:
            ratio_strawberry = 0

        fruit = ''

        if ratio_orange > 0.8:
            fruit = 'orange'
        elif  ratio_strawberry > ratio_banana:
            fruit = 'strawberry'
        else:
            fruit = 'banana'


        print(fruit)


dir = 'strawberry/'
for filename in os.listdir(dir):
    print(filename)
    fr = fruit_detection(dir+filename)
    fr.detect_fruit()

