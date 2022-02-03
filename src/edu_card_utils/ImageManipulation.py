import math
import random
import cv2
import numpy as np


def nathancyEdged(image, blur_kernel):
    random.seed()

    random_id = random.randint(1, 999999999)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"debug/nathan_contour_gray_{random_id}.png", gray)

    # blur = cv2.GaussianBlur(gray, blur_kernel, 0)
    # cv2.imwrite(f"debug/nathan_contour_blur_{random_id}.png", blur)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite(f"debug/nathan_contour_thresh_{random_id}.png", thresh)

    return thresh

def maskeShadowless(img):
        rgb_planes = cv2.split(img)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        # result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)

        # cv2.imwrite('shadows_out.png', result)
        # cv2.imwrite('shadows_out_norm.png', result_norm)

        return result_norm

def lukeContrast(img, clipLimit=3.0):

    #-----Reading the image-----------------------------------------------------
    # img = cv2.imread('Dog.jpg', 1)
    # cv2.imshow("img",img) 

    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    # cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('final', final)

    return final

    #_____END_____#

def thresholdImage(blurred_image, mode = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU, debug = None):
    thresholded = cv2.threshold(blurred_image, 0, 255, mode)[1]

    if (debug is not None):
        cv2.imwrite(f"debug/threshold_{random.randint(0,999999999)}.png", thresholded)
    return thresholded