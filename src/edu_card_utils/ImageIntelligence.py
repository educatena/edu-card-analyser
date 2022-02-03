import math
import random
import cv2
import numpy as np

from edu_card_utils.ImageManipulation import lukeContrast, maskeShadowless, thresholdImage
from edu_card_utils.OpenCVUtils import contourSlice, getSquareContourHeight, sortContour
from edu_card_utils.constants import CONTOUR_HEIGHT_ROUND, CONTOUR_VERTEX_X, CONTOUR_VERTEX_Y, MARK_PERCENT, MARKED, NOT_MARKED, PERIMETER_ARC

def nathancyContours(thresh, debug=None):
    mask = np.zeros(thresh.shape[:2], dtype=np.uint8)

    # Find distorted bounding rect
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
            # Find distorted bounding rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.fillPoly(mask, [box], (255,255,255))

    # Find corners
    corners = cv2.goodFeaturesToTrack(mask,4,0.2,10)
    offset = 15
    if (debug is not None):
        for corner in corners:
            x,y = corner.ravel()
            x, y = int(x), int(y)
            cv2.circle(debug,(x,y),5,(36,255,12),-1)
            cv2.rectangle(debug, (x - offset, y - offset), (x + offset, y + offset), (36,255,12), 3)
            print("({}, {})".format(x,y))

    # cv2.imwrite('debug/thresh.png', thresh)
    cv2.imwrite('debug/nathancy_debug.png', debug)
    cv2.imwrite('debug/nathancy_mask.png', mask)
    # cv2.waitKey()


def averageBlurKernel(width, height, ratio=100):
    imageAverageSide = math.floor((width + height) / 2)
    kernelSize = math.floor(imageAverageSide / ratio)
    if ((kernelSize % 2) == 0): kernelSize += 1

    return ( int(kernelSize), int(kernelSize))

def normalizedImage(image, debug = None):
    width = image.shape[1]
    height = image.shape[0]

    contrast = lukeContrast(image)
    shadowless = maskeShadowless(contrast)
    gray = cv2.cvtColor(shadowless, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, averageBlurKernel(width, height, 200), 0)

    if (debug is not None):
        cv2.imwrite(f"debug/normal_0_contrast_{random.randint(0,999999999)}.png", contrast)
        cv2.imwrite(f"debug/normal_1_shadowless_{random.randint(0,999999999)}.png", shadowless)
        cv2.imwrite(f"debug/normal_2_gray_{random.randint(0,999999999)}.png", gray)
        cv2.imwrite(f"debug/normal_3_blurred_{random.randint(0,999999999)}.png", blurred)

    return blurred

def findContours(source, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE, debug = None):
    threshold_image = thresholdImage(normalizedImage(source, debug=debug), debug=debug)

    contours = cv2.findContours(
        threshold_image,
        mode,
        method
    )[0]

    if contours is not None and len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if debug is not None:
        circled = source.copy()
        for contour in contours:
            for point in contour:
                cv2.circle(circled, (point[CONTOUR_VERTEX_X], point[CONTOUR_VERTEX_Y]), 2, (255,0,0), thickness=1)

        downCircled = cv2.resize(circled, (int(circled.shape[1]/2), int(circled.shape[0]/2)))

        cv2.imwrite(f"debug/contour_points_{random.randint(0,999999999)}.png", downCircled)

    return contours

def findSquares(image_contours, image, debug = None):
    random.seed()

    squaresByHeight = {}
    biggestHeight = 0

    for contour in image_contours:
        perimeter = PERIMETER_ARC * cv2.arcLength(contour, True)
        approximate = cv2.approxPolyDP(contour, perimeter, True)

        if len(approximate) == 4:
            # Lets sort the points so that we have a uniform distribution in x and y
            approximate = sortContour(approximate)

            # We round the height so we have some pixel tolerance for very similar contour heights
            height = np.around(getSquareContourHeight(approximate), CONTOUR_HEIGHT_ROUND)

            # print(f'reading contour height: {height} definition:{self.readableContour(approximate)}')

            if (height > biggestHeight): biggestHeight = height

            if not height in squaresByHeight:
                squaresByHeight[height] = []

            squaresByHeight[height].append(approximate)

            try:
                if (height > 10 and debug):
                    cv2.imwrite(f'debug/squares_{height}_{random.randint(1,99999)}.png', contourSlice(image, approximate))
            except Exception:
                print('Could not save square slice')

    return (squaresByHeight, biggestHeight)


def readDarkness(sourceImage, center, radius = 10, percentage = MARK_PERCENT, mark = MARKED, unmark = NOT_MARKED):

    radius = radius
    centerX = center[0]
    centerY = center[1]

    x1 = centerX - radius
    y1 = centerY - radius
    x2 = centerX + radius
    y2 = centerY + radius

    # cv2.circle(graycopy, (circle[0], circle[1]), circle[2], (0,0,255), thickness=1)

    slice = sourceImage[y1:y2, x1:x2]

    circleHistogram = cv2.calcHist([slice], [0], None, [2], [0,256])

    darks = circleHistogram[0,0]
    brights = circleHistogram[1,0]
    totals = darks + brights

    marked = darks > int(percentage * totals)

    return mark if marked else unmark

def readCircles(sourceImage, circles, percentage = MARK_PERCENT, mark = MARKED, unmark = NOT_MARKED):
    result = []
    for i, circle in enumerate(circles):
        radius = circle[2]
        centerX = circle[0]
        centerY = circle[1]
        result.append(readDarkness(sourceImage, (centerX, centerY), radius))

    return result

def readQRCode(image):
    reader = cv2.QRCodeDetector()

    try:
        decodedText, points, _ = reader.detectAndDecode(image)
    except Exception:
        return 'im slowly going insane'

    return decodedText