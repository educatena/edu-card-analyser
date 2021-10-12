import cv2
import numpy
import random
import math
import collections
from matplotlib import pyplot as plt

"""
BaseSheet represents and collects
basic methods used with computer vision to initiliaze
and find some primitive structures in the image.
"""

class BaseSheet:

    # Gaussian blur parameters
    BLUR_KERNEL_SIZE = (3, 3)
    BLUR_SIGMA_X = 0
    
    # Canny (used for edge detection) parameters
    CANNY_1ST_THRESHOLD = 1
    CANNY_2ND_THRESHOLD = 2

    # Contour parameters
    CONTOUR_MODE = cv2.RETR_EXTERNAL
    CONTOUR_METHOD = cv2.CHAIN_APPROX_SIMPLE

    # Source untouched image
    source = None

    sourceWidth = None
    sourceHeight = None

    # Image prepared to be fed into the edge detection algorithm
    edged = None

    # Python List of contours detected
    contours = None
    
    def __init__(self, image) -> None:
        self.source = image
        self.sourceWidth = self.source.shape[1]
        self.sourceHeight = self.source.shape[0]

        self.BLUR_KERNEL_SIZE = self.getAverageBlurKernel(400)

        self.edged = self.blackWhiteEdgeImage(image)
        self.contours = self.findContours()
    
    def getAverageBlurKernel(self, imageRatio=100):
        imageAverageSide = math.floor((self.sourceWidth + self.sourceHeight) / 2)
        kernelSize = math.floor(imageAverageSide / imageRatio)
        if ((kernelSize % 2) == 0): kernelSize += 1

        return ( int(kernelSize), int(kernelSize))

    # Grayscales, blurs a little bit then uses canny on the image to prepare it
    # on how the contour detection algorithm needs.
    def blackWhiteEdgeImage(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.BLUR_KERNEL_SIZE, self.BLUR_SIGMA_X)
        thresholded = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # edged = cv2.Canny(thresh, self.CANNY_1ST_THRESHOLD, self.CANNY_2ND_THRESHOLD)
        return thresholded

    def findContours(self):

        contours = cv2.findContours(
            self.edged.copy(),
            self.CONTOUR_MODE,
            self.CONTOUR_METHOD
        )[0]

        if contours is not None and len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

        return contours

    def getSquareContourHeight(self, contour):
        y1 = contour[0][0, 1]
        y2 = contour[1][0, 1]

        height = abs(y1 - y2)

        return height

    # Uses a contour to get an image 'slice' of that contour area.
    def getSubImage(self, source, contour):
        x1 = contour[0][0, 0]
        y1 = contour[0][0, 1]
        x2 = contour[2][0, 0]
        y2 = contour[2][0, 1]
        return source[y1:y2, x1:x2]

    # Takes image_contours and extract squares from image whose y/x ratios are above ratio_threshold
    def findSquares(self, image_contours, image):
        random.seed()

        squaresByHeight = {}
        biggestHeight = 0

        for contour in image_contours:
            perimeter = 0.02 * cv2.arcLength(contour, True)
            approximate = cv2.approxPolyDP(contour, perimeter, True)

            if len(approximate) == 4:
                # We round the height so we have some pixel tolerance for very similar contour heights
                height = numpy.around(self.getSquareContourHeight(approximate), -1)

                if (height > biggestHeight): biggestHeight = height

                if not height in squaresByHeight:
                    squaresByHeight[height] = []

                squaresByHeight[height].append(approximate)

        squares = {}
        for contour in squaresByHeight[biggestHeight]:
            slice = self.getSubImage(image, contour)
            y = slice.shape[0] or 1
            x = slice.shape[1] or 1

            if (slice.size != 0):
                squares[contour[0][0,0]] = slice
        
        return squares

    def closest(self, lst, K):
        lst = numpy.asarray(lst)
        idx = (numpy.abs(lst - K)).argmin()
        return lst[idx]

    def yAxisPool(self, circles, deltaDeviation):
        yAxis = numpy.unique(circles[:,1])

        lastY = 0
        for i, y in enumerate(yAxis):
            if (i == 0): lastY = y
            delta = abs(y - lastY)
            if (delta <= deltaDeviation):
                yAxis[i] = lastY
            else:
                lastY = y

        yAxis = sorted(list(set(yAxis)))

        return yAxis

    def snappedYs(self, coordinates, deviation=20):
        yAxisPool = numpy.array(self.yAxisPool(coordinates, deviation))
        
        for coordinate in coordinates:
            originalY = coordinate[1]
            nearY = self.closest(yAxisPool, originalY)
            # print(f'original: {originalY} \t\t near: {nearY}')
        
        x = numpy.subtract.outer(coordinates[:,1], yAxisPool)

        y = numpy.argmin(abs(x), axis=1)
        
        return yAxisPool[y]

    # Detects circles on image
    def findCircles(self, image):
        # TO DO: Extract parameters to either class constants or method parameters.
        x = image.shape[1] or 1

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, self.getAverageBlurKernel(600)[0])
        thresholded = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # 350 625 50
        param1 = int( 350 - (x/625 * 50) ) # I have no idea what this parameter is yet, lets mess around with it
        # 35 625 10
        param2 = int( 35 + (x/625 * 10) ) # Smaller values means possibly detecting more false circles

        # 0.12
        maxDiameter = int(x * 0.12) #9 # Maximum circle diameter
        # 0.8
        minDiameter = int(0.8 * maxDiameter) # Minimum circle diamenter

        maxRadius = math.ceil(maxDiameter / 2)
        minRadius = math.ceil(minDiameter / 2)
        
        # 0.032
        minDist = (minRadius * 2 + math.floor(x * 0.032)) # Minimum distance between each circle center.

        # Adjusting minRadius, maxRadius, param2 and minDist seems to yield more detections if done right.

        circles = cv2.HoughCircles(
            thresholded,
            cv2.HOUGH_GRADIENT,
            2,
            minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        if circles is not None:
            circles = numpy.uint16(
                numpy.fix(circles / (1,10,1)) * (1,10,1)
            )

            #PARAM 20 = ySnapTolerate
            snappedYs = self.snappedYs(circles[0], 20)

            for i in enumerate(circles[0]):
                circles[0][i,1] = snappedYs[i]

            return sorted(circles[0], key=lambda v: [v[1], v[0]])
        else:
            return numpy.array()
    