import os
from time import time
import cv2
import numpy
import random
import math
from matplotlib import pyplot as plt
from datetime import datetime

from edu_card_utils.ImageManipulation import lukeContrast, maskeShadowless, nathancyEdged
from edu_card_utils.ImageIntelligence import nathancyContours
from edu_card_utils.constants import BLUR_KERNEL_RATIO, BLUR_SIGMA_X, CONTOUR_HEIGHT_ROUND, CONTOUR_METHOD, CONTOUR_MODE, CONTOUR_VERTEX_X, CONTOUR_VERTEX_Y, HEIGHT_FIRST_VERTEX, HEIGHT_SECOND_VERTEX, MARK_PERCENT, MARKED, NOT_MARKED, PERIMETER_ARC

"""
BaseSheet represents and collects
basic methods used with computer vision to initiliaze
and find some primitive structures in the image.
"""

class BaseSheet:

    # Source untouched image
    source = None

    sourceWidth = None
    sourceHeight = None

    source_blur_kernel = None

    # Image prepared to be fed into the edge detection algorithm
    edged = None

    # Python List of contours detected
    contours = None

    # Temporary object for meta-info
    meta = {}
    
    def __init__(self, image) -> None:
        self.source = image
        self.sourceWidth = self.source.shape[1]
        self.sourceHeight = self.source.shape[0]

        self.source_blur_kernel = self.getAverageBlurKernel(BLUR_KERNEL_RATIO)

        self.edged = self.blackWhiteEdgeImage(image)
        self.contours = self.findContours()

        self.meta['circle_count'] = 0
    
    def getAverageBlurKernel(self, imageRatio=100):
        imageAverageSide = math.floor((self.sourceWidth + self.sourceHeight) / 2)
        kernelSize = math.floor(imageAverageSide / imageRatio)
        if ((kernelSize % 2) == 0): kernelSize += 1

        return ( int(kernelSize), int(kernelSize))

    # Grayscales, blurs a little bit then uses threshold on the image to prepare it
    # on how the contour detection algorithm needs.
    def blackWhiteEdgeImage(self, image):
        # random.seed(time())

        contrast = lukeContrast(image)
        cv2.imwrite(f"debug/edged_0_contrast_{random.randint(0,999999999)}.png", contrast)
        
        shadowless = maskeShadowless(contrast)
        cv2.imwrite(f"debug/edged_0_shadowless_{random.randint(0,999999999)}.png", shadowless)

        gray = cv2.cvtColor(shadowless, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"debug/edged_0_gray_{random.randint(0,999999999)}.png", gray)

        blurred = cv2.GaussianBlur(gray, self.source_blur_kernel, BLUR_SIGMA_X)
        cv2.imwrite(f"debug/edged_1_blurred_{random.randint(0,999999999)}.png", blurred)

        # return nathancyEdged(shadowless, self.source_blur_kernel)

        # threshold_mode = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        threshold_mode = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU

        thresholded = cv2.threshold(blurred, 0, 255, threshold_mode)[1]
        cv2.imwrite(f"debug/edged_2_threshold_{random.randint(0,999999999)}.png", thresholded)

        # edges = cv2.Canny(shadowless,100, 200)
        # cv2.imwrite(f"debug/edged_1_canny_{random.randint(0,999999999)}.png", edges)

        return thresholded

    def findContours(self):

        contours = cv2.findContours(
            self.edged.copy(),
            CONTOUR_MODE,
            CONTOUR_METHOD
        )[0]

        # contours = nathancyContours(self.edged.copy(), self.source)

        if contours is not None and len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if self.source is not None:
            circled = self.source.copy()
            for contour in contours:
                for point in contour:
                    cv2.circle(circled, (point[CONTOUR_VERTEX_X], point[CONTOUR_VERTEX_Y]), 2, (255,0,0), thickness=1)
            downCircled = cv2.resize(circled, (int(circled.shape[1]/2), int(circled.shape[0]/2)))
            stamp = datetime.now().strftime("%d%H%M%S%f")
            # cv2.imshow(f"{stamp}_circles_{len(circles[0])}.png", downCircled)
            cv2.imwrite(f"debug/{stamp}_contour_points_{random.randint(0,999999999)}.png", downCircled)

        return contours

    def sortContour(self, contour):
        sortedContour = numpy.sort(contour, 1)

        return sortedContour

    def getSquareContourHeight(self, contour):

        y1 = contour[HEIGHT_FIRST_VERTEX][CONTOUR_VERTEX_Y]
        y2 = contour[HEIGHT_SECOND_VERTEX][CONTOUR_VERTEX_Y]

        height = abs(y1 - y2)

        return height

    # Uses a contour to get ''an image 'slice' of that contour area.
    def getSubImage(self, source, contour):

        xySums = [
            (lambda coordinate: [point[0] + point[1] for point in coordinate] + [i])(coordinate) for i, coordinate in enumerate(contour)
        ]

        xySums = sorted(xySums)

        closestPoint = xySums[0]
        furthestPoint = xySums[len(xySums) -1]

        # print('\n\n\n', contour, '\n', f'closest point {closestPoint} {contour[closestPoint[1]]}, furthest point {furthestPoint} {contour[furthestPoint[1]]}' , '\n\n\n')

        x1 = contour[closestPoint[1]][CONTOUR_VERTEX_X]
        y1 = contour[closestPoint[1]][CONTOUR_VERTEX_Y]
        x2 = contour[furthestPoint[1]][CONTOUR_VERTEX_X]
        y2 = contour[furthestPoint[1]][CONTOUR_VERTEX_Y]
        return source[y1:y2, x1:x2]

    # Takes image_contours and extract squares from image whose y/x ratios are above ratio_threshold
    def findSquares(self, image_contours, image):
        random.seed()

        squaresByHeight = {}
        biggestHeight = 0

        for contour in image_contours:
            perimeter = PERIMETER_ARC * cv2.arcLength(contour, True)
            approximate = cv2.approxPolyDP(contour, perimeter, True)

            if len(approximate) == 4:
                # Lets sort the points so that we have a uniform distribution in x and y
                approximate = self.sortContour(approximate)

                # We round the height so we have some pixel tolerance for very similar contour heights
                height = numpy.around(self.getSquareContourHeight(approximate), CONTOUR_HEIGHT_ROUND)

                # print(f'reading contour height: {height} definition:{self.readableContour(approximate)}')

                if (height > biggestHeight): biggestHeight = height

                if not height in squaresByHeight:
                    squaresByHeight[height] = []

                squaresByHeight[height].append(approximate)

                try:
                    if (height > 10):
                        cv2.imwrite(f'debug/squares_{random.randint(1,99999)}.png', self.getSubImage(image, approximate))
                except Exception:
                    print('oh noes, an exception!')

        return (squaresByHeight, biggestHeight)

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
        
        x = numpy.subtract.outer(coordinates[:,1], yAxisPool)

        y = numpy.argmin(abs(x), axis=1)
        
        return yAxisPool[y]

    # Detects circles on image
    def findCircles(self, image, dp, ref_width, distance_ratio, diameter_ratio, y_snap=20, min_diameter=0.8, p1_base=350, p1_grow=50, p2_base=35, p2_grow=-10, imagedbg=None):
        # TO DO: Extract parameters to either class constants or method parameters.
        x = image.shape[1] or 1
        if x < 50: return []

        # 350 625 50
        param1 = int( p1_base - (x/ref_width * p1_grow) ) # I have no idea what this parameter is yet, lets mess around with it
        # 35 625 10
        param2 = int( p2_base + (x/ref_width * p2_grow) ) # Smaller values means possibly detecting more false circles

        # 0.12
        maxDiameter = int(x * diameter_ratio) #9 # Maximum circle diameter
        # 0.8
        minDiameter = int(min_diameter * maxDiameter) # Minimum circle diamenter

        maxRadius = math.ceil(maxDiameter / 2)
        minRadius = math.ceil(minDiameter / 2)
        
        # 0.032
        minDist = (minRadius * 2 + math.floor(x * distance_ratio)) # Minimum distance between each circle center.

        # Adjusting minRadius, maxRadius, param2 and minDist seems to yield more detections if done right.
        
        params = {
            'dp': dp,
            'minDist': minDist,
            'param1': param1,
            'param2': param2,
            'minRadius': minRadius,
            'maxRadius': maxRadius,
            'minDiameter': minDiameter,
            'maxDiameter': maxDiameter,
            'x': x,
            'diameter_ratio': diameter_ratio
        }

        cp = self.meta.get('circle_params') or []
        cp.append(params)
        self.meta['circle_params'] = cp

        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp,
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
            snappedYs = self.snappedYs(circles[0], y_snap)

            for i, circle in enumerate(circles[0]):
                self.meta['circle_count'] += 1
                circles[0][i,1] = snappedYs[i]
            
            if imagedbg is not None:
                circled = imagedbg.copy()
                for circle in circles[0]:
                    cv2.circle(circled, (circle[0], circle[1]), circle[2], (0,0,255), thickness=2)
                downCircled = cv2.resize(circled, (int(circled.shape[1]/2), int(circled.shape[0]/2)))
                stamp = datetime.now().strftime("%d%H%M%S%f")
                # cv2.imshow(f"{stamp}_circles_{len(circles[0])}.png", downCircled)
                cv2.imwrite(f"{stamp}_circles_{len(circles[0])}.png", downCircled)

            return sorted(circles[0], key=lambda v: [v[1], v[0]])
        else:
            return numpy.array([])
    
    def readCircles(self, sourceImage, circles, percentage = MARK_PERCENT, mark = MARKED, unmark = NOT_MARKED):
        result = []

        # graycopy = sourceImage.copy()

        for i, circle in enumerate(circles):

            radius = circle[2]
            centerX = circle[0]
            centerY = circle[1]

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

            result.append(mark if marked else unmark)

        # stamp = datetime.now().strftime("%d%H%M%S%f")
        # cv2.imwrite(f"{stamp}_circles_{len(circles)}.png", graycopy)
        
        return result

    def circleImagePrep(self, image, blur_ratio=400):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, self.getAverageBlurKernel(blur_ratio)[0])
            thresholded = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            return {
                'gray': gray,
                'blurred': blurred,
                'threshold': thresholded
            }
        except Exception:
            print("oh noes, an error, yuwu say??") 
            return {
                'gray': image,
                'blurred': image,
                'threshold': image
            }

    def readableContour(self, contour, native=False):
        if native:
            return str(contour)
        
        return str([
            [f'x{i}:{point[0][0]} y{i}:{point[0][1]}'] for i, point in enumerate(contour.tolist())
        ])

    def readQRCode(self, image):
        reader = cv2.QRCodeDetector()

        try:
            decodedText, points, _ = reader.detectAndDecode(image)
        except Exception:
            return 'im slowly going insane'

        return decodedText