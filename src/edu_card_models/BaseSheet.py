import cv2
import numpy

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
    CANNY_1ST_THRESHOLD = 20
    CANNY_2ND_THRESHOLD = 15

    # Contour parameters
    CONTOUR_MODE = cv2.RETR_EXTERNAL
    CONTOUR_METHOD = cv2.CHAIN_APPROX_SIMPLE

    # Source untouched image
    source = None

    # Image prepared to be fed into the edge detection algorithm
    edged = None

    # Python List of contours detected
    contours = None
    
    def __init__(self, image) -> None:
        self.source = image
        self.edged = self.blackWhiteEdgeImage(image)
        self.contours = self.findContours()

    # Grayscales, blurs a little bit then uses canny on the image to prepare it
    # on how the contour detection algorithm needs.
    def blackWhiteEdgeImage(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.BLUR_KERNEL_SIZE, self.BLUR_SIGMA_X)
        edged = cv2.Canny(blurred, self.CANNY_1ST_THRESHOLD, self.CANNY_2ND_THRESHOLD)
        return edged

    def findContours(self):

        contours = cv2.findContours(
            self.edged.copy(),
            self.CONTOUR_MODE,
            self.CONTOUR_METHOD
        )[0]

        if contours is not None and len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

        return contours

    # Uses a contour to get an image 'slice' of that contour area.
    def getSubImage(self, source, contour):
        x1 = contour[0][0, 0]
        y1 = contour[0][0, 1]
        x2 = contour[2][0, 0]
        y2 = contour[2][0, 1]
        return source[y1:y2, x1:x2]

    # Takes image_contours and extract squares from image whose y/x ratios are above ratio_threshold
    def findSquares(self, image_contours, image, ratio_threshold = 3.0):
        squares = []
        for contour in image_contours:
            perimeter = 0.02 * cv2.arcLength(contour, True)
            approximate = cv2.approxPolyDP(contour, perimeter, True)
            if len(approximate) == 4:
                slice = self.getSubImage(image, approximate)

                y = slice.shape[0] or 1
                x = slice.shape[1] or 1
                ratio = y / x

                if (ratio > ratio_threshold and slice.size != 0):
                    squares.append(slice)
        return squares

    # Detects circles on image
    def findCircles(self, image):
        # TO DO: Extract parameters to either class constants or method parameters.

        # Every circle is appended to this list
        result = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        minDist = 5 # Minimum distance between each circle center.
        param1 = 500 # I have no idea what this parameter is yet, lets mess around with it
        param2 = 30 # Smaller values means possibly detecting more false circles
        minRadius = 7 # Minimum circle radius
        maxRadius = 9 # Maximum circle radius

        # Adjusting minRadius, maxRadius, param2 and minDist seems to yield more detections if done right.

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            2,
            minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        if circles is not None:
            circles = numpy.uint16(numpy.around(circles))

            for i in circles[0,:]:
                result.append(i)

                # Below is temporary code to draw on the source image for us
                # to draw detected circles in the source image.

                # draw the outer circle
                cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)

        # Renders the image on a window called 'circles'
        # If no circles were found, you wont see any...
        cv2.imshow('circles', gray)

        # Makes the previous window await a key before continuing execution.
        # If the window is showing up then disappearing try setting a breakpoint
        # on this function's return statement.
        cv2.waitKey()

        return result
    