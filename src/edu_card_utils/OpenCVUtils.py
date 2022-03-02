from math import floor
import math
import random
import numpy
import cv2

from edu_card_utils.constants import CONTOUR_VERTEX_X, CONTOUR_VERTEX_Y, HEIGHT_FIRST_VERTEX, HEIGHT_SECOND_VERTEX

def imageHeight(image):
    return image.shape[0]

def imageWidth(image):
    return image.shape[1]

def sortContour(contour):
    sortedContour = numpy.sort(contour, 1)

    return sortedContour

def getSquareContourHeight(contour):

    boundingRect = cv2.boundingRect(contour)

    return boundingRect[-1]

    # y1 = contour[HEIGHT_FIRST_VERTEX][CONTOUR_VERTEX_Y]
    # y2 = contour[HEIGHT_SECOND_VERTEX][CONTOUR_VERTEX_Y]

    # height = abs(y1 - y2)

    # return height

def getSquareContourWidth(contour):
    x1 = contour[HEIGHT_FIRST_VERTEX][CONTOUR_VERTEX_X]
    x2 = contour[HEIGHT_SECOND_VERTEX][CONTOUR_VERTEX_X]

    width = abs(x1 - x2)

    return width

def getSquareContourCenter(contour):
    xySums = [
            (lambda coordinate: [point[0] + point[1] for point in coordinate] + [i])(coordinate) for i, coordinate in enumerate(contour)
    ]

    xySums = sorted(xySums)

    closestPoint = xySums[0]
    furthestPoint = xySums[len(xySums) -1]

    # print('\n\n\n', contour, '\n', f'closest point {closestPoint} {contour[closestPoint[1]]}, furthest point {furthestPoint} {contour[furthestPoint[1]]}' , '\n\n\n')

    # x1 = [CONTOUR_VERTEX_X]
    # y1 = contour[closestPoint[1]][CONTOUR_VERTEX_Y]
    # x2 = contour[furthestPoint[1]][CONTOUR_VERTEX_X]
    # y2 = contour[furthestPoint[1]][CONTOUR_VERTEX_Y]

    first_vertex = contour[closestPoint[1]]

    height = getSquareContourHeight(contour)
    width = getSquareContourWidth(contour)
    relative_center = (floor(height/2), floor(width/2))

    return (first_vertex[0,0] + relative_center[1], first_vertex[0,1] + relative_center[0])

# Uses a contour to get ''an image 'slice' of that contour area.
def contourSlice(source, contour):

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


def getLimits(source_img):
    width = source_img.shape[1]
    height = source_img.shape[0]
    return numpy.float32([
        [width,height],
    ])

def getContourDimensions(contour):
    rect = cv2.boundingRect(contour)
    # xySums = [
    #         (lambda coordinate: [point[0] + point[1] for point in coordinate] + [i])(coordinate) for i, coordinate in enumerate(contour)
    # ]

    # xySums = sorted(xySums)

    # closestPoint = xySums[0]
    # furthestPoint = xySums[len(xySums) -1]

    # x1 = contour[closestPoint[1]][CONTOUR_VERTEX_X]
    # y1 = contour[closestPoint[1]][CONTOUR_VERTEX_Y]
    # x2 = contour[furthestPoint[1]][CONTOUR_VERTEX_X]
    # y2 = contour[furthestPoint[1]][CONTOUR_VERTEX_Y]

    # return (abs(x2-x1), abs(y2-y1))
    return (rect[-2], rect[-1])


def drawContourRect(image, contour):
    random.seed()
    red = random.randint(0,255)
    green = random.randint(0,255)
    blue = random.randint(0,255)

    color = (red,green,blue)
    color_highlight = (math.floor(red * 1.2), math.floor(green * 1.2), math.floor(blue * 0.8))

    rect = cv2.boundingRect(contour)

    (x, y, w, h) = rect

    area = w * h
    
    drawBoundingRect(image, rect, color)

    cv2.putText(image, f'w:{str(w)}', (x, math.floor(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_highlight)
    cv2.putText(image, f'a:{str(area)}', (x, math.floor(y+h/2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_highlight)



def getContourRectCenter(contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    return (math.floor(x+w/2), math.floor(y+h/2))


def drawBoundingRect(image, bounding_rect, color=(255,255,255), thickness=1):
    (x, y, w, h) = bounding_rect
    p1 = (x, y)
    p2 = (x+w, y+h)
    cv2.rectangle(image, p1, p2, color, thickness, cv2.LINE_AA)

def rectSlice(image, rect):
    p1 = (rect[0], rect[1])
    p2 = (rect[0] + rect[2], rect[1] + rect[3])
    return image[p1[1]:p2[1], p1[0]:p2[0]]

def getAverageColor(image):
    average_color = numpy.sum(image, axis=(0,1)) / (image.shape[0] * image.shape[1])
    return average_color