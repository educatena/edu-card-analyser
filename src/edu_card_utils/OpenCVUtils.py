from math import floor
import numpy

from edu_card_utils.constants import CONTOUR_VERTEX_X, CONTOUR_VERTEX_Y, HEIGHT_FIRST_VERTEX, HEIGHT_SECOND_VERTEX

def imageHeight(image):
    return image.shape[0]

def imageWidth(image):
    return image.shape[1]

def sortContour(contour):
    sortedContour = numpy.sort(contour, 1)

    return sortedContour

def getSquareContourHeight(contour):

    y1 = contour[HEIGHT_FIRST_VERTEX][CONTOUR_VERTEX_Y]
    y2 = contour[HEIGHT_SECOND_VERTEX][CONTOUR_VERTEX_Y]

    height = abs(y1 - y2)

    return height

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
    xySums = [
            (lambda coordinate: [point[0] + point[1] for point in coordinate] + [i])(coordinate) for i, coordinate in enumerate(contour)
    ]

    xySums = sorted(xySums)

    closestPoint = xySums[0]
    furthestPoint = xySums[len(xySums) -1]

    x1 = contour[closestPoint[1]][CONTOUR_VERTEX_X]
    y1 = contour[closestPoint[1]][CONTOUR_VERTEX_Y]
    x2 = contour[furthestPoint[1]][CONTOUR_VERTEX_X]
    y2 = contour[furthestPoint[1]][CONTOUR_VERTEX_Y]

    return (abs(x2-x1), abs(y2-y1))