
from collections import OrderedDict
from edu_card_utils.ImageIntelligence import chamithDivakalReadCircles, getImageCornerRects, histogram_normal, normalizeLuminance
from edu_card_utils.ImageManipulation import thresholdImage
from edu_card_utils.OpenCVUtils import drawBoundingRect, getAverageColor, rectSlice
import cv2
from scipy.signal import convolve2d
import numpy

def readNibble(nibble):
    nibble_identity = f'nibble:{nibble.shape[0]}-{nibble.shape[1]}'
    nibble = cv2.cvtColor(nibble, cv2.COLOR_BGR2GRAY)
    nibble = (255 - thresholdImage(nibble))
    h, w = nibble.shape[:2]
    nibble_ratio = h/w

    if (nibble_ratio < 0.70 or nibble_ratio >= 1.0 or (h < 10 or w < 10)):
        return 0

    corners = getImageCornerRects(nibble, 0.5)

    # cv2.imwrite(f'debug/{nibble_identity}-d.png', nibble)

    # print(f'{nibble_identity}_r{nibble_ratio}')

    corner_values = OrderedDict({
        'top-left': None,
        'top-right': None,
        'bottom-left': None,
        'bottom-right': None
    })
    corner_colors = OrderedDict({
        'top-left': None,
        'top-right': None,
        'bottom-left': None,
        'bottom-right': None
    })
    corner_states = {
        'top-left': 1,
        'top-right': 2,
        'bottom-left': 3,
        'bottom-right': 4
    }

    # Color Values Calculation

    average_colors = ''

    for corner in dict.keys(corners):

        quadrantImage = rectSlice(nibble, corners[corner])
        average_color = getAverageColor(quadrantImage)
        average_colors = f'{average_colors}\n{corner}:v_{average_color}'

        corner_colors[corner] = average_color
        corner_values[corner] = numpy.sum(average_color) / 3
        # cv2.imwrite(f'debug/{nibble_identity}-{corner}.png', quadrantImage)

    
    # Corner Selection
    
    whitest_corner = max(corner_values, key=corner_values.get)
    # white_delta = numpy.array([255,255,255]) - corner_colors[whitest_corner]
    blackest_corner = min(corner_values, key=corner_values.get)
    # black_delta = numpy.array([1,1,1]) - corner_colors[blackest_corner]


    # for corner in dict.keys(corner_colors):
    #     corner_colors[corner] = corner_colors[corner] + (white_delta + black_delta)
    #     corner_values[corner] = numpy.sum(corner_colors[corner]) / 3

    # Differential Calculations
    average_value = numpy.sum(corner_colors[whitest_corner]) / 3
    other_corners = numpy.array([value for key, value in corner_values.items() if key not in [corner]])
    other_average_value = (other_corners.sum() / 3).astype('uint8')

    # triple_average_ratio = other_average_value / average_value
    differential_ratio = (average_value - other_average_value) / 255

    # print(f'{nibble_identity}_dr:{differential_ratio}')

    
    # Evaluation
    if (differential_ratio >= 0.15):
        # cv2.imwrite(f'debug/{nibble_identity}_{whitest_corner}.jpeg', quadrantImage)
        return corner_states[whitest_corner]
    else:
        return 0