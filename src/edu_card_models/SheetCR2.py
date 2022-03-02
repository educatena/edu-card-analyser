import datetime
import math
from operator import itemgetter
import traceback
import cv2
import numpy

from edu_card_utils.ImageIntelligence import chamithDivakalReadCircles, correct_orientation, decodeMachineCode, findContours, findSquares, getImageCornerRects, isContourInsideRect, normalizedImage, readCircles, readDarkness, readQRCode
from edu_card_utils.ImageManipulation import getSlice, gustavoBrightnessNormalization, perspectiveTransformation, thresholdImage
from edu_card_utils.OpenCVUtils import drawBoundingRect, drawContourRect, getContourDimensions, getContourRectCenter, getLimits, getSquareContourCenter, getSquareContourHeight, getSquareContourWidth, imageHeight, rectSlice
from edu_card_utils.constants import HEIGHT_FIRST_VERTEX, MAX_ANCHOR_AREA_PERCENTAGE, MAX_ANCHOR_WIDTH_PERCENTAGE, MIN_ANCHOR_AREA_PERCENTAGE, MIN_ANCHOR_WIDTH_PERCENTAGE
from edu_card_utils.coordutils import grid
from edu_nibble_code.NibbleReader import readNibble

DEBUG = True
CR2_ANCHOR_HEIGHT_RATIO = 0.017101325352714837
TOPLEFT_ANCHOR = 0
TOPRIGHT_ANCHOR = 1
BOTTOMLEFT_ANCHOR = 2
BOTTOMRIGHT_ANCHOR = 3
ORIGIN_ANCHOR = TOPLEFT_ANCHOR
REFERENCE_QRPANEL = numpy.float32([
    [973.0/1199, 123.0/1599],
    [1168.0/1199, 123.0/1599],
    [973.0/1199, 304.0/1599],
    [1168.0/1199, 304.0/1599]
])
REFERENCE_QUESTIONPANEL = numpy.float32([
    [4.0/1199, 764.0/1599],
    [1194.0/1199, 764.0/1599],
    [4.0/1199, 1549.0/1599],
    [1194.0/1199, 1549.0/1599]
])
OPTION_PER_QUESTION = 5
QUESTION_PER_PANEL = 25
NUMBERING_FUNCTION = lambda panel, question: (question + 1) + (QUESTION_PER_PANEL * panel) - QUESTION_PER_PANEL

class SheetCR2():

    def __init__(self, image, name="test") -> None:
        self.qrData = None
        self.questions = None
        self.name = None

        self.messages = []

        self.name = name

        self.log(f"Input image dimensions: w={image.shape[1]} h={image.shape[0]}")

        # 1. Normalization and Perspective

        # image = correct_orientation(image)

        self.log(f"Input image dimensions after rotation correction: w={image.shape[1]} h={image.shape[0]}")

        # 1.1 - Get coordinates for perspective transformation
        anchors = self.findAnchors(image)
        # 1.2 - Apply perspective transformation based on image aspect ratio
        transformed_img = perspectiveTransformation(image, anchors)

        # 2. Rectangles

        limits = getLimits(transformed_img)

        qrcode_rect = self.getQRCodeRect(limits)
        qpanel_rect = self.getQPanelRect(limits)

        qrcode_img = getSlice(transformed_img, qrcode_rect)
        questions_img = getSlice(transformed_img, qpanel_rect)

        if (DEBUG):
            if (qrcode_img.size > 0):
                cv2.imwrite(f'debug/mcr2_{self.name}_qrcode.jpg', qrcode_img)
            if (questions_img.size > 0):
                cv2.imwrite(f'debug/mcr2_{self.name}_questions.jpg', questions_img)
            if (transformed_img.size > 0):
                cv2.imwrite(f'debug/mcr2_{self.name}_perspective.jpg', transformed_img)

        # 3. Analysis

        self.qrData = self.getQRData(qrcode_img)
        self.questions = self.getQuestions(questions_img)

    def findAnchors(self, image):
        debug = f'{self.name}' if DEBUG is not None else None

        normalized_image_a = normalizedImage(image, debug)
        # brightness_normal_image = cv2.equalizeHist(normalized_image_a)
        threshold_image = thresholdImage(normalized_image_a)

        cv2.imwrite(f'debug/normal.jpg', threshold_image)

        # normal_image = (255 - thresholdImage(normal_image, debug=debug))

        contours = findContours(threshold_image, debug=debug, source=image)
        self.log(f"Found {len(contours)} contours")
        (squares, tallest) = findSquares(contours, image, debug=debug)

        self.log(f"Found squares by heights: {squares.keys()}")

        # self.log(f"Anchor constraints: height: {anchorHeight}, max:{anchorMaxHeight}, min: {anchorMinHeight}")

        anchorCandidates = []

        imageWidth = image.shape[1]
        imageHeight = image.shape[0]

        cornerRects = getImageCornerRects(image)

        quadrantContours = {
            'top-left': [],
            'top-right': [],
            'bottom-left': [],
            'bottom-right': []
        }

        imageArea = image.shape[1] * image.shape[0]

        for i, height in enumerate(squares):
                for candidate in squares[height]:

                    for corner in dict.keys(cornerRects):
                        dbg_img = None
                        if (DEBUG): dbg_img = image.copy()
                        insideRect = isContourInsideRect(candidate, cornerRects[corner], dbg_img)
                        if (insideRect):
                            rect = cv2.boundingRect(candidate)
                            if (rect[2] > 0 and rect[3] > 0):
                                nibble = rectSlice(image, rect)
                                w = nibble.shape[1]
                                h = nibble.shape[0]
                                if (w > 0 and h > 0):
                                    code = readNibble(nibble)
                                    if (code != 0):
                                        anchorCandidates += [candidate]
                                        # print('code')
                                        cv2.imwrite(f'debug/nibble_code_w{w}h{h}_c{code}.jpg', nibble)
                            # quadrantContours[corner] += [candidate]


                    # dimensions = getContourDimensions(candidate)
                    # candidateWidth = dimensions[0]
                    # candidateArea = dimensions[0] * dimensions[1]
                    # areaCoverage = candidateArea / imageArea
                    # widthCoverage = candidateWidth / imageWidth
                    # meetsAreaCriteria = areaCoverage >= MIN_ANCHOR_AREA_PERCENTAGE and areaCoverage <= MAX_ANCHOR_AREA_PERCENTAGE
                    # meetsWidthCriteria = widthCoverage >= MIN_ANCHOR_WIDTH_PERCENTAGE and widthCoverage <= MAX_ANCHOR_WIDTH_PERCENTAGE
                    # if (meetsAreaCriteria and meetsWidthCriteria):
                    #     anchorCandidates += [candidate]
        
        self.log(f"Found {len(anchorCandidates)} anchor candidates", anchorCandidates)

        cv2.imwrite(f'debug/rect_inside_rect.jpg', dbg_img)


        anchors = []

        for i, candidate in enumerate(anchorCandidates):
            (width,height) = getContourDimensions(candidate)

            ratio = width/height
            center = getSquareContourCenter(candidate)

            self.log(f"Candidate {center} has a w/h ratio of {ratio}", {'w':width, 'h':height})

            isQuadrangular = ratio >= 0.9 and ratio <= 1.5

            if (isQuadrangular): anchors += [center]


        if debug is not None:
            for i, anchor in enumerate(anchors):
                cv2.circle(image, (anchor[0], anchor[1]), 10, (255,0,255), thickness=5)
            cv2.imwrite(f"debug/mcr2_{debug}_anchors.png", image)

        sorted_anchors = sorted([
            (
                lambda point: [point[0] + point[1]] + [i]
            )(anchor)
            for i, anchor in enumerate(anchors)
        ])

        anchors = [
            (
                lambda position: anchors[position] 
            )(sort_index[1])
            for sort_index in sorted_anchors
        ]

        self.log(f"After sorting and grabbing anchors, {anchors} remained.", anchors)

        anchors = numpy.float32(anchors)

        return anchors

        # return numpy.sort(anchors, axis=1)

    def getQRCodeRect(self, anchors):
        transformed = numpy.array(anchors) * REFERENCE_QRPANEL

        self.log("Extracted QRCode Rect", transformed)

        return transformed.astype(int)

    def getQPanelRect(self, anchors):
        transformed = numpy.array(anchors) * REFERENCE_QUESTIONPANEL

        self.log("Extracted Question Panel Rect", transformed)

        return transformed.astype(int)


    def getQuestions(self, image):
        numberedQuestions = {}

        ref_width = 1189
        ref_height = 784
        real_height = image.shape[0]
        real_width = image.shape[1]

        panel_count = 5
        start = [
            math.floor(56.0/ref_width * real_width),
            math.floor(18.0/ref_height * real_height)
        ]
        panel_distance = [
            math.floor(245/ref_width * real_width),
            math.floor(243/ref_width * real_width),
            math.floor(243/ref_width * real_width),
            math.floor(241/ref_width * real_width),
            0
        ]
        circle_center_distance = [
            math.floor(33/ref_width * real_width),
            math.floor(31.7/ref_height * real_height)
        ]
        panel_start = [start[0], start[1]]

        circle_radius = math.floor(12/ref_width * real_width)

        self.log("Making options matrix with parameters:", {
            'real_h': real_height,
            'real_w': real_width,
            'ref_h': ref_height,
            'ref_w': ref_width,
            'start': start,
            'panel_count': panel_count,
            'panel_distance': panel_distance,
            'circle_center_distance': circle_center_distance,
            'circle_radius': circle_radius
        })

        gray = image

        panels_circles = None

        for panel in range(0, panel_count):
            circles = numpy.array(grid(panel_start, circle_center_distance, 25, 5, z=circle_radius), ndmin=3)[0]
            panels_circles = circles if panel == 0 else numpy.concatenate((panels_circles, circles))

            panel_start[0] += panel_distance[panel]

        option_marks = chamithDivakalReadCircles(panels_circles, image, debug=self.name if DEBUG else None, logger=self.log)

        multiplier = 1
        for panel in range(0, panel_count):
            min = (multiplier * 125) - 125
            max = min + 125

            panel_circles = panels_circles[min:max]
            circleMarks = option_marks[min:max]

            circleMarks = numpy.where(circleMarks == True, 'X', 'O')

            if (DEBUG):
                debug = gray.copy()

                for circle in panel_circles:
                    cv2.circle(debug, (circle[0], circle[1]), circle_radius, (255,0,0), 2)
                    
                cv2.imwrite(f'debug/{self.name}_circles_{panel}.png', debug)

            questions = self.circleMatrix(OPTION_PER_QUESTION, circleMarks)

            for i, question in enumerate(questions):
                numberedQuestions[NUMBERING_FUNCTION(multiplier, i)] = question

            multiplier += 1

        if (len(numberedQuestions) == 0): return None 
        else: return numberedQuestions

    def circleMatrix(self, per_row, circlesArray):
        questions = []
        question = []
        for option in circlesArray:
            question.append(option)
            if (len(question) == per_row):
                questions.append(question)
                question = []
        return questions

    def getQRData(self, source):
        readText = decodeMachineCode(source)

        self.log(f'Got QRCode data: {readText}')
        
        return readText

    def getAnchorContourCenter(self, contour):
        first_vertex = contour[HEIGHT_FIRST_VERTEX]

        # height = getSquareContourHeight(contour)
        # width = getSquareContourWidth(contour)
        # relative_center = (math.floor(height/2), math.floor(width/2))

        return (first_vertex[0,0], first_vertex[0,1])

    def toDict(self):
        information = {}
        
        try:

            questions = self.questions
            qrData = self.qrData
            # information['meta'] = self.meta

            information['logs'] = self.messages
            information['data'] = {
                'questions': questions,
                'qr': qrData[0].data.decode('utf8'),
                'version': 'CR1'
            }

        except Exception as error:
            information['logs'] = self.messages
            information['error'] = {
                'message': str(error),
                'detailed': traceback.format_exc(),
            }
        
        return information

    def log(self, message, data = {}):
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.messages.append({'message': message, 'data': data.__str__(), 'datetime': date})