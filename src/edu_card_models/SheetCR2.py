import datetime
import math
from operator import itemgetter
import traceback
import cv2
import numpy

from edu_card_utils.ImageIntelligence import decodeMachineCode, findContours, findSquares, readCircles, readDarkness, readQRCode
from edu_card_utils.ImageManipulation import getSlice, gustavoBrightnessNormalization, perspectiveTransformation, thresholdImage
from edu_card_utils.OpenCVUtils import getContourDimensions, getLimits, getSquareContourCenter, getSquareContourHeight, getSquareContourWidth, imageHeight
from edu_card_utils.constants import HEIGHT_FIRST_VERTEX
from edu_card_utils.coordutils import grid
from tester3 import correct_orientation

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
    [10.0/2549, 1686.0/3509],
    [2536.0/2549, 1686.0/3509],
    [10.0/2549, 3400.0/3509],
    [2536.0/2549, 3400.0/3509]
])
OPTION_PER_QUESTION = 5
QUESTION_PER_PANEL = 25
NUMBERING_FUNCTION = lambda panel, question: (question + 1) + (QUESTION_PER_PANEL * panel) - QUESTION_PER_PANEL

class SheetCR2():

    qrData = None
    questions = None
    name = None

    messages = []

    def __init__(self, image, name="test") -> None:
        self.name = name

        self.log(f"Input image dimensions: w={image.shape[1]} h={image.shape[0]}")

        # 1. Normalization and Perspective

        image = correct_orientation(image)

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

        contours = findContours(image, debug=debug)
        self.log(f"Found {len(contours)} contours")
        (squares, tallest) = findSquares(contours, image, debug=debug)

        self.log(f"Found squares by heights: {squares.keys()}")

        anchorHeight = int(CR2_ANCHOR_HEIGHT_RATIO * imageHeight(image))
        anchorMinHeight = (anchorHeight * 0.75)
        anchorMaxHeight = (anchorHeight * 1.50)

        self.log(f"Anchor constraints: height: {anchorHeight}, max:{anchorMaxHeight}, min: {anchorMinHeight}")

        anchorCandidates = []

        for i, height in enumerate(squares):
            # print(height, anchorMinHeight, anchorMaxHeight)
            if (height > anchorMinHeight and height < anchorMaxHeight ):
                for candidate in squares[height]:
                    anchorCandidates = anchorCandidates + [candidate]

        self.log(f"Found {len(anchorCandidates)} anchor candidates", anchorCandidates)

        
        threshold = thresholdImage(image, debug=f'anchors_threshold_{debug}' if debug is not None else None, mode=cv2.THRESH_BINARY_INV)
                    

        anchors = []
        main_anchor = None

        for i, candidate in enumerate(anchorCandidates):
            (width,height) = getContourDimensions(candidate)
            ratio = width/height
            center = getSquareContourCenter(candidate)
            # position = self.getAnchorContourCenter(candidate)

            self.log(f"Candidate {center} has a w/h ratio of {ratio}", {'w':width, 'h':height})

            isQuadrangular = ratio >= 0.9 and ratio <= 1.5
            isRectangle = ratio >= 1.9 and ratio <= 2.70
            dark = readDarkness(threshold, center, radius=anchorHeight, percentage=0.3)

            if (dark == 'X' and isQuadrangular):
                anchors = anchors + [center]
            else:
                self.log(f"Candidate {center} is not fully black or quadrangular, ignored.", candidate)
            # elif (dark == 'X' and isRectangle):
            #     main_anchor = [position]

        # if (len(anchors) == 3 and main_anchor is not None):
        #     m = numpy.int32(main_anchor)
        #     a = numpy.int32(anchors[0])
        #     b = numpy.int32(anchors[1])
        #     c = numpy.int32(anchors[2])
        #     m_to_a = numpy.linalg.norm(m-a)
        #     m_to_b = numpy.linalg.norm(m-b)
        #     m_to_c = numpy.linalg.norm(m-c)

        #     dists = numpy.int32([[0,m_to_a], [1,m_to_b], [2, m_to_c]])
        #     s_dists = numpy.sort(dists, axis=1)[::-1]

        #     anchors = [anchors[distance[0]] for distance in s_dists]

        #     anchors = main_anchor + anchors


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

        self.log("Processed QRCode Rect", transformed)

        return transformed.astype(int)

    def getQPanelRect(self, anchors):
        transformed = numpy.array(anchors) * REFERENCE_QUESTIONPANEL

        self.log("Processed Question Panel Rect", transformed)

        return transformed.astype(int)


    def getQuestions(self, image):
        numberedQuestions = {}

        ref_width = 2525
        ref_height = 1713
        real_height = image.shape[0]
        real_width = image.shape[1]

        panel_count = 5
        start = [
            math.floor(120.0/ref_width * real_width),
            math.floor(40.0/ref_height * real_height)
        ]
        panel_distance = [
            math.floor(518/ref_width * real_width),
            math.floor(515/ref_width * real_width),
            math.floor(520/ref_width * real_width),
            math.floor(510/ref_width * real_width),
            0
        ]
        circle_center_distance = [
            math.floor(72/ref_width * real_width),
            math.floor(70/ref_height * real_height)
        ]
        panel_start = [start[0], start[1]]

        circle_radius = math.floor(28/ref_width * real_width)

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

        multiplier = 1
        for panel in range(0, panel_count):
            panel_circles = grid(panel_start, circle_center_distance, 25, 5, z=circle_radius)

            circleMarks = readCircles(gray, panel_circles, logger=self.log)

            if (DEBUG):
                debug = gray.copy()

                for circle in panel_circles:
                    cv2.circle(debug, (circle[0], circle[1]), circle_radius, (255,0,0), 2)
                    
                cv2.imwrite(f'debug/{self.name}_circles_{panel}.png', debug)

            questions = self.circleMatrix(OPTION_PER_QUESTION, circleMarks)
            panel_start[0] += panel_distance[panel]

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
        self.messages.append({'message': message, 'data': data, 'datetime': date})