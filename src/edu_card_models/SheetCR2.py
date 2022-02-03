import math
from operator import itemgetter
import cv2
import numpy
import traceback

from edu_card_utils.ImageIntelligence import findContours, findSquares, normalizedImage, readCircles, readDarkness, readQRCode
from edu_card_utils.OpenCVUtils import contourSlice, getSquareContourCenter, imageHeight
from edu_card_utils.coordutils import grid

DEBUG = True
CR2_ANCHOR_HEIGHT_RATIO = 0.017101325352714837
TOPLEFT_ANCHOR = 0
TOPRIGHT_ANCHOR = 1
BOTTOMLEFT_ANCHOR = 2
BOTTOMRIGHT_ANCHOR = 3
ORIGIN_ANCHOR = TOPLEFT_ANCHOR
REFERENCE_ANCHORS = numpy.array([
    (83.0, 78.0),
    (1569.0, 78.0),
    (82.0, 2223.0),
    (1569.0, 2223.0)
])
REFERENCE_QRPANEL = numpy.array([
    (1288.0, 244.0),
    (1528.0, 244.0),
    (1288.0, 484.0),
    (1528.0, 484.0)
])
REFERENCE_QUESTIONPANEL = numpy.array([
    (54.0, 1040.0),
    (1598.0, 1040.0),
    (54.0, 2181.0),
    (1598.0, 2181.0)
])
OPTION_PER_QUESTION = 5
QUESTION_PER_PANEL = 25
NUMBERING_FUNCTION = lambda panel, question: (question + 1) + (QUESTION_PER_PANEL * panel) - QUESTION_PER_PANEL

class SheetCR2():

    questions = None
    qrData = None

    def __init__(self, image) -> None:
        anchors = self.findAnchors(image)
        qrCode = self.getQRCodeOrigin(anchors)
        questionPanel = self.getQuestionPanelOrigin(anchors)

        qrcode_img = image[qrCode[0,1]:qrCode[3,1], qrCode[0,0]:qrCode[3,0]]
        questions_img = image[questionPanel[0,1]:questionPanel[3,1], questionPanel[0,0]:questionPanel[3,0]]

        if (DEBUG):
            cv2.imwrite('debug/qrcode.jpg', qrcode_img)
            cv2.imwrite('debug/questions.jpg', questions_img)

        self.qrData = self.getQRData(qrcode_img)
        self.questions = self.getQuestions(questions_img)

    def findAnchors(self, image, debug=DEBUG):
        contours = findContours(image, debug=debug)
        (squares, tallest) = findSquares(contours, image, debug=debug)

        anchorHeight = int(CR2_ANCHOR_HEIGHT_RATIO * imageHeight(image))
        anchorMinHeight = (anchorHeight * 0.9)
        anchorMaxHeight = (anchorHeight * 1.1)

        anchorCandidates = []

        for i, height in enumerate(squares):
            if (height > anchorMinHeight and height < anchorMaxHeight ):
                anchorCandidates = anchorCandidates + squares[height]

        anchors = []

        for i, candidate in enumerate(anchorCandidates):
            center = getSquareContourCenter(candidate)
            dark = readDarkness(image, center, radius=anchorHeight, percentage=0.80)
            if (dark):
                anchors = anchors + [center]

        if debug is not None:
            for i, anchor in enumerate(anchors):
                cv2.circle(image, (anchor[0], anchor[1]), 10, (255,0,255), thickness=5)
            cv2.imwrite(f"debug/anchors.png", image)

        return anchors

    def getQRCodeOrigin(self, anchors):
        transformed = numpy.array(anchors) * (
            REFERENCE_QRPANEL / REFERENCE_ANCHORS
        )

        return transformed.astype(int)

    def getQuestionPanelOrigin(self, anchors):
        transformed = numpy.array(anchors) * (
            REFERENCE_QUESTIONPANEL / REFERENCE_ANCHORS
        )

        return transformed.astype(int)


    def getQuestions(self, image):
        numberedQuestions = {}

        ref_width = 1545
        ref_height = 1142
        real_height = image.shape[0]
        real_width = image.shape[1]

        panel_count = 5
        start = [
            math.floor(103/ref_width * real_width),
            math.floor(92/ref_height * real_height)
        ]
        panel_distance = [
            math.floor(303/ref_width * real_width),
            math.floor(300/ref_width * real_width),
            math.floor(303/ref_width * real_width),
            math.floor(297/ref_width * real_width),
            0
        ]
        circle_center_distance = [
            math.floor(43/ref_width * real_width),
            math.floor(43/ref_height * real_height)
        ]
        panel_start = [start[0], start[1]]

        circle_radius = math.floor(16/ref_width * real_width)

        gray = normalizedImage(image, DEBUG)

        multiplier = 1
        for panel in range(0, panel_count):
            panel_circles = grid(panel_start, circle_center_distance, 25, 5, z=circle_radius)

            circleMarks = readCircles(gray, panel_circles)

            if (DEBUG):
                debug = image.copy()

                for circle in panel_circles:
                    cv2.circle(debug, (circle[0], circle[1]), circle_radius, (255,0,0), 2)
                    
                cv2.imwrite(f'debug/circles_{panel}.png', debug)

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
        readText = readQRCode(source)
        
        return readText