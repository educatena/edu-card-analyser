import math
import cv2
import numpy
import traceback
from edu_card_models.BaseSheet import BaseSheet
from operator import itemgetter

from edu_card_utils.coordutils import grid

FIRST_CONTOUR_X = 0,0,0
PANEL_KEY = lambda contour: contour[FIRST_CONTOUR_X]
PANEL_BLUR_RATIO = 600
OPTION_PER_QUESTION = 5
QUESTION_PER_PANEL = 25
NUMBERING_FUNCTION = lambda panel, question: (question + 1) + (QUESTION_PER_PANEL * panel) - QUESTION_PER_PANEL

class SheetCR1(BaseSheet):
    panels = None
    questions = None
    qrData = None
    squares = None
    questionPanel = None

    def __init__(self, image) -> None:
        super().__init__(image)

        self.questionPanel = self.getQuestionsPanel()
        self.questions = self.getQuestions()
        self.qrData = self.getQRData(self.source)

    def getQuestionPanels(self, image):
        edged = self.blackWhiteEdgeImage(image)
        contours = self.findContours(edged.copy())

        return self.getTallestSquares(contours, image)
    
    def getQuestionsPanel(self):
        (zones, tallest) = self.findSquares(self.contours, self.source)

        contours = zones[tallest]

        image = self.getSubImage(self.source, contours[0])

        return image

    def getMeta(self):
        return self.meta

    def getTallestSquares(self, contours, source):
        (zones, tallest) = self.findSquares(contours, source)

        self.meta['tallestSquare'] = tallest.item() if type(tallest) != int else tallest
        self.meta['squares'] = [
            (
                lambda contours: [height.item()] + [
                    self.readableContour(contour) for contour in contours
                ]
            )(zones[height]) for height in zones
        ]

        tallestZones = {}
        for contour in zones[tallest]:
            slice = self.getSubImage(source, contour)
            y = slice.shape[0] or 1
            x = slice.shape[1] or 1

            if (slice.size != 0):
                tallestZones[PANEL_KEY(contour)] = slice

        return (tallestZones, zones)
    
    def getQuestions(self):
        numberedQuestions = {}
        image = self.questionPanel

        ref_width = 1109
        ref_height = 831
        real_height = image.shape[0]
        real_width = image.shape[1]

        panel_count = 5
        start = [
            math.floor(72/ref_width * real_width),
            math.floor(61/ref_height * real_height)
        ]
        panel_distance = [
            math.floor(218/ref_width * real_width),
            math.floor(217/ref_width * real_width),
            math.floor(218/ref_width * real_width),
            math.floor(215/ref_width * real_width),
            0
        ]
        circle_center_distance = [
            math.floor(30.5/ref_width * real_width),
            math.floor(30/ref_height * real_height)
        ]
        panel_start = [start[0], start[1]]

        circle_radius = math.floor(12/ref_width * real_width)

        threshold, gray = itemgetter('threshold', 'gray')(self.circleImagePrep(image, PANEL_BLUR_RATIO))

        multiplier = 1
        for panel in range(0, panel_count):
            panel_circles = grid(panel_start, circle_center_distance, 25, 5, z=circle_radius)

            circleMarks = self.readCircles(gray, panel_circles)
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
        readText = self.readQRCode(source)
        
        return readText

    def toDict(self):
            information = {}
            
            try:

                questions = self.questions
                qrData = self.qrData
                information['meta'] = self.meta

                information['data'] = {
                    'questions': questions,
                    'qr': qrData,
                    'version': 'CR1'
                }

            except Exception as error:
                information['error'] = {
                    'message': str(error),
                    'detailed': traceback.format_exc()
                }
            
            return information
