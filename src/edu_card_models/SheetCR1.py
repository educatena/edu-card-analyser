import numpy
import traceback
from edu_card_models.BaseSheet import BaseSheet
from operator import itemgetter

FIRST_CONTOUR_X = 0,0,0
PANEL_KEY = lambda contour: contour[FIRST_CONTOUR_X]
PANEL_BLUR_RATIO = 600
OPTION_PER_QUESTION = 5
QUESTION_PER_PANEL = 20
NUMBERING_FUNCTION = lambda panel, question: (question + 1) + (QUESTION_PER_PANEL * panel) - QUESTION_PER_PANEL

class SheetCR1(BaseSheet):
    panels = None
    questions = None
    qrData = None
    squares = None
    questionPanel = None

    meta = {}

    def __init__(self, image) -> None:
        super().__init__(image)
        (panels, squares) = self.getTallestSquares(self.contours, self.source)
        self.questionPanel = panels[sorted(panels)[0]]

        (self.panels, self.squares) = self.getQuestionPanels(self.questionPanel)

        self.questions = self.getQuestions()
        self.qrData = self.getQRData(self.source)

    def getQuestionPanels(self, image):
        edged = self.blackWhiteEdgeImage(image)
        contours = self.findContours(edged.copy())
        # (zones, tallest) = self.findSquares(contours, image)

        return self.getTallestSquares(contours, image)
        # return zones#[tallest]

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
        # self.meta['circles'] = {}

        numberedQuestions = {}
        if len(self.panels) != 0:
            multiplier = 1
            for x in sorted(list(self.panels)):
                image = self.panels[x]
                
                threshold, gray = itemgetter('threshold', 'gray')(self.circleImagePrep(image, PANEL_BLUR_RATIO))

                circles = self.findCircles(threshold, 2, 264, 0.040, 0.1317, p2_base=40, p2_grow=-5, min_diameter=0.9, imagedbg=None)
                # self.meta['circles'][multiplier] = [circle.tolist() for circle in  circles]
                circleMarks = self.readCircles(gray, circles)
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
        readText = self.readQRCode(source)
        
        return readText

    def toDict(self):
            information = {}
            
            try:

                questions = self.questions
                student_number = self.qrData
                information['meta'] = self.meta

                information['data'] = {
                    'questions': questions,
                    'qr': student_number,
                    'version': 'CR1'
                }

            except Exception as error:
                information['error'] = {
                    'message': str(error),
                    'detailed': traceback.format_exc()
                }
            
            return information
