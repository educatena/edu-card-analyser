import numpy
from edu_card_models.BaseSheet import BaseSheet
from operator import itemgetter

FIRST_CONTOUR_X = 0,0,0
PANEL_KEY = lambda contour: contour[FIRST_CONTOUR_X]
PANEL_BLUR_RATIO = 600
OPTION_PER_QUESTION = 5
QUESTION_PER_PANEL = 20
NUMBERING_FUNCTION = lambda panel, question: (question + 1) + (QUESTION_PER_PANEL * panel) - QUESTION_PER_PANEL

class SheetV1(BaseSheet):
    panels = None
    questions = None
    studentNumber = None
    squares = None

    def __init__(self, image) -> None:
        super().__init__(image)
        (self.panels, self.squares) = self.getTallestSquares()
        self.questions = self.getQuestions()
        self.studentNumber = self.getStudentNumber()

    def getTallestSquares(self):
        (zones, tallest) = self.findSquares(self.contours, self.source)

        tallestZones = {}
        for contour in zones[tallest]:
            slice = self.getSubImage(self.source, contour)
            y = slice.shape[0] or 1
            x = slice.shape[1] or 1

            if (slice.size != 0):
                tallestZones[PANEL_KEY(contour)] = slice

        return (tallestZones, zones)
    
    def getQuestions(self):
        numberedQuestions = {}
        if len(self.panels) != 0:
            multiplier = 1
            for x in sorted(list(self.panels)):
                image = self.panels[x]
                
                threshold, gray = itemgetter('threshold', 'gray')(self.circleImagePrep(image, PANEL_BLUR_RATIO))

                circles = self.findCircles(threshold, 2, 625, 0.032, 0.12)
                circleMarks = self.readCircles(gray, circles)
                questions = self.circleMatrix(OPTION_PER_QUESTION, circleMarks)

                for i, question in enumerate(questions):
                    numberedQuestions[NUMBERING_FUNCTION(multiplier, i)] = question
                multiplier += 1
        
        return numberedQuestions

    def circleMatrix(self, per_row, circlesArray):
        questions = []
        question = []
        for option in circlesArray:
            question.append(option)
            if (len(question) == per_row):
                questions.append(question)
                question = []
        return questions

    def getStudentNumber(self):
        whichSquare = list(self.squares)[1]
        square = self.squares[whichSquare][0]
        slice = self.getSubImage(self.source, square)

        REF_WIDTH = 762
        DIST_RATIO = 0.02
        DIAMETER_RATIO = 72 / 762
        MIN_DIAMETER_RATIO = 0.5

        threshold, gray = itemgetter('threshold', 'gray')(self.circleImagePrep(slice, 600))
        circles = self.findCircles(
            threshold,
            1,
            REF_WIDTH,
            DIST_RATIO,
            DIAMETER_RATIO,
            min_diameter=MIN_DIAMETER_RATIO,
            p2_base=30,
            p2_grow=-8
        )
        marks = self.readCircles(gray, circles)
        marksFound = len(marks)

        if (marksFound == 90):
            markMatrix = self.circleMatrix(9, marks)
            places = [None, None, None, None, None, None, None, None, None]
            for i, markRow in enumerate(markMatrix):
                for mark_place, mark in enumerate(markRow):
                    if mark == 'O':
                        continue
                    else:
                        if (places[mark_place] == None): places[mark_place] = 0
                        places[mark_place] = places[mark_place] + (i+1)
                    
            convert = lambda x: 'X' if x is None else str(x)
            number = ''
            number = number.join([
                convert(e) for e in places
            ])
            return number
        else:
            raise Exception('Não consegui encontrar todos os círculos no quadro de número do aluno.')