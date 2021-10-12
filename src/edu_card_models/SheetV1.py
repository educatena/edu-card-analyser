import cv2
from edu_card_models.BaseSheet import BaseSheet


class SheetV1(BaseSheet):
    zones = None
    questions = None

    def __init__(self, image) -> None:
        super().__init__(image)
        self.zones = self.getQuestionZones()
        self.questions = self.getQuestions(20)

    def getQuestionZones(self):
        zones = self.findSquares(self.contours, self.source)

        return zones
    
    def getQuestions(self, perPanel):
        numberedQuestions = {}
        if len(self.zones) != 0:
            multiplier = 1
            for x in sorted(list(self.zones)):

                circles = self.findCircles(self.zones[x])
                circleMarks = self.readCircles(self.zones[x], circles)
                questions = self.getQuestionMatrix(5, circleMarks)

                for i, question in enumerate(questions):
                    numberedQuestions[(i + 1) + (perPanel * multiplier) - perPanel] = question
                multiplier += 1
        
        return numberedQuestions

    def getQuestionMatrix(self, option_per_question, optionArray):
        questions = []
        question = []
        for option in optionArray:
            question.append(option)
            if (len(question) == option_per_question):
                questions.append(question)
                question = []
        return questions
    
    def readCircles(self, sourceImage, circles):
        result = []

        for i, circle in enumerate(circles):

            radius = circle[2]
            centerX = circle[0]
            centerY = circle[1]

            x1 = centerX - radius
            y1 = centerY - radius
            x2 = centerX + radius
            y2 = centerY + radius

            slice = sourceImage[y1:y2, x1:x2]

            circleHistogram = cv2.calcHist([slice], [0], None, [2], [0,256])

            darks = circleHistogram[0,0]
            brights = circleHistogram[1,0]
            totals = darks + brights

            marked = darks > int(0.30 * totals)

            result.append('X' if marked else 'O')
        
        return result