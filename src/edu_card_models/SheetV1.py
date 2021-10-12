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
                # circles = self.findCircles(self.zones[x])
                questions = self.findCircles(self.zones[x])
                for i, question in enumerate(questions):
                    numberedQuestions[(i + 1) + (perPanel * multiplier) - perPanel] = question
                multiplier += 1
        
        return numberedQuestions