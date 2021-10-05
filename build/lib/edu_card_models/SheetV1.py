from edu_card_models.BaseSheet import BaseSheet


class SheetV1(BaseSheet):
    zones = None
    alternatives = None

    def __init__(self, image) -> None:
        super().__init__(image)
        self.zones = self.getQuestionZones()
        self.alternatives = self.getAlternatives()

    def getQuestionZones(self):
        zones = self.findSquares(self.contours, self.source, 3.0)

        return zones
    
    def getAlternatives(self):
        alternatives = []
        if len(self.zones) != 0:
            for zone in self.zones:
                circles = self.findCircles(zone)
                print(circles)