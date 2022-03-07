import datetime
import math
import traceback
import cv2
import numpy
from edu_card_models.SheetCR2ROIs import SheetCR2ROIs

from edu_card_utils.ImageIntelligence import chamithDivakalReadCircles, decodeMachineCode
from edu_card_utils.ImageManipulation import perspectiveTransformation
from edu_card_utils.constants import HEIGHT_FIRST_VERTEX
from edu_card_utils.coordutils import grid

DEBUG = True
OPTION_PER_QUESTION = 5
QUESTION_PER_PANEL = 25
NUMBERING_FUNCTION = lambda panel, question: (question + 1) + (QUESTION_PER_PANEL * panel) - QUESTION_PER_PANEL

class SheetCR2():

    def __init__(self, image, name="cr2_card") -> None:
        self.qrData = None
        self.questions = None
        self.name = None

        self.messages = []

        self.name = name
        self.rois = SheetCR2ROIs(self.log)

        self.log(f"Input image dimensions: w={image.shape[1]} h={image.shape[0]}")

        # ----------------------- NORMALIZATION AND PERSPECTIVE ---------------------- #

        transformed_img = self.normalizeArea(image)

        # ---------------------------- REGIONS OF INTEREST --------------------------- #

        rois = self.rois.get(transformed_img)

        # -------------------------- INFORMATION EXTRACTION -------------------------- #

        self.qrData = self.getQRData(rois['qr_code'])
        self.questions = self.getQuestions(rois['questions'])

    def normalizeArea(self, image):
        # 1.1 - Get coordinates for perspective transformation
        anchors = self.rois.findAnchors(image)
        # 1.2 - Apply perspective transformation based on image aspect ratio
        transformed_img = perspectiveTransformation(image, anchors)
        return transformed_img

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