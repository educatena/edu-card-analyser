import numpy
import cv2
from edu_card_utils.ImageIntelligence import findContours, findSquares, getImageCornerRects, isContourInsideRect, normalizedImage

from edu_card_utils.ImageManipulation import getSlice, thresholdImage
from edu_card_utils.OpenCVUtils import getContourDimensions, getLimits, getSquareContourCenter, rectSlice
from edu_nibble_code.NibbleReader import readNibble

DEFAULT_LOGGER = lambda x,y: 0

class SheetCR2ROIs():
    DEBUG = True
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

    def __init__(self, logger = DEFAULT_LOGGER) -> None:
        self.logger = logger

    def get(self, image, name="cr2_rois"):
        limits = getLimits(image)

        # Calculates the rects for our ROIs based on the corners of the image (limits)
        qrcode_rect = self.getQRCodeRect(limits)
        qpanel_rect = self.getQPanelRect(limits)

        qrcode_img = getSlice(image, qrcode_rect)
        questions_img = getSlice(image, qpanel_rect)

        if (SheetCR2ROIs.DEBUG):
            if (qrcode_img.size > 0):
                cv2.imwrite(f'debug/mcr2_{name}_qrcode.jpg', qrcode_img)
            if (questions_img.size > 0):
                cv2.imwrite(f'debug/mcr2_{name}_questions.jpg', questions_img)
            if (image.size > 0):
                cv2.imwrite(f'debug/mcr2_{name}_perspective.jpg', image)

        return {
            'qr_code': qrcode_img,
            'questions': questions_img
        }

    def findAnchors(self, image, name="cr2_rois_anchors"):
        debug = f'{name}' if SheetCR2ROIs.DEBUG is not None else None

        normalized_image_a = normalizedImage(image, debug)
        threshold_image = thresholdImage(normalized_image_a)

        cv2.imwrite(f'debug/normal.jpg', threshold_image)

        contours = findContours(threshold_image, debug=debug, source=image)
        self.logger(f"Found {len(contours)} contours")
        (squares, tallest) = findSquares(contours, image, debug=debug)

        self.logger(f"Found squares by heights: {squares.keys()}")

        anchorCandidates = []

        cornerRects = getImageCornerRects(image)

        for i, height in enumerate(squares):
                for candidate in squares[height]:

                    for corner in dict.keys(cornerRects):
                        dbg_img = None
                        if (SheetCR2ROIs.DEBUG): dbg_img = image.copy()
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
                                        cv2.imwrite(f'debug/nibble_code_w{w}h{h}_c{code}.jpg', nibble)

        self.logger(f"Found {len(anchorCandidates)} anchor candidates", anchorCandidates)

        cv2.imwrite(f'debug/rect_inside_rect.jpg', dbg_img)


        anchors = []

        for i, candidate in enumerate(anchorCandidates):
            (width,height) = getContourDimensions(candidate)

            ratio = width/height
            center = getSquareContourCenter(candidate)

            self.logger(f"Candidate {center} has a w/h ratio of {ratio}", {'w':width, 'h':height})

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

        self.logger(f"After sorting and grabbing anchors, {anchors} remained.", anchors)

        anchors = numpy.float32(anchors)

        return anchors

    def getQRCodeRect(self, anchors):
        transformed = numpy.array(anchors) * SheetCR2ROIs.REFERENCE_QRPANEL

        self.logger("Extracted QRCode Rect", transformed)

        return transformed.astype(int)

    def getQPanelRect(self, anchors):
        transformed = numpy.array(anchors) * SheetCR2ROIs.REFERENCE_QUESTIONPANEL

        self.logger("Extracted Question Panel Rect", transformed)

        return transformed.astype(int)