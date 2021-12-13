import cv2
from edu_card_models.SheetCR1 import SheetCR1
from edu_card_models.SheetV1 import SheetV1
from edu_card_utils.cvutils import image_convert
import os

# sheet = SheetV1(cv2.imread('test300x.jpg'))
print(os.getcwd())

sheetcr1 = SheetCR1(cv2.imread('renderedv2.jpg'))

cv2.waitKey()