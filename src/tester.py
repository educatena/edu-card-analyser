import cv2
from edu_card_models.SheetCR1 import SheetCR1
from edu_card_models.SheetCR2 import SheetCR2
from edu_card_models.SheetV1 import SheetV1
from edu_card_utils.cvutils import image_convert
import os

os.system('rm -dr debug/*')

# sheet = SheetV1(cv2.imread('test300x.jpg'))
print(os.getcwd())

bigger = '/home/tetra/Downloads/test.jpeg'

image = cv2.imread('/home/tetra/Downloads/cr2_clean.jpg')

# nathancyContours(image)

sheetcr1 = SheetCR2(image)

cv2.waitKey()