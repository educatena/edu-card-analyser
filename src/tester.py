import cv2
from edu_card_models.SheetCR1 import SheetCR1
from edu_card_models.SheetV1 import SheetV1
from edu_card_utils.cvutils import image_convert
import os

# sheet = SheetV1(cv2.imread('test300x.jpg'))
print(os.getcwd())

bigger = '/home/tetra/Downloads/cartao_teste_1.pdf_1 _bigger.jpg'

image = cv2.imread('/home/tetra/Downloads/testeqr.pdf_1.jpg')

sheetcr1 = SheetCR1(image)

cv2.waitKey()