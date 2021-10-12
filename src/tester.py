import cv2
from edu_card_models.SheetV1 import SheetV1

sheet = SheetV1(cv2.imread('test300x.jpg'))

cv2.waitKey()