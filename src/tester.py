import cv2
from edu_card_models.SheetV1 import SheetV1

sheet = SheetV1(cv2.imread('test.jpeg'))

cv2.imshow('Test', sheet.preparedImage(sheet.zones[0]))
cv2.waitKey()