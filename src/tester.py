import cv2
from edu_card_models.SheetV1 import SheetV1
from edu_card_utils.cvutils import image_convert

sheet = SheetV1(cv2.imread('test300x.jpg'))


img = cv2.imread('test300x.jpg')
print('unga bunga', image_convert)

cv2.waitKey()