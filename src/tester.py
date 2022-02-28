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

# filePath = '/home/tetra/Downloads/cr2_clean_qrbig.jpg'
filePath = '/home/tetra/Downloads/121b6ed1-a8b2-41ad-beb8-f1c08b2a2693.jpeg'

image = cv2.imread(filePath)

images = []

ratio = 0.5

for i in range(0, 1):
    if (ratio == 4.0): break
    images.append([ratio, cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))])
    ratio += 0.25

sheets = []

for img in images:
    try:
        sheets.append(SheetCR2(img[1], f"{img[0]}"))
    except Exception:
        print(f'Could not use ratio {img[0]}')



print('done')
# cv2.waitKey()