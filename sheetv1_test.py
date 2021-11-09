import cv2
from src.edu_card_models.SheetV1 import SheetV1
from os import listdir

def test_reading():
    sheets = []
    exceptions = []
    basepath = './assets/sheetv1-test'
    images = listdir(basepath)
    if len(images) > 0:
        for image in images:
            try:
                file = cv2.imread(f'{basepath}/{image}')
                print('opened image file', file, )
                sheet = SheetV1(file)
                sheets.append(sheet)
            except Exception as error:
                exceptions.append({
                    'exception': error,
                    'image': image
                })
    
    return {'exceptions': exceptions, 'sheets': sheets}


result = test_reading()

print(result)