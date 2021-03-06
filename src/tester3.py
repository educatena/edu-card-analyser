import os
import cv2
import numpy as np
import imutils
# from skimage import io              # Only needed for web grabbing images; for local images, use cv2.imread(...)

def correct_orientation(img, logger=None):

    h = img.shape[0]
    w = img.shape[1]

    if (w > h):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h = img.shape[0]
        w = img.shape[1]
        if (logger): logger('Rotated img by 90 degrees.', {'width': w, 'height': h})

    summed = np.sum(255-img, axis=0)
    aSum = np.sum(summed[30:130])
    bSum = np.sum(summed[w-130:w-30])

    if (aSum < bSum):
        img = cv2.rotate(img, cv2.ROTATE_180)
        if (logger): logger('Rotated img by 180 degrees.', {'sum': summed, 'aSum': aSum, 'bSum': bSum, 'equation': 'aSum < bSum'})

    return img

os.system('rm -dr debug/*')
filePath = '/home/tetra/Downloads/escaneados/1x.jpeg'

images = []
img = cv2.imread(filePath)
images.append([0, img])
angle = 45
for i in range(0,8):
    if (angle == 360): break
    images.append([angle, imutils.rotate_bound(img.copy(), angle)])
    angle += 45

results = []

for i, img in enumerate(images):
    cv2.imwrite(f'debug/a{img[0]}_rotated.png', correct_orientation(img[1]))
