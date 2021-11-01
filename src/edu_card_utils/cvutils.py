import numpy as np
import cv2

# Converts binary string image to NDArray image
def image_convert(binary_string):

    # CV2
    nparr = np.fromstring(binary_string, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # CV
    # img_ipl = cv.CreateImageHeader((img_np.shape[1], img_np.shape[0]), cv.IPL_DEPTH_8U, 3)
    # cv.SetData(img_ipl, img_np.tostring(), img_np.dtype.itemsize * 3 * img_np.shape[1])

    return img_np
