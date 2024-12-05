import cv2
import numpy as np

def power(image, power_value):
    copied_image = np.copy(image)

    normalized_image = cv2.normalize(copied_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normalized_image = np.power(normalized_image, power_value)
    copied_image = np.uint8(normalized_image * 255)

    return copied_image
