import numpy as np
import cv2

val_images = np.load('train_and_eval\\segmentation\\datasets_as_numpy\\val_images.npy')



image1 = np.tile(val_images[0], reps=(3, 1, 1))
image2 = cv2.cvtColor(val_images[0], cv2.COLOR_GRAY2BGR).transpose((2, 0, 1))

print(np.array_equal(image1, image2))
