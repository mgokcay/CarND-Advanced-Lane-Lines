import cv2
import numpy as np
import matplotlib.pyplot as plt

def warp(img):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[(img_size[0] / 2) - 75, img_size[1] / 2 + 100],
         [((img_size[0] / 2) - 545), img_size[1]],
         [(img_size[0] / 2) + 545, img_size[1]],
         [(img_size[0] / 2 + 75), img_size[1] / 2 + 100]])

    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # img_marked = img.copy()
    # cv2.polylines(img_marked, [src.astype(int)], True, (255, 120, 255), 3)
    # # Plot the result
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    #
    # ax1.imshow(img_marked)
    # ax1.set_title('Original Image', fontsize=40)
    #
    # ax2.imshow(warped)
    # ax2.set_title('Threshold', fontsize=40)
    #
    # plt.show()

    return warped
