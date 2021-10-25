import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from camera_calib import calibrateCamera
from threshold import threshold
from warp import warp
from lane_detection import find_lane_polynomials
#%matplotlib qt

# Make a list of calibration images
calib_images = glob.glob('./camera_cal/calibration*.jpg')
#calibrate camera
cam_mtx, dist_coeff = calibrateCamera(calib_images)

test_images = glob.glob('./test_images/*.jpg')

for image in test_images:
    img_orig = cv2.imread(image)
    img_undistorted = cv2.undistort(img_orig, cam_mtx, dist_coeff, None, cam_mtx)

    cv2.imwrite('./output_images/undistorted/' + os.path.basename(image), img_undistorted)

    img_threshold = threshold(img_undistorted)
    cv2.imwrite('./output_images/threshold/' + os.path.basename(image), img_threshold * 255)

    Minv, img_threshold_warped = warp(img_threshold)
    cv2.imwrite('./output_images/warped/' + os.path.basename(image), img_threshold_warped * 255)

    left_fit, right_fit, ploty, out_img = find_lane_polynomials(img_threshold_warped)
    cv2.imwrite('./output_images/lane_lines/' + os.path.basename(image), out_img)

    img_size = (out_img.shape[1], out_img.shape[0])
    img_lane = cv2.warpPerspective(out_img, Minv, img_size)
    img_final = cv2.addWeighted(img_undistorted, 1.0, img_lane, 0.4, 0)
    cv2.imwrite('./output_images/final/' + os.path.basename(image), img_final)

    # # Plot the result
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    # f.tight_layout()
    #
    # ax1.imshow(img_orig)
    # ax1.set_title('Original Image', fontsize=40)
    #
    # ax2.imshow(img_threshold)
    # ax2.set_title('Threshold', fontsize=40)
    #
    # ax3.imshow(img_threshold_warped)
    # ax3.set_title('Warped', fontsize=40)
    #
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()
    print("ok")


