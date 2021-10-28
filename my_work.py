import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from camera_calib import calibrateCamera
from threshold import threshold
from warp import warp
from lane_detection import find_lane_polynomials
from lane_detection import measure_curvature, measure_position
from moviepy.editor import VideoFileClip
#%matplotlib qt


def process_image(img, cam_mtx, dist_coeff, write_outputs=False, output_file_name=''):

    img_undistorted = cv2.undistort(img, cam_mtx, dist_coeff, None, cam_mtx)
    if write_outputs:
        cv2.imwrite('./output_images/undistorted/' + output_file_name, img_undistorted)

    img_threshold, img_threshold_colored = threshold(img_undistorted)
    if write_outputs:
        cv2.imwrite('./output_images/threshold/' + output_file_name, img_threshold * 255)

    M_inv, img_threshold_warped = warp(img_threshold)
    if write_outputs:
        cv2.imwrite('./output_images/warped/' + output_file_name, img_threshold_warped * 255)

    left_fit, right_fit, ploty, out_img, img_poly = find_lane_polynomials(img_threshold_warped)
    if write_outputs:
        cv2.imwrite('./output_images/lane_lines/' + output_file_name, out_img)

    left_curverad = measure_curvature(np.max(ploty), left_fit)
    right_curverad = measure_curvature(np.max(ploty), right_fit)
    vehicle_position = measure_position(np.max(ploty), img.shape[1], left_fit, right_fit)

    img_size = (out_img.shape[1], out_img.shape[0])
    img_lane = cv2.warpPerspective(out_img, M_inv, img_size)
    img_final = cv2.addWeighted(img_undistorted, 1.0, img_lane, 0.4, 0)
    cv2.putText(img_final,
                "Radious of curvature: " + str(int((left_curverad+right_curverad)/2)) + " m",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_final,
                "Vehicle position: " + "{:.2f}".format(vehicle_position) + " m",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    if write_outputs:
        cv2.imwrite('./output_images/final/' + output_file_name, img_final)

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

    vis1 = np.concatenate((img_final, img_threshold_colored), axis=1)
    warped_color = np.dstack((img_threshold_warped, img_threshold_warped, img_threshold_warped)) * 255
    vis2 = np.concatenate((img_poly, out_img), axis=1)
    vis = np.concatenate((vis1, vis2), axis=0)

    if write_outputs:
        cv2.imwrite('./output_images/debug/' + output_file_name, vis)
    return vis

# Make a list of calibration images
calib_images = glob.glob('./camera_cal/calibration*.jpg')
# calibrate camera
cam_mtx, dist_coeff = calibrateCamera(calib_images)

test_images = glob.glob('./test_images/*.jpg')

# for image in test_images:
#
#     img_orig = cv2.imread(image)
#     img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
#     process_image(img_rgb, cam_mtx, dist_coeff, True, os.path.basename(image))

# clip1 = VideoFileClip("project_video.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
result_clip = clip1.fl_image(lambda image: process_image(image, cam_mtx, dist_coeff))
result_clip.write_videofile("project_video_output.mp4", audio=False)

print("ok")


