import cv2
import numpy as np
import matplotlib.pyplot as plt


def threshold(img):

    u_threshold = 150
    s_threshold = 240
    l_threshold = 175
    sx_thresh = (30, 200)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]
    u_channel = yuv[:, :, 1]
    v_channel = yuv[:, :, 2]

    # Sobel x
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)  # Take the derivative in x
    abs_sobel_x = np.absolute(sobel_x)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    # Threshold x gradient
    sobel_x_binary = np.zeros_like(scaled_sobel)
    sobel_x_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel > l_threshold)] = 1

    # s_binary = np.zeros_like(s_channel)
    # s_binary[(s_channel > 220)] = 1

    u_binary = np.zeros_like(u_channel)
    u_binary[(u_channel > u_threshold)] = 1

    # Stack each channel
    combined_color = np.dstack((sobel_x_binary, l_binary, u_binary)) * 255

    # Combine the binary thresholds
    combined_binary = np.zeros_like(sobel_x_binary)
    combined_binary[(sobel_x_binary == 1) | (l_binary == 1) | (u_binary == 1)] = 1

    # # Plot the result
    l_channel_threshold = np.copy(l_channel)
    l_channel_threshold[(l_channel_threshold < l_threshold)] = 0

    s_channel_threshold = np.copy(s_channel)
    s_channel_threshold[(s_channel_threshold < s_threshold)] = 0

    u_channel_threshold = np.copy(u_channel)
    u_channel_threshold[(u_channel_threshold < u_threshold)] = 0

    # f,  ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
    # f.tight_layout()
    #
    # ax1.imshow(img)
    # ax1.set_title('image', fontsize=40)
    #
    # ax2.imshow(combined_color)
    # ax2.set_title('combined', fontsize=40)
    #
    # ax3.imshow(combined_binary)
    # ax3.set_title('combined binary', fontsize=40)
    #
    # ax4.imshow(sobel_x_binary)
    # ax4.set_title('sobel', fontsize=40)
    #
    # ax5.imshow(l_binary)
    # ax5.set_title('l', fontsize=40)
    #
    # ax6.imshow(u_binary)
    # ax6.set_title('u', fontsize=40)
    #
    # plt.show()

    return combined_binary, combined_color
