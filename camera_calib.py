import numpy as np
import cv2
import os


def calibrateCamera(images):
    nx = 9  #number of inside corners in x
    ny = 6  #number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_point = np.zeros((ny * nx, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for image in images:
        img_orig = cv2.imread(image)
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            obj_points.append(obj_point)
            img_points.append(corners)

            # Draw and display the corners
            img_orig = cv2.drawChessboardCorners(img_orig, (nx, ny), corners, ret)
            # cv2.imshow('img', img_orig)
            # cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_gray.shape[::-1], None, None)

    for image in images:
        img_orig = cv2.imread(image)
        img_undistorted = cv2.undistort(img_orig, mtx, dist, None, mtx)

        cv2.imwrite('./output_images/calib/' + os.path.basename(image), img_undistorted)

    cv2.destroyAllWindows()

    return mtx, dist

