## Advanced Lane Finding

### This is the writeup of Udacity Self Driving Car Nano Degree Project - Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./WriteupImages/Calibration.png "Undistorted"
[image2]: ./test_images/test1.jpg "Original Image"
[image3]: ./output_images/undistorted/test1.jpg "Undistorted"
[image4]: ./WriteupImages/Threshold.png "Threshold Details"
[image5]: ./output_images/threshold/test1.jpg "Threshold"
[image6]: ./output_images/warped/test1.jpg "Warped"
[image7]: ./output_images/lane_lines/test1.jpg "Lines"
[image8]: ./output_images/debug/test1.jpg "Debug"
[image9]: ./output_images/final/test1.jpg "Output"

[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is the wirteup of the Advanced Lane Finding project.  
All stages of the pipeline is called from my_work.py file. 
Almost each stage of the pipeline is implemented in a separate 
file which details are given in following sections.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the camera_calib.py file.
openCV's `cv2.findChessboardCorners()` function is used to detect chessboard corners.  
I start by preparing "object points", which will be the (x, y, z) coordinates
of the chessboard corners in the world. Here I am assuming the chessboard is fixed
on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be 
appended with a copy of it every time I successfully detect all chessboard corners
in a test image.  `imgpoints` will be appended with the (x, y) pixel position of 
each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration
and distortion coefficients using the `cv2.calibrateCamera()` function. 
I applied this distortion correction to the test image using the `cv2.undistort()` 
function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

In all below steps i will demonstrate the output of the pipeline stage with the following image:

![alt text][image2]

#### 1. Provide an example of a distortion-corrected image.

I used `cv2.undistort()` function with the results of calibration.
Correction to the test image is as follows:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image
(thresholding steps are at `threshold.py`).  
I used sobel x operator, l channel in HLS space and u channel in YUV space.
A detailed image for these thersholds are:
![alt text][image4]

And here's the the output of the example image.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the `warp.py` file.

I chose the calculate the source and destination points from image size in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 45, img_size[1] / 2 + 85],
     [((img_size[0] / 2) - 545), img_size[1]],
     [(img_size[0] / 2) + 545, img_size[1]],
     [(img_size[0] / 2 + 45), img_size[1] / 2 + 85]])

dst = np.float32(
    [[(img_size[0] / 4), 0],
     [(img_size[0] / 4), img_size[1]],
     [(img_size[0] * 3 / 4), img_size[1]],
     [(img_size[0] * 3 / 4), 0]])
```

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I detect lanes with the functions in `lane_detection.py`
In the first image i used the sliding windows aproach and than in the other images i searched around the previous
polynomial. (This is meaningful in the video not in the test images)

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this also in `lane_detection.py`  
An oveall overview of the image in several steps is as follows:

![alt text][image8]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I did this step in `my_work.py` lines #37-39

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the thresholding step i optimize the code for this specific video. So it may not perform well in different scenarios.  
Also i did not use any filtering with line polynomials. A low pass or kalman filter and a mechanism to reject some
bad detections may be added as well.
