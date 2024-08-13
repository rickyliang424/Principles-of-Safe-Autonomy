import time
import math
import numpy as np
import cv2
import rospy
import os
import pathlib

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
# from skimage import morphology

""" 
Author: Jacky
Revise by Ricky
- Remember to change rospy.Subscriber in lanenet_detector.
- For Rosbags: run >> studentVision.py
- For Gazebo: run >> python3 studentVision.py --perspective_pts 200,433,275,80 --val_thresh 80,255 --sim_mode
- For 0830_clip.bag: run >> python3 studentVision.py --val_thresh 80,255 --sat_thresh 0,255 --gradient_thresh 40,120 --perspective_pts 512,721,368,0
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sim_mode', action='store_true')
parser.add_argument('--gradient_thresh', '-g', type=str, default='75,150')
parser.add_argument('--sat_thresh', type=str, default='80,255')
parser.add_argument('--val_thresh', type=str, default='100,255')
parser.add_argument('--dilate_size', type=int, default=5)
parser.add_argument('--laplacian_thres', '-l', type=int, default=0)
parser.add_argument('--save_freq', type=int, default=-1)
parser.add_argument('--perspective_pts', '-p',
                    type=str, default='481,786,224,0')

args = parser.parse_args()

OUTPUT_DIR = './output'
TEST_DIR = '/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/assignments/MP1/test_imgs'
# TMP_DIR = './vis_{}'.format(args.append_str)
grad_thres_min, grad_thres_max = args.gradient_thresh.split(',')
grad_thres_min, grad_thres_max = int(grad_thres_min), int(grad_thres_max)
assert grad_thres_min < grad_thres_max

val_thres_min, val_thres_max = args.val_thresh.split(',')
val_thres_min, val_thres_max = int(val_thres_min), int(val_thres_max)
assert val_thres_min < val_thres_max

sat_thres_min, sat_thres_max = args.sat_thresh.split(',')
sat_thres_min, sat_thres_max = int(sat_thres_min), int(sat_thres_max)
assert sat_thres_min < sat_thres_max

src_leftx, src_rightx, laney, offsety = args.perspective_pts.split(',')
src_leftx, src_rightx, laney, offsety = int(
    src_leftx), int(src_rightx), int(laney), int(offsety)


def imshow(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class lanenet_detector():
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        if not self.test_mode:
            self.bridge = CvBridge()
            # NOTE
            if args.sim_mode:
                print ("sim_mode")
                # Uncomment this line for lane detection of GEM car in Gazebo
                self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
            else:
                # Uncomment this line for lane detection of videos in rosbags
                self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
                # Uncomment this line for lane detection of videos in rosbag "0830_clip.bag"
                self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
            self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
            self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
            self.left_line = Line(n=5)
            self.right_line = Line(n=5)
            self.detected = False
            self.hist = True
            self.counter = 0

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

    def gradient_thresh(self, img, thresh_min=grad_thres_min, thresh_max=grad_thres_max):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        # 1. Convert the image to gray scale
        # 2. Gaussian blur the image
        # 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        # 4. Use cv2.addWeighted() to combine the results
        # 5. Convert each pixel to uint8, then apply threshold to get binary image

        # Step 1: Load the image and convert it to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply Gaussian blur to the grayscale image
        blurred_image = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Step 3: Use cv2.Sobel() to find derivatives for both X and Y axes
        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

        # Step 4: Combine the results using cv2.addWeighted()
        sobel_combined = cv2.addWeighted(np.absolute(
            sobel_x), 0.5, np.absolute(sobel_y), 0.5, 0)

        # Step 5: Convert each pixel to uint8 and apply a threshold to get a binary image
        sobel_combined = np.uint8(sobel_combined)
        binary_output = np.zeros_like(sobel_combined)
        binary_output[(thresh_min < sobel_combined) &
                      (sobel_combined < thresh_max)] = 1

        # vis
        # vis = cv2.cvtColor(binary_output*255, cv2.COLOR_GRAY2BGR)
        # imshow("binary_output", cv2.hconcat([img, vis]))

        return binary_output

    def color_thresh(self, img, val_thres_min, val_thres_max, sat_thres_min, sat_thres_max):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        # 1. Convert the image from RGB to HSL
        # 2. Apply threshold on S channel to get binary image
        # Hint: threshold on H to remove green grass
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # For HSL
        # ref: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_hls
        #   image format: (8-bit images) V ← 255⋅V, S ← 255⋅S, H ← H/2(to fit to 0 to 255)

        # Step 2: Apply threshold on the S (Saturation) channel to get a binary image
        h, l, s = cv2.split(hls_img)
        binary_output = np.zeros_like(l)
        sat_cond = ((sat_thres_min <= s) & (s <= sat_thres_max)) | (s == 0)
        val_cond = (val_thres_min <= l) & (l <= val_thres_max)
        binary_output[val_cond & sat_cond] = 1

        return binary_output

    def laplacian_thres(self, img, laplacian_thres=args.laplacian_thres):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)

        # Convert the result to an absolute value
        laplacian_abs = np.abs(laplacian).astype(np.uint8)
        binary_output = np.zeros_like(laplacian_abs)
        binary_output[laplacian_abs > laplacian_thres] = 1
        return binary_output

    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # 1. Apply sobel filter and color filter on input image
        # 2. Combine the outputs
        # Here you can use as many methods as you want.

        if args.save_freq > 0 and self.counter % args.save_freq == 0:
            pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(
                OUTPUT_DIR, '{}.png').format(self.counter), img)
        self.counter += 1

        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(
            img, val_thres_min, val_thres_max, sat_thres_min, sat_thres_max)

        # imshow("ColorOutput", ColorOutput*255)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.dilate_size, args.dilate_size))
        ColorOutput = cv2.dilate(ColorOutput, kernel, iterations=1)
        # imshow("dilate ColorOutput", ColorOutput*255)

        LaplacianOutput = self.laplacian_thres(img)

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput == 1) & (SobelOutput == 1)
                    & (LaplacianOutput == 1)] = 1

        # Remove noise from binary image
        # binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

        return binaryImage

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        # 1. Visually determine 4 source points and 4 destination points
        # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        # 3. Generate warped image in bird view using cv2.warpPerspective()

        # Define four points as (x, y) coordinates
        src_height, src_width = img.shape[:2]

        src_pts = np.array([[src_leftx, laney],
                            [0, src_height - offsety],
                            [src_width, src_height - offsety],
                            [src_rightx, laney]], dtype=np.int32)

        # dst_width, dst_height = 720, 1250
        dst_width, dst_height = src_width, src_height
        dst_pts = np.array([[0, 0],
                            [0, dst_height],
                            [dst_width, dst_height],
                            [dst_width, 0]], dtype=np.int32)

        def calc_warp_points():
            src = np.float32(src_pts)
            dst = np.float32(dst_pts)

            return src, dst

        src, dst = calc_warp_points()
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        # keep same size as input image
        warped_img = cv2.warpPerspective(
            img, M, (dst_width, dst_height), flags=cv2.INTER_NEAREST)

        return warped_img, M, Minv

    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
                print("Lanes detected!")
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    if __name__ == '__main__':
        # init args
        rospy.init_node('lanenet_node', anonymous=True)
        lanenet_detector()
        while not rospy.core.is_shutdown():
            rospy.rostime.wallsleep(0.5)
