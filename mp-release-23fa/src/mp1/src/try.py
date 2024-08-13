# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 02:51:38 2023
@author: liang
"""
import cv2
import numpy as np
from PIL import Image
from skimage import morphology
import matplotlib.pyplot as plt

#%%
def gradient_thresh(img, thresh_min=100, thresh_max=255):
    """
    Apply sobel edge detection on input image in x, y direction
    """
    #1. Convert the image to gray scale
    #2. Gaussian blur the image
    #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
    #4. Use cv2.addWeighted() to combine the results
    #5. Convert each pixel to uint8, then apply threshold to get binary image
    ## TODO
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Image.fromarray(img_gray).show()
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    # Image.fromarray(img_blur).show()
    sobel_X = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    # Image.fromarray(sobel_X).show()
    sobel_Y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    # Image.fromarray(sobel_Y).show()
    img_sobel = cv2.addWeighted(abs(sobel_X), 0.5, abs(sobel_Y), 0.5, 0).astype(np.uint8)
    # Image.fromarray(img_sobel).show()
    binary_output =  np.zeros_like(img_sobel)
    binary_output[(thresh_min <= img_sobel) & (img_sobel <= thresh_max)] = 1
    # Image.fromarray(binary_output*255).show()
    ####
    return binary_output

#%%
def color_thresh(img, thresh=(100, 255)):
    """
    Convert RGB to HSL and threshold to binary image using S channel
    """
    #1. Convert the image from RGB to HSL
    #2. Apply threshold on S channel to get binary image
    #Hint: threshold on H to remove green grass
    ## TODO
    # Image.fromarray(np.flip(img,axis=2)).show()
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_S = img_HLS[:,:,2]
    img_L = img_HLS[:,:,1]
    img_H = img_HLS[:,:,0]
    # Image.fromarray(np.concatenate([img_S, img_L, img_H], axis=1)).show()
    img_S_th = np.zeros_like(img_S)
    img_L_th = np.zeros_like(img_L)
    img_H_th = np.zeros_like(img_H)
    img_S_th[(225 <= img_S) & (img_S <= 255)] = 1
    img_L_th[(150 <= img_L) & (img_L <= 200)] = 1
    img_H_th[(0 <= img_H) & (img_H <= 35)] = 1
    # Image.fromarray(np.concatenate([img_S_th, img_L_th, img_H_th], axis=1)*255).show()
    binary_output = np.zeros_like(img_S)
    binary_output[((img_S_th == 1) | (img_L_th == 1)) & (img_H_th == 1)] = 1    
    # fig = plt.figure(figsize=(10,10), facecolor='#eeeeee')
    # fig.add_subplot(1,3,1)
    # plt.imshow(np.dstack([img_S]*3))
    # # plt.imshow(np.dstack([img_S_th]*3)*255)
    # plt.title('S channel')
    # plt.axis('off')
    # fig.add_subplot(1,3,2)
    # plt.imshow(np.dstack([img_L]*3))
    # # plt.imshow(np.dstack([img_L_th]*3)*255)
    # plt.title('L channel')
    # plt.axis('off')
    # fig.add_subplot(1,3,3)
    # plt.imshow(np.dstack([img_H]*3))
    # # plt.imshow(np.dstack([img_H_th]*3)*255)
    # plt.title('H channel')
    # plt.axis('off')
    # Image.fromarray(binary_output*255).show()
    # Image.fromarray(binary_output*255).save('C:/Users/liang/Desktop/color_thresh.png')
    ####
    return binary_output


#%%
def combinedBinaryImage(img):
    """
    Get combined binary image from color filter and sobel filter
    """
    #1. Apply sobel filter and color filter on input image
    #2. Combine the outputs
    ## Here you can use as many methods as you want.
    ## TODO
    SobelOutput = gradient_thresh(img, thresh_min=100, thresh_max=255)
    # Image.fromarray(SobelOutput*255).show()
    ColorOutput = color_thresh(img, thresh=(100, 255))
    # Image.fromarray(ColorOutput*255).show()
    ####
    binaryImage = np.zeros_like(SobelOutput)
    binaryImage[(ColorOutput==1) | (SobelOutput==1)] = 1  # ColorOutput | SobelOutput
    # Image.fromarray(binaryImage*255).show()
    # Remove noise from binary image
    binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2).astype(np.uint8)
    # Image.fromarray(binaryImage*255).show()
    return binaryImage

#%%
def perspective_transform(img, verbose=False):
    """
    Get bird's eye view from input image
    """
    #1. Visually determine 4 source points and 4 destination points
    #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
    #3. Generate warped image in bird view using cv2.warpPerspective()
    ## TODO
    pt_ul = [img.shape[1]*0.40, img.shape[0]*0.55]
    pt_ll = [img.shape[1]*0.00, img.shape[0]*0.75]
    pt_lr = [img.shape[1]*1.00, img.shape[0]*0.75]
    pt_ur = [img.shape[1]*0.60, img.shape[0]*0.55]
    pts_1 = np.float32([pt_ul, pt_ll, pt_lr, pt_ur])
    pts_2 = np.float32([[0,0], [0,img.shape[0]], [img.shape[1],img.shape[0]], [img.shape[1],0]])
    M = cv2.getPerspectiveTransform(pts_1, pts_2)
    Minv = cv2.getPerspectiveTransform(pts_2, pts_1)
    warped_img = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    # Image.fromarray(warped_img).show()
    ####
    return warped_img, M, Minv

#%%
# feel free to adjust the parameters in the code if necessary
def line_fit(binary_warped):
    """
    Find and fit lane lines
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//3*2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int_(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[50:midpoint]) + 50
    rightx_base = np.argmax(histogram[midpoint:-50]) + midpoint
    # Choose the number of sliding windows
    nwindows = 20
    # Set height of windows
    window_height = np.int_(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 25
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        ## TODO
        left_p1 = [leftx_current-margin, binary_warped.shape[0]-window*window_height]
        left_p2 = [leftx_current+margin, binary_warped.shape[0]-(window+2)*window_height]
        right_p1 = [rightx_current-margin, binary_warped.shape[0]-window*window_height]
        right_p2 = [rightx_current+margin, binary_warped.shape[0]-(window+2)*window_height]
        ####
        # Draw the windows on the visualization image using cv2.rectangle()
        ## TODO
        img_win = cv2.rectangle(out_img, left_p1, left_p2, (255,0,0), 2)
        img_win = cv2.rectangle(img_win, right_p1, right_p2, (255,0,0), 2)
        ####
        # Identify the nonzero pixels in x and y within the window
        ## TODO
        left_nonzero = np.where((left_p1[0] <= nonzerox) & (nonzerox <= left_p2[0]) & 
                                (left_p2[1] <= nonzeroy) & (nonzeroy <= left_p1[1]))
        right_nonzero = np.where((right_p1[0] <= nonzerox) & (nonzerox <= right_p2[0]) & 
                                 (right_p2[1] <= nonzeroy) & (nonzeroy <= right_p1[1]))
        ####
        # Append these indices to the lists
        ## TODO
        left_lane_inds.append(left_nonzero[0])
        right_lane_inds.append(right_nonzero[0])        
        ####
        # If you found > minpix pixels, recenter next window on their mean position
        ## TODO
        if len(left_nonzero[0]) > minpix:
            leftx_current = int(np.mean(nonzerox[left_nonzero]))
        if len(right_nonzero[0]) > minpix:
            rightx_current = int(np.mean(nonzerox[right_nonzero]))
        ####
        pass
    # Image.fromarray(img_win).show()
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # plt.scatter(leftx, lefty,s=0.1)
    # plt.scatter(rightx, righty,s=0.1)
    # plt.gca().invert_yaxis()
    # Fit a second order polynomial to each using np.polyfit()
    # If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
    # the second order polynomial is unable to be sovled.
    # Thus, it is unable to detect edges.
    try:
    ## TODO
        left_fit = np.polyfit(lefty, leftx, deg=2)
        right_fit = np.polyfit(righty, rightx, deg=2)
    ####
    except TypeError:
        print("Unable to detect lanes")
        return None
    # Return a dict of relevant variables
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['out_img'] = out_img
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds
    return ret

#%%
def bird_fit(binary_warped, ret, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	return result, [left_fitx, right_fitx, ploty]

#%%
def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	return result

#%%
img = cv2.imread("C:/Users/liang/Desktop/test_imgs/test.png")
img = cv2.imread("C:/Users/liang/Desktop/test_imgs/0011_0.png")
binary_img = combinedBinaryImage(img)
img_birdeye, M, Minv = perspective_transform(binary_img)
# Fit lane without previous result
ret = line_fit(img_birdeye)
left_fit = ret['left_fit']
right_fit = ret['right_fit']
nonzerox = ret['nonzerox']
nonzeroy = ret['nonzeroy']
left_lane_inds = ret['left_lane_inds']
right_lane_inds = ret['right_lane_inds']
# Annotate original image
bird_fit_img = None
combine_fit_img = None
if ret is not None:
    bird_fit_img, plot = bird_fit(img_birdeye, ret, save_file=None)
    combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
# Plot images
fig = plt.figure(figsize=(10,12))
fig.add_subplot(3,2,1)
plt.imshow(np.flip(img,axis=2))
plt.title('img')
plt.axis('off')
fig.add_subplot(3,2,3)
plt.imshow(np.dstack([binary_img]*3)*255)
plt.title('binary')
plt.axis('off')
fig.add_subplot(3,2,5)
plt.imshow(np.dstack([img_birdeye]*3)*255)
plt.title('bird-eye')
plt.axis('off')
fig.add_subplot(3,2,2)
plt.imshow(bird_fit_img)
plt.plot(plot[0], plot[2], color='yellow')
plt.plot(plot[1], plot[2], color='yellow')
plt.title('bird-eye fit')
plt.axis('off')
fig.add_subplot(3,2,4)
plt.imshow(combine_fit_img)
plt.title('combine fit')
plt.axis('off')
