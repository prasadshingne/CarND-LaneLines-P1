#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 18:37:22 2020

@author: sprasad

frame29044.jpg
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# import math
import os
from moviepy.editor import VideoFileClip

y_down= 615
y_up  = 460
# verts = [[[250,y_down],[530,y_up],[800,y_up],[1100, y_down]]]
verts = [[[300,y_down],[580,y_up],[800,y_up],[1100, y_down]]]

v     = np.asarray(verts)
v1    = y_down
v2    = y_up
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, mask

def average_pos(lines):
    
    def compute_coeffs(line):
        line = list(line.flatten())
        x1,y1,x2,y2 = line
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        return m, b
    
    def m_b_points(m,b):
        
        # y1 = 590
        y1 = v1
        x1 = (y1-b)/ m
        
        # y2 = 450
        y2 = v2
        x2 = (y2-b)/ m
        return int(x1),int(y1),int(x2),int(y2)

    left_line  = list()
    right_line = list()
    for line in lines:
        m, b = compute_coeffs(line)
        if m > 0:
            left_line.append((m,b))
        else:
            right_line.append((m,b))
            
    left_line  = np.array(left_line)
    right_line = np.array(right_line)
    left_line  = np.median(left_line,axis=0)
    right_line = np.median(right_line,axis=0)
    left_line  = m_b_points(*left_line)
    right_line = m_b_points(*right_line)
    return np.array([left_line,right_line])
            

##def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
def draw_lines(img, lines, color=[255, 0, 0], thickness=8):

    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    lines = average_pos(lines)
    
    ##line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        ## average
        ##for x1,y1,x2,y2 in line:
        x1,y1,x2,y2 = list(line)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    ## average
    lines = average_pos(lines)    
    ##
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    
    draw_lines(line_img, lines)
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)



def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    image_gray                 = grayscale(image)
    image_gaus                 = gaussian_blur(image_gray,7)
    image_canny                = canny(image_gaus, 50, 150)

    # poly_vert                  = np.array([[[150,625],[525,460],[755,460],[1130, 625]]],dtype=np.int32)
    # poly_vert                  = np.array([[[150,625],[525,460],[755,460],[1130, 625]]],dtype=np.int32)
    poly_vert                  = np.array(verts,dtype=np.int32)

    image_canny_filtered, mask = region_of_interest(image_canny,poly_vert) 
    # mask1                      = mask.copy()

    # hough_img,lines = hough_lines(image_canny_filtered,rho=2,theta=np.pi/180,threshold=10, min_line_len=2,max_line_gap=45)
    hough_img,lines = hough_lines(image_canny_filtered,rho=2,theta=np.pi/180,threshold=15, min_line_len=15,max_line_gap=45)

    # you should return the final output (image where lines are drawn on lanes)
    
    return weighted_img(image,hough_img)


# %%
fig_ind  = 4
dir_name = "./frames_20201229_test/"
# file_list = os.listdir("./hwy880_test_frames/")
file_list = os.listdir(dir_name)

# image = mpimg.imread("./hwy880_test_frames/"+ file_list[fig_ind])
image = mpimg.imread(dir_name+ file_list[fig_ind])

print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
# image     = mpimg.imread("./hwy880_test_frames/"+file_list[2])
# image     = mpimg.imread("./hwy880_test_frames/"+ file_list[fig_ind])
image     = mpimg.imread(dir_name+ file_list[fig_ind])


image_gray                 = grayscale(image)
plt.imshow(image_gray)
image_gaus                 = gaussian_blur(image_gray,7)
plt.imshow(image_gaus)
# image_canny                = canny(image_gaus, 50,150)
image_canny                = canny(image_gaus, 100,250)

# poly_vert                  = np.array([[[150,625],[525,460],[755,460],[1130, 625]]],dtype=np.int32)
# poly_vert                  = np.array([[[200,625],[530,460],[750,460],[1080, 625]]],dtype=np.int32)
# 20201227
poly_vert                  = np.array(verts,dtype=np.int32)

image_canny_filtered, mask = region_of_interest(image_canny,poly_vert) 
# mask1                      = mask.copy()

hough_img,lines = hough_lines(image_canny_filtered,rho=2,theta=np.pi/180,threshold=15, min_line_len=15,max_line_gap=45)

plt.figure(figsize=(20,14))
plt.subplot(3,2,1)
plt.title('original image')
plt.imshow(image)
plt.subplot(3,2,2)
plt.title('canny image')
plt.imshow(image_canny)
plt.subplot(3,2,3)
plt.title('mask')
plt.imshow(mask)
plt.subplot(3,2,4)
plt.title('filtered canny image')
plt.imshow(image_canny_filtered)
plt.subplot(3,2,5)
plt.title('hough lines')
plt.imshow(hough_img)
plt.subplot(3,2,6)
plt.title('combined results')
plt.imshow(weighted_img(image,hough_img))
plt.savefig('pipeline.png')

# %%

# out_video = "Rec_20201223_113509_test_out.mp4"
# clip1 = VideoFileClip("Rec_20201223_113509_test.mp4")
# out_clip = clip1.fl_image(process_image) 

# out_clip.write_videofile(out_video, audio=False)

out_video = "Rec_20201229_162513_test_out.mp4"
clip1 = VideoFileClip("Rec_20201229_162513_test.mp4")
out_clip = clip1.fl_image(process_image) 
out_clip.write_videofile(out_video, audio=False)

