# **Finding Lane Lines on the Road** 

### Modified from template.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./pipeline.png "Pipeline"

---

### Reflection

### 1. Pipeline description - 

My pipeline is described visually in the below image. 

1. Convert "orignal image" to grayscale, apply gaussian blur and then canny transform to get the "canny image"

2. Apply the "mask" defined by array describing a polygon on the image

3. Identify lines using HoughLinesP. 

4. Additional functions average_pos() and m_b_points() identify lines with negative and positive slopes. All lines positive slope and positive slope are averaged to get the single left and right lines respectively. hough_lines() and draw_lines() are modified to reflect this change.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image2]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming of the current pipeline could be that it may mis identify other features as lines on a sharp turn. This could be avoided by changing the mask shape based on certain conditions.


### 3. Suggest possible improvements to your pipeline

Couple of improvements come to mind for the current pipeline

1. Currently the left and right lines appear jittery. This could be smoothed by averaging over multiple frames. Care must be taken to not smear out detail in doing so.

2. Canny may misidentify lane lines due to large intensity changes present in other locations in the image than the lane. 