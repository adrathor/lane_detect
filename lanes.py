import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test_image.jpg')

# Gradient: measure of change of brightness over adjacent pixels
# convert image into grayscale because a color picture has 3 channels i.e 3 intensity values whereas a grayscale image has only 1 intensity value

lane_image= np.copy(image)
#gray= cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)

# Gaussian Blur: reduce noise
# each of the pixels for a grayscale image is described by a singe number that describes the brightness of the pixel. In order to smoothen an image, the typical answe would be to modify the value of a pixel with the avg value
# of the pixel intensities around it
# Averaging out the pixels to reduce the noise will be done by a kernel. This kernal of normally distributed numbers(np.array([[1,2,3],[4,5,6],[7,8,9]]) is run across our entire image and sets each pixel value equal to the weighted avg
# of its neighboring pixels, thus smoothening our image. In our case we will apply a 5x5 gaussian kernal

#blur= cv2.GaussianBlur(gray,(5,5),0)


# Apply Canny method to detect edges in our image
# An egde corresponds to a region in an image where there is a sharp change in the intensity/color between adjacet pixels in the image. A strong gradient is a steep change and vice versa is a shallow change.
# So in a way we can say an image is a stack of matrix with rows and columns of intensities
# this means that we can also represent an image in 2D coordinate space, x axis traverses the width(columns) and y axis goes along the image height(rows)
# Canny fn performs a derivative on the x and y axis thereby measuring chang in intensities wrt adjacent pixels. In other words we are computig the gradient (which is change in brightness) in all directions.
# It then traces the strongest gradients with a series of white pixels
# cv2.Canny(image, low_threshold,high_theshold)
# the low_threshold,high_theshold allow us to isolate the adjacent pixels that follow the strongest gradient. If the gradient is larger than the upper threshod then it is accepted as an edge pixel, if its below the loe threshold then it
# is rejected. If the gradient is between the threshold then it is accepted only if its connected to a strong edge.
# Areas where its completely black correspond to low changes in intensity between adjacent pixels whereas the white line represents a region in the image where there is a high change in intensity exceeding the threshold.

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region(image):
    height= image.shape[0]
    poly= np.array([[(200,height),(1100,height),(550,250)]])
    mask= np.zeros_like(image)
    cv2.fillPoly(mask, poly,255)
    masked_image= cv2.bitwise_and(image,mask)
    return masked_image

def display_lines(image, lines):
    line_image= np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    if lines is not None:

        for line in lines:
            x1,y1,x2,y2= line.reshape(4)
            parameters= np.polyfit((x1,x2),(y1,y2),1)
            slope= parameters[0]
            intercept= parameters[1]
            if slope<0:
                left_fit.append((slope,intercept))
            else:
                right_fit.append((slope,intercept))
        lfavg= np.average(left_fit, axis=0)
        rfavg= np.average(right_fit,axis=0)
        left_line = make_points(image, lfavg)
        right_line = make_points(image, rfavg)
        averaged_lines = [left_line, right_line]
        return averaged_lines

'''
canny_image= canny(lane_image)
cropped= region(canny_image)
lines= cv2.HoughLinesP(cropped,2,np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines= average_slope_intercept(lane_image,lines)
line_image= display_lines(lane_image,lines)
combo_image= cv2.addWeighted(lane_image,0.8,line_image, 1, 1)
cv2.imshow('result',combo_image)
cv2.waitKey(0)


'''
cap = cv2.VideoCapture("test2.mp4")

while(cap.isOpened()):
    cap.read()
    _, frame= cap.read()
    canny_image = canny(frame)
    cropped = region(canny_image)
    lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    cv2.waitKey(1)
