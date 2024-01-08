# https://youtu.be/1FJWXOO1SRI/

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time

#############################################################################################
# Step 1. Dataset.
cap = cv.VideoCapture("Square_Rotation_resize.mp4")
# cap = cv.VideoCapture("RedBox_Resize.mp4")


#############################################################################################
# Step 2. Feature Detector.
# 2.1 Extract two frames from the video
success,image = cap.read()
count = 0
while count < 2:
  cv.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = cap.read()
  print('Read a new frame: ', success)
  count += 1


#############################################################################################

# 2.2 Detect the feature

img0 = cv.imread('frame0.jpg')
img1 = cv.imread('frame1.jpg')


###############################################

#  Using Harris COrner Detector

gray = cv.cvtColor(img0,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img0[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst',img0)

if cv.waitKey(0) & 0xff == ord("q"):
    cv.destroyAllWindows()

###############################################
#  Using SIFT
sift = cv.SIFT_create()

# for image0
gray0 = cv.cvtColor(img0,cv.COLOR_BGR2GRAY)
kp0 = sift.detect(gray0,None)
kp0, desc0 = sift.compute(gray0, kp0)
imgKp0 = cv.drawKeypoints(gray0,kp0,img0)
cv.imwrite('sift_keypoints.jpg',imgKp0)
imgKp0=cv.drawKeypoints(gray0,kp0,img0,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # draw a circle with size of keypoint and show its orientation
cv.imwrite('sift_keypoints_2.jpg',imgKp0)

print(f"{kp0[0].pt[0]}") # x: It is the position of the keypoint with respect to the x-axis
print(f"{kp0[0].pt[1]}") # y: It is the position of the keypoint with respect to the y-axis
print(f"{kp0[0].size}") # size: It is the diameter of the keypoint
print(f"{kp0[0].angle}") # angle: It is the orientation of the keypoint
print(f"{kp0[0].response}") # response: Also known as the strength of the keypoint, it is the keypoint detector response on the keypoint
print(f"{kp0[0].octave}") # octave: It is the pyramid octave in which the keypoint has been detected
print(f"{desc0[0]}") # keypoint-descriptor

# for image1
gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
kp1 = sift.detect(gray1,None)
kp1, desc1 = sift.compute(gray1, kp1)
imgKp1 = cv.drawKeypoints(gray1,kp1,img1)

# Calculate Euclidean distance
print(desc0[0].size)
print(desc1[0].size)
normType=cv.NORM_L2
# dist = cv.norm(desc0,desc1,cv.NORM_L2)
dist = cv.norm(desc0[0],desc1[0],cv.NORM_L2)
print(dist)

# Combine images
img_combine = np.concatenate((imgKp0, imgKp1), axis=0)
cv.imshow('combine',img_combine)
if cv.waitKey(0) & 0xff == ord("q"):
    cv.destroyAllWindows()

###############################################
# Canny Edge Detection in OpenCV
img = cv.imread('frame0.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
if cv.waitKey(0) & 0xff == ord("q"):
    cv.destroyAllWindows()

###############################################
# # Colour Detection
# bgr_img = cv.imread('frame0.jpg')
# hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)

# def red_hsv():
#     # H -> 0 - 10
#     # S -> 175 - 255
#     # V -> 20 - 255
#     lower_hsv_1 = np.array([0, 175, 20])
#     higher_hsv_1 = np.array([10, 255, 255])
    
#     # H -> 170 - 10
#     # S -> 175 - 255
#     # V -> 20 - 255
#     lower_hsv_2 = np.array([170, 175, 20])
#     higher_hsv_2 = np.array([180, 255, 255])
    
#     # generating mask for red color
#     mask_1 = cv.inRange(hsv_img, lower_hsv_1, higher_hsv_1)
#     mask_2 = cv.inRange(hsv_img, lower_hsv_2, higher_hsv_2)
#     return mask_1 + mask_2

# mask = red_hsv()

# # detected_img
# detected_img = cv.bitwise_and(bgr_img, bgr_img, mask = mask)

# cv.imshow("Detected Images Colour", detected_img)
# if cv.waitKey(0) & 0xff == ord("q"):
#     cv.destroyAllWindows()


# ###############################################
# # Draw a rectangle on the object
# img = cv.imread('frame1.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # binarize the image
# ret, bw = cv.threshold(gray, 128, 255, 
# cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# # find connected components
# connectivity = 1
# nb_components, output, stats, centroids = cv.connectedComponentsWithStats(bw, connectivity, cv.CV_32S)
# sizes = stats[1:, -1]
# # nb_components = nb_components - 1
# min_size = 5 #threshhold value for objects in scene

# img2 = np.zeros((img.shape), np.uint8)

# for i in range(0, nb_components):
#     # use if sizes[i] >= min_size: to identify objects
#     color = np.random.randint(255,size=3)
#     # draw the bounding rectangele around each object
#     cv.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
#     img2[output == i + 1] = color

# cv.imshow("Detected Images Colour", img2)
# if cv.waitKey(0) & 0xff == ord("q"):
#     cv.destroyAllWindows()

# make the video
cap = cv.VideoCapture("Square_Rotation_resize.mp4")
# cap = cv.VideoCapture("RedBox_Resize.mp4")

# cap = cv.VideoCapture(0)


TextFontSize = 0.3
TextBold= 1
while True:

    timer = cv.getTickCount()
    
    success, img = cap.read()
    
    if success:
        # cv.putText(img2,"Object found", (10,45),cv.FONT_HERSHEY_COMPLEX, TextFontSize, (0,0,255), TextBold)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # binarize the image
        ret, bw = cv.threshold(gray, 128, 255, 
        cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        # find connected components
        connectivity = 1
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(bw, connectivity, cv.CV_32S)
        sizes = stats[1:, -1]
        # nb_components = nb_components - 1
        min_size = 5 #threshhold value for objects in scene

        img2 = np.zeros((img.shape), np.uint8)

        for i in range(0, nb_components):
            # use if sizes[i] >= min_size: to identify objects
            # color = np.random.randint(255,size=3)
            color = (0,0,255)
            # draw the bounding rectangele around each object
            cv.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
            img2[output == i + 1] = color
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        cv.putText(img2,"FPS: "+str(int(fps)),(10,20),cv.FONT_HERSHEY_COMPLEX, TextFontSize, (0,0,255), TextBold)
        objectPositionX = (stats[1][0] + stats[1][2])/2
        objectPositionY = (stats[1][1] + stats[1][3])/2
        cv.putText(img2,"Object Detected: x="+str(objectPositionX)+", y="+str(objectPositionY), (10,35),cv.FONT_HERSHEY_COMPLEX, TextFontSize, (0,255,0), TextBold)

    # else:
        # cv.putText(img2,"Object Lost", (10,35),cv.FONT_HERSHEY_COMPLEX, TextFontSize, (0,0,255), TextBold)

    
    
    cv.imshow("Tracking", img2)

    time.sleep(0.01)
  
    if cv.waitKey(1) & 0xff == ord("q"):
        cv.destroyAllWindows()
        break

exit()


# Set display params
TextFontSize = 0.5
TextBold= 2

tracker = cv.TrackerCSRT_create()

success, img = cap.read()
bbox = cv.selectROI("Tracking",img, False)
tracker.init(img,bbox)


def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv.rectangle(img,(x,y),((x+w),(y+h)), (255, 0, 255), 3, 1)
    cv.putText(img,"Object Detected: x="+str(x)+", y="+str(y), (75,75),cv.FONT_HERSHEY_COMPLEX, TextFontSize, (0,255,0), TextBold)
    pass



while True:
    
    timer = cv.getTickCount()
    
    success, img = cap.read()
    
    success, bbox = tracker.update(img)
    # print(type(bbox))
    
    if success:
        drawBox(img, bbox)
    else:
        cv.putText(img,"Object Lost", (75,75),cv.FONT_HERSHEY_COMPLEX, TextFontSize, (0,0,255), TextBold)

    
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
    cv.putText(img,"FPS: "+str(int(fps)),(75,50),cv.FONT_HERSHEY_COMPLEX, TextFontSize, (0,0,255), TextBold)
    cv.imshow("Tracking", img)
  
    if cv.waitKey(1) & 0xff == ord("q"):
        break

