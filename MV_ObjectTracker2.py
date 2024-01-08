# https://youtu.be/1FJWXOO1SRI/

import cv2

TextFontSize = 0.5;
TextBold= 2;

cap = cv2.VideoCapture("Square_Rotation_resize.mp4")
# cap = cv2.VideoCapture("RedBox_Resize.mp4")

tracker = cv2.TrackerCSRT_create()

success, img = cap.read()
bbox = cv2.selectROI("Tracking",img, False)
tracker.init(img,bbox)

def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)), (255, 0, 255), 3, 1)
    cv2.putText(img,"Object Detected: x="+str(x)+", y="+str(y), (75,75),cv2.FONT_HERSHEY_COMPLEX, TextFontSize, (0,255,0), TextBold)
    pass

while True:
    
    timer = cv2.getTickCount()
    
    success, img = cap.read()
    
    success, bbox = tracker.update(img)
    # print(type(bbox))
    
    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img,"Object Lost", (75,75),cv2.FONT_HERSHEY_COMPLEX, TextFontSize, (0,0,255), TextBold)

    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img,"FPS: "+str(int(fps)),(75,50),cv2.FONT_HERSHEY_COMPLEX, TextFontSize, (0,0,255), TextBold)
    cv2.imshow("Tracking", img)
  
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

