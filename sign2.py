import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  
offset = 20
imgSize = 300
folder = "C:\\Users\\2005g\\OneDrive\\Desktop\\python\\rough"
counter = 0
capture_images = False 

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
  
    imgWhite = np.ones((imgSize, imgSize * 2, 3), np.uint8) * 255

    if hands:
     
        hands = sorted(hands, key=lambda hand: hand['bbox'][0])

        for i, hand in enumerate(hands):
            x, y, w, h = hand['bbox']

            y1, y2 = max(0, y - offset), min(y + h + offset, img.shape[0])
            x1, x2 = max(0, x - offset), min(x + w + offset, img.shape[1])

            
            imgCrop = img[y1:y2, x1:x2]
            imgCropShape = imgCrop.shape

            if imgCropShape[0] > 0 and imgCropShape[1] > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    position = i * imgSize  
                    imgWhite[:, position + wGap:position + wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    position = i * imgSize  
                    imgWhite[hGap:hCal + hGap, position:position + imgSize] = imgResize
        
        
        cv2.imshow("ImageWhite", imgWhite)

       
        if capture_images:
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f"Image saved: {counter}")
            time.sleep(0.5)  # ye time set karlena as per ease
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"): #isko change karlena apne hisab se
        capture_images = True  

    if key == ord("q"): #isko bhi
        capture_images = False  
        break

cap.release()
cv2.destroyAllWindows()
