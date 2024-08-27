import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")
eng = pyttsx3.init()
offset = 20
imgSize = 300
labels = ["hello", "correct", "no", "A", "B"]
speak = ""
prev_label = ""
debounce_time = 1  # Minimum time between predictions in seconds
last_prediction_time = time.time()  # Last time prediction was made

while True:
    # Capture frame
    success, img = cap.read()
    if not success:
        break
    
    imgOutput = img.copy()
    

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        
        if imgCrop.size == 0:
            continue
        
      
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        
      
        current_time = time.time()
        if current_time - last_prediction_time > debounce_time:
        
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]
            
      
            if label != prev_label:
                eng.say(label)
                eng.runAndWait()
                prev_label = label
            
            last_prediction_time = current_time
        
  
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, prev_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", imgOutput)
    

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
