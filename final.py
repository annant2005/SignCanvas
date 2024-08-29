import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2) 
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")
eng = pyttsx3.init()
offset = 20
imgSize = 400
labels = ["hello", "correct", "no", "A", "B"]
prev_label = ""
debounce_time = 0.5
last_prediction_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)


    imgWhite = np.ones((imgSize, imgSize * 2, 3), np.uint8) * 255

    if hands:
        for i, hand in enumerate(hands):
            x, y, w, h = hand['bbox']

       
            y1, y2 = max(0, y - offset), min(y + h + offset, img.shape[0])
            x1, x2 = max(0, x - offset), min(x + w + offset, img.shape[1])

          
            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size == 0:
                continue

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

    cv2.imshow("Image", imgOutput)
    cv2.imshow("ImageWhite", imgWhite)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
