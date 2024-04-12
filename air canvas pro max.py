import cv2
import numpy as np
import mediapipe as mp
from collections import deque

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Update colors list to include black color
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
colorIndex = 0

paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (10,1), (60,35), (0,0,0), -1)  # Filled rectangle
paintWindow = cv2.rectangle(paintWindow, (80,1), (130,35), (255,0,0), -1)  # Filled rectangle
paintWindow = cv2.rectangle(paintWindow, (150,1), (200,35), (0,255,0), -1)  # Filled rectangle
paintWindow = cv2.rectangle(paintWindow, (220,1), (270,35), (0,0,255), -1)  # Filled rectangle
paintWindow = cv2.rectangle(paintWindow, (290,1), (340,35), (0,0,0), -1)  # Filled rectangle
paintWindow = cv2.rectangle(paintWindow, (420,1), (470,35), (255,255,255), -1) # Filled rectangle
paintWindow = cv2.rectangle(paintWindow, (520,1), (570,35), (255,255,255), -1) # Filled rectangle

# Load the image
clear_img = cv2.imread('C:\\Users\\2005g\\OneDrive\\Documents\\python\\e.png')
clear_img = cv2.resize(clear_img, (45, 30))

cv2.putText(paintWindow, "BLUE", (85, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (155, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (225, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "BLACK", (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "PEN", (425, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "SHAPE", (525, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

# Add clear image
paintWindow[5:35, 15:60] = clear_img

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ret = True

thickness = 2
drawing = True
shape = ""

while ret:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame[:40,:] = (255, 255, 255)

    frame = cv2.rectangle(frame, (10,1), (60,35), (0,0,0), 2)
    frame = cv2.rectangle(frame, (80,1), (130,35), (255,0,0), -1)
    frame = cv2.rectangle(frame, (150,1), (200,35), (0,255,0), -1)
    frame = cv2.rectangle(frame, (220,1), (270,35), (0,0,255), -1)
    frame = cv2.rectangle(frame, (290,1), (340,35), (0,0,0), -1)
    frame = cv2.rectangle(frame, (420,1), (470,35), (255,255,255), -1) # Pen button
    frame = cv2.rectangle(frame, (520,1), (570,35), (255,255,255), -1) # Shape button

    cv2.putText(frame, "Clr ALL", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (85, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (155, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "RED", (225, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (295, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "PEN", (425, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "SHAPE", (525, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        if len(result.multi_hand_landmarks) == 2:
            thumb_2 = (landmarks[4][0],landmarks[4][1])
            index_2 = (landmarks[8][0],landmarks[8][1])
            middle_2 = (landmarks[12][0],landmarks[12][1])

            if np.sqrt((thumb_2[0]-index_2[0])**2 + (thumb_2[1]-index_2[1])**2) < 30:
                thickness -= 1 if thickness > 1 else 0
            elif np.sqrt((thumb_2[0]-middle_2[0])**2 + (thumb_2[1]-middle_2[1])**2) < 30:
                thickness += 1 if thickness < 10 else 0

            # Check if the "Shape" button is clicked
            if not drawing:
                if np.sqrt((thumb_2[0]-index_2[0])**2 + (thumb_2[1]-index_2[1])**2) < 30:
                    shape = "circle"
                elif np.sqrt((thumb_2[0]-middle_2[0])**2 + (thumb_2[1]-middle_2[1])**2) < 30:
                    shape = "rectangle"
            else:
                shape = ""

        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 9, (0,255,0),-1)

        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 35:
            if 10 <= center[0] <= 60:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 80 <= center[0] <= 130:
                colorIndex = 0
            elif 150 <= center[0] <= 200:
                colorIndex = 1
            elif 220 <= center[0] <= 270:
                colorIndex = 2
            elif 290 <= center[0] <= 340:
                colorIndex = 3
            elif 420 <= center[0] <= 470: # Pen button
                drawing = True
            elif 520 <= center[0] <= 570: # Shape button
                drawing = False
                # Reset shape
                shape = ""
        else:
            if drawing:
                # Check if the shape button is clicked
                if shape == "circle":
                    # Draw a circle only once
                    cv2.circle(frame, center, 20, colors[colorIndex], thickness)
                    cv2.circle(paintWindow, center, 20, colors[colorIndex], thickness)
                elif shape == "rectangle":
                    cv2.rectangle(frame, (center[0]-20, center[1]-20), (center[0]+20, center[1]+20), colors[colorIndex], thickness)
                    cv2.rectangle(paintWindow, (center[0]-20, center[1]-20), (center[0]+20, center[1]+20), colors[colorIndex], thickness)
                else:
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)

    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], thickness)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], thickness)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKeyEx(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

