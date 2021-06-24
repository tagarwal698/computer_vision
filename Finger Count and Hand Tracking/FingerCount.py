import cv2
import time
import os
import HandtrackingModule as htm

wcam, hcam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

folderPath = "fingercount"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))
# print(myList)

pTime = 0

detector = htm.handDetector(detectionCon= 0.75)
tipID = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:

        fingers = []
        # Thumb
        if lmList[tipID[0]][1] > lmList[tipID[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for i in range(1,5):
            if lmList[tipID[i]][2] < lmList[tipID[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        totalfingers = fingers.count(1)
        cv2.putText(img, f'{int(totalfingers)}', (50,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,0,0), 2)


    # h, w, c = overlayList[0].shape
    # print(h)
    # img[0:h, 0:w] = overlayList[0]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (400,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,0,0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)