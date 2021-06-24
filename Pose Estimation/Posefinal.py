import cv2
import mediapipe as mp
import time
import PoseModule as pmod

cap = cv2.VideoCapture('PoseVideos/5.mp4')
ptime = 0

detector = pmod.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmlist = detector.findPosition(img)

    if len(lmlist) != 0:
        print(lmlist)

    cTime = time.time()
    fps = 1 / cTime - ptime
    ptime = cTime

    cv2.putText(img, str(int(fps)), (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
