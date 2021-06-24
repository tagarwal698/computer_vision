import cv2
import time
import numpy as np
import HandtrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#####################
wCam, hCam = 640, 480
#####################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
vol = 0
volBR = 400
detector = htm.handDetector(detectionCon=0.75)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
#  volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()
minVol = volrange[0]
maxVol = volrange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw = False)




    if(len(lmlist)!=0):
        # print(lmlist[4],lmlist[8])
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        # cv2.circle(img, (x1,y1), 10, (255,0,0), cv2.FILLED )

        # cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        # cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 3)

        cx, cy = (x1+x2)//2, (y1+y2)//2
        # cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        length = math.hypot((x2-x1), (y2-y1))
        # print(length)
        # Hand range is 20 to 250
        # Volume range is from -65 to 0
        vol = np.interp(length, [20, 220], [minVol, maxVol])
        volBR = np.interp(length, [20, 220], [400, 150])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 20:
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBR)), (85, 400), (255, 255, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
