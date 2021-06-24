import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            for id, lms in enumerate(facelms.landmark):
                # print(lms)
                ih, iw, ic = img.shape
                x, y = int(lms.x * iw), int(lms.y * ih)
                print(id, x, y)




    cTime = time.time()
    fps = 1/cTime - pTime
    pTime = cTime
    cv2.putText(img, f'FPS :  {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)