import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, mode = False, maxnum = 2, minDetectionCon = 0.5, minTrackingCon = 0.5 ):
        self.mode = mode
        self.maxnum = maxnum
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.FaceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxnum, self.minDetectionCon, self.minTrackingCon)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        finallist = []
        results = self.FaceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            finallist = []
            for facelms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                face = []
                for id, lms in enumerate(facelms.landmark):
                    # print(lms)
                    ih, iw, ic = img.shape
                    x, y = int(lms.x * iw), int(lms.y * ih)
                    face.append([id, x, y])
                finallist.append(face)
        return img, finallist




def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, finallist = detector.findFaceMesh(img)
        cTime = time.time()
        fps = 1 / cTime - pTime
        pTime = cTime
        cv2.putText(img, f'FPS :  {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        # if(len(finallist)!=0):
        #     print(len(finallist))   #this prints the number of faces
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()