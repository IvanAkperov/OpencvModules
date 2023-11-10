import cv2
import time
import numpy as np
from HandTrackingModule import HandDetector
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math


class HandVolume:

    def __init__(self):
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = self.interface.QueryInterface(IAudioEndpointVolume)
        self.volRange = self.volume.GetVolumeRange()
        self.detector = HandDetector(min_detection_confidence=0.75)

    def volume_gesture(self, img):
        self.detector.find_hands(img)
        lm_list = self.detector.find_position(img)
        if len(lm_list) != 0:
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]
            cv2.circle(img, (x1, y1), 10, (25, 25, 112), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (25, 25, 112), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (25, 25, 112), 2)
            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, (50, 300), (-96, 0))
            self.volume.SetMasterVolumeLevel(vol, None)
        return True


def main():
    h_cam, w_cam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, h_cam)
    cap.set(4, w_cam)
    p_time = 0
    volume = HandVolume()
    while True:
        success, img = cap.read()
        volume.volume_gesture(img)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f"FPS: {str(int(round(fps, 2)))}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()