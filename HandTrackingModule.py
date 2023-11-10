import time
import cv2
import mediapipe as mp
import pyautogui
import numpy as np


class HandDetector:

    def __init__(self, static_image_mode=False,
                 max_num_hands=1,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.static_image_mode,
                                        self.max_num_hands,
                                        self.model_complexity,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        self.tipsIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        landmarks_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for hand_id, landmark in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmarks_list.append([hand_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (0, 0, 0), cv2.FILLED)
        return landmarks_list

    def count_fingers(self, img):
        lm_list = self.find_position(img, draw=False)
        fingers_list = []
        if len(lm_list) != 0:
            #  thumb
            if lm_list[self.tipsIds[0]][1] > lm_list[self.tipsIds[0] - 2][1]:
                fingers_list.append(1)
            else:
                fingers_list.append(0)
            #  4 fingers
            for ind in range(1, 5):
                if lm_list[self.tipsIds[ind]][2] < lm_list[self.tipsIds[ind] - 2][2]:
                    fingers_list.append(1)
                else:
                    fingers_list.append(0)
        return fingers_list

    def draw_hand_box(self, img, draw=True):
        landmarks_list = self.find_position(img, draw=False)
        if len(landmarks_list) != 0:
            x_coords = [landmark[1] for landmark in landmarks_list]
            y_coords = [landmark[2] for landmark in landmarks_list]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            if draw:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img

    @staticmethod
    def run():
        cap = cv2.VideoCapture(0)
        width, height = 640, 480
        frameR = 100
        smooth = 8
        ploc_x = ploc_y = 0
        cap.set(3, width)
        cap.set(4, height)
        wScr, hScr = pyautogui.size()
        p_time = 0
        hand_detection = HandDetector()
        while True:
            success, img = cap.read()
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            hand_detection.find_hands(img=img, draw=False)
            hand_detection.draw_hand_box(img)
            finger_positions = hand_detection.find_position(img, draw=False)
            if len(finger_positions) != 0:
                x1, y1 = finger_positions[8][1:]
                finger_list = hand_detection.count_fingers(img=img)
                cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)
                if finger_list[2] and finger_list[1] and finger_list.count(1) <= 2:
                    pyautogui.click()
                    time.sleep(0.4)
                    pass
                elif finger_list[1] == 1 and finger_list.count(1) <= 1:
                    x3 = np.interp(x1, (frameR, width - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, height - frameR), (0, hScr))
                    cloc_x = ploc_x + (x3 - ploc_x) / smooth
                    cloc_y = ploc_y + (y3 - ploc_y) / smooth
                    pyautogui.moveTo(wScr - cloc_x, cloc_y, _pause=False, tween=pyautogui.easeInOutQuad)
                    ploc_x, ploc_y = cloc_x, cloc_y
                elif finger_list[4] == 1 and finger_list.count(1) == 1:
                    pyautogui.hotkey('win', 'r')
                    # Задержка на 1 секунду
                    time.sleep(1)
                    # Ввод текста "osk"
                    pyautogui.typewrite('osk')
                    # Нажатие клавиши Enter
                    pyautogui.press('enter')
            cv2.putText(img, str(int(round(fps, 2))), (10, 40), 2, 1, (0, 255, 255), 1)
            cv2.imshow("Image", img)
            cv2.waitKey(1)


if __name__ == '__main__':
    detector = HandDetector()
    detector.run()