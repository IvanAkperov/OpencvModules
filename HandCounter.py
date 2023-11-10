import time
import cv2
import HandTrackingModule as htm
import os


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
folder_name = "fingers"
my_list = os.listdir(folder_name)
overlay_list = []
tipsIds = [4, 8, 12, 16, 20]
p_time = 0
hand_detector = htm.HandDetector(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

for img in my_list:
    image = cv2.imread(f"{folder_name}/{img}")
    overlay_list.append(image)

while True:
    _, img = cap.read()
    detector = hand_detector.find_hands(img, draw=False)
    lm_list = hand_detector.find_position(img, draw=False)
    temp_lst = []
    if len(lm_list) != 0:

        #  thumb
        if lm_list[tipsIds[0]][1] > lm_list[tipsIds[0] - 2][1]:
            temp_lst.append(1)
        else:
            temp_lst.append(0)

        #  4 fingers
        for ind in range(1, 5):
            if lm_list[tipsIds[ind]][2] < lm_list[tipsIds[ind] - 2][2]:
                temp_lst.append(1)
            else:
                temp_lst.append(0)
    cv2.rectangle(img, (20, 225), (170, 425), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, f"{(str(temp_lst.count(1)))}", (75, 350), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 3)
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    img[0:200, 0:200] = overlay_list[(temp_lst.count(1))]
    cv2.putText(img, f"FPS: {int(fps)}", (220, 30), cv2.FONT_HERSHEY_PLAIN, 2, (128, 0, 0), 2)
    cv2.imshow("Video", img)
    cv2.waitKey(1)
