import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


class PoseDetector:
    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode, self.model_complexity,
                                      self.smooth_landmarks, self.enable_segmentation,
                                      self.smooth_segmentation, self.min_detection_confidence,
                                      self.min_detection_confidence)

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if draw:
            if self.results.pose_landmarks:
                mp_draw.draw_landmarks(img, self.results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return self.results

    def get_position(self, img, draw=True):
        landmarks_list = []
        if self.results.pose_landmarks:
            for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmarks_list.append([landmark_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return landmarks_list


def main():
    cap = cv2.VideoCapture("PoseVideos/training_girl.mp4")
    p_time = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        detector.find_pose(img)
        lm_list = detector.get_position(img, draw=False)
        if len(lm_list) != 0:
            print(lm_list[14])
            cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(round(fps, 2))), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()