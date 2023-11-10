import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face = self.mp_face_detection.FaceDetection(self.min_detection_confidence, self.model_selection)

    def find_face(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox_list = []
        results = self.face.process(img_rgb)
        if results.detections:
            for face_id, detection in enumerate(results.detections):
                boxes = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(boxes.xmin * w), int(boxes.ymin * h), int(boxes.width * w), int(boxes.height * h)
                bbox_list.append([face_id, bbox, detection.score])
                if draw:
                    img = self.rectangle_draw(img, bbox)
                    cv2.putText(img, f"{str(int(round(detection.score[0] * 100, 2)))}%",
                                (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 248, 220), 2)
        return img, bbox_list

    @staticmethod
    def rectangle_draw(img, bbox, length=10, thickness=2):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 0, 255), 1)
        cv2.line(img, (x, y), (x + length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y + length), (255, 0, 255), thickness)

        cv2.line(img, (x1, y), (x1 - length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + length), (255, 0, 255), thickness)

        cv2.line(img, (x, y1), (x + length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - length), (255, 0, 255), thickness)

        cv2.line(img, (x1, y1), (x1 - length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness)

        return img


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceDetector(0.5, 0)
    while True:
        success, img = cap.read()
        img, bbox = detector.find_face(img)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f"FPS: {str(int(round(fps, 2)))}", (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow("Faces", img)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()
