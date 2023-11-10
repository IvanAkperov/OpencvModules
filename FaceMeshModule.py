import os
import cv2
import mediapipe as mp
import time


class FaceMeshing:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_img = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_det_conf = min_detection_confidence
        self.min_track_conf = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = mp.solutions.face_mesh
        self.mesh = self.faceMesh.FaceMesh(self.static_img, self.max_num_faces, self.refine_landmarks,
                                           self.min_det_conf, self.min_track_conf)

        self.drawSpec = self.mpDraw.DrawingSpec((0, 0, 255), 1, 1)

    def mesh_face(self, img, draw=True):
        faces_list = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(imgRGB)
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, face_lms, self.faceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                for face_id, face in enumerate(face_lms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(face.x * w), int(face.y * h)

                    faces_list.append([face_id, cx, cy])
        return img, faces_list

    @staticmethod
    def folder():
        overlay_list = []
        folder_name = "faces"
        my_list = os.listdir(folder_name)
        for img in my_list:
            image = cv2.imread(f"{folder_name}/{img}")
            overlay_list.append(image)
        return overlay_list

    def draw_line(self, img, draw=True):
        _, mesh_list = self.mesh_face(img)
        faces_list = self.folder()
        if draw:
            if len(mesh_list) != 0:
                x1, y1 = mesh_list[61][1], mesh_list[61][2]
                x2, y2 = mesh_list[291][1], mesh_list[291][2]
                face_height = mesh_list[9][2] - mesh_list[152][2]
                lips_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                normalized_nose_lip_distance = abs(lips_distance / face_height) * 100
                if 40 < normalized_nose_lip_distance < 50:
                    img[0:200, 0:200] = faces_list[0]
                elif normalized_nose_lip_distance < 40:
                    img[0:200, 0:200] = faces_list[1]
                elif normalized_nose_lip_distance > 50:
                    img[0:200, 0:200] = faces_list[2]
                return img


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceMeshing()
    while True:
        success, img = cap.read()
        img = detector.draw_line(img)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(round(fps, 2))), (200, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
            cv2.imshow("Video", img)
            cv2.waitKey(10)


if __name__ == '__main__':
    main()
