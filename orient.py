import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_keypoints(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    # Use indexes: left_eye (33), right_eye (263)
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    h, w, _ = image.shape
    left_eye = (int(left_eye.x * w), int(left_eye.y * h))
    right_eye = (int(right_eye.x * w), int(right_eye.y * h))

    return left_eye, right_eye

def align_face(image):
    left_eye, right_eye = get_keypoints(image)
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    M = cv2.getRotationMatrix2D(eye_center, angle, 1)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC)
    return aligned