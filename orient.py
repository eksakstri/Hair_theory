# # import mediapipe as mp
# import face_alignment
# import cv2
# import numpy as np

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# def get_keypoints(image):
#     img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # results = face_mesh.process(img_rgb)

#     # if not results.multi_face_landmarks:
#     #     return None

#     # landmarks = results.multi_face_landmarks[0].landmark
    
#     landmarks = fa.get_landmarks(img_rgb)[0]

#     # Use indexes: left_eye (33), right_eye (263)
#     # left_eye = landmarks[33]
#     # right_eye = landmarks[263]

#     # h, w, _ = image.shape
#     # left_eye = (int(left_eye.x * w), int(left_eye.y * h))
#     # right_eye = (int(right_eye.x * w), int(right_eye.y * h))

#     left_eye = landmarks[36:42] 
#     right_eye = landmarks[42:48]

#     return left_eye, right_eye

# def align_face(image):
#     left_eye, right_eye = get_keypoints(image)
#     eye_center = ((left_eye[0] + right_eye[0]) // 2,
#                   (left_eye[1] + right_eye[1]) // 2)

#     dx = right_eye[0] - left_eye[0]
#     dy = right_eye[1] - left_eye[1]
#     angle = np.degrees(np.arctan2(dy, dx))

#     M = cv2.getRotationMatrix2D(eye_center, angle, 1)
#     aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
#                              flags=cv2.INTER_CUBIC)
#     return aligned

import face_alignment
import cv2
from skimage import io
import numpy as np

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

def align_face(img):
    landmarks = fa.get_landmarks(img)[0]

    # Get eye points
    left_eye  = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)

    # Align by rotating the image
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(landmarks.mean(axis=0)), angle, 1)
    aligned = cv2.warpAffine(img, M, (w, h))

    # cv2.imwrite("aligned.jpg", cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    return aligned