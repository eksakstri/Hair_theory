import streamlit as st
from PIL import Image
import mediapipe
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask import Flask, request, jsonify
import base64
from orient import align_face

app = Flask(__name__)

def images_to_base64_dict(cropped_objects):
    base64_dict = {}
    for cls_name, img in cropped_objects.items():
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        base64_dict[cls_name] = img_base64
    return base64_dict

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    return loss

def normalize_hsv(image_bgr):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    image_hsv[..., 0] /= 179.0
    image_hsv[..., 1] /= 255.0
    image_hsv[..., 2] /= 255.0
    return image_hsv

def pred(image):
    image = align_face(image)
    model = YOLO('best.pt')
    results = model(image)
    cropped_objects = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            crop_img = image[y1:y2, x1:x2].copy()

            cropped_objects[cls_name] = crop_img

    copy = cropped_objects.copy()

    if 'lear' not in cropped_objects.keys():
        cropped_objects['lear'] = cv2.flip(cropped_objects['rear'],1)

    if 'rear' not in cropped_objects.keys():
        cropped_objects['rear'] = cv2.flip(cropped_objects['lear'],1)

    keys_order = ["lear", "hairs", "rear"]
    images = [cropped_objects[k] for k in keys_order if k in cropped_objects]
    max_height = max(img.shape[0] for img in images)

    padded_images = []
    for img in images:
        h, w, _ = img.shape
        top_pad = (max_height - h) // 2
        bottom_pad = max_height - h - top_pad

        padded = cv2.copyMakeBorder(img, top_pad, bottom_pad, 0, 0,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        padded_images.append(padded)

    cropped_objects['forehead'] = cv2.hconcat(padded_images)
    resize_shape = (128, 128)
    type_img = cv2.resize(cropped_objects['forehead'], resize_shape)
    gray = cv2.cvtColor(type_img, cv2.COLOR_BGR2GRAY)
    all = cv2.Canny(gray, threshold1=80, threshold2=120)
    all = np.expand_dims(all, axis=-1)
    all = np.expand_dims(all, axis=0)
    col = cv2.cvtColor(type_img, cv2.COLOR_RGB2HSV)
    col = normalize_hsv(col)
    col = np.expand_dims(col, axis=0)
    all = all.astype(np.float32) / 255.0

    beard_model = load_model("Beard_1757324749.keras", custom_objects={'custom_loss': focal_loss})
    col_model = load_model("Col_1757327821.keras")
    len_model = load_model("Length_1757331097.keras", custom_objects={'custom_loss': focal_loss})
    must_model = load_model("Must_1757333090.keras", custom_objects={'custom_loss': focal_loss})
    type_model = load_model("Type_1757390267.keras", custom_objects={'custom_loss': focal_loss})

    beard = ['clean', 'full', 'mid', 'short']
    color = ['black', 'blonde', 'brown', 'none', 'red', 'white']
    length = ['bald', 'full', 'mid', 'short']
    must = ['clean', 'full', 'mid', 'short']
    type = ['curly', 'none', 'straight', 'wavy']
    beard_pred = beard_model.predict(all)
    col_pred = col_model.predict(col)
    len_pred = len_model.predict(all)
    must_pred = must_model.predict(all)
    type_pred = type_model.predict(all)

    result = images_to_base64_dict(copy)

    result["beard_name"] =  beard[np.argmax(beard_pred)]
    result["color_name"] =  color[np.argmax(col_pred)]
    result["length_name"] =  length[np.argmax(len_pred)]
    result["moustache_name"] =  must[np.argmax(must_pred)]
    result["type_name"] =  type[np.argmax(type_pred)]

    return result

@app.route('/pred', methods=['POST'])
def pred_api():
    data = request.get_json()
    base64_str = data['image']
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return pred(image)

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host='0.0.0.0',debug=True, port = port)