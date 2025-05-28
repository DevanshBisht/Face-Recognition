import cv2
import math
import numpy as np
from retinaface import RetinaFace
from PIL import Image
#import matplotlib.pyplot as plt
from scipy.spatial import distance
import Facenet
from FacialRecognition import FacialRecognition
from typing import Union, Tuple
from pathlib import Path
import YuNet
from Detector import Detector

def model_build(model_name):
    if model_name=='Facenet':
        if not "model_obj" in globals():
            model_obj = {}

        if not 'Facenet' in model_obj.keys():
            model = Facenet.FaceNet512dClient
            if model:
                model_obj['Facenet'] = model()
            else:
                raise ValueError(f"Invalid model_name passed - {'Facenet'}")

        return model_obj['Facenet']
    else:
        if not "model_obj" in globals():
            model_obj = {}

        if not 'YuNet' in model_obj.keys():
            model = YuNet.YuNetClient
            if model:
                model_obj['YuNet'] = model()
            else:
                raise ValueError(f"Invalid model_name passed - {'YuNet'}")

        return model_obj['YuNet']
face_detector: Detector = model_build('else')
model: FacialRecognition = model_build('Facenet')


# def detect_and_align_face(img_path):
#     try:
#         img = cv2.imread(img_path)
#         img = img[..., ::-1]  # Convert BGR to RGB
#         faces = face_detector.detect_faces(img)
#
#         if not faces:
#             return [-1], -1  # No face detected
#
#         detected_face = ''
#         for facial_area in faces:
#             x = facial_area.x
#             y = facial_area.y
#             w = facial_area.w
#             h = facial_area.h
#             left_eye = facial_area.left_eye
#             right_eye = facial_area.right_eye
#             confidence = facial_area.confidence
#             detected_face = img[int(y):int(y + h), int(x):int(x + w)]
#
#         angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
#         aligned_img = np.array(Image.fromarray(detected_face).rotate(angle))
#
#         return aligned_img, len(faces)
#     except:
#         return [-1], -1
#


def detect_and_align_face(img_path):
    try:
        img = cv2.imread(img_path)

        # Check if the image is loaded correctly
        if img is None:
            print(f"Error: Could not load image from path: {img_path}")
            return None, -1  # Invalid image path

        img = img[..., ::-1]  # Convert BGR to RGB
        faces = face_detector.detect_faces(img)

        # Handle no faces detected
        if not faces:
            print("No faces detected!")
            return None, 0  # Return 0 for no faces

        # Handle multiple faces detected
        if len(faces) > 1:
            print(f"Multiple faces detected: {len(faces)} faces.")
            return None, len(faces)  # Return the count of faces

        detected_face = ''
        for facial_area in faces:
            x = facial_area.x
            y = facial_area.y
            w = facial_area.w
            h = facial_area.h
            left_eye = facial_area.left_eye
            right_eye = facial_area.right_eye
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # Crop the detected face

        if detected_face is None:
            print("No face region extracted!")
            return None, -1  # No face region extracted

        # Calculate angle for alignment
        angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
        aligned_img = np.array(Image.fromarray(detected_face).rotate(angle))

        return aligned_img, 1  # Only one face detected
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None, -1


def extract_embedding(image_array,target_size=(160,160)):

    img = image_array
    embeddings = []

    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = img.astype(np.float32)
    img_objs = [
        {
            "face": img,
            "facial_area": {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]},
            "confidence": 0,
        }
    ]
    # ---------------------------------

    for img_obj in img_objs:
        img = img_obj["face"]
        region = img_obj["facial_area"]
        confidence = img_obj["confidence"]
        img /= 127.5
        img -= 1

        embedding = model.find_embeddings(img)

    img_embedding = embedding
    embeddings.append(img_embedding)
    return embeddings

def find_distance(source_representation,test_representation):
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def face_compare(image_path1,image_path2):
    image1_aligned,face_count1 = detect_and_align_face(image_path1)
    image2_aligned,face_count2 = detect_and_align_face(image_path2)
    # if face_count1==1 and face_count2==1:

    if face_count1 == 1 and face_count2 == 1 and isinstance(image1_aligned, np.ndarray) and isinstance(image2_aligned,
                                                                                                       np.ndarray):

        img1_embeddings= extract_embedding(image_array=image1_aligned)
        img2_embeddings= extract_embedding(image_array=image2_aligned)
        distances = []
        for idx, img1_embedding in enumerate(img1_embeddings):
            for idy, img2_embedding in enumerate(img2_embeddings):
                distance = find_distance(img1_embedding, img2_embedding)
                distances.append(distance)
        result = round(float(100 - (float(min(distances)) * 100)), 2)
        return result
    else:
        return -200
