# import os
# from typing import Any, List
# import cv2
# import numpy as np
# import gdown
# from Detector import Detector, FacialAreaRegion
#
#
#
# class YuNetClient(Detector):
#     def __init__(self):
#         self.model = self.build_model()
#
#     def build_model(self) -> Any:
#
#
#         file_name = "face_detection_yunet_2023mar.onnx"
#         face_detector = cv2.FaceDetectorYN_create('./face_detection_yunet_2023mar.onnx', "", (0, 0))
#
#         return face_detector
import os
from typing import Any, List
import cv2
import numpy as np
from Detector import Detector, FacialAreaRegion

class YuNetClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        # Get the absolute path of the ONNX model
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of YuNet.py
        model_path = os.path.join(script_dir, "face_detection_yunet_2023mar.onnx")

        # Check if the file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at: {model_path}")

        # Load the model using OpenCV
        face_detector = cv2.FaceDetectorYN_create(model_path, "", (0, 0))
        return face_detector

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with yunet

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        # FaceDetector.detect_faces does not support score_threshold parameter.
        # We can set it via environment variable.
        score_threshold = float(os.environ.get("yunet_score_threshold", "0.9"))
        resp = []
        faces = []
        height, width = img.shape[0], img.shape[1]
        # resize image if it is too large (Yunet fails to detect faces on large input sometimes)
        # I picked 640 as a threshold because it is the default value of max_size in Yunet.
        resized = False
        r = 1  # resize factor
        if height > 640 or width > 640:
            r = 640.0 / max(height, width)
            img = cv2.resize(img, (int(width * r), int(height * r)))
            height, width = img.shape[0], img.shape[1]
            resized = True
        self.model.setInputSize((width, height))
        self.model.setScoreThreshold(score_threshold)
        _, faces = self.model.detect(img)
        if faces is None:
            return resp
        for face in faces:
            # pylint: disable=W0105
            """
            The detection output faces is a two-dimension array of type CV_32F,
            whose rows are the detected face instances, columns are the location
            of a face and 5 facial landmarks.
            The format of each row is as follows:
            x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
            x_rcm, y_rcm, x_lcm, y_lcm,
            where x1, y1, w, h are the top-left coordinates, width and height of
            the face bounding box,
            {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye,
            left eye, nose tip, the right corner and left corner of the mouth respectively.
            """
            (x, y, w, h, x_re, y_re, x_le, y_le) = list(map(int, face[:8]))

            # YuNet returns negative coordinates if it thinks part of the detected face
            # is outside the frame.
            x = max(x, 0)
            y = max(y, 0)
            if resized:
                x, y, w, h = int(x / r), int(y / r), int(w / r), int(h / r)
                x_re, y_re, x_le, y_le = (
                    int(x_re / r),
                    int(y_re / r),
                    int(x_le / r),
                    int(y_le / r),
                )
            confidence = float(face[-1])

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=confidence,
                left_eye=(x_re, y_re),
                right_eye=(x_le, y_le),
            )
            resp.append(facial_area)
        return resp
