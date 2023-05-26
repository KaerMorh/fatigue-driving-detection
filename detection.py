from scipy.spatial import distance as dist
from imutils import face_utils
import time
import dlib
import numpy as np
import math
import cv2
from main2 import grab_screen
import torch
from utils.augmentations import letterbox
from models.experimental import attempt_load
from detection4 import face_analysis,infer_image
from main2 import grab_screen
import os

from scipy.spatial import distance as dist
from imutils import face_utils
import time
import dlib
import numpy as np
import math
import cv2
import re




import torch
import numpy as np

from aim.utils.general import scale_coords
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.augmentations import letterbox
from models.experimental import attempt_load

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

ANGLE_THRESHOLD = 45.0

model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner

])
weights = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\aim\weights\face_phone_detection.pt'
device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'
stride=32
imgsz = 640
def input_module():
    img0 = grab_screen(region=(0, 0, 960, 540))
    img0 = cv2.resize(img0, (960, 540))
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]
    return img, img0

def get_euler_angle(rotation_vector):
    """
    通过旋转向量获得欧拉角
    :param rotation_vector: 旋转向量
    :return: 欧拉角 3个方向 pitch, yaw, roll
    """
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    w = np.cos(theta / 2)
    x = np.sin(theta / 2) * rotation_vector[0][0] / theta
    y = np.sin(theta / 2) * rotation_vector[1][0] / theta
    z = np.sin(theta / 2) * rotation_vector[2][0] / theta
    yy = np.square(y)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + yy)
    pitch = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (yy + z * z)
    roll = math.atan2(t3, t4)

    # 弧度 -> 角度
    pitch = int(np.rad2deg(pitch))
    yaw = int(np.rad2deg(yaw))
    roll = int(np.rad2deg(roll))

    return pitch, yaw, roll


def mouth_aspect_ratio(mouth):
    """
    计算嘴长宽比例
    :param mouth: 嘴部矩阵
    :return: MAR
    """
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    MAR = (A + B + C) / 3.0
    return MAR


def eye_ratio(eye):
    """
    计算人眼纵横比 (EAR)
    :param eye: 眼部矩阵
    :return: ear
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def eye_detection(threshold, sustain):
    # 人眼纵横比参数
    EAR_THRESHOLD = threshold  # 阈值
    SUSTAIN_FREAMS = sustain  # 当检测到人眼超过50帧还在闭眼状态，说明人正在瞌睡

    # 检测帧次数
    COUNTER = 0

    # 加载人脸68点数据模型
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    print(face_utils.FACIAL_LANDMARKS_IDXS)
    # 获取人眼的坐标
    (left_eye_start_point, left_eye_end_point) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (right_eye_start_point, right_eye_end_point) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    (mouth_start_point, mouth_end_point) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    time.sleep(1.0)
    while True:
        # 从视频中获取图片来检测
        _, frame = input_module()
        # frame = imutils.resize(frame, width=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        point_list = []
        # for rect in rects:
        if len(rects) == 0:
            continue
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        landmarks = list((p.x, p.y) for p in predictor(frame, rect).parts())
        point_list.append(landmarks[30])
        point_list.append(landmarks[8])
        point_list.append(landmarks[36])
        point_list.append(landmarks[45])
        point_list.append(landmarks[48])
        point_list.append(landmarks[54])

        for idx, point in enumerate(point_list):
            cv2.circle(frame, point, 2, color=(0, 255, 255))

        # 提取人眼,嘴坐标，来计算人眼纵横比
        leftEye = shape[left_eye_start_point:left_eye_end_point]
        rightEye = shape[right_eye_start_point:right_eye_end_point]
        mouth = shape[mouth_start_point: mouth_end_point]
        mar = mouth_aspect_ratio(mouth)
        if mar > 25:
            print("Yawn")
        leftEAR = eye_ratio(leftEye)
        rightEAR = eye_ratio(rightEye)

        # 平均左右眼的纵横比
        ear = (leftEAR + rightEAR) / 2.0

        # 显示左右眼
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 显示嘴
        realmouth = cv2.convexHull(mouth)
        cv2.drawContours(frame, [realmouth], -1, (0, 255, 0), 1)

        # 测试头部检测

        image_points = np.array(point_list, dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                    dist_coeffs,
                                                                    flags=cv2.SOLVEPNP_ITERATIVE)

        X1, Y1, Z1 = get_euler_angle(rotation_vector)
        if Y1 <= -ANGLE_THRESHOLD:
            print("Turn Left")
        elif Y1 >= ANGLE_THRESHOLD:
            print("Turn Right")

        # 计算ear是否小于阈值
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= SUSTAIN_FREAMS:
                # 人瞌睡是要处理的函数
                cv2.putText(frame, "You Died", (230, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7)

            # 闭眼时，提醒
            cv2.putText(frame, "WARNING", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            COUNTER = 0  # 当检测不到闭眼
            cv2.putText(frame, 'EAR: {:.2f}'.format(ear), (580, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    eye_detection(0.2, 50)
