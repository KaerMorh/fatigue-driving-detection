import numpy as np
import math
import cv2
# import dlib
from imutils import face_utils
from collections import OrderedDict
from scipy.spatial import distance as dist
from .spiga_inference import spiga_process_frame
from .model_loader import load_spiga_framework

spiga_model = load_spiga_framework()

FICIAL_LANDMARKS_98_WFLW = OrderedDict([
    ("left_eye", (60, 68)),
    ("right_eye", (68, 76)),
    ("mouth_exterior", (76, 88)),
    ("mouth_interior", (88, 96)),
])

model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# processor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 2D utils
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


# 3D utils
def eye_ratio_3d(landmarks):
    """
    计算人眼纵横比 (EAR)
    :param landmarks: 3D关键点数组
    :return: ear
    """
    # 得到左眼右眼的坐标
    left_eye = landmarks[60:68]
    right_eye = landmarks[68:76]

    # initialize the norm_op
    norm = np.linalg.norm
    gn = lambda id1, id2, ob: norm(ob[id1] - ob[id2])

    A1 = gn(1, 7, left_eye)
    A2 = gn(3, 5, left_eye)
    B = gn(2, 6, left_eye)
    C = gn(0, 4, left_eye)
    L_E = (A1 + A2 + B * 0.9) / (3.0 * C)

    A1 = gn(1, 7, right_eye)
    A2 = gn(3, 5, right_eye)
    B = gn(2, 6, right_eye)
    C = gn(0, 4, right_eye)
    R_E = (A1 + A2 + B * 0.9) / (3.0 * C)

    ear = (L_E + R_E) / 2.0  # 取左右眼纵横比的平均值
    return ear


def mouth_aspect_ratio_3d(landmarks):
    """
    计算嘴长宽比例
    :param landmarks: 3D关键点数组
    :return: MAR
    """
    mth_ext = landmarks[76: 88]
    mth_int = landmarks[88: 96]

    # initialize the norm_op
    norm = np.linalg.norm
    gmed = lambda p1, p2: (p1 + p2) / 2
    norm_of_med = lambda i1, i2, i3, i4: norm(gmed(mth_int[i1], mth_ext[i2]) - gmed(mth_int[i3], mth_ext[i4]))

    H = norm_of_med(0, 0, 4, 6)
    A1 = norm_of_med(1, 2, -1, -2)
    A2 = norm_of_med(3, 4, -3, -4)
    B = norm_of_med(2, 3, -2, -3)

    MAR = (A1 + A2 + B) / (3.0 * H)
    return MAR


def pose_estimation_3d(headpose):
    """
    计算姿态标识符
    :param headpose:  3D headpose
    :return:  PE
    """
    rot = headpose[:3]
    trl = headpose[3:]

    # PE = rot / 90.0
    return rot


# 2d process
def face_analysis_2d(image, rect):
    ANGLE_THRESHOLD = 45.0
    EAR_THRESHOLD = 0.2
    YAWN_THRESHOLD = 25

    is_eyes_closed = False
    is_turning_head = 0
    is_yawning = False

    # 将输入的图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    shape = processor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # 提取特征点中的眼睛和嘴巴的点
    (left_eye_start_point, left_eye_end_point) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (right_eye_start_point, right_eye_end_point) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    (mouth_start_point, mouth_end_point) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    leftEye = shape[left_eye_start_point:left_eye_end_point]
    rightEye = shape[right_eye_start_point:right_eye_end_point]
    mouth = shape[mouth_start_point: mouth_end_point]

    # 计算眼睛和嘴巴的纵横比
    leftEAR = eye_ratio(leftEye)
    rightEAR = eye_ratio(rightEye)
    mar = mouth_aspect_ratio(mouth)

    # 平均左右眼的纵横比
    ear = (leftEAR + rightEAR) / 2.0

    # 如果眼睛的纵横比小于阈值，那么判定为闭眼
    if ear < EAR_THRESHOLD:
        is_eyes_closed = True

    # 如果嘴巴的纵横比大于阈值，那么判定为打哈欠
    if mar > YAWN_THRESHOLD:
        is_yawning = True

    return is_eyes_closed, is_yawning


# total
def face_analysis(image, rect, test=0):
    # set threshold
    ANGLE_THRESHOLD = 35
    EAR_THRESHOLD = 0.2
    YAWN_THRESHOLD = 0.4

    landmarks, headpose, _ = spiga_process_frame(spiga_model, image, rect)

    # 计算纵横比
    ear = eye_ratio_3d(landmarks)
    mar = mouth_aspect_ratio_3d(landmarks)
    pose = pose_estimation_3d(headpose)
    if test == 1:
        return pose, mar, ear

    is_turning_head = True if np.abs(pose[[0, 2]]).max() > ANGLE_THRESHOLD else False
    is_yawning = True if mar > YAWN_THRESHOLD else False
    is_eyes_closed = True if ear < EAR_THRESHOLD else False


    # if np.abs(pose[[0, 2]]).max() > ANGLE_THRESHOLD:
    # else:
    #     is_turning_head = True
    #     # is_eyes_closed, is_yawning = face_analysis_2d(image, rect)
    #     is_eyes_closed, is_yawning = False, False

    return is_turning_head, is_yawning, is_eyes_closed


