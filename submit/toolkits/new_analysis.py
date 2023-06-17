import cv2
import numpy as np
from .inference import landmarks_68_2d_inference, spiga_frame_inference
from .arithmetic_3d import eye_aspect_ratio_3d, mouth_aspect_ratio_3d, pose_estimation_3d ,angle_aspect_ratio_3d,eye_aspect_ratio_3d_detail
def face_analysis(image, rect, landmarks_mem=None,get_detail=False):
    # set threshold
    # ANGLE_THRESHOLD = 30
    # EAR_THRESHOLD = 0.2f
    # YAWN_THRESHOLD = 0.4


    landmarks, headpose, _ = spiga_frame_inference(image, rect) # 人脸关键点检测
    if len(landmarks_mem) > 1  :
        landmarks_mem.pop(0)    # 删除第一帧
    landmarks_mem.append(landmarks) # 保存前后两帧的landmarks

    # 计算纵横比
    landmarks_eur_dis = np.linalg.norm(landmarks_mem[0] - landmarks_mem[1], axis=1).sum() if len(landmarks_mem)> 1 else 0
    landmark_entropy = np.exp(landmarks_eur_dis / 5000 - 1)



    mar = mouth_aspect_ratio_3d(landmarks)
    pose = pose_estimation_3d(headpose)
    aar = angle_aspect_ratio_3d(headpose)


    if not get_detail:
        ear = eye_aspect_ratio_3d(landmarks)
        return landmark_entropy, mar, ear, aar
    else:
        ear,LE,RE = eye_aspect_ratio_3d_detail(landmarks)
        return landmark_entropy, mar, ear, aar, LE, RE