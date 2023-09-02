#from x.2

# import torch
import cv2
import numpy as np
import copy
from grabscreen import grab_screen
# yolo_model = YOLO('best.pt')
# side_model = YOLO(r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\submit\side.pt')
device = 'cpu'
half = device != 'cpu'
from tracker import Tracker
from EAR import eye_aspect_ratio
from ultralytics import YOLO
from MAR import mouth_aspect_ratio
from time import time
from datetime import datetime
import time
import json

def modify_duration(json_obj, new_duration):
    if "result" in json_obj and "duration" in json_obj["result"]:
        json_obj["result"]["duration"] = new_duration
    else:
        raise KeyError("Duration key not found in the JSON object")
    return json_obj


def single_frame_detection(frame,yolo_model, side_model, tracker,YAWN_THRESHOLD,EAR_THRESHOLD,sensitivity):

    phone_around_face = False
    is_eyes_closed = False
    is_turning_head = False
    is_yawning = False
    is_moving = False
    yolo1 = True
    yolo2 = None
    frame_result = 0
    lStart = 42
    lEnd = 48
    # (rStart, rEnd) = (36, 42)
    rStart = 36
    rEnd = 42
    results = yolo_model(frame)
    if results is None:
        return -1
    img0 = frame.copy()
    # 获取图像的宽度
    img_height = frame.shape[0]

    img_width = frame.shape[1]

    # 获取所有的边界框
    boxes = results[0].boxes

    # 获取所有类别
    classes = boxes.cls

    # 获取所有的置信度
    confidences = boxes.conf

    # 初始化最靠右的框和最靠右的手机
    rightmost_box = None
    rightmost_phone = None

    # 遍历所有的边界框
    for box, cls, conf in zip(boxes.xyxy, classes, confidences):
        # 如果类别为1（手机）且在图片的右2/3区域内
        if cls == 1 and box[0] > img_width * 1 / 3:
            # 如果还没有找到最靠右的手机或者这个手机更靠右
            if rightmost_phone is None or box[0] > rightmost_phone[0]:
                rightmost_phone = box

        # 如果类别为0（驾驶员）且在图片的右2/3区域内
        if cls == 0 and box[0] > img_width * 1 / 3:
            # 如果还没有找到最靠右的框或者这个框更靠右
            if rightmost_box is None or box[0] > rightmost_box[0]:
                rightmost_box = box

    # 如果没有找到有效的检测框，返回img1为img0的右3/5区域
    if rightmost_box is None:
        yolo1 = None
        is_turning_head = True
        print('no human box found')

        m1 = int(img_width * 2 / 5)
        n1 = 0
        # 右下角的坐标
        m2 = img_width
        n2 = img_height

    # 否则，返回img1仅拥有最靠右的框内的图片
    else:
        x1, y1, x2, y2 = rightmost_box
        x1 = max(0, int(x1 - 0.1 * (x2 - x1)))
        y1 = max(0, int(y1 - 0.1 * (y2 - y1)))
        x2 = min(img_width, int(x2 + 0.1 * (x2 - x1)))
        y2 = min(img_width, int(y2 + 0.1 * (y2 - y1)))  # 把框扩大10%
        m1, n1, m2, n2 = x1, y1, x2, y2

        # 计算交集的面积
    if rightmost_phone is not None and rightmost_box is not None:
        # 计算两个框的IoU
        overlap = iou(rightmost_box, rightmost_phone)
        # if isinstance(overlap, torch.Tensor):
        #     overlap = overlap.item()
        print(overlap)
        # 如果IoU大于阈值，打印警告
        if overlap > sensitivity:
            phone_around_face = True
    if phone_around_face == False:
        # 五个指标
        faces = tracker.predict(img0)
        if len(faces) > 0:  # 关键点检测部分
            face_num = None
            max_x = 0
            for face_num_index, f in enumerate(faces):
                if max_x <= (f.bbox[1]+f.bbox[3])/2:
                    face_num = face_num_index
                    max_x = (f.bbox[1]+f.bbox[3])/2
            if face_num is not None:
                f = faces[face_num]
                f = copy.copy(f)
                print(f'box[0]:{max_x}')

                # 检测是否转头
                # if np.abs(standard_pose[0] - f.euler[0]) >= 45 or np.abs(standard_pose[1] - f.euler[1]) >= 45 or \
                #         np.abs(standard_pose[2] - f.euler[2]) >= 45:
                #     is_turning_head = True
                # else:
                #     is_turning_head = False

                # 检测是否闭眼
                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                # 检测是否张嘴
                mar = mouth_aspect_ratio(f.lms)
                leftEye = f.lms[lStart:lEnd]
                rightEye = f.lms[rStart:rEnd]
                L_E = eye_aspect_ratio(leftEye)
                R_E = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (L_E + R_E) / 2.0

                is_yawning = True if mar > YAWN_THRESHOLD else False
                close_start_cnt = None
                if (ear < EAR_THRESHOLD):
                    is_eyes_closed = True
                else:
                    is_eyes_closed = False

            ################################################  转头的yolo判断
            # 这个地方需要有两种策略:如果这里传入frame,则不需要进行在画幅右4/9的判断.如果传入的img0则需要判断.
        side_results = side_model(img0)
        side_boxes = side_results[0].boxes

        # 获取所有类别
        side_classes = side_boxes.cls
        # 获取所有的置信度
        side_confidences = side_boxes.conf

        # 初始化最靠右的框和最靠右的手机
        rightmost_box = None

        rightmost_cls = None

        # 遍历所有的边界框
        for box, cls, conf in zip(side_boxes.xyxy, side_classes, side_confidences):
            if box[0] > img_width * 0.477:
                if rightmost_box is None or box[0] > rightmost_box[0]:
                    rightmost_box = box
                    rightmost_cls = cls

            # if rightmost_box is None or box[0] > rightmost_box[0]:
            #     rightmost_box = box
            #     rightmost_cls = cls

        if rightmost_box is None:
            yolo2 = None
            is_turning_head = True
        elif rightmost_cls != 0 and rightmost_cls is not None:
            yolo2 = True
            is_turning_head = True
        else:
            yolo2 = False
            is_turning_head = False

            # is_moving = True if landmark_entropy > 50 else False
            ################################################################################

    if is_eyes_closed:
        frame_result = 1
    if is_yawning:  # 2
        frame_result = 2
    if is_moving:
        eyes_closed_frame = 0
    if is_turning_head:
        frame_result = 4
    if phone_around_face:  # 3
        print(f'overlap:{overlap}')
        frame_result = 3
    return frame_result



def find_nearest_smaller_cnt(cnt, cnt_photo_buff):
    low, high = 0, len(cnt_photo_buff) - 1
    result = -1  # 初始化为一个不可能的值，表示未找到
    position = -1  # 位置的初始化值，表示未找到

    while low <= high:
        mid = (low + high) // 2

        if cnt_photo_buff[mid] == cnt:
            if mid - 1 >= 0:
                return cnt_photo_buff[mid - 1], mid - 1
            else:
                return result, position  # 如果没有更小的值，返回-1或其他适当的默认值及其位置
        elif cnt_photo_buff[mid] < cnt:
            result = cnt_photo_buff[mid]
            position = mid
            low = mid + 1
        else:
            high = mid - 1

    return result, position


def insert_drowsy_behavior(json_obj, category, start_time, end_time):
    if "result" in json_obj and "drowsy" in json_obj["result"]:
        behavior = {
            "period": [start_time, end_time],
            "category": int(category)
        }
        json_obj["result"]["drowsy"].append(behavior)
    else:
        raise KeyError("Drowsy key not found in the JSON object")
    return json_obj


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # calculate the overlap coordinates
    xx1, yy1 = max(x1, x3), max(y1, y3)
    xx2, yy2 = min(x2, x4), min(y2, y4)

    # compute the width and height of the overlap area
    overlap_width, overlap_height = max(0, xx2 - xx1), max(0, yy2 - yy1)
    overlap_area = overlap_width * overlap_height

    # compute the area of both bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    # compute IoU
    iou = overlap_area / float(box1_area + box2_area - overlap_area)
    return iou

#修改均值由0.8-0.9
def check_sequence(ear_list, result_list, diff_threshold=0.04):
    ear_array = np.array(ear_list)
    result_array = np.array(result_list)

    # 检查长度
    if len(ear_array) < 6 or len(ear_array) != len(result_array):
        return False

    # 计算EAR序列的差分
    ear_diff = np.abs(np.diff(ear_array))

    print(ear_diff)

    for i in range(len(ear_diff)-5):
        current_diff = ear_diff[i:i+5]
        current_result = result_array[i:i+6]

        if np.any(current_diff > diff_threshold): #突变且不够小的数存在
            if not np.all(ear_array[i:i+6] < 0.18):
                continue

        # 检查是否有888
        if np.any(ear_array[i:i+6] == 888):
            continue

        if not (np.all(np.logical_or(current_result == 0, current_result == 1))):
            continue

        # 剔除888
        other_ear = ear_array[
            np.logical_and(np.logical_and(np.arange(len(ear_array)) != i, np.arange(len(ear_array)) != i + 5),
                           ear_array != 888)]

        if len(other_ear) == 0:
            continue

        mean1 = np.mean(ear_array[i:i+6])
        mean2 = np.mean(other_ear)
        # 计算平均值并比较
        if mean1 < mean2 * 0.9 :
            # 在窗口内的变动是否超过阈值
            return True

    return False
def ear_realtime_check_sequence(ear_list, result_list, new_ear, diff_threshold=0.04):
    i = len(ear_list) - 1
    ear_window = []
    if new_ear < 0.18:
        # 初始化滑动窗口 ear_window
        ear_window = [new_ear]

        n = 0
        while i - n >= 0:
            # 扩充条件
            current_ear = ear_list[i - n]


            # 根据给定的条件检查当前 ear 值
            if (
                    current_ear < 0.18 and
                    result_list[i - n] in [0, 1] and
                    (n == 0 or abs(current_ear - ear_list[i - n + 1]) < diff_threshold)):
                # 如果满足条件，则将当前 ear 添加到 ear_window
                ear_window.insert(0, current_ear)
                n += 1
            else:
                # 如果不满足条件，则结束
                break

        ear_window_mean = sum(ear_window) / len(ear_window) if ear_window else 0
        # ear_list 中除去 ear_window 的其他的值的平均值
        other_ears = [ear for idx, ear in enumerate(ear_list) if idx < i - n or idx >= i - n + len(ear_window)]
        other_ears_mean = sum(other_ears) / len(other_ears) if other_ears else 0
        ear_window_mean >= 0.9 * other_ears_mean
        # 返回从当前帧开始的最早的连续闭眼帧的位置
    return i - n + 1 if ear_window else i


def run_video(video_path, save_path,yolo_model,side_model,tracker):
    cap = cv2.VideoCapture(video_path)

    # 初始化所有模型

    # yolo_model = YOLO('bestface.pt')
    # side_model = YOLO('best ce_face.pt')
    #
    # tracker = Tracker(1920, 1080, threshold=None, max_threads=4, max_faces=4,
    #                   discard_after=10, scan_every=3, silent=True, model_type=3,
    #                   model_dir=None, no_gaze=False, detection_threshold=0.6,
    #                   use_retinaface=0, max_feature_updates=900,
    #                   static_model=True, try_hard=False)
    # device = 'cpu'
    # results = yolo_model(cv2.imread('bus.jpg'))
    # results = side_model(cv2.imread('bus.jpg'))


    cnt = 0  # 开始处理的帧数
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 待处理的总帧数
    on_behavior = 0  # 是否处于任何行为
    temp_start_time = None  # 临时开始时间
    last_frame_result = 0


    # 获取当前视频的帧率与宽高，设置同样的格式，以确保相同帧率与宽高的视频输出
    fps = cap.get(cv2.CAP_PROP_FPS)
    tickness = int(fps / 2) #每秒测几
    buff_tickness = int(tickness/2)
    photo_buff = []
    cnt_photo_buff = []

    fps_3s = int(fps / tickness) * 3#
    alpha = 1#9帧中取7帧
    mar_alpha = 0.85 #mar的检测可以用5/6帧

    # real_fps_3s = int(fps_3s * alpha)#放在进程处单独检测了


    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ######################################################################################################
    # result = {
    #             "result":
    #                 {
    #                     "duration":6000,
    #                     "drowsy":
    #                     [
    #                         {
    #                             "period":[3200,7100],
    #                             "caregory":1,
    #                         },
    #                         {
    #                             "period":[8000,9000],
    #                             "caregory":2,
    #                         }
    #                     ]
    #                 }
    #         }
    result = {
                "result":
                    {
                        "duration":6000,
                        "drowsy":
                        [

                        ]
                    }
            }




    eyes_closed_frame = 0
    mouth_open_frame = 0
    use_phone_frame = 0
    max_phone = 0
    max_eyes = 0
    max_mouth = 0
    max_wandering = 0
    look_around_frame = 0
    result_list = []
    result_cnt_list = []
    ear_list = []
    mar_list = []
    L_E_list = []
    R_E_list = []
    iou_list = []
    side_list = []
    yolo1_list = []
    yolo2_list = []
    poses_list = []
    last_cnt = False
    inactivations = []
    last_confirm = []
    # 用于判断是否在看手机
    front_face = True
    last_turning_head = False
    turning_num = 0


    sensitivity = 0.0005
    time_read_video = 0 #1-0
    time_pred1 = 0 #3-2
    time_phone_process = 0 #5-4
    time_face_retina = 0 #7-6
    time_face_data_process = 0
    time_side_yolo = 0
    time_face_data_process_after_side = 0
    time_total_result_process = 0


    ##人脸方位角初始化
    stander_poses = []#poses[0] - poses[2] 为人脸方位角 poses[3]为box[0]的位置 区间为100

    now = time.time()  # 读取视频与加载模型的时间不被计时（？）

    while cap.isOpened():
        #让time0到time10都置零
        time0 = 0
        time1 = 0
        time2 = 0
        time3 = 0
        time4 = 0
        time5 = 0
        time6 = 0
        time7 = 0
        time8 = 0
        time9 = 0
        time10 = 0
        time0 = time.time()

        if cnt >= frames:
            break
        phone_around_face = False
        is_eyes_closed = False
        is_turning_head = False
        is_yawning = False
        is_moving = False
        yolo1 = True
        yolo2 = None
        frame_result = 0

        landmark_entropy, trl_entropy, mar, ear, aar = 0,0,0,888,0
        L_E, R_E = 888,888  # 左右眼睛的状态
        overlap = 0
        standard_pose = [180, 10, 80]
        lStart = 42
        lEnd = 48
        # (rStart, rEnd) = (36, 42)
        rStart = 36
        rEnd = 42
        # (mStart, mEnd) = (49, 66)
        mStart = 49
        mEnd = 66
        EAR_THRESHOLD = 0.16
        YAWN_THRESHOLD = 0.6
        face_num = 0
        poses = [0,0,0,0,0]

        new_test = 0

        # if eyes_closed_frame == 6-1:  #如果连续5帧都是闭眼的，则在第2.8s加一个快查
        #     new_test = int(cnt + 0.1 * fps_3s)
        #     last_confirm.append(new_test)



        cnt += 1
        ret, frame = cap.read()
        time1 = time.time()
        time_read_video = time1 - time0
        last_confirm = [1]
        if (cnt + fps * 0.5) > frames:
            last_cnt = True

        if cnt % tickness != 0 and  (cnt != 1):
            if (cnt-buff_tickness) % tickness == 0 and (cnt - buff_tickness) != 0: #buff_tickness = int(tickness/2)
                photo_buff.append(frame)
                cnt_photo_buff.append(cnt)
            continue
        if cnt + int(fps * 0.8) > frames and on_behavior == 0:  # 最后三秒，若没有异常则不判断了
            #如果result_list最后一位是0
            if len(result_list) > 0 and result_list[-1] == 0:
                break




        print(f'video {cnt}/{frames} {save_path}')  # delete
        # process the image with the yolo and get the person list
        time2 = time.time()
        results = yolo_model(frame)
        time3 = time.time()
        time_pred1= time3 - time2
        if results is None:
            continue
        time4 = time.time()
        # 创建img0的副本,并进行faceimg的裁剪
        img0 = frame.copy()
        # width = img0.shape[1]
        # start_col = int(width * 8 / 30)
        # face_img = img0[:, start_col:, :]

        # 获取图像的宽度
        img_height = frame.shape[0]

        img_width = frame.shape[1]

        # 获取所有的边界框
        boxes = results[0].boxes

        # 获取所有类别
        classes = boxes.cls

        # 获取所有的置信度
        confidences = boxes.conf

        # 初始化最靠右的框和最靠右的手机
        rightmost_box = None
        rightmost_phone = None

        # 遍历所有的边界框
        for box, cls, conf in zip(boxes.xyxy, classes, confidences):
            # 如果类别为1（手机）且在图片的右2/3区域内
            if cls == 1 and box[0] > img_width * 1 / 3:
                # 如果还没有找到最靠右的手机或者这个手机更靠右
                if rightmost_phone is None or box[0] > rightmost_phone[0]:
                    rightmost_phone = box

            # 如果类别为0（驾驶员）且在图片的右2/3区域内
            if cls == 0 and box[0] > img_width * 1 / 3:
                # 如果还没有找到最靠右的框或者这个框更靠右
                if rightmost_box is None or box[0] > rightmost_box[0]:
                    rightmost_box = box


        # 如果没有找到有效的检测框，返回img1为img0的右3/5区域
        if rightmost_box is None:
            yolo1 = None
            is_turning_head = True
            print('no human box found')

            m1 = int(img_width * 2 / 5)
            n1 = 0
            # 右下角的坐标
            m2 = img_width
            n2 = img_height

        # 否则，返回img1仅拥有最靠右的框内的图片
        else:
            x1, y1, x2, y2 = rightmost_box
            x1 = max(0, int(x1 - 0.1 * (x2 - x1)))
            y1 = max(0, int(y1 - 0.1 * (y2 - y1)))
            x2 = min(img_width, int(x2 + 0.1 * (x2 - x1)))
            y2 = min(img_width, int(y2 + 0.1 * (y2 - y1))) # 把框扩大10%
            m1, n1, m2, n2 = x1, y1, x2, y2


            # 计算交集的面积
            if rightmost_phone is not None and rightmost_box is not None:
                # 计算两个框的IoU
                overlap = iou(rightmost_box, rightmost_phone)
                # if isinstance(overlap, torch.Tensor):
                #     overlap = overlap.item()
                print(overlap)
                # 如果IoU大于阈值，打印警告
                if overlap > sensitivity:
                    phone_around_face = True
                                        ##判断手机



        # # frame = frame[y1:y2, x1:x2]
        # frame = frame[n1:n2, m1:m2]
        # img1 = img0[:, 600:1920, :]
        time5 = time.time()
        frame_for_tracker = frame[:, int(frame.shape[1] / 2):, :]
        time_phone_process += time5 - time4
        if phone_around_face == False:
            # 五个指标
            time6 = time.time()
            faces = tracker.predict(frame_for_tracker)
            time7 = time.time()
            time_face_retina += time7 - time6
            if len(faces) > 0:              #关键点检测部分
                face_num = None
                max_x = 0
                face_area = 0
                for face_num_index, f in enumerate(faces):
                    if max_x <= (f.bbox[1] + f.bbox[3]) / 2:
                        face_num = face_num_index
                        max_x = (f.bbox[1] + f.bbox[3]) / 2

                if face_num is not None:
                    f = faces[face_num]
                    f = copy.copy(f)
                    face_area = np.abs(int(f.bbox[2] * f.bbox[3]))
                    print(f'box[0]:{max_x}')

                    # 检测是否转头
                    # if np.abs(standard_pose[0] - f.euler[0]) >= 45 or np.abs(standard_pose[1] - f.euler[1]) >= 45 or \
                    #         np.abs(standard_pose[2] - f.euler[2]) >= 45:
                    #     is_turning_head = True
                    # else:
                    #     is_turning_head = False


                    if len(stander_poses) == 0:
                        stander_poses.append(np.abs(f.euler[0]))
                        stander_poses.append(f.euler[1])
                        stander_poses.append(f.euler[2])
                        stander_poses.append(max_x)
                        stander_poses.append(face_area)
                        # stander_poses[1] = (f.euler[1] + 180 )/2
                        # stander_poses[2] = (f.euler[2] + 180 )/2
                        # stander_poses[3] = max_x

                    mar = mouth_aspect_ratio(f.lms)
                    leftEye = f.lms[lStart:lEnd]
                    rightEye = f.lms[rStart:rEnd]
                    L_E = eye_aspect_ratio(leftEye)
                    R_E = eye_aspect_ratio(rightEye)
                    # average the eye aspect ratio together for both eyes
                    ear = (L_E + R_E) / 2.0

                    poses[0] = np.abs(f.euler[0])
                    poses[1] = (f.euler[1])
                    poses[2] = (f.euler[2])
                    poses[3] = max_x
                    poses[4] = face_area

            time8 = time.time()
            time_face_data_process += time8 - time7
        ################################################  转头的yolo判断
            #这个地方需要有两种策略:如果这里传入frame,则不需要进行在画幅右4/9的判断.如果传入的img0则需要判断.
            if len(stander_poses) * poses[4] == 0:
                is_turning_head = True
                #poses[3]的绝对值和max_x相差过大
            elif np.abs(stander_poses[3] - max_x) >= 80:
                is_turning_head = True
            elif  np.abs(stander_poses[1] - poses[1]) >= 30 or np.abs(stander_poses[2] - poses[2]) >= 30 or (np.abs(stander_poses[4]) * 0.75 > np.abs(poses[4])): #np.abs(stander_poses[0] - poses[0]) >= 45 or
                is_turning_head = True
            else:
                is_turning_head = False



            # is_moving = True if landmark_entropy > 50 else False
################################################################################
            is_yawning = True if mar > YAWN_THRESHOLD else False
            close_start_cnt = None
            if (ear < EAR_THRESHOLD):
                is_eyes_closed = True
            else:
                close_start_cnt = ear_realtime_check_sequence(ear_list, result_list, ear, diff_threshold=0.04)
                if close_start_cnt != len(ear_list) and close_start_cnt is not None and close_start_cnt != len(ear_list) -1:
                    is_eyes_closed = True
                    temp_start_time = result_cnt_list[close_start_cnt]/fps*1000


            # if cnt == 2 and ear < EAR_THRESHOLD: # 第二帧就识别成小眼
            #     EAR_THRESHOLD = 0.15
            # is_eyes_closed = True if (L_E < EAR_THRESHOLD or R_E < EAR_THRESHOLD) else False
            time10 = time.time()
            time_face_data_process_after_side += time10 - time9
        # if not (cnt in inactivations): #如果不在不活跃的帧数里
        #如果results的长度等于零，则将is_yawning，is_eyes_closed，is_turning_head，is_moving都置为False
        time11 = time.time()
        if len(results) == 0:
            is_yawning = False
            is_eyes_closed = False
            is_turning_head = False
            is_moving = False

        if on_behavior != 0:
            #如果动作改变
            if last_cnt or (on_behavior == 1 and not (is_eyes_closed)) or (on_behavior == 3 and not (phone_around_face)) or (on_behavior == 4 and not (is_turning_head)) or  (on_behavior == 2 and not (is_yawning)):
                temp_start_cnt = temp_start_time/1000 * fps #这几个变量都没有预先定义，只限定在此if区域使用
                end_cnt = cnt
                recheck_start_frame = 0
                recheck_end_frame = 0
                start_time = int(temp_start_time) #-350 -200
                end_time = int(cnt / fps * 1000)
                frame1_cnt = None
                frame2_cnt = None
                try:
                    frame1_cnt,frame1_cnt_i = find_nearest_smaller_cnt(temp_start_cnt, cnt_photo_buff)
                    print('fps', fps)
                    print('temp_start_cnt', temp_start_cnt)
                    print('frame1_cnt', frame1_cnt)
                    frame1 = photo_buff[frame1_cnt_i]

                    recheck_start_frame = single_frame_detection(frame1, yolo_model, side_model, tracker, YAWN_THRESHOLD, EAR_THRESHOLD,
                                       sensitivity)
                    print('recheck_start_frame', recheck_start_frame)
                except ValueError as e:
                    recheck_recheck_start_frame = 0
                    print(f'recheck start error {e}')

                try:
                    frame2_cnt,frame2_cnt_i = find_nearest_smaller_cnt(cnt, cnt_photo_buff)
                    print('cnt', cnt)
                    print('frame2_cnt', frame2_cnt)
                    frame2 = photo_buff[frame2_cnt_i]
                    recheck_end_frame = single_frame_detection(frame2, yolo_model, side_model, tracker, YAWN_THRESHOLD, EAR_THRESHOLD,
                                        sensitivity)
                    print('recheck_end_frame', recheck_end_frame)
                    if recheck_end_frame == 2:
                        recheck_end_frame = 0
                except ValueError as e:
                    recheck_end_frame = 0
                    print(f'recheck end error {e}')

                if recheck_start_frame == on_behavior and frame1_cnt is not None:
                    # 如果小于0则令其等于0
                    start_time = 0 if int(frame1_cnt / fps * 1000) - 70 < 0 else int(frame1_cnt / fps * 1000) - 70

                if recheck_end_frame == on_behavior and frame2_cnt is not None:
                    end_time = 0 if int(frame2_cnt / fps * 1000) -50 < 0 else int(frame2_cnt / fps * 1000) -50



                if end_time - start_time >= 2500:
                    #小于两秒五的不插入
                    insert_drowsy_behavior(result,on_behavior,start_time, end_time)
                    # if (int(time.time())%4 == 0):
                    #     insert_drowsy_behavior(result, on_behavior, start_time+50, end_time+50)
                on_behavior = 0
                temp_start_time = None

        if is_eyes_closed:
            print(f'ear:{ear}')
            eyes_closed_frame += 1
            frame_result = 1
        else:
            eyes_closed_frame = 0


        if is_yawning: #2
            print(f'mar:{mar}')
            frame_result = 2
            mouth_open_frame += 1  #帧数
            eyes_closed_frame = 0  # 有打哈欠则把闭眼和转头置零
            # look_around_frame = 0
            if mouth_open_frame > max_mouth:
                max_mouth = mouth_open_frame
        else:
            mouth_open_frame = 0

        if is_moving:
            eyes_closed_frame = 0
        if is_turning_head:
            frame_result = 4
            look_around_frame += 1  #帧数
            mouth_open_frame = 0
            eyes_closed_frame = 0
            if look_around_frame > max_wandering:
                max_wandering = look_around_frame

        else:
            look_around_frame = 0

        if phone_around_face: #3
            print(f'overlap:{overlap}')
            frame_result = 3
            use_phone_frame += 1
            mouth_open_frame = 0
            look_around_frame = 0
            eyes_closed_frame = 0  # 有手机则把其他都置零
            front_face = True
            if use_phone_frame > max_phone:
                max_phone = use_phone_frame

        else:
            use_phone_frame = 0

        if frame_result == 0:
            temp_start_time = None
        elif frame_result != last_frame_result or temp_start_time is None:
            temp_start_time = cnt/fps*1000


        if not (cnt in inactivations):
            result_list.append(frame_result)
            result_cnt_list.append(cnt)
            ear_list.append(ear)
            mar_list.append(mar)
            L_E_list.append(L_E)
            R_E_list.append(R_E)
            side_list.append(is_turning_head)
            iou_list.append(overlap)
            yolo1_list.append(yolo1)
            yolo2_list.append(yolo2)
            last_frame_result = frame_result

            poses_list.append(poses)
        real_fps_3s = int(fps_3s * alpha)
        if use_phone_frame >= int(fps_3s * mar_alpha):  #帧数
            on_behavior = 3

        elif look_around_frame >= (fps_3s * mar_alpha):  #帧数
            on_behavior = 4


        elif mouth_open_frame >= int(fps_3s * mar_alpha):#帧数
            on_behavior = 2

        elif eyes_closed_frame >= real_fps_3s:  #帧数
            on_behavior = 1

        time12 = time.time()
        time_total_result_process += time12 - time11

        # if temp_start_time is None and on_behavior != 0:
        #     if on_behavior == 1: #zanshi
        #         temp_start_time = (cnt-tickness) / fps * 1000 - 3000 + 350
        #     else:
        #         temp_start_time = (cnt-tickness) / fps * 1000 - 3000 * mar_alpha + 350



    #对ear_list进行处理
    '''
    对于ear_list进行重新判断。重新遍历ear_list
    '''
    # if len(ear_list) >= fps_3s-1:
    #     ear_judge_list = []
    #     ear_open_total = ear_list[0]
    #     i = 1
    #     while i < len(ear_list):
    #         ear_avg = ear_open_total / i
    #         if ear_list[i] >= ear_avg * 0.83: #0.22 to 0.182
    #             ear_open_total += ear_list[i]
    #             ear_judge_list.append(0)
    #         else:
    #             ear_judge_list.append(1)
    #         i += 1
    # if result['result']['category'] == 0:
    #     try:
    #         result['result']['category'] = 1 if check_sequence(ear_list, result_list) else 0
    #     except:
    #         result['result']['category'] = 0
    final_time = time.time()
    duration = int(np.round((final_time - now) * 1000))
    print(f'len(result_list)={len(result_list)}')
    print(f'result_list={result_list}')
    print(f'result_cnt_lis={result_cnt_list}')
    print(f'cnt_photo_buff={cnt_photo_buff}')
    print(f'mar_list={mar_list}')
    print(f'ear_list={ear_list}')
    print(f'L_E_list={L_E_list}')
    print(f'R_E_list={R_E_list}')
    print(f'iou_list={iou_list}')
    print(f'side_list={side_list}')
    print(f'yolo1_list={yolo1_list}')
    print(f'yolo2_list={yolo2_list}')
    print(f'poses_list={poses_list}')
    time_count_result = {
        'time_read_video': time_read_video,
        'time_pred1': time_pred1,
        'time_phone_process': time_phone_process,
        'time_face_retina': time_face_retina,
        'time_face_data_process': time_face_data_process,
        'time_side_yolo': time_side_yolo,
        'time_face_data_process_after_side': time_face_data_process_after_side,
        'time_total_result_process': time_total_result_process
    }



    cap.release()

    result['result']['duration'] = duration
    print(result)


    return result, result_list, result_cnt_list, mar_list, ear_list, L_E_list, R_E_list,time_count_result

def accumulate_time_results(time_count_result, time_final_result):
    for key, value in time_count_result.items():
        time_final_result[key] += value
    return time_final_result
def main():
    # video_dir = r'F:\ChallengeCup\an'
    # video_dir = r'D:\0000000\new_dataset\bo'
    video_dir = r'F:\ccp1\close'
    save_dir = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\output'

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    # video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".json")]
    yolo_model = YOLO('bestface.pt')
    side_model = YOLO('best ce_face.pt')


    device = 'cpu'
    results = yolo_model(cv2.imread('bus.jpg'))
    results = side_model(cv2.imread('bus.jpg'))
    # Create a new log file with the current time
    log_file = os.path.join(save_dir, datetime.now().strftime("%Y%m%d%H%M%S") + '_log.json')
    log = {}
    time_final_result = {
        'time_read_video': 0,
        'time_pred1': 0,
        'time_phone_process': 0,
        'time_face_retina': 0,
        'time_face_data_process': 0,
        'time_side_yolo': 0,
        'time_face_data_process_after_side': 0,
        'time_total_result_process': 0
    }

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        save_path = os.path.join(save_dir, video_file)
        print(video_file)

        try:
            tracker = Tracker(960, 1080, threshold=None, max_threads=4, max_faces=4,
                              discard_after=10, scan_every=3, silent=True, model_type=3,
                              model_dir=None, no_gaze=False, detection_threshold=0.6,
                              use_retinaface=0, max_feature_updates=900,
                              static_model=True, try_hard=False)
            result, result_list, result_cnt_list, mar_list, ear_list, L_E_list, R_E_list, time_count_result = run_video(video_path, save_path, yolo_model, side_model, tracker)
            json_save_path = save_path.rsplit('.', 1)[0] + '.json'
            accumulate_time_results(time_count_result, time_final_result)


            # with open(json_save_path, 'w') as json_file:
            #     json.dump(result, json_file)

            # Update the log and write it to the log file
            log[video_file] = result
            with open(log_file, 'w') as log_json:
                json.dump(log, log_json)


        except ValueError as error:
            print(error)

    print(f'time_final_result:{time_final_result}')




if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    main()