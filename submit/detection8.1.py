from ultralytics import YOLO
from toolkits.new_analysis import face_analysis
from toolkits.inference import spiga_frame_inference

import cv2
import os

import numpy as np
from datetime import datetime
import time
import json


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


def scan_file(path):
    filelist = os.listdir(path)
    file_d = {path: filelist}
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            file_d[filepath] = scan_file(filepath)
    return file_d


def run_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)

    # 初始化所有模型
    yolo_model = YOLO('best.pt')
    side_model = YOLO('120best.pt')
    device = 'cpu'
    results = yolo_model(cv2.imread('bus.jpg'))

    cnt = 0  # 开始处理的帧数
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 待处理的总帧数

    # 获取当前视频的帧率与宽高，设置同样的格式，以确保相同帧率与宽高的视频输出
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ######################################################################################################
    result = {"result": {"category": 0, "duration": 6000}}

    ANGLE_THRESHOLD = 35
    EAR_THRESHOLD = 0.18
    YAWN_THRESHOLD = 0.4

    eyes_closed_frame = 0
    mouth_open_frame = 0
    use_phone_frame = 0
    max_phone = 0
    max_eyes = 0
    max_mouth = 0
    max_wandering = 0
    look_around_frame = 0
    result_list = []

    front_face = True
    last_turning_head = False
    turning_num = 0


    sensitivity = 0.001
    inactivations = [1,28,52]
    landmarks_mem = []


    now = time.time()  # 读取视频与加载模型的时间不被计时（？）

    while cap.isOpened():
        while len(landmarks_mem) > 2:
            landmarks_mem.pop(0)

        if cnt >= frames:
            break
        phone_around_face = False
        is_eyes_closed = False
        is_turning_head = False
        is_yawning = False
        is_moving = False
        frame_result = 0

        landmark_entropy, trl_entropy, mar, ear, aar = 0,0,0,888,0
        L_E, R_E = 888,888  # 左右眼睛的状态
        overlap = 0

        cnt += 1
        ret, frame = cap.read()
        if cnt % 21 != 0 and cnt != 2 and not(cnt in inactivations):  #帧数
            continue
        if cnt + 80 > frames:  # 最后三秒不判断了
            break

        print(f'video {cnt}/{frames} {save_path}')  # delete
        # process the image with the yolo and get the person list
        results = yolo_model(frame)

        # 创建img0的副本
        img1 = frame.copy()

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
            is_turning_head = True
            img1 = img1
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
            y2 = min(img_width, int(y2 + 0.1 * (y2 - y1)))
            # img1 = img1[y1:y2, x1:x2]
            m1, n1, m2, n2 = x1, y1, x2, y2
            # x1, y1, x2, y2 = rightmost_box
            # #将图片的增大百分之20
            # dw, dh = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
            # x1, y1, x2, y2 = max(0, x1 - dw), max(0, y1 - dh), min(w, x2 + dw), min(h, y2 + dh)
            # # img1 = img1[y1:y2, x1:x2]
            # m1, n1, m2, n2 = x1, y1, x2, y2

            # 计算交集的面积
            if rightmost_phone is not None and rightmost_box is not None:
                # 计算两个框的IoU
                overlap = iou(rightmost_box, rightmost_phone)

                # cv2.putText(img1, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(overlap)
                # 如果IoU大于阈值，打印警告
                if overlap > sensitivity:
                    phone_around_face = True  ##判断手机

        # 计算边界框的宽度和高度
        w = m2 - m1
        h = n2 - n1

        # 创建 bbox
        bbox = [m1, n1, w, h]

        # frame = spig_process_frame(frame, bbox)  # TODO: 删除
        if phone_around_face == False:
            # 五个指标
            landmark_entropy, mar, ear, aar, L_E, R_E = face_analysis(frame, bbox, landmarks_mem, get_detail=True)  # TODO: 添加功能
        ################################################  转头的yolo判断
            side_results = side_model(frame)
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
                if box[0] > img_width * 4/9:
                    if rightmost_box is None or box[0] > rightmost_box[0]:
                        rightmost_box = box
                        rightmost_cls = cls

            if rightmost_cls != 0 or rightmost_box is None:
                is_turning_head = True
            else:
                is_turning_head = False

            is_moving = True if landmark_entropy > 50 else False
################################################################################
            is_yawning = True if mar > YAWN_THRESHOLD else False
            # is_eyes_closed = True if (ear < EAR_THRESHOLD) else False
            if cnt == 2 and ear < EAR_THRESHOLD: # 第二帧就识别成小眼
                EAR_THRESHOLD = 0.15
            is_eyes_closed = True if (L_E < EAR_THRESHOLD or R_E < EAR_THRESHOLD) else False

        if not (cnt in inactivations): #如果不在不活跃的帧数里
            if is_eyes_closed:
                eyes_closed_frame += 1
                frame_result = 1
            else:
                eyes_closed_frame = 0


            if is_yawning: #2
                print(mar)
                frame_result = 2
                mouth_open_frame += 1  #帧数
                eyes_closed_frame = 0  # 有打哈欠则把闭眼和转头置零
                # look_around_frame = 0
                if mouth_open_frame > max_mouth:
                    max_mouth = mouth_open_frame
            else:
                mouth_open_frame = 0

            # if is_turning_head:
            if is_moving:
                eyes_closed_frame = 0
            if is_turning_head:
                frame_result = 3
                look_around_frame += 1  #帧数
                mouth_open_frame = 0
                eyes_closed_frame = 0
                if look_around_frame > max_wandering:
                    max_wandering = look_around_frame

            else:
                look_around_frame = 0

            if phone_around_face: #3
                print(overlap)
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

        else:
            if phone_around_face: #3
                mouth_open_frame = 0
                look_around_frame = 0
                eyes_closed_frame = 0
            if is_turning_head:
                mouth_open_frame = 0
                eyes_closed_frame = 0
            if not is_eyes_closed:
                eyes_closed_frame = 0
            if not is_yawning:
                mouth_open_frame = 0

            ###################################################
            # im0 = display_results(img0, det, names,is_eyes_closed, is_turning_head, is_yawning)
            # # write video
            # vid_writer.write(im0)
        result_list.append(frame_result)

        if use_phone_frame >= 4:  #帧数
            result['result']['category'] = 3
            break

        elif look_around_frame >= 4:  #帧数
            result['result']['category'] = 4
            break

        elif mouth_open_frame >= 4:  #帧数
            result['result']['category'] = 2
            break

        elif eyes_closed_frame >= 4:  #帧数
            result['result']['category'] = 1
            break

    final_time = time.time()
    duration = int(np.round((final_time - now) * 1000))

    cap.release()

    result['result']['duration'] = duration

    return result

def main():
    video_dir = r'F:\ChallengeCup'
    save_dir = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\output'

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    # video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".json")]

    # Create a new log file with the current time
    log_file = os.path.join(save_dir, datetime.now().strftime("%Y%m%d%H%M%S") + '_log.json')
    log = {}

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        save_path = os.path.join(save_dir, video_file)
        print(video_file)

        try:
            result = run_video(video_path, save_path)
            json_save_path = save_path.rsplit('.', 1)[0] + '.json'

            with open(json_save_path, 'w') as json_file:
                json.dump(result, json_file)

            # Update the log and write it to the log file
            log[video_file] = result
            with open(log_file, 'w') as log_json:
                json.dump(log, log_json)


        except ValueError as error:
            print(error)


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()