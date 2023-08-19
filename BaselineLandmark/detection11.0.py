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

def insert_drowsy_behavior(json_obj, category, start_time, end_time):
    if "result" in json_obj and "drowsy" in json_obj["result"]:
        behavior = {
            "period": [start_time, end_time],
            "caregory": int(category)
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



def run_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)

    # 初始化所有模型
    # yolo_model = YOLO('best.pt')
    yolo_model = YOLO('bestface.pt')
    side_model = YOLO('best ce_face.pt')

    tracker = Tracker(1920, 1080, threshold=None, max_threads=4, max_faces=4,
                      discard_after=10, scan_every=3, silent=True, model_type=3,
                      model_dir=None, no_gaze=False, detection_threshold=0.6,
                      use_retinaface=0, max_feature_updates=900,
                      static_model=True, try_hard=False)
    device = 'cpu'
    results = yolo_model(cv2.imread('bus.jpg'))
    results = side_model(cv2.imread('bus.jpg'))


    cnt = 0  # 开始处理的帧数
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 待处理的总帧数
    on_behavior = 0  # 是否处于任何行为
    temp_start_time = None  # 临时开始时间
    last_frame_result = 0


    # 获取当前视频的帧率与宽高，设置同样的格式，以确保相同帧率与宽高的视频输出
    fps = cap.get(cv2.CAP_PROP_FPS)
    tickness = int(fps / 2) #每秒测几
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
                            # {
                            #     "period":[3200,7100],
                            #     "caregory":1,
                            # },
                            # {
                            #     "period":[8000,9000],
                            #     "caregory":2,
                            # }
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

    inactivations = [1,28,52]
    last_confirm = []
    # 用于判断是否在看手机
    front_face = True
    last_turning_head = False
    turning_num = 0


    sensitivity = 0.0005




    now = time.time()  # 读取视频与加载模型的时间不被计时（？）

    while cap.isOpened():
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
        standard_pose = [180, 40, 80]
        lStart = 42
        lEnd = 48
        # (rStart, rEnd) = (36, 42)
        rStart = 36
        rEnd = 42
        # (mStart, mEnd) = (49, 66)
        mStart = 49
        mEnd = 66
        EAR_THRESHOLD = 0.16
        YAWN_THRESHOLD = 0.5
        face_num = 0

        new_test = 0

        if eyes_closed_frame == 6-1:  #如果连续5帧都是闭眼的，则在第2.8s加一个快查
            new_test = int(cnt + 0.1 * fps_3s)
            last_confirm.append(new_test)



        cnt += 1
        ret, frame = cap.read()
        # if cnt % 10 != 0 and cnt != 2 and not(cnt in inactivations):  #帧数
        if cnt % tickness != 0 and not (cnt in last_confirm):
            continue
        # if cnt + int(fps_3s*8/9) > frames:
        if cnt + int(fps * 0.2) > frames :  # 最后三秒，若没有异常则不判断了#TODO:
            #如果result_list最后一位是0
            if len(result_list) > 0 and result_list[-1] == 0:
                break




        print(f'video {cnt}/{frames} {save_path}')  # delete
        # process the image with the yolo and get the person list
        results = yolo_model(frame)
        if results is None:
            continue

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

        if phone_around_face == False:
            # 五个指标
            faces = tracker.predict(img0)

            if len(faces) > 0:              #关键点检测部分
                face_num = None
                max_x = 0
                for face_num_index, f in enumerate(faces):
                    if max_x <= f.bbox[3]:
                        face_num = face_num_index
                        max_x = f.bbox[3]
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


        ################################################  转头的yolo判断
            #这个地方需要有两种策略:如果这里传入frame,则不需要进行在画幅右4/9的判断.如果传入的img0则需要判断.
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
            is_yawning = True if mar > YAWN_THRESHOLD else False
            is_eyes_closed = True if (ear < EAR_THRESHOLD) else False
            # if cnt == 2 and ear < EAR_THRESHOLD: # 第二帧就识别成小眼
            #     EAR_THRESHOLD = 0.15
            # is_eyes_closed = True if (L_E < EAR_THRESHOLD or R_E < EAR_THRESHOLD) else False

        # if not (cnt in inactivations): #如果不在不活跃的帧数里
        #如果results的长度等于零，则将is_yawning，is_eyes_closed，is_turning_head，is_moving都置为False
        if len(results) == 0:
            is_yawning = False
            is_eyes_closed = False
            is_turning_head = False
            is_moving = False

        if on_behavior != 0:
            #如果动作改变
            if (on_behavior == 1 and not (is_eyes_closed)) or (on_behavior == 3 and not (phone_around_face)) or (on_behavior == 4 and not (is_turning_head)) or  (on_behavior == 2 and not (is_yawning)):
                start_time = int(temp_start_time) - 350
                end_time = int(cnt/fps*1000) -200
                insert_drowsy_behavior(result,on_behavior,start_time, end_time)
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

        real_fps_3s = int(fps_3s * alpha)
        if use_phone_frame >= int(fps_3s * mar_alpha):  #帧数
            on_behavior = 3

        elif look_around_frame >= (fps_3s * mar_alpha):  #帧数
            on_behavior = 4


        elif mouth_open_frame >= int(fps_3s * mar_alpha):#帧数
            on_behavior = 2

        elif eyes_closed_frame >= real_fps_3s:  #帧数
            on_behavior = 1



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
    print(f'len(result_list):{len(result_list)}')
    print(f'result_list:{result_list}')
    print(f'result_cnt_list:{result_cnt_list}')
    print(f'mar_list:{mar_list}')
    print(f'ear_list:{ear_list}')
    print(f'L_E_list:{L_E_list}')
    print(f'R_E_list:{R_E_list}')
    print(f'iou_list:{iou_list}')
    print(f'side_list:{side_list}')
    print(f'yolo1_list:{yolo1_list}')
    print(f'yolo2_list:{yolo2_list}')



    cap.release()

    result['result']['duration'] = duration
    print(result)

    return result, result_list, result_cnt_list, mar_list, ear_list, L_E_list, R_E_list


def main():
    # video_dir = r'F:\ccp1\close'
    video_dir = r'D:\0000000\video\2'
    # video_dir = r'F:\ccp2\interference\check'
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
            result, result_list, result_cnt_list, mar_list, ear_list, L_E_list, R_E_list = run_video(video_path, save_path)
            json_save_path = save_path.rsplit('.', 1)[0] + '.json'

            # with open(json_save_path, 'w') as json_file:
            #     json.dump(result, json_file)

            # Update the log and write it to the log file
            log[video_file] = result
            with open(log_file, 'w') as log_json:
                json.dump(log, log_json)


        except ValueError as error:
            print(error)




if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    main()