import os
import time
import copy
import random
import traceback
import numpy as np


# from model_service.pytorch_model_service import PTServingBaseService

from input_reader import InputReader
from tracker import Tracker
from face_utils import iou, mouth_aspect_ratio, eye_aspect_ratio, check_sequence

from ultralytics import YOLO



class fatigue_driving_detection:
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.model_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        print(f"model_base_path: {self.model_base_path}")

        self.capture = 'test.mp4'
        self.width = 1920
        self.height = 1080
        self.need_reinit = 0
        self.failures = 0
        self.fps = 30

        print("-----------init begin!-----------")
        # set the three yolo models
        start_time = time.time()
        self.yolo_model = YOLO(model_path)
        self.side_model = YOLO(os.path.join(self.model_base_path, 'side_yolo'))
        # self.eyes_model = YOLO(os.path.join(self.model_base_path, 'eyes_yolo'))
        # set tracker
        self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=4, max_faces=4,
                          discard_after=10, scan_every=3, silent=True, model_type=3,
                          model_dir=None, no_gaze=False, detection_threshold=0.6,
                          use_retinaface=0, max_feature_updates=900,
                          static_model=True, try_hard=False)
        # warm up
        x = np.random.randn(384, 640, 3)
        self.yolo_model(x)
        self.side_model(x)
        # self.eyes_model(x)
        print(f'triple-yolo and tracker load done:{time.time() - start_time:2f}s')

        print("-----------init done!-----------")

    def _preprocess(self, data):
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def inference(self, data):
        print("-----------inference begin!-----------")
        print(data)
        # set input_reader
        self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps)
        print(f"width, height: {self.input_reader.get_video_dimensions()}")
        source_name = self.input_reader.name
        frames = self.input_reader.get_total_frames()
        fps = self.input_reader.get_fps()

        print(f"fps: {fps}")
        tickness = int(fps / 2)
        fps_3s = int(fps / tickness) * 3
        alpha = 1
        mar_alpha = 0.89

        eyes_closed_frame = 0
        mouth_open_frame = 0
        use_phone_frame = 0
        max_phone = 0
        max_eyes = 0
        max_mouth = 0
        max_wandering = 0
        look_around_frame = 0
        result_list = []
        result_yolo_list = []
        result_cnt_list = []
        result_yolo_cnt_list = []
        ear_list = []
        mar_list = []
        L_E_list = []
        R_E_list = []
        iou_list = []
        side_list = []
        yolo1_list = []
        yolo2_list = []

        inactivations = [1, 28, 52]
        last_confirm = []
        # 用于判断是否在看手机
        front_face = True
        last_turning_head = False
        turning_num = 0

        sensitivity = 0.0005

        cnt = 0
        result = {"result": {"category": 0, "duration": 6000}}
        now = time.time()

        while self.input_reader.is_open():
            # 读取视频单帧操作
            if not self.input_reader.is_open() or self.need_reinit == 1:
                self.input_reader = InputReader(self.capture,
                                                0,
                                                self.width,
                                                self.height,
                                                self.fps,
                                                use_dshowcapture=False,
                                                dcap=None)
                if self.input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {self.input_reader.name} instead of {source_name}.")
                    # sys.exit(1)
                self.need_reinit = 2
                time.sleep(0.02)
                continue
            if not self.input_reader.is_ready():
                time.sleep(0.02)
                continue

            phone_around_face = False
            is_eyes_closed = False
            is_turning_head = False
            is_yawning = False
            is_moving = False
            yolo1 = True
            yolo2 = None
            frame_result = 0

            landmark_entropy, trl_entropy, mar, ear, aar = 0, 0, 0, 888, 0
            L_E, R_E = 888, 888  # 左右眼睛的状态
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

            run_yolo = False
            run_tracker = False

            if eyes_closed_frame == 4:  # 如果连续5帧都是闭眼的，则在第2.8s加一个快查
                new_test = int(cnt + 0.1 * fps_3s)
                last_confirm.append(new_test)

            # read the frame
            ret, frame = self.input_reader.read()
            cnt += 1
            # if cnt % 10 != 0 and cnt != 2 and not(cnt in inactivations):  #帧数
            if cnt % tickness != 0 and not (cnt in last_confirm):
                continue
            if cnt + int(fps_3s * 8 / 9) > frames:  # 最后三秒，若没有异常则不判断了
                # 如果result_list最后一位是0
                if len(result_list) > 0 and result_list[-1] == 0:
                    break

            self.need_reinit = 0

            # __inference__ main
            try:

                # copy
                results = self.yolo_model(frame)
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
                            ##判断手机

                # # frame = frame[y1:y2, x1:x2]
                # frame = frame[n1:n2, m1:m2]
                # img1 = img0[:, 600:1920, :]
                phone_around_face = False  # TODO: delete
                if phone_around_face == False:
                    # 五个指标
                    faces = self.tracker.predict(img0)

                    if len(faces) > 0:  # 关键点检测部分
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
                    # 这个地方需要有两种策略:如果这里传入frame,则不需要进行在画幅右4/9的判断.如果传入的img0则需要判断.
                    side_results = self.side_model(img0)
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
                        if box[0] > img_width * 4 / 9:
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
                # 如果results的长度等于零，则将is_yawning，is_eyes_closed，is_turning_head，is_moving都置为False
                if len(results) == 0:
                    is_yawning = False
                    is_eyes_closed = False
                    is_turning_head = False
                    is_moving = False

                if is_eyes_closed:
                    print(f'ear:{ear}')
                    eyes_closed_frame += 1
                    frame_result = 1
                else:
                    eyes_closed_frame = 0

                if is_yawning:  # 2
                    print(f'mar:{mar}')
                    frame_result = 2
                    mouth_open_frame += 1  # 帧数
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
                    look_around_frame += 1  # 帧数
                    mouth_open_frame = 0
                    eyes_closed_frame = 0
                    if look_around_frame > max_wandering:
                        max_wandering = look_around_frame

                else:
                    look_around_frame = 0

                if phone_around_face:  # 3
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

                # else:
                #     if phone_around_face: #3
                #         mouth_open_frame = 0
                #         look_around_frame = 0
                #         eyes_closed_frame = 0
                #     if is_turning_head:
                #         mouth_open_frame = 0
                #         eyes_closed_frame = 0
                #     if not is_eyes_closed:
                #         eyes_closed_frame = 0
                #     if not is_yawning:
                #         mouth_open_frame = 0

                ###################################################
                # im0 = display_results(img0, det, names,is_eyes_closed, is_turning_head, is_yawning)
                # # write video
                # vid_writer.write(im0)
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

                real_fps_3s = int(fps_3s * alpha)
                if use_phone_frame >= real_fps_3s:  # 帧数
                    result['result']['category'] = 3
                    break

                elif look_around_frame >= real_fps_3s:  # 帧数
                    result['result']['category'] = 4
                    break

                elif mouth_open_frame >= int(fps_3s * mar_alpha):  # 帧数
                    result['result']['category'] = 2
                    break

                elif eyes_closed_frame >= real_fps_3s:  # 帧数
                    result['result']['category'] = 1
                    break

            except Exception as e:
                traceback.print_exc()
                self.failures += 1
                if self.failures > 30:  # 失败超过30次就默认返回
                    result['result']['category'] = random.randint(0, 4)

            del frame
        if result['result']['category'] == 0:
            try:
                result['result']['category'] = 1 if check_sequence(ear_list, result_list) else 0
            except:
                result['result']['category'] = 0
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))
        print("-----------inference done!-----------")


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
        print("-----------check the results!-----------")

        result['result']['duration'] = duration
        print(result)
        return result

    def _postprocess(self, data):
        os.remove(self.capture)
        print(data)
        return data


if __name__ == '__main__':
    processor = fatigue_driving_detection("yolo2spiga", "best.pt")
    processor.inference('')

    # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    print("initialize succeeded")
