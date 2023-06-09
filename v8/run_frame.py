from v8interference import iou
import cv2
import numpy as np
from ultralytics import YOLO
from submit.toolkits.utils import face_analysis

def run_frame(frame_path,mode = 0):
    #mode = 0 只画司机
    #mode = 1 都画
    frame = cv2.imread(frame_path)
    yolo_model = YOLO('best.pt')
    device = 'cpu'
    half = device != 'cpu'

    cnt = 0  # 开始处理的帧数


    # 获取当前视频的帧率与宽高，设置同样的格式，以确保相同帧率与宽高的视频输出

    # save_path += os.path.basename(video_path)
    # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    ######################################################################################################
    result = {"result": {"category": 0, "duration": 6000}}
    eyes_closed_frame = 0
    mouth_open_frame = 0
    use_phone_frame = 0
    max_phone = 0
    max_eyes = 0
    max_mouth = 0
    max_wandering = 0
    look_around_frame = 0

    sensitivity = 0.001




    phone_around_face = False
    overlap = 0


    # process the image with the yolo and get the person list
    results = yolo_model(frame)

    # img1,bbox = process_results(results, frame, sensitivity)

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
    if mode == 1:
        person_list = []
        for box,cls in zip(boxes.xyxy,classes):
            if cls == 0:
                person_list.append(box)
            else:
                continue

        # process the image with the spig_model
        for box in person_list:
            m1, n1, m2, n2 = box
            w = m2 - m1
            h = n2 - n1

            # 创建 bbox
            bbox = [m1, n1, w, h]
            face_analysis(frame, bbox, test=2)

    # 遍历所有的边界框
    if mode == 0:
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
            img1 = img1[y1:y2, x1:x2]
            m1, n1, m2, n2 = x1, y1, x2, y2

            # 计算交集的面积
            if rightmost_phone is not None and rightmost_box is not None:
                # 计算两个框的IoU
                overlap = iou(rightmost_box, rightmost_phone)

                cv2.putText(img1, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 如果IoU大于阈值，打印警告
                if overlap > sensitivity:
                    print("Warning: phone and driver overlap!")

        # 计算边界框的宽度和高度
        w = m2 - m1
        h = n2 - n1

        # 创建 bbox
        bbox = [m1, n1, w, h]

        ANGLE_THRESHOLD = 30
        EAR_THRESHOLD = 0.2
        YAWN_THRESHOLD = 0.4

        pose, mar, ear = 0, 0, 0

        pose, mar, ear = face_analysis(frame, bbox, test=2)
        np.abs(pose[[0, 2]]).max()
        mar > YAWN_THRESHOLD
        ear < EAR_THRESHOLD

        # frame = spig_process_frame(frame, bbox)
        # cv2.putText(frame, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Pose max: {np.abs(pose[[0, 2]]).max():.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {np.abs(pose[0]):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if mar > YAWN_THRESHOLD:
            cv2.putText(frame, f"MAR: {mar}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"MAR: {mar}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if ear < EAR_THRESHOLD:
            cv2.putText(frame, f"EAR: {ear}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"EAR: {ear}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # if overlap:
            cv2.putText(frame, f"IoU: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # continue_loop = output_module(frame)
    res_plotted = results[0].plot()
    return res_plotted


from tqdm import tqdm


def photo_select(frame_dir, output_dir):
    '''
    从frame_dir读取jpg图片，并且用yolov8的图像识别进行处理，判断结果以分别处理。
    我们要判断图片中人（cls==0）的数量并分别讲他们复制到不同的输出路径。
    在output中创建三个文件夹：one,two,three
    如果有一个人就将output中的同名图片图片复制进one，以此类推。
    '''
    # 创建人数对应的文件夹
    for i in range(0, 4):
        path = os.path.join(output_dir, str(i))
        if not os.path.exists(path):
            os.makedirs(path)

    yolo_model = YOLO('best.pt')

    # 获取所有jpg文件
    all_files = [file for file in os.listdir(frame_dir) if file.endswith('.jpg')]

    # 使用tqdm创建一个进度条，并在循环中更新进度
    for i in tqdm(range(len(all_files)), desc='处理图片中'):
        filename = all_files[i]

        # 拼接完整的文件路径
        full_frame_path = os.path.join(frame_dir, filename)
        full_frame1_path = os.path.join(output_dir, filename)

        try:
            frame = cv2.imread(full_frame_path)
            frame1 = cv2.imread(full_frame1_path)
            results = yolo_model(frame)

            # 获取所有的边界框
            boxes = results[0].boxes

            # 获取所有类别
            classes = boxes.cls

            # 计算图片中人的数量
            person_count = list(classes).count(0)

            # 根据人的数量将图片复制到对应的文件夹
            if 1 <= person_count <= 3:
                target_path = os.path.join(output_dir, str(person_count), filename)
                cv2.imwrite(target_path, frame1)
            else:
                target_path = os.path.join(output_dir, str(0), filename)
                cv2.imwrite(target_path, frame1)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

def photo_select_many(frame_dir,many_check_dir, output_dir):
    for i in range(0, 4):
        path = os.path.join(output_dir, str(i))
        if not os.path.exists(path):
            os.makedirs(path)

    yolo_model = YOLO('best.pt')

    # 获取所有jpg文件
    all_files = [file for file in os.listdir(frame_dir) if file.endswith('.jpg')]

    # 使用tqdm创建一个进度条，并在循环中更新进度
    for i in tqdm(range(len(all_files)), desc='处理图片中'):
        filename = all_files[i]

        # 拼接完整的文件路径
        full_frame_path = os.path.join(frame_dir, filename)
        check_path = os.path.join(many_check_dir, filename)
        full_frame1_path = os.path.join(output_dir, filename)

        try:
            frame = cv2.imread(full_frame_path)
            frame1 = cv2.imread(full_frame1_path)
            results = yolo_model(frame)
            result = run_frame(full_frame_path)

            # 获取所有的边界框
            boxes = results[0].boxes

            # 获取所有类别
            classes = boxes.cls

            # 计算图片中人的数量
            person_count = list(classes).count(0)

            # 根据人的数量将图片复制到对应的文件夹
            cv2.imwrite(full_frame1_path, result)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue





def main(frame_dir, output_dir,check_dir = None):
    # 首先，我们检查并创建输出目录，如果它不存在的话
    #如果有checkdir 则从frame_dir中抽取checkdir指定的图片。
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有jpg文件
    all_files = [file for file in os.listdir(frame_dir) if file.endswith('.jpg')]
    if check_dir:
        all_files = [file for file in os.listdir(check_dir) if file.endswith('.jpg')]

    # 获取已经处理过的文件列表
    processed_files = [file for file in os.listdir(output_dir) if file.endswith('.jpg')]

    # 使用tqdm创建一个进度条，并在循环中更新进度
    for i in tqdm(range(len(all_files)), desc='处理图片中'):
        filename = all_files[i]

        # 如果该文件已经被处理过（在输出目录中存在），则跳过
        if filename in processed_files:
            continue

        # 拼接完整的文件路径
        full_frame_path = os.path.join(frame_dir, filename)
        if check_dir:
            full_check_path = os.path.join(check_dir, filename)

        try:
            # 使用run_frame函数处理图片
            result = run_frame(full_frame_path,mode=1)

            # 创建新的文件路径，用于保存结果
            full_output_path = os.path.join(output_dir, filename)

            # 保存结果
            cv2.imwrite(full_output_path, result)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

def save_img(frame, cap, cnt):

    '''
    调用run_frame函数，处理frame
    将传入的frame保存到save_path路径下的与cap同名的文件夹中
    '''
    save_path = r'F:\ccp1\la\test'
    frame = run_frame(frame)
    if not os.path.exists(os.path.join(save_path, cap)):
        os.makedirs(os.path.join(save_path, cap))
    cv2.imwrite(os.path.join(save_path, cap, str(cnt) + '.jpg'), frame)




if __name__ == '__main__':
    import os
    video_path = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\vedio\day_man_001_30_2.mp4'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    frame_dir = r'F:\ccp1\dataset\train\images'
    check_dir = r'F:\ccp1\dataset\output\2'
    output_dir = r'F:\ccp1\dataset\output\2.1'
    main(frame_dir,output_dir,check_dir=check_dir)
    # photo_select(frame_dir,output_dir)
