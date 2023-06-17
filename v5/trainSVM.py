
import os

from tqdm import tqdm

from ultralytics import YOLO
# from submit.toolkits.utils import face_analysis
import cv2
from v8.spiga_model_processing import spig_process_frame
import datetime
import glob
import numpy as np

from sklearn.svm import SVC


def normalize_landmarks(face_landmarks):
    # 验证输入数据
    assert face_landmarks.shape == (98, 2), "输入的人脸关键点数组应该为(98, 3)的形状"

    # 将关键点转换为相对于鼻尖的坐标
    nose_tip = face_landmarks[53]  # 鼻尖是第54个关键点，索引为53
    face_landmarks = face_landmarks - nose_tip

    # # 计算眼睛的关键点的方向
    # left_eye = face_landmarks[95]  # 左眼是第96个关键点，索引为95
    # right_eye = face_landmarks[96]  # 右眼是第97个关键点，索引为96
    # eye_direction = right_eye - left_eye
    # eye_direction = eye_direction / np.linalg.norm(eye_direction)  # 归一化
    #
    # # 计算旋转矩阵
    # x = np.array([1, 0, 0])  # x轴
    # rotation_axis = np.cross(x, eye_direction)  # 旋转轴
    # rotation_angle = np.arccos(np.dot(x, eye_direction))  # 旋转角度
    # rotation_matrix = np.array([
    #     [np.cos(rotation_angle) + rotation_axis[0]**2*(1 - np.cos(rotation_angle)), rotation_axis[0]*rotation_axis[1]*(1 - np.cos(rotation_angle)) - rotation_axis[2]*np.sin(rotation_angle), rotation_axis[0]*rotation_axis[2]*(1 - np.cos(rotation_angle)) + rotation_axis[1]*np.sin(rotation_angle)],
    #     [rotation_axis[1]*rotation_axis[0]*(1 - np.cos(rotation_angle)) + rotation_axis[2]*np.sin(rotation_angle), np.cos(rotation_angle) + rotation_axis[1]**2*(1 - np.cos(rotation_angle)), rotation_axis[1]*rotation_axis[2]*(1 - np.cos(rotation_angle)) - rotation_axis[0]*np.sin(rotation_angle)],
    #     [rotation_axis[2]*rotation_axis[0]*(1 - np.cos(rotation_angle)) - rotation_axis[1]*np.sin(rotation_angle), rotation_axis[2]*rotation_axis[1]*(1 - np.cos(rotation_angle)) + rotation_axis[0]*np.sin(rotation_angle), np.cos(rotation_angle) + rotation_axis[2]**2*(1 - np.cos(rotation_angle))]
    # ])  # 使用罗德里格斯公式计算旋转矩阵
    #
    # # 旋转关键点
    # face_landmarks = np.dot(face_landmarks, rotation_matrix.T)

    return face_landmarks




def run_frame(input):
    frame = input.copy()
    yolo_model = YOLO(r'/v8/best.pt')
    device = 'cpu'
    half = device != 'cpu'

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


    # 计算边界框的宽度和高度
    w = m2 - m1
    h = n2 - n1

    # 创建 bbox
    bbox = [m1, n1, w, h]

    ANGLE_THRESHOLD = 30
    EAR_THRESHOLD = 0.2
    YAWN_THRESHOLD = 0.4


    landmarks = spig_process_frame(frame, bbox,landmark = 1)
    print(landmarks.shape)
    return landmarks


def get_diff_jpg(path1, path2):
# 获取第一个目录中的所有jpg文件
    jpgs_path1 = set(glob.glob(os.path.join(path1, '*.jpg')))

    # 获取第二个目录中的所有jpg文件
    jpgs_path2 = set(glob.glob(os.path.join(path2, '*.jpg')))

    # 获取第一个目录中有而第二个目录中没有的jpg文件
    diff = jpgs_path1 - jpgs_path2

    return diff

def save_lm(check_path, dataset_path, output_path,path1 = None,output2 = None):
    # 首先读取check_path下的所有jpg文件的文件名
    check_files = glob.glob(os.path.join(check_path, "*.jpg"))
    check_files = [os.path.basename(f) for f in check_files]
    # if path1:
    #     files0 = get_diff_jpg(path1, check_path)
    #     for f in tqdm(files0):
    #         # 检查output_path下是否已经存在对应的landmark数据，如果存在则跳过
    #         output_file = os.path.join(output2, os.path.splitext(f)[0] + ".npy")
    #         if os.path.exists(output_file):
    #             continue
    #
    #         # 从dataset_path读取相应的图片
    #         img_file = os.path.join(dataset_path, f)
    #         img = cv2.imread(img_file)
    #
    #         # 如果图片不存在或者读取失败，则跳过
    #         if img is None:
    #             continue
    #
    #         # 使用run_frame函数处理图片，获取landmark数据
    #         landmarks = run_frame(img)
    #
    #         # 保存landmark数据到output_path
    #         np.save(output_file, landmarks)
    #         print(output_file)



    # 使用tqdm显示进度条
    for f in tqdm(check_files):
        # 检查output_path下是否已经存在对应的landmark数据，如果存在则跳过
        output_file = os.path.join(output_path, os.path.splitext(f)[0] + ".npy")
        if os.path.exists(output_file):
            continue

        # 从dataset_path读取相应的图片
        img_file = os.path.join(dataset_path, f)
        img = cv2.imread(img_file)

        # 如果图片不存在或者读取失败，则跳过
        if img is None:
            continue

        # 使用run_frame函数处理图片，获取landmark数据
        landmarks = run_frame(img)

        # 保存landmark数据到output_path
        np.save(output_file, landmarks)




def load_data(lm_0_path, lm_1_path):
    # 获取所有的landmark文件
    lm_0_files = glob.glob(os.path.join(lm_0_path, "*.npy"))
    lm_1_files = glob.glob(os.path.join(lm_1_path, "*.npy"))

    # 初始化一个空的特征矩阵和目标向量
    X = []
    y = []

    # 读取并处理lm_0_path中的landmark数据
    for f in lm_0_files:
        # 读取landmark数据
        landmarks = np.load(f)
        print(landmarks.shape)

        # 对landmark数据进行标准化处理
        landmarks = normalize_landmarks(landmarks)

        # 将处理后的landmark数据添加到特征矩阵
        X.append(landmarks.flatten())  # 这里假设你的模型可以接受flatten后的landmark数据作为特征

        # 为这个landmark数据添加对应的标签
        y.append(0)

    # 读取并处理lm_1_path中的landmark数据
    for f in lm_1_files:
        # 读取landmark数据
        landmarks = np.load(f)

        # 对landmark数据进行标准化处理
        landmarks = normalize_landmarks(landmarks)

        # 将处理后的landmark数据添加到特征矩阵
        X.append(landmarks.flatten())  # 这里假设你的模型可以接受flatten后的landmark数据作为特征

        # 为这个landmark数据添加对应的标签
        y.append(1)

    # 将特征矩阵和目标向量转化为numpy数组
    X = np.array(X)
    y = np.array(y)

    return X, y

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # 需要填充的变量
    check_0_path = r"F:\ccp1\dataset\output\1"  # 检查文件路径
    check_1_path = r"F:\ccp1\dataset\output\1\side"
    dataset_path = r"F:\ccp1\dataset\train\images"  # 数据集路径

    lm_0_path = r"F:\ccp1\dataset\SVM\front"  # 标签为0的landmark数据路径
    lm_1_path = r"F:\ccp1\dataset\SVM\side"  # 标签为1的landmark数据路径
    model_output_path = "F:\ccp1\dataset\SVM"  # 训练后的模型保存路径

    # 从check_path中读取图片名称，并从dataset_path中取出图片，处理并保存landmark数据到output_path
    # print("Saving landmarks...")
    # save_lm(check_0_path, dataset_path, lm_0_path)
    save_lm(check_1_path, dataset_path, lm_1_path)
    # # 从lm_0_path和lm_1_path中读取并处理landmark数据，获取特征矩阵和目标向量
    print("Loading data...")
    X, y = load_data(lm_0_path, lm_1_path)

    # 分割数据集
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=39)

    # 训练一个支持向量机模型
    print("Training SVM...")
    svm = SVC(kernel='poly', C=1.0, degree=3)

    svm.fit(X_train, y_train)

    # 预测测试集并评估模型
    print("Evaluating model...")
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # 保存模型
    print("Saving model...")
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"svm_{date_str}_ac_{accuracy*100:.2f}%.pkl"
    model_filepath = os.path.join(model_output_path, model_filename)
    with open(model_filepath, 'wb') as f:
        pickle.dump(svm, f)

    print("Training completed.")

import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model



if __name__ == '__main__':
    # main()
    #
    path = cv2.imread(r'F:\ccp1\dataset\train\images\dts_20348.jpg')
    model_path = "F:\ccp1\dataset\SVM\svm_20230607_123807_ac_93.75%.pkl"  # 模型文件路径
    landmarks = run_frame(path)  # 这里是你的特征向量
    X = normalize_landmarks(landmarks)

    # 加载模型
    model = load_model(model_path)
    X = normalize_landmarks(landmarks).flatten().reshape(1, -1)
    # 使用模型进行预测
    y_pred = model.predict(X)
    print(y_pred)

# import os
#
# import shutil
#
#
# def copy_diff_npy(all_path, side2_path, front2_path):
#     # 获取all_path下的所有npy文件
#     all_files = set(f for f in os.listdir(all_path) if f.endswith('.npy'))
#
#     # 获取side2_path下的所有npy文件
#     side2_files = set(f for f in os.listdir(side2_path) if f.endswith('.npy'))
#
#     # 找出在all_path中存在但在side2_path中不存在的文件
#     diff_files = all_files - side2_files
#
#     # 将这些文件复制到front2_path
#     for file in diff_files:
#         shutil.copy(os.path.join(all_path, file), front2_path)
#
# all_path = r'F:\ccp1\dataset\SVM\all'
# side2 = r'F:\ccp1\dataset\SVM\side2'
# front2 = r'F:\ccp1\dataset\SVM\front2'
# # 使用函数
# copy_diff_npy(all_path , side2, front2 )