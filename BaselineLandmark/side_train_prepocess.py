import cv2
import os
#photo_dir
#photo_output_dir
#label_dir
#label_output_dir
length = 1920
width = 1080
hvw = length/width

'''
不要手机，其他三项都要，且只允许主驾驶位存在。
tradeoff：其他两项没有标注侧脸，贸然使用可能会导致侧脸不行。
1.把3开头的去除 select_photo
2.读取photo_dir的所有photo与label_dir的所有label，若photo的编号在label中，则将其复制到photo_output_dir中，同时将其同名的label复制到label_output_dir中。
将输入图片裁剪为指定比例的图片，不改变高度，只从左边开始进行裁剪。
同时对其同名的label文件进行裁剪与坐标变换，若有x1不在图片内的，将其去除。   输出到指定文件夹。
change_photo

3.遍历label_dir中已经处理过的标签，先将其中类别为1的标签删去，若删去后没有任何标签，则记录在0_list中并在结尾输出。
再对在check_photo_dir中存在同名图片的标签进行处理：将其中为0的项改为1.若改完后存在的标签数量不等于一，则记录在1_list中并在结尾输出。

'''
import os
from PIL import Image


def parse_yolo_label(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        label, x, y, w, h = map(float, line.strip().split())
        labels.append((label, x, y, w, h))

    return labels

#
# def change_label(label_path, l_output_path, width, height, L):
#     labels = parse_yolo_label(label_path)
#
#     # 寻找最左下角的人脸框
#     leftmost_face = min(labels, key=lambda x: (x[1] - x[3] / 2, -x[2] + x[4] / 2))
#
#     # 计算新的左边界
#     new_left = max(0, leftmost_face[1] - leftmost_face[3] / 2 - L / 2 / width)
#
#     new_labels = []
#     for label in labels:
#         # 调整标签的位置和宽度
#         x = (label[1] - new_left) * width / L
#         w = label[3] * width / L
#
#         if 0 < x + w / 2 and x - w / 2 < 1:  # 确保标签在新图像范围内
#             new_labels.append((label[0], x, label[2], w, label[4]))
#
#     # 写入新的标签文件
#     with open(l_output_path, 'w') as file:
#         for label in new_labels:
#             file.write(' '.join(map(str, label)) + '\n')
#
#     # 也需要返回新的左边界，以便调整图像
#     return new_left * width


import os
import numpy as np

def change_label(label_path, l_output_path, w, w1, h):
    try:
        with open(label_path, 'r') as f:
            labels = f.readlines()

        new_labels = []
        for label in labels:
            data = label.split()
            label_type, x, y, dw, dh = data[0], float(data[1]), float(data[2]), float(data[3]), float(data[4])

            # 转化为绝对坐标
            x_abs = x * w
            dw_abs = dw * w

            # 找到目标框的左右两端坐标
            x1 = x_abs - dw_abs / 2
            x2 = x_abs + dw_abs / 2

            # 裁剪框的右端坐标
            x1_new = x1 - w + w1

            # 检查目标框是否在裁剪框内部
            if x1_new <= 0:
                continue

            # 计算新的坐标和宽度
            x_abs_new = x1_new + dw_abs/2
            x2_new = x1_new + dw

            # 转化回相对坐标
            x_new = x_abs_new / w1
            dw_new = dw_abs / w1

            # 确保坐标不为负数
            if x_new < 0 or dw_new < 0:
                print(f'low:{label_path}')
                break

            new_labels.append(f"{label_type} {x_new} {y} {dw_new} {dh}\n")

        with open(l_output_path, 'w') as f:
            f.writelines(new_labels)

        return True

    except Exception as e:
        print(e)
        return False




def change_photo(photo_dir,p_output_dir, label_dir, l_output_dir,hvw):
    #读取photo_dir的所有photo与label_dir的所有label
    photo_files = [f for f in os.listdir(photo_dir) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    #去除以上两个列表中各个项的后四位
    photo_files_names = [f[:-4] for f in photo_files]
    label_files_names = [f[:-4] for f in label_files]
    fail_photos1 = []
    #若photo的编号在label中，则将其复制到photo_output_dir中，同时将其同名的label复制到label_output_dir中。
    for i in range(len(photo_files_names)):
        photo_name = photo_files_names[i]
        if photo_name in label_files_names:
            #修改图片尺寸并将其复制到photo_output_dir中
            photo = os.path.join(photo_dir,photo_name+'.jpg')
            photo_output = os.path.join(p_output_dir,photo_name+'.jpg')
            img = cv2.imread(photo)
            h = img.shape[0]
            w = img.shape[1]
            if h/w < hvw:
                w1 = int(h/hvw)
                #裁剪,将输入图片裁剪为指定比例的图片，不改变高度，右边不变，从左边开始裁剪。
                img = img[:, w-w1:w]

                #读取标签文件，对其同名的label文件进行裁剪与坐标变换，若有x1不在图片内的，将其去除。
                label = os.path.join(label_dir,photo_name+'.txt')
                label_output = os.path.join(l_output_dir,photo_name+'.txt')
                if not change_label(label,label_output,w,w1,h):
                    fail_photos1.append(photo_name)

            else:
                fail_photos1.append(photo_name)
                continue
            cv2.imwrite(photo_output,img)


def process_labels(label_output_dir, check_photo_dir):
    label_files = [f for f in os.listdir(label_output_dir) if f.endswith('.txt')]
    check_photo_files = [f[:-4] for f in os.listdir(check_photo_dir) if f.endswith('.jpg')]

    # 文件名存储在两个列表中
    zero_list = []
    one_list = []

    for label_file in label_files:
        label_file_path = os.path.join(label_output_dir, label_file)
        with open(label_file_path, 'r') as f:
            labels = f.readlines()

        # 删除标签类别为1的标签
        labels = [label for label in labels if label.split()[0] != '1']

        if not labels:  # 如果删除后没有任何标签
            zero_list.append(label_file[:-4])
        else:
            if label_file[:-4] in check_photo_files:  # 对存在同名图片的标签进行处理
                labels = [' '.join(['1'] + label.split()[1:]) if label.split()[0] == '0' else label for label in labels] # 将标签类别为0的标签修改为1
                if len(labels) != 1:  # 如果修改后存在的标签数量不等于一
                    one_list.append(label_file[:-4])

        # 将处理后的标签写回文件
        with open(label_file_path, 'w') as f:
            f.writelines(labels)

        for file in zero_list + one_list:
            file_path = os.path.join(label_output_dir, file + '.txt')
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f'File {file_path} not found.')



    # 输出文件名列表
    print('zero_list:', zero_list)
    print('one_list:', one_list)


import os
import shutil

def copy_unique_files(path1, path2, target_path):
    # 获取两个路径中所有.jpg图片的文件名
    path1_files = {f for f in os.listdir(path1) if f.endswith('.jpg')}
    path2_files = {f for f in os.listdir(path2) if f.endswith('.jpg')}

    # 获取只在 path1 中存在的文件
    unique_files = path1_files - path2_files

    # 将这些文件复制到 target_path
    for file in unique_files:
        shutil.copy(os.path.join(path1, file), os.path.join(target_path, file))





def main():
    photo_dir = r'F:\ccp2\side_photo'
    photo_output_dir = r'F:\ccp2\side_photo\croped'
    label_dir = r'F:\ccp2\labels\labels'
    label_output_dir = r'F:\ccp2\side_photo\label_output'
    check_photo_dir =r'F:\ccp2\side_photo\croped'
    hvw = 1080 / 1152


    # change_photo(photo_dir, photo_output_dir, label_dir, label_output_dir, hvw)
    process_labels(label_output_dir, check_photo_dir)

    # photo_dir = r'F:\ccp2\photo\front4'
    # photo_output_dir = r'F:\ccp2\side_photo\crop0'
    # label_dir = r'F:\ccp2\labels\labels'
    # label_output_dir = r'F:\ccp2\side_photo\label0'
    #
    # # change_photo(photo_dir, photo_output_dir, label_dir, label_output_dir, hvw)
    #
    # dir0 = r'F:\ccp2\photo\0'
    # change_photo(dir0, photo_output_dir, label_dir, label_output_dir, hvw)
    # process_labels(label_output_dir, photo_output_dir)
    #
    # # dir2 = r'F:\ccp2\photo\1'
    # # photo_output_dir = r'F:\ccp2\side_photo\crop12'
    # # label2 = r'F:\ccp2\side_photo\label12'
    # # change_photo(dir2 , photo_output_dir, label_dir, label2, hvw)
    # # process_labels(label2, photo_output_dir)
    #
    #
    # # path1 = r'F:\ccp2\photo\4'
    # # path2 = r'F:\ccp2\side_photo'
    # # path3 = r'F:\ccp2\photo\front4'
    # # copy_unique_files(path1,path2,path3)

if __name__ == '__main__':
    main()