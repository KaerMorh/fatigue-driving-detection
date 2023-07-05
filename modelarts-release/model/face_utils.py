from scipy.spatial import distance as dist
import numpy as np

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

        # 剔除888
        other_ear = ear_array[np.logical_and(np.arange(len(ear_array)) != i, np.arange(len(ear_array)) != i+5)]

        if len(other_ear) == 0:
            continue

        mean1 = np.mean(ear_array[i:i+6])
        mean2 = np.mean(other_ear)
        # 计算平均值并比较
        if mean1 < mean2 * 0.9 and np.all(np.logical_or(current_result == 0, current_result == 1)):
            # 在窗口内的变动是否超过阈值
            return True

    return False


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[49], mouth[56])  # 49, 56
    B = dist.euclidean(mouth[51], mouth[54])  # 51, 54

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[58], mouth[62])  # 58, 62

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


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
