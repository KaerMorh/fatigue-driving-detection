import math
vtime = 9
runtimes = [5.3, 2 ,11.8, 3.3, 6.8 ,5.0]

#求出runtimes的平均值再进行sigmoud计算
runtime = sum(runtimes)/len(runtimes)

sigmoid1 = 1/(1+math.exp(-(vtime/runtime)))
#先对每一项求runtimes/vtime 求sigmoid再求平均值
sigmoid2 = sum([1/(1+math.exp(-(vtime/runtime))) for runtime in runtimes])/len(runtimes)
# Import the math module
from moviepy.editor import VideoFileClip
import os
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def calculate_average_video_length(video_path):
    total_length = 0
    video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]

    # 使用tqdm显示进度
    for filename in tqdm(video_files, desc='Calculating average video length'):
        video = VideoFileClip(os.path.join(video_path, filename))
        total_length += video.duration  # 以秒为单位的视频长度

    # 计算并返回平均长度
    average_length = total_length / len(video_files) if video_files else None
    return average_length


# Define the sigmoid function
def sigmoid(x):
    # Calculate the sigmoid of x
    return 1 / (1 + math.exp(-x))

import numpy as np

print(sigmoid1,sigmoid2)
video_path = r'F:\ChallengeCup'
# print(calculate_average_video_length(video_path))
def check_sequence(ear_list, result_list, diff_threshold=0.04):
    ear_array = np.array(ear_list)
    result_array = np.array(result_list)

    if len(ear_array) < 6 or len(ear_array) > len(result_array):
        return False

    for window_size in range(6, len(ear_array)):  # 控制滑动窗口大小
        ear_diff = np.abs(np.diff(ear_array))

        for i in range(len(ear_array) - window_size + 1):  # 控制滑动窗口位置
            current_diff = ear_diff[i:i+window_size-1]
            current_result = result_array[i:i+window_size]
            current_ear = ear_array[i:i+window_size]

            # 如果滑动窗口包含数组的最后一个元素
            if i + window_size == len(ear_array):
                # 仅对最后一个元素进行特殊处理
                if current_ear[-1] > 0.18 or current_ear[-1] == 888:
                    continue

            if np.any(current_diff > diff_threshold):
                if not np.all(current_ear < 0.18):
                    continue

            if np.any(current_ear[:-1] == 888):
                continue

            other_ear = ear_array[np.logical_and(
                np.logical_and(np.arange(len(ear_array)) != i, np.arange(len(ear_array)) != i + window_size - 1),
                ear_array != 888)]

            if len(other_ear) == 0:
                continue

            mean1 = np.mean(current_ear)
            mean2 = np.mean(other_ear)

            if mean1 < mean2 * 0.9 and np.all(np.logical_or(current_result == 0, current_result == 1)):
                return True

    return False




result_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ear_list2 = [0.15874990320727161, 0.2743422818790768, 0.2500754995674946, 0.2596978711498933, 0.19864541979831626, 0.17959440462261683, 0.16215223737015516, 0.15184655624251897, 0.1462487899335471, 0.14210700919774205, 0.2562845870727066, 0.2430776172556589]
ear_list = [0.2707649474427509, 0.2638965997042301, 0.2763695188802266, 0.27429404937449453, 0.27467687193368495, 0.19703379291616252, 0.17583044119514513, 0.18637746000634497, 0.1707038468185434, 0.16772192441183637, 0.17312798384943812]
print(check_sequence(ear_list, result_list))
print(sigmoid(2))
