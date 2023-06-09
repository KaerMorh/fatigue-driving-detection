import math
vtime = 9
runtime = 10.9
sigmoid = 1/(1+math.exp(-(vtime/runtime)))
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

print(sigmoid(vtime/runtime))
video_path = r'F:\ChallengeCup'
# print(calculate_average_video_length(video_path))