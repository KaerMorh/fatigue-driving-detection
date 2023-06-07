import json

path = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\output'
# 读取日志文件
import json
import glob
import os
def check_video_logs(path,mode = 0):
    # 读取log文件

    if mode == 1:
        all_files = [file for file in glob.glob(os.path.join(path, '*.json'))]
        all_files.sort(key=os.path.getmtime)
        path = all_files[-1]  # use the latest file


    with open(path) as json_file:
        data = json.load(json_file)

    correct = 0
    incorrect = 0
    incorrect_video_names = []

    # 遍历所有的日志条目
    for video_name, video_data in data.items():
        category = video_data['result']['category']

        # 在视频名字中找到对应的数字
        category_num = int(video_name.split('_')[3])

        # 检查category的值是否与视频名字中的数字对应
        if (category_num == 31 and category != 0) or (category_num == 30 and category != 3) or(category_num == 20 and category != 2) or (category_num == 31 and category != 0):
            incorrect += 1
            incorrect_video_names.append(video_name.join(('',str(category))))

        elif (category_num == 11 and category != 0) or (category_num == 10 and category != 1) :
            incorrect += 1
            incorrect_video_names.append(video_name.join(('',str(category))))

        elif (category_num == 41 and category != 0) or (category_num == 40 and category != 4) :
            incorrect += 1
            incorrect_video_names.append(video_name.join(('',str(category))))

        else:
            correct += 1
    print(f"file:{path}")
    print(f"正确的个数: {correct}")
    print(f"错误的个数: {incorrect}")
    if incorrect > 0:
        print("错误的视频名称:")
        for name in incorrect_video_names:
            print(name)


def true_cls(line):
    category_num = int(line.split('_')[3])
    if (category_num == 31 or category_num == 21 or category_num == 11 or category_num == 41):
        return 0
    if category_num == 30:
        return 3
    if category_num == 20:
        return 2
    if category_num == 10:
        return 1
    if category_num == 40:
        return 4


def calculate_score(filename):
    # 类别的总数
    total_counts = [9+14+8+9+36, 11, 8, 18, 6+4+1+4+5+7]

    # 初始化计数器
    error_counts = [0, 0, 0, 0, 0]
    false_positive = [0, 0, 0, 0, 0]
    false_negative = [0, 0, 0, 0, 0]

    with open(filename, 'r') as file:
        for line in file:
            # 解析文件名
            fields = line.strip().split('_')
            true_class = true_cls(line)
            predicted_class = int(fields[-1][-1])

            if true_class != predicted_class:
                false_positive[predicted_class] += 1
                false_negative[true_class] += 1
                error_counts[true_class] += 1

    # 计算正确分类的视频数量
    true_positive = [total - error for total, error in zip(total_counts, error_counts)]

    # 计算每个类别的 F1-Score
    f1_scores = []
    for i in range(5):
        if true_positive[i] + false_positive[i] != 0:
            precision = true_positive[i] / (true_positive[i] + false_positive[i])
        else:
            precision = 0

        if true_positive[i] + false_negative[i] != 0:
            recall = true_positive[i] / (true_positive[i] + false_negative[i])
        else:
            recall = 0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        f1_scores.append(f1)

    # 计算最后的得分
    total_videos = sum(total_counts)
    final_score = sum(f1 * count / total_videos for f1, count in zip(f1_scores, total_counts))

    return final_score


# 运行函数
check_video_logs(path,1)
print(calculate_score('file_list.txt'))
# call the function with your json file
