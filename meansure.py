import json

path = r'D:\0---Program\Projects\aimbot\yolov5-master\yolov5-master\output\close1.json'
# 读取日志文件
import json

def check_video_logs(path):
    # 读取log文件
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
            incorrect_video_names.append(video_name)

        elif (category_num == 11 and category != 0) or (category_num == 10 and category != 1) :
            incorrect += 1
            incorrect_video_names.append(video_name)

        else:
            correct += 1

    print(f"正确的个数: {correct}")
    print(f"错误的个数: {incorrect}")
    if incorrect > 0:
        print("错误的视频名称:")
        for name in incorrect_video_names:
            print(name)

# 运行函数
check_video_logs(path)

# call the function with your json file
