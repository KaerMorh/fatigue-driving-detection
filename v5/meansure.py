import json

path = r'/output'
# 读取日志文件
import json
import glob
import os


def comparision(incorrect_videos, compare_path):
    '''
    输入一个列表incorrect_videos，和一个txt文件compare_path，先对incorrect_video的数据处理，将每一项的末尾数字存入categories中，
    再生成一个新的列表videos_names，将incorrect_videos中的每一项的末尾数字去掉，存入videos_names中
    从compare_path(一个txt文件，包含如night_man_002_10_1.mp40的列表)读取数据，将每一项的末尾数字存入compare_categories中，
    再生成一个新的列表compare_videos，将compare_path中的每一项的末尾数字去掉，存入compare_videos中
    将两个名称列表中的名称相同项对应的标签进行比较，如果不同则打印出来(按照标签进行排序)
    如果有compare_videos中的名称不在videos_names中，则将其打印出来（少错误项）
    如果有videos_names中的名称不在compare_videos中，则将其打印出来（新增错误项）
    '''
    with open(compare_path  , 'r') as file:
        compare_videos = file.read().splitlines()
    compare_categories = []
    new_wrong = []
    new_wrong_categories = []

    for i in range(len(compare_videos)):
        compare_categories.append(compare_videos[i][-1])
        compare_videos[i] = compare_videos[i][:-1]
    videos_names = []
    categories = []
    for i in range(len(incorrect_videos)):
        categories.append(incorrect_videos[i][-1])
        videos_names.append(incorrect_videos[i][:-1])
    for i in range(len(videos_names)):
        name = videos_names[i]
        if videos_names[i] in compare_videos:
            #将videos_names[i]在compare_videos中的位置存入com_cls
            com_cls = compare_videos.index(videos_names[i])
            a1 = compare_videos[com_cls]
            a2 = compare_categories[com_cls]
            if categories[i] != compare_categories[com_cls]:
                # 打印出来名称，原标签，新标签，以一段字符形式
                print('变动错误项',videos_names[i], '新标签：', categories[i], '原标签：',compare_categories[compare_videos.index(videos_names[i])])


            else:
                #打印出相同的错误，给出名称与标签
                print('相同错误项：',videos_names[i],'标签：',categories[i])

            #删除compare_videos中的相同项
            compare_videos.pop(com_cls)
            compare_categories.pop(com_cls)

        else:
            #加入到新增错误项中
            new_wrong.append(videos_names[i])
            new_wrong_categories.append(categories[i])
    #打印出新增错误项,每行一个
    for i in range(len(new_wrong)):
        print('新增错误项：',new_wrong[i],'标签：',new_wrong_categories[i])


    #打印出少错误项
    for i in range(len(compare_videos)):
        print('少错误项：',compare_videos[i],'标签：',compare_categories[i])








def check_video_logs(path,mode = 0, compare_path = None ):
    if mode == 1:
        all_files = [file for file in glob.glob(os.path.join(path, '*.json'))]
        all_files.sort(key=os.path.getmtime)
        path = all_files[-1]  # use the latest file


    print(path)
    with open(path) as json_file:
        data = json.load(json_file)

    correct = 0
    incorrect = 0
    incorrect_video_names = []

    # 遍历所有的日志条目
    for video_name, video_data in data.items():
        # print(video_name)
        category = video_data['result']['category']


        # 在视频名字中找到对应的数字
        category_num = int(video_name.split('_')[3])
        #将错误的视频加进to_compare_videos中
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

    if compare_path is not None:
        try:
            comparision(incorrect_video_names, compare_path)
        except Exception as e:
            print(e)

    print(f"file:{path}")
    print(f"正确的个数: {correct}")
    print(f"错误的个数: {incorrect}")
    if incorrect > 0:
        print("错误的视频名称:")
        for name in incorrect_video_names:
            print(name)


def true_cls(line):
    try:
        category_num = int(line.split('_')[3])
    except Exception as e:
        print(e)
        print(line)
        cataegory_num = 0
    if (category_num == 31 or category_num == 21 or category_num == 11 or category_num == 41 or category_num == 0):
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

def calculate_average_duration(path, mode=0):
    # 读取json文件
    if mode == 1:
        all_files = [file for file in glob.glob(os.path.join(path, '*.json'))]
        all_files.sort(key=os.path.getmtime)
        path = all_files[-1]  # use the latest file

    with open(path) as json_file:
        data = json.load(json_file)

    total_duration = 0
    video_count = 0
    less_3s_count = 0
    more_8s_count = 0

    # 遍历所有的日志条目
    for video_name, video_data in data.items():
        duration = video_data['result']['duration']
        total_duration += duration
        if duration < 3000:
            less_3s_count += 1
        if duration > 8000:
            more_8s_count += 1
        video_count += 1

    print(f'less than 3s: {less_3s_count}')
    print(f'more than 8s: {more_8s_count}')
    # 计算并返回平均用时
    average_duration = total_duration / video_count if video_count > 0 else 0

    return average_duration



# 运行函数
# check_video_logs(path,1)
check_video_logs(path, 1, '../file_list.txt')
print(calculate_average_duration(path,1))
print(calculate_score('../file_list.txt'))
# call the function with your json file
