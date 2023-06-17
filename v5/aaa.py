import numpy as np

# 读取文件
with open(r"/file_list.txt") as f:
    files = f.read().splitlines()

# 初始化文件类型字典
files_dict = {'mp4': [], 'json': []}

# 将文件名分配到相应的列表中
for file in files:
    filetype = file.split('.')[-1]
    if filetype in files_dict:
        files_dict[filetype].append(file)

# 去掉文件类型后的文件名列表
basename_json = [name.split('.json')[0] for name in files_dict['json']]
basename_mp4 = [name.split('.mp4')[0] for name in files_dict['mp4']]

# 找出存在mp4但没有json的文件
mp4_without_json = np.setdiff1d(basename_mp4, basename_json)

# 输出结果
print(f"Number of mp4 files: {len(files_dict['mp4'])}")
print(f"Number of json files: {len(files_dict['json'])}")
print("mp4 files without corresponding json:")
for file in mp4_without_json:
    print(f"{file}.mp4")
