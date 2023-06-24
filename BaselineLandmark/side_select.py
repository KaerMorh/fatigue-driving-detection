import os
import shutil



def from_check_select_pic(check_dir, pic_dir, output_dir):
    '''
    从check_dir读取视频的名称，读取其姓名编号，存入列表num_list。
    随后遍历pic_dir中所有的图片，如果图片的姓名编号在check_files中，则将该图片复制到output_dir中。
    '''
    # 遍历check_dir中所有的.mp4文件
    target_files = []
    num_list = []
    check_files = [f for f in os.listdir(check_dir) if f.endswith('.mp4')]
    for video_name in check_files:
        num = int(video_name.split('_')[2])
        num_list.append(num)
    pic_files = [f for f in os.listdir(pic_dir) if f.endswith('.jpg')]
    for pic_name in pic_files:
        num = int(pic_name.split('_')[2])
        if num in num_list:
            target_files.append(pic_name)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将target_files中的文件复制到output_dir中
    for file in target_files:
        shutil.copy(os.path.join(pic_dir, file), os.path.join(output_dir, file))

def photo_select_4(video_dir,output_dir):
    #遍历vido_dir中所有的.jpg文件
    target_files = []
    failed_files = []
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
    for video_name in video_files:
        try:
            category_num = int(video_name.split('_')[3])
            if category_num == 40 or category_num == 41:
            # if category_num ==0:
                target_files.append(video_name)
        except Exception as e:
            print(e)
            failed_files.append(video_name)
            continue

    #确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #将target_files中的文件复制到output_dir中
    for file in target_files:
        shutil.copy(os.path.join(video_dir,file),os.path.join(output_dir,file))

    print(f'failed_files:{failed_files}')


def video_select_4(video_dir,output_dir):
    #遍历vido_dir中所有的.mp4文件
    target_files = []
    failed_files = []
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    for video_name in video_files:
        try:
            category_num = int(video_name.split('_')[3])
            if category_num == 30 or category_num == 31:
            # if category_num == 0:
                target_files.append(video_name)

        except Exception as e:
            print(e)
            failed_files.append(video_name)
            continue

    #确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #将target_files中的文件复制到output_dir中
    for file in target_files:
        shutil.copy(os.path.join(video_dir,file),os.path.join(output_dir,file))

    print(f'failed_files:{failed_files}')

video_dir = r'F:\ccp2\dataset'
output_dir = r'F:\ccp2\photo\4'
photo_dir = r'F:\ccp2\photo'

# video_select_4(video_dir,output_dir)
# from_check_select_pic(check_dir=output_dir, pic_dir=photo_dir,output_dir=r'F:\ccp2\dataset_4_pic' )
photo_select_4(photo_dir,output_dir)