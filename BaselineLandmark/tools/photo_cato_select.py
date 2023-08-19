
import os
import shutil


def photo_select(video_dir,catogory,output_dir):
    #遍历vido_dir中所有的.jpg文件
    target_files = []
    failed_files = []
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
    for video_name in video_files:
        try:
            category_num = int(video_name.split('_')[3])
            if category_num == catogory*10 or category_num == (catogory*10+1):
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


###main
if __name__ == '__main__':
    input_path = r''
    output_path = r''
    #if there is no fold named 0,1,2,3,4 in output_path, create them
    for i in range(5):
        if not os.path.exists(os.path.join(output_path,str(i))):
            os.makedirs(os.path.join(output_path,str(i)))
        photo_select(input_path, i, output_path)

