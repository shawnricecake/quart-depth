import os
import shutil

# organize nyuv2 dataset
# go to NYU_Depth_V2/official_split/test/
# python3 organize_data.py

# 定义目标文件夹
rgb_target_dir = 'rgb'
depth_target_dir = 'depth'

# 确保目标文件夹存在，如果不存在则创建
os.makedirs(rgb_target_dir, exist_ok=True)
os.makedirs(depth_target_dir, exist_ok=True)

# 需要遍历的类文件夹
class_folders = [
    'bathroom', 'bookstore', 'computer_lab', 'dining_room', 'home_office',
    'living_room', 'office_kitchen', 'reception_room', 'study', 'bedroom',
    'classroom', 'foyer', 'kitchen', 'office', 'playroom', 'study_room'
]

# 遍历每个类文件夹
for folder in class_folders:
    # 检查文件夹是否存在
    if os.path.exists(folder):
        # 遍历文件夹中的文件
        for file_name in os.listdir(folder):
            # 处理 rgb_*.jpg 文件
            if file_name.startswith('rgb_') and file_name.endswith('.jpg'):
                source_path = os.path.join(folder, file_name)
                target_path = os.path.join(rgb_target_dir, file_name)
                shutil.copy(source_path, target_path)  # 使用 copy 代替 move
                print(f"Copied {source_path} to {target_path}")
            # 处理 sync_depth_*.png 文件
            elif file_name.startswith('sync_depth_') and file_name.endswith('.png'):
                source_path = os.path.join(folder, file_name)
                target_path = os.path.join(depth_target_dir, file_name)
                shutil.copy(source_path, target_path)  # 使用 copy 代替 move
                print(f"Copied {source_path} to {target_path}")

print("Files have been copied successfully.")
