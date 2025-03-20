if __name__ == '__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import json


    # 读取文件并生成两个list
    def extract_paths_from_txt(file_path, base_path):
        rgb_list = []
        depth_list = []

        with open(file_path, 'r') as file:
            for line in file:
                # 拆分每行的数据，提取左边的RGB路径和右边的Depth路径
                parts = line.strip().split()
                if len(parts) >= 2:
                    rgb_path = parts[0]  # 左边是RGB路径
                    depth_path = parts[1]  # 右边是Depth路径
                    rgb_list.append(base_path + '/raw_data/' + rgb_path)
                    depth_list.append(base_path + '/data_depth_annotated/' + depth_path)

        return rgb_list, depth_list


    # 示例使用
    code_root = "your_code_path"
    base_path = 'your_dataset_path/kitti/'
    file_path = 'kitti_demo/kitti_eigen_test_files_with_gt.txt'
    rgb_paths, depth_paths = extract_paths_from_txt(file_path, base_path)

    files = []
    # rgb_root = osp.join(data_root, 'rgb')
    # depth_root = osp.join(data_root, 'depth')
    for i in range(len(rgb_paths)):
        if 'None' in depth_paths[i]:
            continue
        rgb_path = rgb_paths[i]
        depth_path = depth_paths[i]
        cam_in = [707.0493, 707.0493, 604.0814, 180.5066]
        depth_scale = 256.

        meta_data = {}
        meta_data['cam_in'] = cam_in
        meta_data['rgb'] = rgb_path
        meta_data['depth'] = depth_path
        meta_data['depth_scale'] = depth_scale
        files.append(meta_data)
    files_dict = dict(files=files)

    with open(osp.join(code_root, 'data/kitti_demo/test_annotations.json'), 'w') as f:
        json.dump(files_dict, f)