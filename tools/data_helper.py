import os


def get_all_file_from_dir(path_dir, image_path_list, label_list, set_label):
    # 遍历文件夹，输出文件及子文件夹内文件
    if os.path.exists(path_dir):
        path_dir = os.path.abspath(path_dir)
        for i in os.listdir(path_dir):
            path_i = os.path.join(path_dir, i)
            if os.path.isfile(path_i):
                image_path_list.append(path_i)
                label_list.append(set_label)
            else:
                get_all_file_from_dir(path_i, image_path_list, label_list, set_label)
