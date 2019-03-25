import argparse


class TrainNetArgs:
    gpu_device = 0
    output_dir = "./logs"
    output_name = "default"
    train_pos_dir = r'C:\public_data\指纹识别\train_data_20190318_1045\positive'
    train_neg_dir = r'C:\public_data\指纹识别\train_data_20190318_1045\negative'
    image_shape = (60, 60, 1)
    classes_num = 2

    max_epochs = 999
    epoch_size = 1000
    batch_size = 256
    learning_rate = -1
    learning_rate_file = r'./data/learning_rate_test.txt'
    tf_data_map_num = 8
    tf_data_shuffle = -1  # 指定-1时默认全打乱, 在单个样本和数据量很大时不推荐如此做
    per_image = 0  # 0: (x-mean)/adjusted_stddev,  1: (x-mean)/(max-min),  2: (x-127.5)/128


def train_net_args(argv):
    """

    :param argv:
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_device', help='指定显卡')
    parser.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.')

    return parser.parse_args(argv)
