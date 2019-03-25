import os
import sys
import time
from datetime import datetime

import tensorflow as tf

from tools import data_helper, utils
from models import cnn_net_keras
from args_management.tain_net_params import TrainNetArgs, train_net_args


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_image(image_string)
    image.set_shape([None, None, None])  # 防止image没有shape而报错
    image = tf.image.resize_images(image, (60, 60), method=2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if args.per_image == 0:
        image = tf.image.per_image_standardization(image)
    elif args.per_image == 2:
        image = (tf.cast(image, tf.float32) - 127.5) / 128
    return image, tf.one_hot(label, 2)


def main():
    if str(args.gpu_device):  # 指定gpu设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    # 导入指纹训练数据集
    train_image_list = []
    train_label_list = []
    data_helper.get_all_file_from_dir(args.train_pos_dir, train_image_list, train_label_list, 1)
    data_helper.get_all_file_from_dir(args.train_neg_dir, train_image_list, train_label_list, 0)
    # 创建tf.data数据流
    dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    if args.tf_data_shuffle < 0:
        dataset = dataset.shuffle(len(train_image_list))
    else:
        dataset = dataset.shuffle(args.tf_data_shuffle)
    dataset = dataset.map(_parse_function, num_parallel_calls=args.tf_data_map_num)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # 创建占位符
    inputs = tf.placeholder(tf.float32, shape=(None,) + args.image_shape, name="inputs")
    labels = tf.placeholder(tf.float32, shape=(None,) + (args.classes_num,), name="labels")
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    out = cnn_net_keras.cnn_net(inputs)
    # 计算损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=labels))
    # 创建优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # 计算精度
    correct_pred = tf.equal(tf.argmax(labels, axis=1), tf.argmax(out, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('losses/total_loss', loss)
    tf.summary.scalar('train_acc', accuracy)
    tf.summary.scalar('learning_rate', learning_rate)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=2)

    # 创建日志保存目录
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    output_dir = os.path.join(os.path.expanduser(args.output_dir), args.output_name + "_" + subdir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    model_dir = os.path.join(output_dir, 'save_models')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    utils.write_arguments_to_file(args, os.path.join(output_dir, 'arguments.txt'))

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
        global_step = 0
        for epoch in range(args.max_epochs):
            if args.learning_rate > 0.0:
                lr = args.learning_rate
            else:
                lr = utils.get_learning_rate_from_file(args.learning_rate_file, epoch)
            if lr <= 0:
                return False
            batch_number = 0
            while batch_number < args.epoch_size:
                start_time = time.time()
                image_batch, label_batch = sess.run(next_element)
                feed_dict = {inputs: image_batch, labels: label_batch, keep_prob: 0.8, learning_rate: lr}
                if batch_number % 100 == 0:
                    loss_, acc_, _, summary_str_ = sess.run([loss, accuracy, optimizer, merged], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str_, global_step)
                else:
                    loss_, acc_, _ = sess.run([loss, accuracy, optimizer], feed_dict=feed_dict)
                run_time = time.time() - start_time
                batch_number += 1
                global_step += 1
                sys.stdout.write(
                    "\repoch: {:d}\tglobal_step: {:d}\ttotal_loss: {:f}\ttrain_acc: {:f}\tlr: {}\trun_time: {:f}".format(
                        epoch, global_step, loss_, acc_, lr, run_time))
            print("\t\tsave model...")
            checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % args.output_name)
            saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
            metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % args.output_name)
            if not os.path.exists(metagraph_filename):
                print('Saving metagraph')
                saver.export_meta_graph(metagraph_filename)


if __name__ == '__main__':
    argv = sys.argv[1:]  # 获取命令行参数
    args = train_net_args(argv) if argv else TrainNetArgs()  # 有命令行参数就用命令行参数
    main()
