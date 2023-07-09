# 工具类  分割数据集
import os
import random
import shutil
from shutil import copy2


def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.2):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
    :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    # 获取划分前数据
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        # 根据 train val  test 分别 放入 image 和mask
        split_path = os.path.join(target_data_folder, split_name)
        # 有了文件夹就跳过 没有就创建 智能化
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹  image 和mask
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历


    # 要保证img 和mask每个类别的图片对应
    # for class_name in class_names:
    current_class_data_path = os.path.join(src_data_folder, 'images')
    # 所有image
    current_all_data = os.listdir(current_class_data_path)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)
    train_folder = os.path.join(os.path.join(target_data_folder, 'train'), 'images')
    train_folder_mask = os.path.join(os.path.join(target_data_folder, 'train'), 'masks')
    val_folder = os.path.join(os.path.join(target_data_folder, 'val'), 'images')
    val_folder_mask = os.path.join(os.path.join(target_data_folder, 'val'), 'masks')
    test_folder = os.path.join(os.path.join(target_data_folder, 'test'), 'images')
    test_folder_mask = os.path.join(os.path.join(target_data_folder, 'test'), 'masks')
    train_stop_flag = round(current_data_length * train_scale)
    val_stop_flag = round(current_data_length * (train_scale + val_scale))
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        src_img_path = os.path.join(current_class_data_path, current_all_data[i])
        src_msk_path = src_img_path.replace('images', 'masks')
        src_msk_path = src_msk_path.replace('jpg', 'png')
        if current_idx < train_stop_flag:
            copy2(src_img_path, train_folder)
            copy2(src_msk_path, train_folder_mask)

            # print("{}复制到了{}".format(src_img_path, train_folder))
            train_num = train_num + 1
        elif current_idx < val_stop_flag:
            copy2(src_img_path, val_folder)
            copy2(src_msk_path, val_folder_mask)
            # print("{}复制到了{}".format(src_img_path, val_folder))
            val_num = val_num + 1
        else:
            copy2(src_img_path, test_folder)
            copy2(src_msk_path, test_folder_mask)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1

        current_idx = current_idx + 1

        # print("*********************************{}*************************************".format('images'))
        # print(
        #     "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format('images', train_scale, val_scale, test_scale, current_data_length))
        # print("训练集{}：{}张".format(train_folder, train_num))
        # print("验证集{}：{}张".format(val_folder, val_num))
        # print("测试集{}：{}张".format(test_folder, test_num))
    print("训练集{}：{}张".format(test_folder, train_stop_flag))
    print("验证集{}：{}张".format(test_folder, val_stop_flag-train_stop_flag))
    print("测试集{}：{}张".format(test_folder, current_data_length -val_stop_flag))

if __name__ == '__main__':
    src_data_folder = "D:\good"
    target_data_folder = "D:/project/unet-pp-Medical-cell-segmentation/inputs/"
    data_set_split(src_data_folder, target_data_folder)

