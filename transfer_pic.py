# 机 构：中国科学院大学
# 程序员：李浩东
# 时 间：2023/3/27 23:05

import numpy as np
import struct  # 解析二进制数据
from PIL import Image
import os


# # 训练集
# # 指定MNIST数据集所在路径和文件名
# dataset_path = 'D:/DL/实验一 手写数字识别/data/'  # 解压的数据集所在文件夹
# data_file = dataset_path + 'train-images.idx3-ubyte'
#
# # It's 47040016B, but we should set to 47040000B
# data_file_size = 47040016
# data_file_size = str(data_file_size - 16) + 'B'
#
# # 读取图像数据，并将其转换为numpy数组类型
# data_buf = open(data_file, 'rb').read()  # 以二进制模式打开文件，并将文件内容读取到data_buf变量中，此时data_buf是一个字节串
# magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)  # 使用struct模块按照大端字节序（>）将字节串中前4个字节解压缩成4个整数（分别为magic、numImages、numRows和numColumns），这些数字描述了数据集的基本信息
# datas = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII'))  # 根据之前获取的magic等信息计算出每张图片所占的字节数，然后按照大端字节序将剩余的字节串解压缩成一组uint8类型的数据
# datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)  # 将上一步得到的一维数组转换成一个四维数组，其中第一维表示样本数，第二维固定为1（因为每个样本只有一幅图像），第三维和第四维分别为图像的高度和宽度。最终返回一个ndarray类型的变量datas
#
# # 指定MNIST数据集中标签文件的路径和文件名
# label_file = dataset_path + 'train-labels.idx1-ubyte'
#
# # It's 60008B, but we should set to 60000B
# label_file_size = 60008
# label_file_size = str(label_file_size - 8) + 'B'
#
# # 读取标签数据，并将其转换为numpy数组类型
# label_buf = open(label_file, 'rb').read()
# magic, numLabels = struct.unpack_from('>II', label_buf, 0)
# labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
# labels = np.array(labels).astype(np.int64)
#
# # 创建存放转换后的图片的文件夹
# train_path = dataset_path + 'mnist_train'  # 转换后的训练集所在路径
# if not os.path.exists(train_path):
#     os.mkdir(train_path)
#
# # 新建0~9的十个子文件夹，用于存放对应标签的图片
# for i in range(10):
#     file_name = train_path + os.sep + str(i)
#     if not os.path.exists(file_name):
#         os.mkdir(file_name)
#
# # 对每张图像进行处理，将其转换为PIL图像格式，并保存到对应的标签文件夹
# for ii in range(numLabels):
#     img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
#     label = labels[ii]
#     file_name = train_path + os.sep + str(label) + os.sep + str(ii) + '.png'
#     img.save(file_name)


# 测试集
# 指定MNIST数据集所在路径和文件名
dataset_path = 'D:/DL/实验一 手写数字识别/data/'  # 解压的数据集所在文件夹
data_file = dataset_path + 't10k-images.idx3-ubyte'

# It's 7840016B, but we should set to 7840000B
data_file_size = 7840016
data_file_size = str(data_file_size - 16) + 'B'

# 读取图像数据，并将其转换为numpy数组类型
data_buf = open(data_file, 'rb').read()  # 以二进制模式打开文件，并将文件内容读取到data_buf变量中，此时data_buf是一个字节串
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)  # 使用struct模块按照大端字节序（>）将字节串中前4个字节解压缩成4个整数（分别为magic、numImages、numRows和numColumns），这些数字描述了数据集的基本信息
datas = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII')) # 根据之前获取的magic等信息计算出每张图片所占的字节数，然后按照大端字节序将剩余的字节串解压缩成一组uint8类型的数据
datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)  # 将上一步得到的一维数组转换成一个四维数组，其中第一维表示样本数，第二维固定为1（因为每个样本只有一幅图像），第三维和第四维分别为图像的高度和宽度。最终返回一个ndarray类型的变量datas

# 指定MNIST数据集中标签文件的路径和文件名
label_file = dataset_path + 't10k-labels.idx1-ubyte'

# It's 10008B, but we should set to 10000B
label_file_size = 10008
label_file_size = str(label_file_size - 8) + 'B'

# 读取标签数据，并将其转换为numpy数组类型
label_buf = open(label_file, 'rb').read()
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

# 创建存放转换后的图片的文件夹
test_path = dataset_path + 'mnist_test1'  # 转换后的测试集所在路径
if not os.path.exists(test_path):
    os.mkdir(test_path)

# 新建0~9的十个子文件夹，用于存放对应标签的图片
for i in range(10):
    file_name = test_path + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

# 对每张图像进行处理，将其转换为PIL图像格式，并保存到对应的标签文件夹
for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = test_path + os.sep + str(label) + os.sep + str(ii) + '.png'
    img.save(file_name)

