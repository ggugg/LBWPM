#!/usr/bin/python
#-*- encoding: utf-8 -*-
import os
from jpype import *
import cv2 as cv
import numpy as np
from scipy.fftpack import fft
import scipy.signal as signal
from bresenham import bresenham


def coarseCornerDetection(img_path):
    # 启动JVM
    '''
    return (TopLeftPoint,BottomRightPoint),
    剪切后的图片
    '''
    filepath = os.getcwd()
    zxing_path = "./javase.jar;./core.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", ("-Djava.class.path=%s" %
                                              zxing_path))

    File = JClass("java.io.File")
    ImageIO = JClass("javax.imageio.ImageIO")
    BufferedImageLuminanceSource = JClass(
        "com.google.zxing.client.j2se.BufferedImageLuminanceSource")
    HybridBinarizer = JClass("com.google.zxing.common.HybridBinarizer")
    BinaryBitmap = JClass("com.google.zxing.BinaryBitmap")
    HashMap = JClass("java.util.HashMap")
    MultiFormatReader = JClass("com.google.zxing.MultiFormatReader")
    DataMatrixReader = JClass("com.google.zxing.datamatrix.DataMatrixReader")
    DecodeHintType = JClass("com.google.zxing.DecodeHintType")
    img = ImageIO.read(File(img_path))
    binaryBitmap = BinaryBitmap(HybridBinarizer(
        BufferedImageLuminanceSource(img)))
    image = binaryBitmap.getBlackMatrix()
    # print(image.getTopLeftOnBit(), image.getBottomRightOnBit())
    # print(image.getEnclosingRectangle())
    location = [(int(image.getTopLeftOnBit()[0]),
                 int(image.getTopLeftOnBit()[1])),
                (int(image.getBottomRightOnBit()[0]),
                 int(image.getBottomRightOnBit()[1]))]
    # Detector = JClass("com.google.zxing.datamatrix.detector.Detector")
    # d = Detector(image)
    # r = d.detect()
    # print(r.getPoints()[0], r.getPoints()[1],
    #       r.getPoints()[2], r.getPoints()[3])
    # print(r.getPoints()[0].getX(), type(r.getPoints()[0]))
    # location = []
    # for i in range(4):
    #     location.append(
    #         (int(r.getPoints()[i].getX()), int(r.getPoints()[i].getY())))
    # print(location)
    shutdownJVM()
    img = cv.imread(img_path)
    img_new = img[location[0][1]:location[1][1], location[0][0]:location[1][0]]
    return location, img_new
    # return location


def bresenhamLine(img, x1, y1, x2, y2, color):
    """
    img:[height,width,channels]
    x1,y1:first point x ,y
    x2,y2:first point x ,y
    color: (r,g,b)
    return img_add_line,直线的Y分量
    """
    Y_line = []
    yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    Y, _, _ = cv.split(yuv)
    line = bresenham(x1, y1, x2, y2)
    for i in line:
        img[i] = color
        Y_line.append(Y[i])
    return img, Y_line


def getModuleSize(connection_line):
    '''
    connection_line:直线的Y分量
    根据直线的Y分量进行模块大小的估计
    return:module size

    '''
    line_length = len(connection_line)
    sampling_rate = line_length  # 采样频率
    Amplitudes = np.abs(np.fft.rfft(connection_line) / line_length)
    # 利用np.fft.rfft()进行FFT计算，rfft()是为了更方便对实数信号进行变换，由公式可知/fft_size为了正确显示波形能量
    # rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
    # 于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
    Frequencys = np.linspace(0, sampling_rate // 2, line_length // 2 + 1)

    Frequency = signal.argrelextrema(Amplitudes, np.greater)[0][0]

    module_size = int(line_length // (2 * Frequency))
    return module_size


def pointList(connection_line, module_size):
    '''
    根据连接线的Y分量和粗估计的模块大小确定模块的位置
    return: the list of point
    '''
    Filter_Template = np.zeros(module_size * 4)
    Filter_Template[:module_size] = 0
    Filter_Template[module_size:module_size * 2] = 255
    Filter_Template[module_size * 2:module_size * 3] = 0
    Filter_Template[module_size * 3:] = 255
    same = signal.convolve(
        connection_line, Filter_Template, mode='same')
    greater_index = signal.argrelextrema(same, np.greater_equal)[0].tolist()
    less_index = signal.argrelextrema(same, np.less_equal)[0].tolist()
    flag = 1
    index = [0]

    # print(less_index, greater_index)
    while greater_index != [] and less_index != []:
        if flag == 0:
            temp = less_index.pop(0)
            # print("temp", temp)
            if temp > index[-1]:
                index.append(temp)
                flag = 1
        if flag == 1:
            temp = greater_index.pop(0)
            # print(temp)
            if temp > index[-1]:
                index.append(temp)
                flag = 0
    return index


def getDataArea(img_name):
    '''
    return img of data, module_size,x_locations_list,y_locations_list

    '''
    location, img_new = coarseCornerDetection(img_name)
    img_x, connection_line = bresenhamLine(
        img_new, 0, 0, 0, img_new.shape[1] - 1, (255, 0, 0))
    x_modulesize = getModuleSize(connection_line)
    x_point = pointList(connection_line, x_modulesize)
    # print(x_point)
    img_y, connection_line = bresenhamLine(
        img_new, 0, img_new.shape[1] - 1, img_new.shape[0] - 1, img_new.shape[1] - 1, (255, 0, 0))
    # print(connection_line)
    y_modulesize = getModuleSize(connection_line)
    y_point = pointList(connection_line, x_modulesize)
    # print(y_point)
    module_size = x_modulesize
    img1 = img_new[x_point[1]:x_point[len(
        x_point) - 2], y_point[1]: y_point[len(x_point) - 2], ]
    # plt.imshow(img1)
    # plt.show()
    # print(x_point)
    x_point2 = [(i - x_point[1]) for i in x_point[1:-1]]
    y_point2 = [(i - y_point[1]) for i in y_point[1:-1]]
    return img1, module_size, x_point2, y_point2


if __name__ == '__main__':
    img_name = './result.jpg'
    img_data, module_size, x_locations_list, y_locations_list = getDataArea(img_name)
    print(module_size, x_locations_list, y_locations_list)
