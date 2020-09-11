import cv2 as cv
import numpy as np
from scipy.fftpack import fft
import scipy.signal as signal
import math

def Contrastevaluation(img, k):
    '''
    return  50% and 150% of the median contrast of all modules in the barcode image

    '''
    MedianContrast = []
    height, width = img.shape[0], img.shape[1]
    for i in range(0, height, k):
        for j in range(0, width, k):
            module = img[i:i + k, j:j + k]
            MedianContrast.append(np.max(module) - np.min(module))
    MedianContrast = np.array(MedianContrast)
    MedianContrast = np.median(MedianContrast)
    delta_1 = MedianContrast * 0.5
    delta_2 = MedianContrast * 1.5
    return delta_1, delta_2


def ContrastBased(nparray, ModuleSize):
    '''
    C ∈ [Δ1, Δ2]
    contrast-based demodulation
    '''
    move = int(ModuleSize / 4 - 3)
    inner_size = int(ModuleSize / 2 - 2)
    inner_start = int(ModuleSize / 2 + move)
    inner = nparray[inner_start:inner_start +
                    inner_size, inner_start:inner_start + inner_size]
    inner_1 = np.mean(inner)
    outer_size = int(ModuleSize / 2 + 2)
    outer_start = int(inner_start - 2)
    outer = nparray[outer_start:outer_start +
                    outer_size, outer_start:outer_start + outer_size]
    middle = nparray[inner_start - 1:inner_start + inner_size +
                     1, inner_start - 1:inner_start + inner_size + 1]
    outer_1 = (np.sum(outer) - np.sum(middle)) / (outer_size *
                                                  outer_size - ((inner_size + 2) * (inner_size + 2)))
    if inner_1 > outer_1:
        return '1'
    elif outer_1 > inner_1:
        return '0'


def MatchedFiltering(nparray, ModuleSize):
    '''
    C < Δ1
    matched filter-based
    '''
    nparray = nparray - nparray.mean()
    nparray = nparray / nparray.max()
    filter_size = math.ceil(1.5 * ModuleSize)
    filter_temp = np.zeros((filter_size, filter_size))
    temp = math.ceil(0.25 * ModuleSize)
    filter_temp[temp:temp + ModuleSize, temp:temp + ModuleSize] = 255
    valid = signal.convolve2d(
        nparray, filter_temp, mode='valid')
    line = valid.flatten()
    line_max = np.max(line)
    line_min = np.min(line)
    if abs(line_max) > abs(line_min):
        return "1"
    else:
        return "0"


def GradientBased(nparray, ModuleSize):
    '''
    C > Δ2
    gradient-based
    '''
    # 内外部比较记录
    in_num, out_num = 0, 0
    inner_start = int(ModuleSize / 4)
    size = int(ModuleSize / 2)
    outer_start = int(ModuleSize / 4 - 1)
    Z_inner = [inner_start, inner_start + size]
    Z_outer = [outer_start, outer_start + size]
    in_row = nparray[Z_inner, inner_start:inner_start + size]
    out_row = nparray[Z_inner, outer_start:outer_start + size]
    in_col = nparray[inner_start:inner_start + size, Z_inner]
    out_col = nparray[outer_start:outer_start + size, Z_outer]
    for i in range(len(in_row)):
        for j in range(len(in_row[0])):
            if in_row[i][j] > out_row[i][j]:
                in_num += 1
            if in_row[i][j] < out_row[i][j]:
                out_num += 1
    for i in range(len(in_col)):
        for j in range(len(in_col[0])):
            if in_col[i][j] > out_col[i][j]:
                in_num += 1
            if in_col[i][j] < out_col[i][j]:
                out_num += 1
    # 输出结果
    if in_num > out_num:
        return '1'
    elif in_num < out_num:
        return '0'
    elif in_num == out_num:
        in_sum = np.sum(in_row) + np.sum(in_col)
        out_sum = np.sum(out_row) + np.sum(out_col)
        if in_sum > out_sum:
            return '1'
        elif in_sum < out_sum:
            return '0'


def decodemodule(nparray, module_size, delta_1, delta_2):
    """
    根据传入的模块进行分情况解码，
    C ∈ [Δ1, Δ2]
    contrast-based demodulation
    C < Δ1
    matched filter-based
    C > Δ2
    gradient-based
    """
    extend = cv.resize(nparray, (module_size * 2, module_size * 2),
                       interpolation=cv.INTER_LINEAR)
    bit = ''
    start = int(module_size / 2)
    inner = extend[start:start + module_size, start:start + module_size]
    inner = np.sum(inner)
    outer = np.sum(extend) - inner
    inner = int(inner / int(module_size * module_size))
    outer = int(outer / int(3 * module_size * module_size))
    contrast = abs(inner - outer)
    if contrast >= delta_1 and contrast <= delta_2:
        bit = ContrastBased(extend, module_size)
        decodetype = '1'
    elif contrast < delta_1:
        bit = MatchedFiltering(extend, module_size)
        decodetype = '2'
    elif contrast > delta_2:
        bit = GradientBased(extend, module_size)
        decodetype = '3'
    return bit, decodetype


def decode(img, module_size, x_locations_list, y_locations_list):
    yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    Y, U, V = cv.split(yuv)
    delta_1, delta_2 = Contrastevaluation(img, module_size)
    decode_message = ''
    for x in range(len(x_locations_list) - 1):
        for y in range(len(y_locations_list) - 1):
            decode_message += decodemodule(Y[x_locations_list[x]: x_locations_list[x + 1],
                                             y_locations_list[y]: y_locations_list[y + 1]], module_size, delta_1, delta_2)[0]
    return decode_message

