import sys
sys.path.append('../decoder')
sys.path.append('../encoder')
from encoder import encode, msgchange
from decoder import greenplot
import matplotlib.pyplot as plt
from reedsolo import RSCodec
import cv2 as cv
import numpy as np
import random

def rsencode(msg):
    '''
    msg: 二进制流
    '''
    # msg = '00110001001100100011001100110100001101010011011000110111'
    # return msg
    while len(msg) % 7:
        msg += '0'
    temp = ''
    for i in range(0, len(msg), 7):
        temp += chr(int(msg[i:i + 7], 2))
    ecc = RSCodec(96)
    byte_msg = ecc.encode(temp)
    rscode = byte_msg[len(temp):]
    binary_code = ''.join(format(x, '08b') for x in rscode)
    return msg + binary_code


def rsdecode(rscode, msglen):
    '''
    msg:rs二进制码
    '''
    ecc = RSCodec(96)
    length = len(rscode)
    padmsglen = (msglen // 7 + 1) * 7
    rslen = padmsglen + 96 * 8
    temp = []
    for i in range(0, padmsglen, 7):
        temp.append(int(rscode[i:i + 7], 2))
    for i in range(padmsglen, rslen, 8):
        temp.append(int(rscode[i:i + 8], 2))
    temp = bytearray(temp)
    try:
        rscode = ecc.decode(temp)
        rscode = ''.join(format(x, '08b') for x in rscode)
    finally:
        if len(rscode) == length:
            print('too many bits errors!')
            return rscode[:msglen]
        else:
            temp = ''
            for i in range(0, padmsglen // 7 * 8, 8):
                temp += rscode[i + 1:i + 8]
            return temp[:msglen]
                

def createWatermarkSequence(img, watermark_path):
    # 读取水印图片得到水印序列
    img_watermark = cv.imread(watermark_path, cv.IMREAD_GRAYSCALE)
    ret, dst = cv.threshold(img_watermark[:, :], 0, 255, cv.THRESH_OTSU)
    msg_watermark = ''
    for i in img_watermark.flatten():
        if i >= ret:
            msg_watermark += '1'
        else:
            msg_watermark += '0'
    width, height = img_watermark.shape[0], img_watermark.shape[1]
    len_width, len_height = bin(width)[2:], bin(height)[2:]
    while len(len_width) < 16:
        len_width = '0' + len_width
    while len(len_height) < 16:
        len_height = '0' + len_height
    return len_width + len_height + msg_watermark


def addWatermark(img, message, watermarkedMSG, k, Lambda, channel='R'):
    width, height = img.shape[0], img.shape[1]
    assert(height == width)

    # 模块数
    block_num = int(height * width / (k * k))
    # 消息长度不足补0
    while len(message) < block_num:
        message += '0'

    # 分离R,G,B分量
    B, G, R = cv.split(img)
    bgr = {'B': B, 'G': G, 'R': R}

    # 改变RGB通道
    BGR = []
    for cnl in ['B', 'G', 'R']:
        if cnl == channel:
            BGR.append(encode.change(bgr[cnl], watermarkedMSG, k, Lambda))
        else:
            BGR.append(encode.change(bgr[cnl], message, k, Lambda))

    # 合并Y,U,V分离得到新图片
    new_img = cv.merge(BGR)

    return new_img

def getWatermark(img, k, origin, point_1, point_2, channel='R'):

    # 分离R,G,B分量
    B, G, R = cv.split(img)
    bgr = {'B': B, 'G': G, 'R': R}

    # 获取对应通道的解码字符串
    decode_msg = encode.channel_decode(bgr[channel], k)

    # 根据绿点得到水印序列
    decodedMSG = greenplot.getHiddenMessage(
        decode_msg, origin, point_1, point_2, img1, k)
    return decodedMSG


k = 4
encode_type = "RGB"
Lambda = 25

# 产生格的原始向量
origin = np.array([0, 0])
point_1 = np.array([0, 1])
point_2 = np.array([1, 0])

# 读取背景图片
img = cv.imread('./lena.jpg')
len_msg = (int(img.shape[0]) // k)**2

# 随机生成公开信息
message = '0000011100010101100100000011000000100010010100100000110100000110011001000001001000000111101110111111011011001101001110010000010000100000011110110001101101111000101100111010011100001011000001111010111011101000010101010110010111011110011100000111001010111000000011110101111110111001111001100011010101000101111010010000001010010111011000111111000001010001100000010011100100011111101010010001111110000100111001000111010111001101011001111010011000101111100111001101001111001110000011001011000010111101011110111100000011100111100110011111101001110010101001011001111100101100101101111101000011010110010011100110011101110001111001010001101011101001001011000111100001100011000011111011101010000101010011101010010010010011100101000010010111111110110001010001110001110101001010000001111100101010011111101001010001001111100110011011111010011011100010000'
#   message +="0"
# 生成绿点
z = msgchange.greenplot_create(origin, point_1, point_2, img, k)

# 获取水印序列
watermarkSequence = createWatermarkSequence(img, './jnu.png')
originmes = watermarkSequence[32:544]
rsc = rsencode(originmes)
watermarkSequence = watermarkSequence[:32] + rsc
# 添加了水印序列后的信息
watermarkedMSG = msgchange.change_message(message, z, img, watermarkSequence, k)

# 添加水印
img1 = addWatermark(img, message, watermarkedMSG, k, Lambda, channel='G')
cv.imwrite('./addpicture.jpg', img1)

# 解码公开信息
decode_msg = encode.decode(img1, k, encode_type)

# 得到水印长度+水印序列
decodedMSG = getWatermark(img1, k, origin, point_1, point_2, channel='G')

# 1-16位是水印的宽,17-32位是水印的高
len_width, len_height = int(decodedMSG[:16], 2), int(decodedMSG[16:32], 2)
len_width, len_height = 16, 32
decodedwatermarkSequence = decodedMSG[32:]
decodedwatermarkSequence = rsdecode(decodedwatermarkSequence, 512)
decodedwatermarkSequence = [255 if i == '1' else 0 for i in list(decodedwatermarkSequence)]

# 一维矩阵转二维矩阵
decodedwatermarkSequence = np.array(
	decodedwatermarkSequence).reshape(len_width, len_height)
# 保存水印
cv.imwrite('./watermark.png', decodedwatermarkSequence)

