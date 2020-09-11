import numpy as np
import math
import sympy
from encoder import *
import random


def greenplot_create(origin, point_1, point_2, img, k):
    # 确定绿点位置
    greenplot_list = []
    # vector_1 = point_1 - origin
    # vector_2 = point_2 - origin
    vector_1 = np.array((point_1[1] - origin[1], origin[0] - point_1[0]))
    vector_2 = np.array((point_2[1] - origin[1], origin[0] - point_2[0]))
    width, height = img.shape[0], img.shape[1]
    # print(type(vector_1[0]),vector_2)
# 新的坐标系
    x_max = width / k - origin[1] - 1
    # print(x_max)
    x_min = -origin[1]
    y_max = origin[0]
    y_min = - (height / k - origin[0] - 1)
    # print(x_max, x_min, y_max, y_min)
    for i in range(0, height, k):
        for j in range(0, width, k):
            # for x in range(0,k):
                # if flag:
                #     break
                # # for y in range(0,k):
                #     if flag:
                #         break
            vector_x = np.array((j / k - origin[1], origin[0] - i / k))
            # print(vector_x)
            m, n = sympy.symbols("m n", integer=True)
            a = sympy.solve([int(vector_1[0]) * m + int(vector_2[0]) * n - int(vector_x[0]),
                             int(vector_1[1]) * m + int(vector_2[1]) * n - int(vector_x[1])], [m, n])
            # print(a)
            if isinstance(a, dict):
                a_1 = float(a[m])
                b_1 = float(a[n])
                # if a_1<0 or b_1<0:
                # if (a_1-int(a[m]))==0 and (b_1-int(a[n]))==0:
                #     print(a_1,b_1)

                # print(a_1,b_1)
                # if vector_1[0]*a[m] > x_max or vector_1[0]*a[m] <x_min or vector_2[0]*a[n] >x_max or vector_2[0]*a[n] <x_min or vector_1[1]*a[m] > y_max or vector_1[1]*a[m] <y_min or vector_2[1]*a[n] > y_max or vector_2[1]*a[n] <y_min :
                # if (a_1-int(a[m]))!=0 or (b_1-int(a[n]))!=0:
                # continue
                # else:
                greenplot_list.append(
                    (int((-vector_x[1] + y_max) * k), int((vector_x[0] - x_min) * k)))
    # print(greenplot_list)
    return np.array(greenplot_list)


# #直角，锐角，钝角
#    ange_x = math.ceil(
#              max(abs(x_max),abs(x_min)) / max(abs(vector_1[1] + vector_2[1]), abs(vector_1[1]), abs(vector_2[1])))
 # if vector_1[1]==0 or vector_2[1]==0 or :
#         if vector_1[1]==0 and vector_2[1]==0:
#             return 0
#         else:
#             range_x=math.ceil(max(abs(x_max),abs(x_min)) / max(abs(vector_1[1]),abs(vector_2[1])))
#     else:
#         r
#     if vector_1[0]==0 or vector_2[0]==0 :
#         if vector_1[0]==0 and vector_2[0]==0:
#             return 0
#         else:
#             print("1")
#             range_y=math.ceil(max(abs(y_max),abs(y_min)) / max(abs(vector_1[0]),abs(vector_2[0])))
#     else:
#         print("2")
#         range_y = math.ceil(
#             max(abs(y_max),abs(y_min)) / min(abs(vector_1[0] + vector_2[0]), abs(vector_1[0]), abs(vector_2[0])))
#     print(range_x, range_y)
# 找符合的点
#     range_x=math.floor(max(abs(x_max),abs(x_min)) / max(abs(vector_1[0]),abs(vector_2[0])))
#     range_y=math.floor(max(abs(y_max),abs(y_min)) / max(abs(vector_1[1]),abs(vector_2[1])))
#     print(range_x, range_y)
#     greenplot_1=np.array((1,2))
#     greenplot_2=np.array((1,2))
#     for n in range(-range_x - 1, +range_x + 1):
#         for m in range(-range_y - 1, +range_y + 1):
#             greenplot_1[0] = vector_1[0] + vector_2[0] * n
#             greenplot_1[1] = vector_1[1] + vector_2[1] * n
#             greenplot_2[0] = vector_2[0] + vector_1[0] * n
#             greenplot_2[1] = vector_2[1] + vector_1[1] * m
#             if greenplot_1[1] <= x_max and greenplot_1[1] >= x_min and greenplot_1[0] <= y_max and greenplot_1[0] >= y_min:
#                 # b = greenplot + origin
#                 # b_1 = b.tolist()
#                 # img[b_1[0], b_1[1]] = [0, 0, 0]
#                 greenplot_list.append((-greenplot_1[0] + y_max,greenplot_1[1] - x_min))
#             if greenplot_2[1] <= x_max and greenplot_2[1] >= x_min and greenplot_2[0] <= y_max and greenplot_2[0] >= y_min:
#                 greenplot_list.append((-greenplot_1[0] + y_max,greenplot_1[1] - x_min))
#     greenplot_list=list(set(greenplot_list))
#     print(greenplot_list)
#     return np.array(greenplot_list)
# # 修改信息对应比特位


def change_message(message, greenplot_list, img, message_enc, k=16):
    # 根据绿点修改信息串
    print('len(greenplot):', len(greenplot_list))
    width, height = img.shape[0], img.shape[1]
  #  print(len(message),int(height * width / (k * k)))
    # 填充加密信息
    while len(message_enc) < len(greenplot_list):
        message_enc += str(random.randint(0, 1))
    message = ''
    while len(message) < int(height * width / (k * k)):
        message += str(random.randint(0, 1))
    # 由点定位到块
    # print(message)
    greenplot_block = np.floor(greenplot_list / k).astype(int)
    # 由块定位到点
    message_list = list(message)
    index_list = []
    index_set = set()
    for i in range(len(greenplot_block)):
        index = int(greenplot_block[i][0] *
                    (width / k)) + int(greenplot_block[i][1])
        index_list.append(index)
        index_set.add(index)

        # print("change", index, "before", message_list[index])
        message_list[index] = message_enc[i]
        # print("after", message_list[index])
    new_message = ''.join(message_list)
    # print(len(index_set))
    # print(len(index_list))
    return new_message
