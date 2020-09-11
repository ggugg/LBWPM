import numpy as np
import math
import sympy

def greenplot_create(origin, point_1, point_2, img,k):
    # 确定绿点位置
    greenplot_list = []
    vector_1 =np.array((point_1[1] - origin[1],origin[0]-point_1[0]))
    vector_2 =np.array((point_2[1] - origin[1],origin[0]-point_2[0]))
    width, height = img.shape[0], img.shape[1]
    # 新的坐标系
    x_max = width/k - origin[1] - 1
    x_min = -origin[1]
    y_max = origin[0]
    y_min = - (height/k - origin[0] - 1)
    for i in range(0,height,k):
        for j in range(0,width,k):
            vector_x=np.array((j/k - origin[1], origin[0] - i/k))
            m,n=sympy.symbols("m n",integer=True)
            a=sympy.solve([ int(vector_1[0])*m+int(vector_2[0])*n -int(vector_x[0]), int(vector_1[1])*m+int(vector_2[1])*n-int(vector_x[1]) ],[m,n])
            if isinstance(a,dict):
                a_1=float(a[m])
                b_1=float(a[n])
                greenplot_list.append((int((-vector_x[1] + y_max)*k),int((vector_x[0] - x_min)*k)))
    return np.array(greenplot_list)


def getHiddenMessage(decode_message,origin,point_1,point_2,img,module_size):
	#把消息串变为29*29
    width,height=img.shape[0],img.shape[1]
    dlist=list(decode_message)
    darr = np.array(dlist)
    daresh = darr.reshape(width//module_size,height//module_size)
    img1 = np.array(img)
    #得到秘密信息位置
    points = greenplot_create(origin,point_1,point_2,img1,module_size)
    msg=''
    for i in range(len(points)):
        #模块坐标转换成信息矩阵坐标
        p0=int(points[i][0]/module_size)
        p1=int(points[i][1]/module_size)
        msg += '' + daresh[p0][p1]
    return msg

