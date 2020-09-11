import os
import math
import numpy
import cv2 as cv
import random

#常数R
R=3

#常数η
eta=0.1

#加密的消息
messages = {
    'low': '0000011100010101100100000011000000100010010100100000110100000110011001000001001000000111101110111111011011001101001110010000010000100000011110110001101101111000101100111010011100001011000001111010111011101000010101010110010111011110011100000111001010111000000011110101111110111001111001100011010101000101111010010000001010010111011000111111000001010001100000010011100100011111101010010001111110000100111001000111010111001101011001111010011000101111100111001101001111001110000011001011000010111101011110111100000011100111100110011111101001110010101001011001111100101100101101111101000011010110010011100110011101110001111001010001101011101001001011000111100001100011000011111011101010000101010011101010010010010011100101000010010111111110110001010001110001110101001010000001111100101010011111101001010001001111100110011011111010011011100010000',
    'high': '1110111110110111011111011011001111101100110111110111011011111011111111111111111001111111111111111111100111111111111111111100111111110101010101110101101010111100110110100010011010101001010011100111111111111110011111110011110011100111100111111111110000110011111011010101010111101010110010011100100101111010101010111011001001011111111110000111100000101100111101011111111110001111111111010000111101010110100010011100101011010101010110101001001010010101011010111111111000011110010011110011111111111001111000111111111111001001111011011011111001110101011011001010011101101011010000110101101111100001100000011000000000000011000001100001111100001100111110001011110101100100101010110010010100100101001010101101011111001110011110111111011111111111001111111100011111111111110010100011000001111101001010011101100101110111010111100111011010110101110011010111011010010100111011001010101110101011001110110101101011100110101111110001111111110011000001111111111100000000110001100000011000000011110110101000001100101001001010101101111111001001101111100111010111100111111111001100011001100110000011110011000000011000000100110110011010101000110010011111011111110111111100111010000101011100111110100000111101110110110000010000000000011001011101111111111101111001110100100101011110111110111111110110101010010100101010101101111000000011111111111000010000000000000100000001111111111111110011110111001100001010101010111001000100000010101010010001011010101111000000110011000000000000011000000011110000000001111000011111110111011011001111001101010111101101011100001101011000000010101011011100011100111011001111111001100100000011001111100111111111111100111001101011000010101001101110101011010110101010010101101001001110110011010110100101010011011101010110101101010100101001010010011111111111100100000000000000001111000000011000000000000000000000000111010101101011101011101011010100101010111001101010111110101010101111000110011110011111110000100000100011111110001111000000111110101100110110011001101010110110100000000100100000001011101011010111111101111000010111011101010111001000111001100000111101000000011011110101111010111001101111011101001010011110010000101100101010011011111111100001011001110101011111100011110110000111110000000001101111010010100110100110111101101011000110110110101111000011010110111111110000011111100111000000000011011000000111000000110011111000101110110101111101101010101000101100011000010110001010001100111010111000111110010001111111111100001101100110000101111111001100100001111100100110110111101111011011010001010010010011001010101011100111111001001101101111011110110110100010100100100110010101010111001011100000111110011010001001101111100100001101000110001001111100111110011100101110101110111000011010011101110000000110000110101100111100011111001000101001111001100000111100110000011000000000001101011101101011011010111001011110011100101111001000011011010101001111111111111100101100111010011110011011000000111010100111100000001111101010101101110011011110001001110011011011010111100110010111001111111110110010111111101001111001101100000111101110011110110010111111011100101100101110100100011001001010001011011001110101101100111100000011111100101000011110011100111111000010100110000001100011011001101011010011110101011101011100101001001000010100010110101011111001111100110010111111100000010011111111000001011010000111111111100111010101110011001001111001100010100011001000100010010111001111001110101010100111010011110001100101001110010111010100101110010110110001111011011000111001101100000001111001110011111111000111111111001001011100101110101111010010101001000110110000010100100101111011000000000100000100110000111111100110011111110100001111111010111100101011110001001100111111010101100011101010110111110101011111100011100110011011001100111111111111100010100000111100110001100111101101100111101001001101010101110101100101010000100011001001101110000111011011111001010011011101011111101011001011110011000111111101001010101101011111001000110000010100011011110101101011010111111111000000000000011100111100110001111110011111000000000011111010111101011010000101110101000011010110100001010100101000110101010111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111'
}

#参数
#Lambdas=[10,25,40]
#Capacitys=['low','high']

#加密类型
#encode_types=["Y","RGB"]

#颜色
colors={"Y":"yellow","RGB":"black"}

def cal_delta_I(contrast,eta,Lambda):
	"""计算delta_I"""
	return math.ceil((contrast+eta)/Lambda)*Lambda

def add_location(img,k):
	"""添加定位图案"""
	width,height=img.shape[0],img.shape[1]

	img_bottom=numpy.ones((k,width),dtype=numpy.uint8)
	img_bottom=cv.cvtColor(img_bottom,cv.COLOR_GRAY2BGR)
	img_bottom[:,:,:]=0
	img_top=img_bottom.copy()
	for i in range(int(width/k)):
		if i%2==0:
			img_top[0:k,k*i:k*(i+1),:]=255

	img_left=numpy.ones((height+k*2,k),dtype=numpy.uint8)
	img_left=cv.cvtColor(img_left,cv.COLOR_GRAY2BGR)
	img_left[:,:,:]=0
	img_right=img_left.copy()
	for i in range(int((height+k*2)/k)):
		if i%2==1:
			img_right[k*i:k*(i+1),0:k,:]=255

	#纵向拼接
	image=numpy.vstack((img_top,img))
	image=numpy.vstack((image,img_bottom))
	#横向拼接
	image=numpy.concatenate([img_left,image], axis=1)
	image=numpy.concatenate([image,img_right], axis=1)
	return image

def change(Y,message,k,Lambda):
	"""根据编码规则改变单个通道"""
	width,height=len(Y[0]),len(Y)
	for i in range(0,height,k):
		for j in range(0,width,k):
			#计算当前为第几个模块
			index=int((i/k)*(width/k))+int(j/k)
			
			#嵌入当前模块的比特
			B=int(message[index])
			
			#OSTU计算阈值
			ret,dst=cv.threshold(Y[i:i+k,j:j+k],0,255,cv.THRESH_OTSU)
			
			#计算对比度
			contrast,bright_pixs_sum,dark_pixs_sum,bright_pixs_num,dark_pixs_num=0,0,0,0,0
			for x in range(k):
				for y in range(k):
					if int(Y[i+x][j+y])>ret:
						bright_pixs_sum+=int(Y[i+x][j+y])
						bright_pixs_num+=1
					else:
						dark_pixs_sum+=int(Y[i+x][j+y])
						dark_pixs_num+=1
			if bright_pixs_num!=0 and dark_pixs_num!=0:
				contrast=((bright_pixs_sum//bright_pixs_num)-(dark_pixs_sum//dark_pixs_num))//8
			contrast=0
			assert(0<=contrast and contrast<=255)
			
			#计算ΔI
			delta_I=cal_delta_I(contrast,eta,Lambda)
			
			#修改内外像素的亮度
			ε_I_sum,ε_O_sum,ε_I_num,ε_O_num=0,0,0,0
			for x in range(k):
				for y in range(k):
					if x>=int(k/4) and x<int(k*3/4) and y>=int(k/4) and y<int(k*3/4):
						temp=int(int(Y[i+x][j+y])-((-1)**B)*(delta_I))
						if temp>255:
							Y[i+x][j+y]=255
						elif temp<0:
							Y[i+x][j+y]=0
						else:
							Y[i+x][j+y]=temp
						if abs(temp-int(Y[i+x][j+y]))!=0:
							ε_O_sum+=abs(temp-int(Y[i+x][j+y]))
							ε_O_num+=1
					else:
						temp=int(int(Y[i+x][j+y])+((-1)**B)*math.ceil(delta_I))
						if temp>255:
							Y[i+x][j+y]=255
						elif temp<0:
							Y[i+x][j+y]=0
						else:
							Y[i+x][j+y]=temp
						if abs(temp-int(Y[i+x][j+y]))!=0:
							ε_I_sum+=abs(temp-int(Y[i+x][j+y]))
							ε_I_num+=1
							
			#计算补偿值
			ave_ε_I,ave_ε_O=0,0
			if ε_I_num!=0:
				ave_ε_I=math.ceil(ε_I_sum/(3*k*k//4))
			if ε_O_num!=0:
				ave_ε_O=math.ceil(ε_O_sum/(k*k//4))
				
			#进行补偿
			if ave_ε_I!=0 or ave_ε_O!=0:
				for x in range(k):
					for y in range(k):
						if x>=int(k/4) and x<int(k*3/4) and y>=int(k/4) and y<int(k*3/4):
							if ave_ε_I!=0:
								temp=int(int(Y[i+x][j+y])-((-1)**B)*ave_ε_I)
								if temp>255:
									Y[i+x][j+y]=255
								elif temp<0:
									Y[i+x][j+y]=0
								else:
									Y[i+x][j+y]=temp
						else:
							if ave_ε_O!=0:
								temp=int(int(Y[i+x][j+y])+((-1)**B)*ave_ε_O)
								if temp>255:
									Y[i+x][j+y]=255
								elif temp<0:
									Y[i+x][j+y]=0
								else:
									Y[i+x][j+y]=temp
	return Y

def Y_encode(img,message,k,Lambda):
	"""对完整无损的图片Y分量进行编码嵌入01串"""
	
	width,height=img.shape[0],img.shape[1]
	assert(height==width)
	
	#模块数
	block_num=int(height*width/(k*k))
	#消息长度不足补0
	while len(message)<block_num:
		message+='0'

	#BGR转YUV
	yuv=cv.cvtColor(img,cv.COLOR_BGR2YUV)
	#分离Y,U,V分量
	Y,U,V=cv.split(yuv)
	#改变Y通道
	Y=change(Y,message,k,Lambda)

	#合并Y,U,V分离得到新图片
	new_img_YUV=cv.merge([Y,U,V])
	new_img=cv.cvtColor(new_img_YUV,cv.COLOR_YUV2BGR)
	
	return new_img

def RGB_encode(img,message,k,Lambda):
	"""对完整无损的图片RGB分量进行编码嵌入01串"""
	
	width,height=img.shape[0],img.shape[1]
	assert(height==width)
	
	#模块数
	block_num=int(height*width/(k*k))
	#消息长度不足补0
	while len(message)<block_num:
		message+='0'

	#分离R,G,B分量
	B,G,R=cv.split(img)
	
	#改变RGB通道
	R=change(R,message,k,Lambda)
	G=change(G,message,k,Lambda)
	B=change(B,message,k,Lambda)
	
	#合并Y,U,V分离得到新图片
	new_img=cv.merge([B,G,R])
	
	return new_img

def encode(img,message,k,Lambda,encode_type):
	if encode_type=='Y':
		return Y_encode(img,message,k,Lambda)
	elif encode_type=='RGB':
		return RGB_encode(img,message,k,Lambda)


def decode(img,k,decode_type):
	"""仅测试用的解码"""
	msg=""
	if decode_type=='Y':
		msg=Y_decode(img,k)
	elif decode_type=='RGB':
		msg=RGB_decode(img,k)
	return msg


def Y_decode(img,k):
	"""仅测试用的Y分量解码"""
	width,height=img.shape[0],img.shape[1]
	assert(height==width)

	yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
	Y,_,_ = cv.split(yuv)
							
	#恢复01串
	msg=""
	for i in range(0,height,k):
		for j in range(0,width,k):
			#计算当前为第几个模块
			index=int((i/k)*(width/k))+int(j/k)
			inner,outer=0,0
			for x in range(k):
				for y in range(k):
					if x>=int(k/4) and x<int(k*3/4) and y>=int(k/4) and y<int(k*3/4):
						inner+=int(Y[i+x][j+y])
					else:
						outer+=int(Y[i+x][j+y])
			inner=int(inner/(k*k/4))
			outer=int(outer/(3*k*k/4))
			if inner>=outer:
				msg+='1'
			else:
				msg+='0'
	return msg

def channel_decode(channel,k):
	width,height=channel.shape[0],channel.shape[1]
	assert(height==width)

	#恢复01串
	msg=""
	for i in range(0,height,k):
		for j in range(0,width,k):
			#计算当前为第几个模块
			index=int((i/k)*(width/k))+int(j/k)
			inner,outer=0,0
			for x in range(k):
				for y in range(k):
					if x>=int(k/4) and x<int(k*3/4) and y>=int(k/4) and y<int(k*3/4):
						inner+=int(channel[i+x][j+y])
					else:
						outer+=int(channel[i+x][j+y])
			inner=int(inner/(k*k/4))
			outer=int(outer/(3*k*k/4))
			if inner>=outer:
				msg+='1'
			else:
				msg+='0'
	return msg

def RGB_decode(img,k):
	"""仅测试用的RGB分量解码"""
	
	width,height=img.shape[0],img.shape[1]
	assert(height==width)
	
	#分离B,G,R分量
	B,G,R=cv.split(img)

	#恢复01串
	msg=""
	for i in range(0,height,k):
		for j in range(0,width,k):
			#计算当前为第几个模块
			index=int((i/k)*(width/k))+int(j/k)
			inner_B,outer_B=0,0
			inner_G,outer_G=0,0
			inner_R,outer_R=0,0
			for x in range(k):
				for y in range(k):
					if x>=int(k/4) and x<int(k*3/4) and y>=int(k/4) and y<int(k*3/4):
						inner_B+=int(B[i+x][j+y])
						inner_G+=int(G[i+x][j+y])
						inner_R+=int(R[i+x][j+y])
					else:
						outer_B+=int(B[i+x][j+y])
						outer_G+=int(G[i+x][j+y])
						outer_R+=int(R[i+x][j+y])
			inner_B=int(inner_B/(k*k/4))
			outer_B=int(outer_B/(3*k*k/4))
			inner_G=int(inner_G/(k*k/4))
			outer_G=int(outer_G/(3*k*k/4))
			inner_R=int(inner_R/(k*k/4))
			outer_R=int(outer_R/(3*k*k/4))
			if (inner_B>=outer_B and inner_G>=outer_G) or (inner_B>=outer_B and inner_R>=outer_R) or (inner_G>=outer_G and inner_R>=outer_R):
				msg+='1'
			else:
				msg+='0'
	return msg
