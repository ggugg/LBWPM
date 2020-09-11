from encode import *

if __name__ == '__main__':
	k=4
	Lambda=25
	Capacity='low'
	encode_type='Y'
	change_size=29*k
	message=messages[Capacity]
					
	#读取大小为height×width的图片
	original=cv.imread("./lena.jpg",-1)
	width,height=original.shape[0],original.shape[1]
	assert(width==512)
	assert(height==512)
	original_size=width
	
	#对图片进行缩放
	original_change=original
	if width>change_size:
		#缩小
		original_change=cv.resize(original_change,(change_size,change_size),interpolation=cv.INTER_AREA)
	elif width<change_size:
		#放大
		original_change=cv.resize(original_change,(change_size,change_size),interpolation=cv.INTER_CUBIC)
	
	#编码
	img_no_location=encode(original_change,message,k,Lambda,encode_type)
	
	#添加定位图案
	image_result=add_location(img_no_location,k)
	if image_result.shape[0]>original_size:
		#缩小
		image_result=cv.resize(image_result,(original_size,original_size),interpolation=cv.INTER_AREA)
	elif image_result.shape[0]<original_size:
		#放大
		image_result=cv.resize(image_result,(original_size,original_size),interpolation=cv.INTER_CUBIC)
	
	#保存
	cv.imwrite("./result.jpg",image_result)
	
