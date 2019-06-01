import numpy as np
from osgeo import gdal
import cv2
import torch
def read_img( file_name='v1.jpg'):
	dataset = gdal.Open(file_name)
	if dataset is None:
		print("Invalid Image Path")
		return None
	img_width = dataset.RasterXSize
	img_height = dataset.RasterYSize
	ibands = dataset.RasterCount
	print("Image Info: width :", img_width, "height: ", img_height, "Band: ", ibands)
	for band in range(ibands):
		band += 1
		print("Getting band: ", band)
		srcband = dataset.GetRasterBand(band)
		if srcband is None:
			continue
		dataraster = srcband.ReadAsArray(0, 0, img_width, img_height).astype(np.uint8)
		print(dataraster.shape)
		if band == 1:
			data = dataraster.reshape((img_height, img_width, 1))
		else:
			data = np.append(data, dataraster.reshape((img_height, img_width, 1)), axis=2)
	return data

def split_image(data,width,height,size = 416):
	images = []
	wc = (width - 416) // 350 + 1
	hc = (height - 416) // 350 + 1
	for i in range(10):
		for j in range(10):
			images.append(data[i * 350:i * 350 + size,j * 350 ,j * 350 + size])
	return images

def cut_image(im_data,filename,left,top,right,bottom):
	assert left <= right
	assert bottom >= top
	#print(im_data.dtype)
	#if torch.uint8 == im_data.dtype:
	#	datatype = gdal.GDT_Byte
	#elif torch.uint16 == im_data.dtype:
	#	datatype = gdal.GDT_UInt16
	#else:
	#	datatype = gdal.GDT_Float32
	out_data = np.array(im_data[left:right,top:bottom,:],dtype=np.uint8)
	r = out_data[:,:,2]
	g = out_data[:,:,1]
	b = out_data[:,:,0]
	img = cv2.merge([r,g,b])
	print(img)
	cv2.imwrite(filename,img)

if __name__ == '__main__':
	read_img('v1.jpg')
