import warnings

from grid import *

warnings.filterwarnings("ignore")
from util import *
import os


class Point():
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def add(self, p):
		assert isinstance(p, Point)
		p1 = Point(self.x + p.x, self.y + p.y)
		return p1
	
	def minus(self, p):
		assert isinstance(p, Point)
		p1 = Point(self.x - p.x, self.y - p.y)
		return p1
	
	def divide(self, a):
		p1 = Point(self.x / a, self.y / a)
		return p1
	
	def times(self, a):
		p1 = Point(self.x * a, self.y * a)
		return p1
	
	def __str__(self):
		return str(self.x) + ' ' + str(self.y)


class PointPair:
	def __init__(self, point1, point2, type):
		self.point1 = point1
		self.point2 = point2
		self.type = type
	
	def get_center(self):
		return self.point1.add(self.point2).divide(2.0)
	
	def get_distance(self):
		return self.point2.minus(self.point1)


class Image():
	def __init__(self, data, position, pointlist, shape=416):
		self.data = data
		self.position = position  # 左上角
		self.pointlist = pointlist
		self.type = type
		self.shape = shape
	
	def transform_format(self):
		output = []
		for i in self.pointlist:
			point1 = i.point1
			point2 = i.point2
			# print("format")
			# print(self.position,point1, point2)
			type = i.type
			l = []
			l.append(type - 1)
			
			center = (point1.add(point2).divide(2).minus(self.position).divide(float(self.shape)))
			center_x, center_y = center.x, center.y
			r = (point2.minus(point1).divide(float(self.shape)))
			w, h = r.x, r.y
			# print(center,r)
			
			l.append(center_x)
			l.append(center_y)
			l.append(w)
			l.append(h)
			output.append(l)
		# print(output)
		return output


class dataset():
	
	# self.img:: np.array
	# self.tensor torch.tensor
	# self.data labels
	def __init__(self, csv_file='v1.csv', img_path='v1.jpg', split_size=416, transform=None):
		self.transform = transform
		self.split_size = split_size
		self.loadImage(img_path=img_path)
		self.loadCsv(csv_file)
	
	def loadImage(self, img_path='v1.jpg'):
		self.img = read_img(img_path)
		
		if self.img is None:
			print("None Image")
			return
		if self.transform:
			self.img = self.transform(self.img)
		print("Load finished")
		print("Image Shape:", self.img.shape)
		self.tensor = torch.from_numpy(self.img)
	
	def loadCsv(self, csv_file='v1.csv'):
		file = open(csv_file, 'r')
		line = file.readline()
		self.data = []
		count = 0
		while (True):
			count += 1
			line = file.readline()
			if (line == ''):
				break
			line_data = list(map(int, line.split(',')))
			self.data.append(line_data[1:])
		
		print("csv File Read Finished")
		self.data = np.array(self.data)
		print("Data size:", self.data.shape)
	
	def split_for_train(self, pos, output_dir):
		l = self.img.shape[0] // self.split_size
		h = self.img.shape[1] // self.split_size
		self.img_list = []
		self.length = l * h
		for i in range(l):
			for j in range(h):
				position = Point(i * self.split_size, j * self.split_size)
				img_data = self.img[i * self.split_size:(i + 1) * self.split_size,
				           j * self.split_size:(j + 1) * self.split_size]
				r = self.filter(i * self.split_size, j * self.split_size, (i + 1) * self.split_size,
				                (j + 1) * self.split_size)
				# get trees in this rect
				r = np.array(r).reshape((-1, 5))
				pointlist = []
				# transform into pointlist
				for k in range(r.shape[0]):
					data = r[k, :]
					point1 = Point(data[0], data[1])
					point2 = Point(data[2], data[3])
					pointpair = PointPair(point1, point2, data[4])
					pointlist.append(pointpair)
				p = Image(img_data, position, pointlist)
				self.img_list.append(p)
		self.save_to_dir(output_dir)
	
	def split_for_other(self,pos, output_dir):
		assert isinstance(pos,Point)
		po_x = pos.x
		po_y = pos.y
		l = self.img.shape[0] * 3 // 2 // self.split_size
		h = self.img.shape[1] * 3 // 2 // self.split_size
		self.img_list = []
		self.length = l * h
		for i in range(l):
			for j in range(h):
				left = i * self.split_size // 2
				right = (i + 2) * self.split_size // 2
				top = j * self.split_size // 2
				bottom = (j + 2) * self.split_size // 2
				
				
				
				img_data = self.img[left:right, top:bottom]
				left += po_x
				right += po_x
				top += po_y
				bottom += po_y
				position = Point(left, top)
				
				#shape split_size * split_size * 3
				r = self.filter(left, top, right, bottom)
				
				# get trees in this rect
				r = np.array(r).reshape((-1, 5))
				pointlist = []
				# transform into pointlist
				for k in range(r.shape[0]):
					data = r[k, :]
					point1 = Point(data[0], data[1])
					point2 = Point(data[2], data[3])
					pointpair = PointPair(point1, point2, data[4])
					pointlist.append(pointpair)
				p = Image(img_data, position, pointlist)
				self.img_list.append(p)
		self.save_to_dir(output_dir)
	
	def save_to_dir(self, dir):
		if os.path.exists(dir):
			print("Directory already exists")
		else:
			os.mkdir(dir)
			print("Mkdir succeed")
		
		if not dir.endswith('/'):
			dir += '/'
		
		print(dir)
		self.img_path = []
		for num in range(self.length):
			prefix = dir + 'train_' + str(num)
			filename = prefix + '.jpg'
			self.img_path.append(filename)
			image = self.img_list[num]
			out_data = image.data
			r = out_data[:, :, 2]
			g = out_data[:, :, 1]
			b = out_data[:, :, 0]
			img = cv2.merge([r, g, b])
			cv2.imwrite(filename, img)
			data = image.transform_format()
			labelname = prefix + '.txt'
			with open(labelname, 'w') as f:
				for k in range(len(data)):
					out = list(map(str, data[k]))
					output = ' '.join(out) + '\n'
					f.write(output)
	
	#
	# def __len__(self):
	# 	return self.length
	#
	# def __getitem__(self, idx):
	# 	img = torch.from_numpy(self.img_list[idx])
	# 	landmarks = torch.from_numpy(np.array(self.data_list[idx], dtype=np.uint8))
	# 	return img, landmarks
	#
	def filter(self, left, top, right, bottom, Print=False):
		assert left <= right and bottom >= top
		output = []
		r = np.array([left, top, right, bottom])
		for i in range(self.data.shape[0]):
			if rect_in(self.data[i], r):
				output.append(self.data[i])
		output = np.array(output)
		if Print:
			print("Filter shape:", output.shape)
		return output
	
	def filter_point(self,point1,point2):
		assert isinstance(point1,Point)
		assert isinstance(point2,Point)
		return self.filter(point1.x,point1.y,point2.x,point2.y)
	
	def draw(self, left, bottom, right, top, label, color):
		try:
			c1 = (left, bottom)
			c2 = (right, top)
			# print(c1,c2,self.img)
			cv2.rectangle(self.img, c1, c2, color, 1)
			t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
			c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
			cv2.rectangle(self.img, c1, c2, color, -1)
			cv2.putText(self.img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
		except Exception as e:
			print("Draw Result Failed.")
			print(e)
	
	def create_img_list_file(self, filename, path_prefix):
		if not path_prefix.endswith('/'):
			if path_prefix != '':
				path_prefix += '/'
		with open(filename, 'w') as f:
			for i in self.img_path:
				f.write(i)
				f.write('\n')


if __name__ == '__main__':
	point1 = Point(8320,0)
	point2 = Point(16640,8320)
	
	
	ds = dataset(img_path='valid.jpg')
	print(ds.tensor.shape)
	# ds = dataset(img_path='cut.jpg')
	
	# cut_image(ds.img,'valid.jpg',8320,0,16640,8320)
	
	ds.data = ds.filter_point(point1,point2)
	print(ds.data[:5])
	ds.split_for_other(point1,'test/images')
	ds.create_img_list_file('data/palm_test.txt', '')
