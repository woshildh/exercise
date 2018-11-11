class Rect():
	'''
	定义矩形:first是左上角,second是右下角
	'''
	def __init__(self):
		self.first=(1,1)
		self.second=(1,1)

def iou(r1,r2):
	'''
	求矩形r1和r2的交并比
	'''
	#先求出两个矩形各自的宽和高
	width1=r1.second[0]-r1.first[0]
	height1=r1.second[1]-r1.first[1]

	width2=r2.second[0]-r2.first[0]
	height2=r2.second[1]-r2.first[1]

	#求出最大的宽和最大的高
	xmax=max(r1.second[0],r2.second[0])
	xmin=min(r1.first[0],r2.first[0])
	ymax=max(r1.second[1],r2.second[1])
	ymin=min(r1.first[1],r2.first[1])

	#求最大宽和最大高
	width=xmax-xmin
	height=ymax-ymin

	#求交集部分的宽和高
	inter_width=width1+width2-width
	inter_height=height1+height2-height

	if inter_height<=0 or inter_width<=0:
		return 0
	else:
		area=inter_height*inter_width
		area1=width1*height1
		area2=width2*height2
		iou_num=area/(area1+area2-area)
		return iou_num

if __name__=="__main__":
	r1=Rect()
	r2=Rect()

	r1.first=(100,98)
	r1.second=(324,217)

	r2.first=(99,101)
	r2.second=(312,213)
	print(iou(r1,r2))

