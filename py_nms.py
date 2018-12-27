import numpy as np
class Rect():
	'''
	定义矩形:first是左上角,second是右下角
	'''
	def __init__(self,first,second):
		self.first=first
		self.second=second

def calc_iou(r1,r2):
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

def py_nms(dets,thresh):
	scores = dets[:, 4]
	#获取根据打分获得的排名
	order=scores.argsort()[::-1]
	print(order)
	save_dets=[]
	save_dets.append(dets[order[0]])
	#从下标1开始淘汰
	for i in order[1:]:
		max_iou=0
		for det in save_dets:
			r1=Rect((det[0],det[1]),(det[2],det[3]))
			r2=Rect((dets[i][0],dets[i][1]),(dets[i][2],dets[i][3]))
			iou=calc_iou(r1,r2)
			max_iou=max(max_iou,iou)
		if max_iou<thresh:
			save_dets.append(dets[i])
	return save_dets
if __name__=="__main__":
	dets = np.array([[30, 20, 230, 200, 1], 
					 [50, 50, 260, 220, 0.9],
					 [430, 280, 460, 360, 0.7],
					 [210, 30, 420, 5, 0.8]
					 ])
	thresh = 0.35
	keep_dets = py_nms(dets, thresh)
	print(keep_dets)



