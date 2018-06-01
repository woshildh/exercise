import torch
import torch.nn as nn
from torch.autograd import Variable

class NIN(nn.Module):
	'''
	使用Network in Network 的方式来定义网络
	params:
		classes_num:(int),number of classes
		pool_method:(str),'max' or 'avg'
	'''
	def __init__(self,classes_num=10,pool_method="avg"):
		super(NIN,self).__init__()
		if pool_method=="avg":
			self.pooling=nn.AvgPool2d(kernel_size=(9,9))
		else:
			self.pooling=nn.MaxPool2d(kernel_size=(9,9))

		self.features=nn.Sequential(
			nn.Conv2d(3,192,kernel_size=5,stride=1,padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(192,160,kernel_size=1,stride=1,padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(160,96,kernel_size=1,stride=1,padding=0),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
			nn.Dropout(0.5),

			nn.Conv2d(96,192,kernel_size=5,stride=1,padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(192,192,kernel_size=1,stride=1,padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(192,192,kernel_size=1,stride=1,padding=0),
			nn.ReLU(inplace=True),
			nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
			nn.Dropout(0.5)	,

			nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
			nn.ReLU(inplace=True),
		)
		self.classifier=nn.Sequential(
			self.pooling,
		)
	def forward(self,x):
		x=self.features(x)
		x=self.classifier(x)
		x=x.view(x.size(0),-1)
		return x

