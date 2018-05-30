'''
定义one-hot形式的softmax交叉熵
包括函数和类方法
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

class OneHotSoftmaxCrossEntropy(nn.Module):
	'''
	functional:
		实现one_hot形式的交叉熵
	params:
		weight:定义各个类别
	'''
	def __init__(self,weight=None):
		'''
		params:
			weight:代表 n 类的权重, a Tensor(float or double) , shape:[class_num]
		'''
		super(OneHotSoftmaxCrossEntropy,self).__init__()
		if weight is not None:
			self.weight=torch.autograd.Variable(weight.view(1,-1),requires_grad=False)
		else:
			self.weight=None
	def forward(self,inputs,labels):
		'''
		functional:
			实现one_hot形式的交叉熵
		params:
			inputs:预测的值
			labels:标签,类型也需要是float
		return:
			loss:损失，一个矩阵(batch_size)
		'''
		inputs=torch.clamp(F.softmax(inputs,dim=1),min=1e-8,max=1-1e-8)
		inputs=torch.clamp(torch.log(inputs),min=-100,max=1-1e-8)
		loss=-1*labels*inputs
		
		if self.weight is not None:
			loss=loss*self.weight   #将weight乘到里面
		loss=torch.sum(loss,dim=1)
		
		return loss

def softmax_cross_entropy(inputs,labels,weight=None):
	'''
	functional:
		实现one_hot形式的交叉熵
	params:
		inputs:预测的值, 类型需要是float(or double), a Variable
		labels:标签,    类型需要是float(or double) ,a Variable
		weight:定义了各个类别的weight,类型需要是float(or double), a Tensor
	return:
		loss:损失，一个矩阵(batch_size)
	'''

	inputs= torch.clamp(F.softmax(inputs),min=1e-10,max=1-1e-10) #进行softmax操作并且截断
	inputs=torch.clamp(torch.log(inputs),min=-100,max=0)
	loss=-labels * inputs
	
	if weight is not None:
		weight=torch.autograd.Variable(weight,requires_grad=False) #将weight转为Variable否则无法计算
		weight=weight.view(1,-1)
		loss=loss*weight   #将weight乘到里面
	
	loss=torch.sum(loss,dim=1)
	return loss
	
