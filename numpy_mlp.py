import numpy as np
import torch.nn as nn
import torch
# grads = {}
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

class MLP(object):
	def __init__(self,in_dim,hidden_dim,out_dim):
		self.shape = [in_dim,hidden_dim,out_dim]
		self.w1 = np.random.uniform(low=-0.5,high=0.5,size=[in_dim,hidden_dim]) # in_dim x out_dim
		self.b1 = np.zeros(shape=[1,hidden_dim])
		self.w2 = np.random.uniform(low=-0.5,high=0.5,size=[hidden_dim,out_dim])
		self.b2 = np.zeros(shape=[1,out_dim])
		self.temp = [] #用于保存中间变量x,hidden
	def forward(self,x):
		'''
		x : batch_size x in_dim
		'''
		self.temp.append(x)
		hidden = self.relu(x.dot(self.w1)+self.b1)
		self.temp.append(hidden)
		out = self.relu(hidden.dot(self.w2)+self.b2)
		return out
	def mse(self,y,label):
		'''
		求mse loss
		'''
		return ((y-label)**2).mean()
	def _mse(self,y,label):
		'''
		对mse loss反传，求y的梯度
		'''
		return (2*y - 2*label) / y.shape[0]
	def relu(self,x):
		'''
		激活函数
		'''
		return x * (x>0)
	def _relu(self,x):
		'''
		返回x的梯度
		'''
		return x>0
	def backward(self,x,y,label):
		'''
		反传求所有参数以及中间变量的梯度
		'''
		self.y_ = self._mse(y,label)
		self.y_before_relu_ = self._relu(y) * self.y_
		self.b2_ = self.y_before_relu_.sum()
		self.w2_ = self.y_before_relu_.T.dot(self.temp[1])
		self.hidden_ = self.y_before_relu_.dot(self.w2.T)
		self.hidden_before_relu_ = self._relu(self.temp[1]) * self.hidden_
		self.b1_ = self.hidden_before_relu_.sum()
		self.w1_ = self.hidden_before_relu_.T.dot(self.temp[0]).T
		self.x_ = self.hidden_before_relu_.dot(self.w1.T)

class MLP_torch(nn.Module):
	def __init__(self,in_dim,hidden_dim,out_dim):
		super(MLP_torch,self).__init__()
		self.shape = [in_dim,hidden_dim,out_dim]
		self.fc1 = nn.Linear(in_dim,hidden_dim)
		self.fc2 = nn.Linear(hidden_dim,out_dim)
	def change_parameters(self,mlp):
		'''
		这个函数目的是将numpy 版本的 mlp的所有参数赋值给pytorch版本的
		'''
		# print(self.fc1.weight.data.size(),self.fc2.weight.data.size(),
		# 	self.fc1.bias.data.size(),self.fc2.bias.data.size())
	
		self.fc1.weight.data = torch.from_numpy(mlp.w1).transpose(0,1)
		self.fc1.bias.data = torch.from_numpy(mlp.b1).squeeze(0)
		self.fc2.weight.data = torch.from_numpy(mlp.w2).transpose(0,1)
		self.fc2.bias.data = torch.from_numpy(mlp.b2).squeeze(0)

		# print(self.fc1.weight.data.size(),self.fc2.weight.data.size(),
		# 	self.fc1.bias.data.size(),self.fc2.bias.data.size())
	def forward(self,x):
		'''
		前传
		'''
		hidden = torch.relu(self.fc1(x))
		d = self.fc2(hidden)
		out = torch.relu(d)
		return out
	def mse(self,y,label):
		'''
		求 mse loss
		'''
		return ((y-label)**2).mean()

if __name__=="__main__":
	#定义numpy的多层感知机
	mlp = MLP(20,50,1)
	#定义pytorch版本的多层感知机并将参数设置为mlp的参数
	mlp_torch = MLP_torch(20,50,1)
	mlp_torch.change_parameters(mlp)
	# 生成x,label
	x = np.random.uniform(low=-0.5,high=0.5,size=(16,20))
	label = np.random.uniform(low=-0.5,high=0.5,size=(16,1))
	# #获取前向的结果以及求损失
	
	y1 = mlp.forward(x)
	loss1 = mlp.mse(y1,label)
	
	y2 = mlp_torch(torch.from_numpy(x))
	loss2 = mlp_torch.mse(y2,torch.from_numpy(label))
	
	print("mlp_numpy version loss: ",loss1,"mlp_pytorch version loss: ",loss2.item())

	#获取反传的梯度是不是一样
	loss2.backward()
	mlp.backward(x,y1,label)
	#输出mlp第一个全连接层weight和bias梯度总和
	print("mlp_numpy version:",mlp.w1_.sum(),mlp.b1_.sum()) 
	#输出mlp_torch第一个全连接层weight和bias阿梯度总和
	print("mlp_pytorch version:",mlp_torch.fc1.weight.grad.sum().item(),
		mlp_torch.fc1.bias.grad.sum().item())
