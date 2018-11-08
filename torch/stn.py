import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import SGD

use_cuda=False
epochs=20
best_acc=0
net_type="net"

class STNMNIST(nn.Module):
	def __init__(self):
		super(STNMNIST, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

		# 空间转换本地网络 (Spatial transformer localization-network)
		self.localization = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=7),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True),
			nn.Conv2d(8, 10, kernel_size=5),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True)
		)

		# 3 * 2 仿射矩阵 (affine matrix) 的回归器
		self.fc_loc = nn.Sequential(
			nn.Linear(10 * 3 * 3, 32),
			nn.ReLU(True),
			nn.Linear(32, 3 * 2)
		)

		# 用身份转换 (identity transformation) 初始化权重 (weights) / 偏置 (bias)
		self.fc_loc[2].weight.data.fill_(0)
		self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

	# 空间转换网络的前向函数 (Spatial transformer network forward function)
	def stn(self, x):
		xs = self.localization(x)
		xs = xs.view(-1, 10 * 3 * 3)
		theta = self.fc_loc(xs)
		theta = theta.view(-1, 2, 3)

		grid = F.affine_grid(theta, x.size())
		x = F.grid_sample(x, grid)

		return x

	def forward(self, x):
		# 转换输入
		x = self.stn(x)

		# 执行常规的正向传递
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		# 执行常规的正向传递
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

net=STNMNIST()

if use_cuda:
	net=net.cuda()
transform=transforms.Compose([transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))])
train_loader=DataLoader(MNIST("./data/",download=True,transform=transform),batch_size=128,shuffle=True,
	   num_workers=4)
test_loader=DataLoader(MNIST("./data/",download=True,train=False,transform=transform),batch_size=128,shuffle=True,
	   num_workers=4)

def accuracy(logits,labels):
	batch_size=labels.size(0)
	_,max_pos=torch.max(logits,dim=1)
	max_pos=max_pos.squeeze()
	correct_num=torch.eq(max_pos,labels).sum().data[0]
	acc=correct_num/batch_size
	return acc

op=SGD(net.parameters(),lr=0.01,momentum=0.9)

def main():
	for i in range(epochs):
		print("{} epoch start".format(i))
		step=0
		total_loss=0
		total_acc=0
		for x,y in train_loader:
			x,y=Variable(x),Variable(y)
			if use_cuda:
				x=x.cuda()
				y=y.cuda()
			y_=net(x)
			step_loss=F.nll_loss(y_,y)
			step_acc=accuracy(y_,y)
			op.zero_grad()
			step_loss.backward()
			op.step()
			if step%100==0:
				print("{} epoch,{} step,acc is {:.4f},loss is {:.6f}".format(i,step,step_acc,step_loss.data[0]))
			total_loss+=step_loss.data[0]
			total_acc+=step_acc
			step+=1
		avg_acc=total_acc/step
		avg_loss=total_loss/step
		print("{} epoch train end,acc is {:.4f},loss is {:.6f}".format(i,avg_acc,avg_loss))

		step=0
		total_loss=0
		total_acc=0
		for x,y in test_loader:
			x,y=Variable(x),Variable(y)
			if use_cuda:
				x=x.cuda()
				y=y.cuda()
			y_=net(x)
			step_loss=F.nll_loss(y_,y)
			step_acc=accuracy(y_,y)
			total_loss+=step_loss.data[0]
			total_acc+=step_acc
			step+=1
		avg_acc=total_acc/step
		avg_loss=total_loss/step
		print("{} epoch test end,acc is {:.4f},loss is {:.6f}".format(i,avg_acc,avg_loss))

		if best_acc<avg_acc:
			best_acc=avg_acc

	print("mnist train end,best acc is {}".format(best_acc))

if __name__=="__main__":
	main()

