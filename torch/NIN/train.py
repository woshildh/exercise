import torch
import NIN
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import math

#定义超参数
use_cuda=True
start_epoch=341
epoch_num=50
lr=1e-6
batch_size=64
save_path="./weights/NIN_{}.pth"
load_path="./weights/NIN_340.pth"
log_dir="./tblog/"
pool_method="avg"

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def train():
	'''
	对cifar10进行训练
	'''
	train_transform = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),#转为tensor
                                	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
                                ])
	test_transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
			])
	#加载数据
	trainset=CIFAR10(root="./data/",train=True,transform=train_transform,download=True)
	trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=4)

	testset=CIFAR10(root="./data/",train=False,transform=test_transform,download=True)
	testloader=DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=4)

	#加载模型
	model=NIN.NIN(classes_num=10,pool_method=pool_method)
	if load_path is not None:
		model.load_state_dict(torch.load(load_path))
		print("model weights load succeed . . .")
	else:
		model.apply(weight_init)
		print("params init succeed . . .")

	if use_cuda:
		model=model.cuda()
	#定义优化器
	optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,nesterov=True)
	
	#定义损失函数
	criterion=torch.nn.CrossEntropyLoss()

	#定义写的文件
	writer=SummaryWriter(log_dir=log_dir)
	for i in range(start_epoch,start_epoch+epoch_num):
		print("{} epoch start".format(i))
		train_loss=0
		train_acc=0
		train_step=1
		model.train()
		for x,y in trainloader:
			x=Variable(x)
			y=Variable(y)
			if use_cuda:
				x=x.cuda()
				y=y.cuda()

			y_=model(x) #前向传导
			pred_value,pred_class=torch.max(y_,dim=1)

			step_loss=criterion(y_,y)
			step_acc=torch.eq(pred_class,y).sum().data[0]/batch_size
			
			#反向传播
			optimizer.zero_grad()
			step_loss.backward()
			optimizer.step()

			#输出每一步的结果
			print("{} epoch, {}step, step loss is {},step acc is {}".format(i,train_step,
				step_loss.data[0],step_acc))

			#更新没一步之后的全局记录
			train_loss+=step_loss.data[0]
			train_acc+=step_acc
			train_step+=1
			del(x,y,step_loss,step_acc,pred_class,pred_value)

		torch.save(model.state_dict(),save_path.format(i))
		print("model save succeed...")

		model.eval()
		test_loss=0
		test_acc=0
		test_step=1
		for x,y in testloader:
			x=Variable(x)
			y=Variable(y)
			if use_cuda:
				x=x.cuda()
				y=y.cuda()

			y_=model(x) #前向传导
			pred_value,pred_class=torch.max(y_,dim=1)

			step_loss=criterion(y_,y)
			step_acc=torch.eq(pred_class,y).sum().data[0]/batch_size

			test_loss+=step_loss.data[0]
			test_acc+=step_acc
			test_step+=1
		print("{} epoch, train loss is {},train acc is {},test loss is {} ,test acc is {}".
			format(i,train_loss/train_step,train_acc/train_step,test_loss/test_step,test_acc/test_step))
		
		writer.add_scalar(pool_method+"/Train/loss",train_loss/train_step,global_step=i)
		writer.add_scalar(pool_method+"/Train/acc",train_acc/train_step,global_step=i)
		writer.add_scalar(pool_method+"/Test/loss",test_loss/test_step,global_step=i)
		writer.add_scalar(pool_method+"/Test/acc",test_acc/test_step,global_step=i)

if __name__=="__main__":
	train()

