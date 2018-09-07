import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dst
from torchvision import transforms

use_cuda=False

train_dataset=dst.MNIST(root="./data",train=True,
	transform=transforms.ToTensor(),download=True)
test_dataset=dst.MNIST(root="./data",train=False,
	transform=transforms.ToTensor())

batch_size=100
n_iters=3000
num_epochs=n_iters/(len(train_dataset)/batch_size)
num_epochs=int(num_epochs)

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
	batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
	batch_size=batch_size,shuffle=False)

class LSTMModel(nn.Module):
	def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
		super(LSTMModel,self).__init__()
		self.hidden_dim=hidden_dim
		self.layer_dim=layer_dim
		self.lstm=nn.LSTM(input_dim,hidden_dim,layer_dim,batch_first=True)
		self.fc=nn.Linear(hidden_dim,output_dim)
	def forward(self,x):
		if use_cuda:
			h0=Variable(torch.zeros(self.layer_dim,x.size(0),self.hidden_dim)).cuda()
		else:
			h0=Variable(torch.zeros(self.layer_dim,x.size(0),self.hidden_dim))	
		#initialize cell state
		if use_cuda:
			c0=Variable(torch.zeros(self.layer_dim,x.size(0),self.hidden_dim).cuda())
		else:
			c0=Variable(torch.zeros(self.layer_dim,x.size(0),self.hidden_dim))

		out,(hn,cn)=self.lstm(x,(h0,c0))
		out=self.fc(out[:,-1,:])
		return out

input_dim=28
hidden_dim=100
layer_dim=3
output_dim=10
model=LSTMModel(input_dim,hidden_dim,layer_dim,output_dim)

if use_cuda:
	model=model.cuda()
criterion=nn.CrossEntropyLoss()
learning_rate=0.1
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#Number of steps to unroll
seq_dim=28

iter=0
for epoch in range(num_epochs):
	for i,(images,labels) in enumerate(train_loader):
		if use_cuda:
			images=Variable(images.view(-1,seq_dim,input_dim).cuda())
			labels=Variable(labels.cuda())
		else:
			images=Variable(images.view(-1,seq_dim,input_dim))
			labels=Variable(labels)
		optimizer.zero_grad()

		outputs=model(images)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		iter+=1

		if iter%500==0:
			correct=0
			total=0
			for images,labels in test_loader:
				if use_cuda:
					images=Variable(images.view(-1,seq_dim,input_dim).cuda())
				else:
					images=Variable(images.view(-1,seq_dim,input_dim))
				outputs=model(images)
				_,predicted=torch.max(outputs.data,-1)
				total+=labels.size(0)

				if use_cuda:
					correct+=(predicted.cpu()==labels.cpu()).sum()
				else:
					correct+=(predicted==labels).sum()
			acc=correct/total
			print("Iterion: {}, loss: {}, Acc: {}".format(iter,loss.data[0],acc))
