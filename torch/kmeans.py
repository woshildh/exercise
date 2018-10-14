import torch
import numpy as np
import matplotlib.pyplot as plt
#from pairwise import pairwise_distance

def pairwise_distance(data1, data2=None):
	r'''
	using broadcast mechanism to calculate pairwise ecludian distance of data
	the input data is N*M matrix, where M is the dimension
	we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
	then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
	'''
	if data2 is None:
		data2 = data1 

	#N*1*M
	A = data1.unsqueeze(dim=1)

	#1*N*M
	B = data2.unsqueeze(dim=0)
	dis = (A-B)**2
	#return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis
'''
def forgy(X, n_clusters):
	_len = len(X)
	indices = np.random.choice(_len, n_clusters)
	indices=torch.from_numpy(indices).long()
	if isinstance(X,torch.FloatTensor):
		initial_state = X[indices]
	else:
		d=X.get_device()
		indices=indices.cuda(d)
		initial_state = X[indices]
	print(initial_state)
	return initial_state
'''
def forgy(X, n_clusters):
	initial_state=torch.Tensor([[0.1],[0.3],[0.5]])
	try:
		d=X.get_device()
		initial_state=initial_state.cuda(d)
	except:
		pass
	return initial_state
def lloyd(X, n_clusters, tol=1e-4,max_iter=100):
	X=X.float()
	initial_state = forgy(X, n_clusters)
	ite=0
	while ite < max_iter:
		dis = pairwise_distance(X, initial_state)

		_,choice_cluster = torch.min(dis, dim=1)

		initial_state_pre = initial_state.clone()

		for index in range(n_clusters):
			selected = torch.nonzero(choice_cluster==index).squeeze()
			try:
				selected = torch.index_select(X, 0, selected)
			except:
				continue
			initial_state[index] = selected.mean(dim=0)
		

		center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
		ite+=1
		if (center_shift ** 2) < tol:
			break
	return choice_cluster

if __name__=="__main__":
	'''
	x=np.array([[1,1],[1,2],[2,2],[7,7],[7,9],[7,8],[2,1]])
	x=torch.autograd.Variable(torch.from_numpy(x).float())
	y=lloyd(x,2)
	for i in y:
		print(i)
	'''
	'''
	x_=[np.random.randint(1,4,size=(100,2)),
			np.random.randint(3,9,size=(100,2))]
	x_=np.concatenate(x_,axis=0)
	x=torch.from_numpy(x_).float()
	x=torch.autograd.Variable(x)
	y=lloyd(x,2)
	y=y.data.numpy()
	for i in range(200):
		if y[i]==0:
			plt.scatter(x=x_[i][0],y=x_[i][1],color="red")
		else:
			plt.scatter(x=x_[i][0],y=x_[i][1],color="blue")
	plt.show()
	'''
	map=[0.0614 , 0.0863 , 0.1674 , 0.2148 , 0.1976 , 0.1272 , 0.0755,
	 0.0790 , 0.3088 , 0.5166 , 0.6803 , 0.6498  ,0.4936,  0.2488,
	 0.1736 , 0.5978 , 0.7747 , 0.9600 , 0.8347 , 0.6638,  0.4086,
	 0.2033 ,0.7051 , 0.8608  ,1.0836 , 0.9940 ,0.7293  ,0.4526,
	 0.1985 , 0.5771 , 0.7650,  0.8851 , 0.8256 , 0.7254  ,0.4683,
	 0.0996 , 0.2346 , 0.3589 , 0.4998  ,0.4523 , 0.3625  ,0.2312,
	 0.1077 , 0.1074 , 0.1156 , 0.1233 , 0.1230  ,0.1334 , 0.1351]
	map = torch.Tensor(map).view(-1,1)
	print(map.mean())
	print(map.size())
	res=lloyd(map,3)
	print(res.view(7,7))

