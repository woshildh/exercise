import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

x=Variable(torch.randn(3,5))
y=Variable(torch.LongTensor(3).random_(5))


loss = torch.nn.CrossEntropyLoss()(x,y)
print(loss)
x=F.softmax(x,dim=1)
loss= torch.nn.CrossEntropyLoss()(x,y)
print(loss)


prob = F.log_softmax(x,dim=1)
crossentropy = F.nll_loss(prob,y)
print(crossentropy)

prob = F.softmax(x,dim=1)
log_prob = torch.log(prob)
loss=-1/3.0*(log_prob[0,y[0].data[0]]+log_prob[1,y[1].data[0]]+log_prob[2,y[2].data[0]])
print(loss)



