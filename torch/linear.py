import torch
from torch.autograd import Function
from torch.nn.functional import linear
from torch.autograd import Variable
import torch.nn.functional as F
class MyLinear(Function):
	def forward(ctx,inputs,weights,bias=None):
		ctx.save_for_backward(inputs,weights,bias)
		output = torch.mm(inputs,weights)
		if bias is not None:
			output += bias.unsqueeze(0).expand_as(output)
		return output
	def backward(ctx,grad_out):
		inputs,weights,bias=ctx.saved_tensors
		grad_inputs = grad_weight = grad_bias = None		
		grad_inputs = grad_out.mm(weights.t())
		grad_weights = inputs.t().mm(grad_out)
		if bias is not None:
			grad_bias=grad_out.sum(dim=0).squeeze(0)
		return grad_inputs,grad_weights,grad_bias


if __name__=="__main__":
	linear=MyLinear()
	x=Variable(torch.randn(10,5),requires_grad=True)
	x.register_hook(lambda grad:print(grad))
	weights=Variable(torch.randn(5,2),requires_grad=True)
	bias=Variable(torch.randn(2),requires_grad=True)

	y=linear(x,weights,bias)
	loss=y.sum()
	loss.backward()
	#清空梯度
	bias.grad.data.zero_()
	weights.grad.data.zero_()
	x.grad.data.zero_()
	#先将x和weights转一下形状
	x=x.unsqueeze(dim=0)
	weights=weights.t()
	#求y并且进行反传
	y=F.linear(x,weights,bias)
	loss=y.sum()
	loss.backward()
