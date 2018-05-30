import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import SOFTMAXCE 

#手动写标签
y_scores=np.array([[1.2,-1.2,0.6,3.5],[2.1,0.9,2.9,13],[0.8,0.9,0.6,1.1]])
y_true=np.array([[0,0,0,1],[0,0,0,1],[0,1,0,0]])
#tensorflow下求loss
tf_y_scores=tf.Variable(y_scores)
tf_y_true=tf.Variable(y_true)

tf_loss=tf.nn.softmax_cross_entropy_with_logits(labels=tf_y_true,logits=tf_y_scores)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(tf_loss))
#torch下求损失
t_y_scores=torch.autograd.Variable(torch.DoubleTensor(y_scores))
t_y_true=torch.autograd.Variable(torch.DoubleTensor(y_true))

l1=SOFTMAXCE.softmax_cross_entropy(t_y_scores,t_y_true,weight=torch.DoubleTensor([1,1,1,1]))
print(l1)

criterion=SOFTMAXCE.OneHotSoftmaxCrossEntropy(weight=torch.DoubleTensor([1,1,1,1]))
l2=criterion(t_y_scores,t_y_true)
print(l2)
