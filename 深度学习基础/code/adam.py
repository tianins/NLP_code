#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import copy

"""
基于pytorch的网络编写
手动实现梯度计算和反向传播
加入激活函数
"""

class TorchModel(nn.Module):
    def __init__(self, hidden_size):
        super(TorchModel, self).__init__()
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss  #loss采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.layer(x)
        y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


#自定义模型，接受一个参数矩阵作为入参
class DiyModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        y_pred = np.dot(self.weight, x)
        y_pred = self.diy_sigmoid(y_pred)
        if y is not None:
            return self.diy_mse_loss(y_pred, y)
        else:
            return y_pred

    #sigmoid
    def diy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #手动实现mse，均方差loss
    def diy_mse_loss(self, y_pred, y_true):
        return np.sum(np.square(y_pred - y_true)) / len(y_pred)

    #手动实现梯度计算
    def calculate_grad(self, y_pred, y_true, x):
        #前向过程
        # wx = np.dot(self.weight, x)
        # sigmoid_wx = self.diy_sigmoid(wx)
        # loss = self.diy_mse_loss(sigmoid_wx, y_true)
        #反向过程
        # 均方差函数 (y_pred - y_true) ^ 2 / n 的导数 = 2 * (y_pred - y_true) / n
        grad_loss_sigmoid_wx = 2/len(x) * (y_pred - y_true)
        # sigmoid函数 y = 1/(1+e^(-x)) 的导数 = y * (1 - y)
        grad_sigmoid_wx_wx = y_pred * (1 - y_pred)
        # wx对w求导 = x
        grad_wx_w = x
        #导数链式相乘
        grad = grad_loss_sigmoid_wx * grad_sigmoid_wx_wx
        grad = np.dot(grad.reshape(len(x),1), grad_wx_w.reshape(1,len(x)))
        return grad

#梯度更新
def diy_sgd(grad, weight, learning_rate):
    return weight - learning_rate * grad

#adam梯度更新
def diy_adam(grad, weight):
    #参数应当放在外面，此处为保持后方代码整洁简单实现一步
    alpha = 1e-3  #学习率
    beta1 = 0.9   #超参数
    beta2 = 0.999 #超参数
    eps = 1e-8    #超参数
    t = 0         #初始化
    mt = 0        #初始化
    vt = 0        #初始化
    #开始计算
    t = t + 1
    gt = grad
    mt = beta1 * mt + (1 - beta1) * gt
    vt = beta2 * vt + (1 - beta2) * gt ** 2
    mth = mt / (1 - beta1 ** t)
    vth = vt / (1 - beta2 ** t)
    weight = weight - (alpha / (np.sqrt(vth) + eps)) * mth
    return weight

x = np.array([1, 2, 3, 4])  #输入
y = np.array([0.1,-0.1,0.01,-0.01])  #预期输出

#torch实验
torch_model = TorchModel(len(x))
torch_model_w = torch_model.state_dict()["layer.weight"]
print(torch_model_w, "初始化权重")
numpy_model_w = copy.deepcopy(torch_model_w.numpy())
#numpy array -> torch tensor, unsqueeze的目的是增加一个batchsize维度
torch_x = torch.from_numpy(x).float().unsqueeze(0)
torch_y = torch.from_numpy(y).float().unsqueeze(0)
#torch的前向计算过程，得到loss
torch_loss = torch_model(torch_x, torch_y)
print("torch模型计算loss：", torch_loss)
# #手动实现loss计算
diy_model = DiyModel(numpy_model_w)
diy_loss = diy_model.forward(x, y)
print("diy模型计算loss：", diy_loss)

# #设定优化器
learning_rate = 0.1
optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(torch_model.parameters())
optimizer.zero_grad()
#
# #pytorch的反向传播操作
torch_loss.backward()
print(torch_model.layer.weight.grad, "torch 计算梯度")  #查看某层权重的梯度

# #手动实现反向传播
grad = diy_model.calculate_grad(diy_model.forward(x), y, x)
print(grad, "diy 计算梯度")
#
# #torch梯度更新
# optimizer.step()
# #查看更新后权重
# update_torch_model_w = torch_model.state_dict()["layer.weight"]
# print(update_torch_model_w, "torch更新后权重")
#
# #手动梯度更新
# diy_update_w = diy_sgd(grad, numpy_model_w, learning_rate)
# diy_update_w = diy_adam(grad, numpy_model_w)
# print(diy_update_w, "diy更新权重")