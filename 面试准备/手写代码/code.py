import torch
import torch.nn as nn
import numpy as np
# 用torch实现欧式距离

# Euclidean distance = sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

def euclidean_disteance(x1,x2):
    return torch.sqrt(torch.sum((x1-x2)**2))

x1 = torch.tensor([1,2,3])
x2 = torch.tensor([4,5,6])

# dis = euclidean_disteance(x1,x2)
# print(dis)

# 手动实现softmax
"""
假设我们有一个矩阵 matrix，其中有两行（每行代表一个样本）和三列（每列代表一个类别的得分）。当我们调用 softmax(matrix) 时，首先会计算指数，然后对每行的元素进行求和。axis=1 表示我们要沿着第二个轴（即水平方向）对每行的元素进行求和，这样就得到了每个样本的总指数。keepdims=True 表示保持求和结果的维度和输入矩阵相同，这样在后续计算中可以正确地进行广播操作。

最终，我们将指数除以求和结果，得到每个元素的 softmax 值。
"""
def softmax_n(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

def softmax_t(matrix):
    return torch.exp(matrix) / torch.sum(torch.exp(matrix),dim=1,keepdim=True)
#假设有3个样本，每个都在做3分类
pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.2, 0.9],
                          [0.5, 0.4, 0.2]])
#正确的类别分别为1,2,0
target = torch.LongTensor([1, 2, 0])

# print(softmax_n(pred.numpy()))
# print(softmax_t(pred))
# print(torch.softmax(pred,dim=1))
ce_loss = nn.CrossEntropyLoss()
# loss = ce_loss(pred,target)
# print(loss)

def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i,t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target

"""
交叉熵理解：
交叉熵的本质是 对数损失，pytorh中的实现是，对于每一个样本来说，计算预测为真实
标签的对数损失：-1*log(预测该值为真实标签的概率)
以常见的01分类为例：
    当预测标签为0时，此时的损失为 -1*(log(预测为0的概率))
    当预测标签为1时，此时的损失为 -1*(log(预测为1的概率))
    组合起来就是-1*[真实标签 * log(预测为1的概率) + （1-真实标签） * log(1-预测为1的概率)]
"""
def cross_entropy(pred,target):
    bs,class_nums = pred.shape
    pred = softmax_n(pred)
    target = to_one_hot(target,pred.shape)
    entropy = -np.sum(target*np.log(pred), axis=1)
    return sum(entropy)/bs

def cross_entropy_t(pred,traget):
    bs = pred.shape[0]
    pred = softmax_t(pred)
    traget = to_one_hot(traget,pred.shape)
    traget = torch.tensor(traget)
    entropy = -torch.sum(traget*torch.log(pred),dim=1)
    return sum(entropy)/bs

# print(cross_entropy(pred.numpy(),target.numpy()))
# print(cross_entropy_t(pred,target))


# 手动实现BN
# 例子
input_size = 10
batch_size = 2
max_len = 10

epsilon = 1e-5
a1 = torch.randn(input_size)
b1 = torch.randn(input_size)
def batch_norm(input_tensor):
    mean = torch.mean(input_tensor,dim=0,keepdim=True)
    variance = torch.var(input_tensor,dim=0,keepdim=True)
    
    normalized_input = a1 * (input_tensor-mean)/torch.sqrt(variance+epsilon) + b1
    return normalized_input

def layer_norm(input_tensor):
    mean = torch.mean(input_tensor,dim=-1,keepdim=True)
    variance = torch.var(input_tensor,dim=-1,keepdim=True)
    
    normalized_input = a1 * (input_tensor-mean)/torch.sqrt(variance+epsilon) + b1
    return normalized_input


input_tensor = torch.randn(batch_size, max_len, input_size)
batch_norm_out = batch_norm(input_tensor)
layer_norm_out = layer_norm(input_tensor)

batch_norm_torch = nn.BatchNorm1d(max_len)
layer_norm_torch = nn.LayerNorm(input_size)
print(batch_norm_torch(input_tensor))
print(batch_norm_out)
print("----------------------")
print(layer_norm_torch(input_tensor))
print(layer_norm_out)


