from torch import nn
import torch.nn.functional as F
import torch
import math
class MultiheadAtten(nn.Module):
    def __init__(self,d_model,head_nums):
        super(MultiheadAtten,self).__init__()
        
        self.d_model = d_model
        self.head_nums = head_nums
        self.d_k = d_model//head_nums
        
        self.W_q = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        
    def split_heads(self,x,bs):
        return x.view(bs,-1,self.head_nums,self.d_k).transpose(1,2)
    def forward(self,q,k,v):
        bs = q.shape[0]
        
        q = self.W_q(q)
        k = self.W_q(k)
        v = self.W_q(v)
        # 分割
        q = self.split_heads(q,bs)
        k = self.split_heads(k,bs)
        v = self.split_heads(v,bs)
        
        # 计算多头注意力
        
        qk = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(self.d_k)
        
        qk = F.softmax(qk,dim=-1) # max_len*max_len的注意力得分
        
        qkv = torch.matmul(qk,v)
        
        # 合并多头
        
        qkv = qkv.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        
        atten_out = self.W_o(qkv)
        
        return atten_out
    
bs = 2
d_model = 768
max_len = 10
num_heads = 12

multiheadatten = MultiheadAtten(d_model,num_heads)

q = torch.randn(bs,max_len,d_model)
k = torch.randn(bs,max_len,d_model)
v = torch.randn(bs,max_len,d_model)

atten_out = multiheadatten(q,k,v)

print(atten_out)


        