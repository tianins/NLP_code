import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class MultiHeadAtten(nn.Module):
    def __init__(self,d_model,head_nums) -> None:
        super(MultiHeadAtten,self).__init__()
        self.d_model = d_model
        self.head_nums = head_nums
        self.d_k = d_model // head_nums
        
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        
    def split_heads(self,x,bs):
        return x.view(bs,-1,self.head_nums,self.d_k).transpose(1,2)
    
    def forward(self,q,k,v):
        bs = q.shape[0]
        q = self.W_q(q)
        k = self.W_q(k)
        v = self.W_q(v)
        
        q = self.split_heads(q,bs)
        k = self.split_heads(k,bs)
        v = self.split_heads(v,bs)
        
        qk = torch.matmul(q,k.transpose(-1,-2))/ math.sqrt(self.d_k)
        
        score = F.softmax(qk,dim=-1)
        
        qkv = torch.matmul(score,v)
        
        qkv_view = qkv.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        
        atten_out = self.W_o(qkv_view)
        
        return atten_out
d_model = 768
num_heads = 12
bs = 10
max_len = 10
max_len = 10
max_len = 10
q = torch.randn(bs,max_len,d_model)
k = torch.randn(bs,max_len,d_model)
v = torch.randn(bs,max_len,d_model)
multiheadatten = MultiHeadAtten(d_model,num_heads)

atten = multiheadatten(q,k,v)

print(atten)
        
        
        
        
        
        
        
        