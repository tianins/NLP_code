import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class MultHeadAtten(nn.Module):
    def __init__(self,d_model,head_nums) -> None:
        super(MultHeadAtten,self).__init__()
        
        self.d_model = d_model
        self.head_nums = head_nums
        self.d_k = d_model // head_nums
        
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        
    def split_heads(self,bs,x):
        return x.view(bs,-1,self.head_nums,self.d_k).transpose(1,2)
    
    def forward(self,q,k,v,mask=None):
        bs = q.shape[0]
        
        Q = self.W_q(q)
        K = self.W_q(k)
        V = self.W_q(v)
        
        Q = self.split_heads(bs,Q)
        K = self.split_heads(bs,K)
        V = self.split_heads(bs,V)
        
        QK = torch.matmul(Q,K.transpose(-1,-2)) / math.sqrt(self.d_k)
        
        if mask is not None:
            QK = QK.masked_fill(mask.to(QK.dtype()) == 0, float('-inf'))
        
        score = f.softmax(QK,dim=-1)
        
        atten = torch.matmul(score,V)
        
        atten_concat = atten.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        
        atten_out = self.W_o(atten_concat)
        
        return atten_out
    
bs = 24
d_model = 768
nums_heads = 12
max_len = 18

q = torch.randn(bs,max_len,d_model)
k = torch.randn(bs,max_len,d_model)
v = torch.randn(bs,max_len,d_model)

multheadatten = MultHeadAtten(d_model,nums_heads)

atten = multheadatten(q,k,v)
print(atten)
        
        