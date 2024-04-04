import torch 
import torch.nn as nn
import torch.nn.functional as f
import math

class MultHeadAtten(nn.Module):
    def __init__(self,d_model,head_nums) -> None:
        super(MultHeadAtten,self).__init__()
        self.d_model = d_model
        self.head_nums = head_nums
        self.d_k = d_model//head_nums
        
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        
    def split_heads(self,x,bs):
        return x.view(bs,-1,self.head_nums,self.d_k).transpose(1,2)
    
    def forward(self,q,k,v,mask=None):
        bs = q.shape[0]
        
        q = self.W_q(q)
        k = self.W_q(k)
        v = self.W_q(v)
        
        q = self.split_heads(q,bs)
        k = self.split_heads(k,bs)
        v = self.split_heads(v,bs)
        
        qk = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(self.d_k)
        
        if mask is not None:
            qk = qk.masked_fill(mask.to(qk.dtype()) == 0, float('-inf'))
            
        score = f.softmax(qk,dim=-1)
        
        atten = torch.matmul(score,v)  # 需要注意这里，score(len,len)  v(len,d_k) 一定是s*v
         
        atten_concat = atten.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        
        atten_out = self.W_o(atten_concat)
        
        return atten_out
    
d_model = 768
head_nums = 12
bs = 1
max_len = 10

q = torch.randn(bs,max_len,d_model)
k = torch.randn(bs,max_len,d_model)
v = torch.randn(bs,max_len,d_model)

multheadatten = MultHeadAtten(d_model,head_nums)

atten = multheadatten(q,k,v)

print(atten)
