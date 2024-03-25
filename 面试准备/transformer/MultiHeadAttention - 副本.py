import torch.nn as nn
import math
import torch
import torch.nn.functional as F
class MultiHeadAttention_my(nn.Module):
    def __init__(self,d_model,nums_heads):
        super(MultiHeadAttention_my, self).__init__()
        self.num_heads = nums_heads
        self.d_model = d_model
        assert d_model % nums_heads == 0
        
        self.d_k = d_model // nums_heads
        
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
    def split_heads(self,x,bs):
        x = x.view(bs,-1,self.num_heads,self.d_k)
        x = x.transpose(1,2)
        return x
            
    def forward(self,q,k,v,mask=None):
        bs = q.shape[0]
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = self.split_heads(q,bs)
        k = self.split_heads(k,bs)
        v = self.split_heads(v,bs)
        
        qk = torch.matmul(q, k.transpose(-1,-2))/ math.sqrt(self.d_k)
        
        if mask is not None:
            qk = qk.masked_fill(mask.to(qk.type)==0,float('-inf'))
            
        weight = F.softmax(qk,dim=-1)
        
        attention = torch.matmul(weight,v)
        
        attention_concat = attention.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        attention_output = self.W_o(attention_concat)
        
        return attention_output    

nums_heads = 12
d_model = 768
attention = MultiHeadAttention_my(d_model,nums_heads) 
bs = 2
max_len = 10
q = torch.randn(bs,max_len,d_model)   
k = torch.randn(bs,max_len,d_model)
v = torch.randn(bs,max_len,d_model)

attention_output = attention(q,k,v)

print(attention_output)
    

        