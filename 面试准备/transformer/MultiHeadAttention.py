import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention_my(nn.Module):
    def __init__(self, d_model, num_heads, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b):
        super(MultiHeadAttention_my, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        # Define the dimension of each head or subspace
        self.d_k = d_model // self.num_heads

        # These are still of dimension d_model. They will be split into number of heads 
        # 这三个矩阵得作用是将输入X转换为三个相同形状不同的向量，分别为q,k,v
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Outputs of all sub-layers need to be of dimension d_model
        self.W_o = nn.Linear(d_model, d_model)
        
        # Set custom weights and biases for W_q
        with torch.no_grad():
            self.W_q.weight.copy_(torch.tensor(q_w.T).clone().detach())
            self.W_q.bias.copy_(torch.tensor(q_b).clone().detach())

        # Set custom weights and biases for W_k
        with torch.no_grad():
            self.W_k.weight.copy_(torch.tensor(k_w.T).clone().detach())
            self.W_k.bias.copy_(torch.tensor(k_b).clone().detach())

        # Set custom weights and biases for W_v
        with torch.no_grad():
            self.W_v.weight.copy_(torch.tensor(v_w.T).clone().detach())
            self.W_v.bias.copy_(torch.tensor(v_b).clone().detach())
            
        with torch.no_grad():
            self.W_o.weight.copy_(torch.tensor(o_w.T).clone().detach())
            self.W_o.bias.copy_(torch.tensor(o_b).clone().detach())
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        
        batch_size = Q.size(0)
        K_length = K.size(-2)
        # QKV形状是bs,num_heads,max_len,d_k
        # / math.sqrt(self.d_k)防止softmax在反向传播时梯度接近0，导致梯度消失的问题
        # K.transpose(-2, -1)交换倒数1、2维度，为矩阵乘法做准备
        QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
        # Apply the mask
        if mask is not None:
            QK = QK.masked_fill(mask.to(QK.dtype) == 0, float('-inf'))
    
        # Calculate the attention weights (softmax over the last dimension)
        weights = F.softmax(QK, dim=-1)
    
        # Apply the self attention to the values
        attention = torch.matmul(weights, V)
    
        return attention, weights


    def split_heads(self, x, batch_size):
        """
        The original tensor with dimension batch_size * seq_length * d_model is split into num_heads 
        so we now have batch_size * num_heads * seq_length * d_k
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # linear layers
        # 这里的qkv在selfatten中应该是同一个值X，X的形状是bs，max_len，hidden_size
        # 相同的x经过三个形状相同的矩阵（hidden_size，hidden_size），得到3个不同的qkv（bs，max_len，hidden_size）
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        # split into multiple heads
        # 多头机制，拆分qkv，拆分后，qkv的形状为bs,num_heads,max_len,d_k(将hidden_size拆分成num_heads份)
        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  

        # self attention
        # scores,weights (bs,head_nums,max_len,d_k)
        scores, weights = self.scaled_dot_product_attention(q, k, v, mask)

        # concatenate heads
        # 将拆分的heads拼接起来
        # concat形状分析，.transpose(1,2)交换1,2维顺序，得到bs,max_len,head_nums,d_k
        # .contiguous()这个方法是为了确保张量在内存中是连续的,以提高之后的view操作效率。它不改变张量的形状和值。
        # view(batch_size, -1, self.d_model)，
        # -1指的是根据剩余元素自动计算当前维度的大小，元素总数：bs*max_len*head_nums*d_k 固定元素总数：batch_size*d_model
        # 剩余元素：bs*max_len*head_nums*d_k/batch_size*d_model = max_len
        # concat（bs,max_len,hidden_size）
        # 这里与我看到的技术博客不同，那里说是拼接所有的head结果组合成一个大的矩阵，再用矩阵映射回去
        # 而实际的操作是直接进行矩阵变化，然后经过一个不改变形状的全连接网络
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        # final linear layer
        output = self.W_o(concat)

        return output, weights
    
    
# 创建一些随机输入张量作为示例
batch_size = 2 
seq_len = 10
d_model = 512
num_heads = 8

q_w = torch.randn(d_model, d_model)
q_b = torch.randn(d_model)
k_w = torch.randn(d_model, d_model)
k_b = torch.randn(d_model)
v_w = torch.randn(d_model, d_model)
v_b = torch.randn(d_model)
o_w = torch.randn(d_model, d_model)
o_b = torch.randn(d_model)

q = torch.randn(batch_size, seq_len, d_model)
k = torch.randn(batch_size, seq_len, d_model)
v = torch.randn(batch_size, seq_len, d_model)

# 创建一个MultiHeadAttention实例
attention = MultiHeadAttention_my(d_model, num_heads, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b)

# 执行前向传播
output, weights = attention(q, k, v)

print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)