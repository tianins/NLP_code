import torch
import math
import numpy as np
from transformers import BertModel
import torch.nn as nn
import MultiHeadAttention

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''
bs_pos = 1
bert = BertModel.from_pretrained(r"E:\data\hub\bert_base_chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
if bs_pos:
    x = np.array([[2450, 15486, 102, 2110],[2450, 15486, 102, 2110]]) #通过vocab对应输入：深度学习
    torch_x = torch.LongTensor(x)  #pytorch形式输入
else:
    x = np.array([2450, 15486, 102, 2110]) #通过vocab对应输入：深度学习
    torch_x = torch.LongTensor([x])  #pytorch形式输入



# seqence_output, pooler_output = bert(torch_x)
# print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

# print(bert.state_dict().keys())  #查看所有的权值矩阵名称

# 这里面的问题是好像没有考虑bs这一维度，默认为1了

#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class DiyBert:
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 12  # 注意这里要和本地bert的config保持一致
        self.layer_norm_eps = 1e-12
        self.load_weights(state_dict)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()


    #bert embedding，使用3层叠加，在经过一个embedding层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        # 这里不理解get_embedding的作用是什么？
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        if bs_pos:
            # position embeding的输入 [0, 1, 2, 3]
            pe = self.get_embedding(self.position_embeddings, np.array([list(range(len(x[0])))]))  # shpae: [max_len, hidden_size]
            # token type embedding,单输入的情况下为[0, 0, 0, 0]
            te = self.get_embedding(self.token_type_embeddings, np.array([[0] * len(x[0])]))  # shpae: [max_len, hidden_size]
        else:
            pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x))))) 
            te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))
        embedding = we + pe + te
        # 加和后有一个归一化层
        # embedding = self.LayerNorm(embedding)
     
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)

        return embedding

    #embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵，感觉问题可能是出在这里
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    #执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #self attention层
        
        
        # 创建一个MultiHeadAttention实例
        attention = MultiHeadAttention.MultiHeadAttention_my(self.hidden_size, self.num_attention_heads, q_w, q_b, k_w, k_b, v_w, v_b,attention_output_weight,attention_output_bias)
        q=k=v=torch.FloatTensor(x)
        # 执行前向传播
        # 两种方法的结果对不上
        output, weights = attention(q, k, v)
        
        
        attention_output = self.self_attention(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #bn层，并使用了残差机制，进入前馈层之前经过一层残差和layer_norm
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed forward层，将原始维度放大4倍，再缩放回去，中间经过一次gelu激活函数
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制，在输出前经过一次残差和layer_norm
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # self attention的计算
    def self_attention(self,
                       x,
                       q_w,
                       q_b,
                       k_w,
                       k_b,
                       v_w,
                       v_b,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        
        # x分别经过3个不同的线性层，得到QKV
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
        k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
        batch_size = q.shape[0]
        # 多头机制，分割原始的QVK
        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        
        # atten的计算公式，Q*K的转置/根号attention_head_size
        # qk.shape = num_attention_heads, max_len, max_len
        # 得到文本长度*文本长度的矩阵
        qk = np.matmul(q, k.swapaxes(-1, -2))
        qk /= np.sqrt(attention_head_size)
        # 计算结果再经过softmax，归一化，得到的加和为1，的0到1之间的数值
        qk = softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        # 再*v，得到max_len, hidden_size的结果
        qkv = np.matmul(qk, v)
        
        # qkv，（head_nums,max_len,d_k）,bs去哪里了？bs为1？
        
        # qkv.shape = max_len, hidden_size，改变形状,
        # 这里其实有点问题qkv的形状应该是bs,head_nums,max_len,d_k，之后.swapaxes(0, 1)交换0、1维，得到head_nums,bs,max_len,d_k
        # 再.reshape(-1, hidden_size)得到bs*max_len，hidden_size，这里面的交换维度好像没有实际的作用
        # 这里其实是有作用的，只是最初版本里面没有bs维度，原始版本的形状head_nums,max_len,d_K，交换维度后max_len，head_nums，d_k
        # 然后再变换形状得到max_len，hidden_szie
        # 现在增加bs维度，形状是bs,head_nums,max_len,d_k，交换1，2维，再变形
        
        
        if bs_pos:
            qkv = qkv.swapaxes(1, 2).reshape(batch_size,-1, hidden_size)
        else:
            qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        
        
        # 之后接一个全连接层(768*768)，hs*hs，得到bs，这里的形状好像有问题，没搞清楚
        # attention.shape = bs,max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    #多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        if bs_pos:
        
            batch_size, max_len, hidden_size = x.shape
            x = x.reshape(batch_size, max_len, num_attention_heads, attention_head_size)
            x = x.swapaxes(1, 2)  # output shape = [batch_size, num_attention_heads, max_len, attention_head_size]
        else:
            max_len, hidden_size = x.shape
            x = x.reshape(max_len, num_attention_heads, attention_head_size)
            x = x.swapaxes(0, 1)
        return x

    #前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    #归一化层
    def layer_norm(self, x, w, b):
        if bs_pos:
            x = (x - np.mean(x, axis=2, keepdims=True)) / np.std(x, axis=2, keepdims=True) +self.layer_norm_eps
        else:
            x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True) +self.layer_norm_eps
        x = x * w + b
        return x

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    #最终输出
    def forward(self, x):
        # x: 4 max_len  -> 1,4 bs,max_len
        # 经过embedding后 x：4，768 max_len,hidden_size -> 1,4,768 bs,max_len,hidden_size 
        # 但是这里少了bs维度
        # 增加bs维度
        x = self.embedding_forward(x) 
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output


#自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print("==============================")
print(torch_sequence_output)

"""
增加bs维度后，结果与原始bert对不上
猜测可能是embedding时出了问题
"""

# print(diy_pooler_output)
# print(torch_pooler_output)