

Transformer采用Query-Key-Value（QKV）模型的注意力机制。给定查询矩阵表示$\mathbf{Q} \in\mathbb{R}^{N \times D_{k}}$，键的表示$\mathbf{K} \in \mathbb{R}^{M \times D_{k}}$和值的表示$\mathbf{V} \in \mathbb{R}^{M \times D_{v}}$，Transformer使用的缩放点积注意力定义如下：

$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q K}^{\top}}{\sqrt{D_{k}}}\right) \mathbf{V}=\mathbf{A} \mathbf{V}$

其中，𝑁和𝑀表示查询和键（或值）的长度；$D_{k}$和$D_{v}$表示键（或查询）和值的维度；$\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q K}^{\top}}{\sqrt{D_{k}}}\right)$通常被称为注意力矩阵，其中softmax是以逐行的方式对注意力矩阵进行归一化。查询和键的点积除以$\sqrt{D_k}$，以缓解softmax函数可能带来的梯度消失问题。

Transformer 不是简单地应用单个注意力函数，而是使用多头注意力机制，其中 $D_m$ 维的原始查询、键和值分别通过 H 个不同的学习投影集投影到 Dk、Dk 和 Dv 维度。对于每个预测的查询、键和值以及输出，根据等式(1)进行注意计算。然后，该模型拼接所有多头输出拼接成一个长向量，之后再使用一个全连接网络将它们投影回 $D_m$ 维表示。
$$
\begin{array}{l}\text { MultiHeadAttn }(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\text { Concat }\left(\operatorname{head}_{1}, \ldots, \text { head }_{H}\right) \mathbf{W}^{O}, \\ \text { where } \operatorname{head}_{i}=\operatorname{Attention}\left(\mathbf{Q} \mathbf{W}_{i}^{Q}, \mathbf{K} \mathbf{W}_{i}^{K}, \mathbf{V} \mathbf{W}_{i}^{V}\right)  \\\end{array}
$$
在 Transformer 中，根据查询和键值对的来源可以分为三种类型的注意力：

自注意力机制（Self-attention）：在Transformer 的编码器中，公式2的Q=K=V=X，其中X为前一层的输出结果。

掩码自注意力机制（Masked Self-attention）：在 Transformer 解码器中，自注意力受到限制，每个位置的查询只能关注到并包括该位置的所有键值对。为了实现并行训练，通常通过将掩码函数应用于非归一化注意力矩阵$\hat{\mathbf{A}}=\exp \left(\frac{\mathbf{Q K}^{\top}}{\sqrt{D_{k}}}\right)$来完成，当其中通过设置$\hat{A}_{i j}=-\infty \quad if \quad i <j$，用这种方式屏蔽非法位置。这种自注意力通常被称为自回归或因果注意力。

交叉注意力机制（Cross-attention）：与自注意力（Self-attention）机制不同，交叉注意力机制允许解码器在生成序列中的每个位置时将注意力集中在编码器输出序列的相关部分。这在序列到序列学习任务中尤其有用，例如机器翻译，其中解码器需要在生成目标语言的单词时理解输入序列的含义。

