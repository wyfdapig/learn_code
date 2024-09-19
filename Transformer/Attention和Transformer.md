# Attention

## 本质

![Attention的本质](https://easyai.tech/wp-content/uploads/2022/08/624d4-2019-11-06-benzhi.png)

## NLP、BERT、GPT、Transformer

![Attention的位置](https://easyai.tech/wp-content/uploads/2022/08/d3164-2019-11-07-weizhi.png)

Attention的发展脉络：

RNN 时代是死记硬背的时期，attention 的模型学会了提纲挈领，进化到 transformer，融汇贯通，具备优秀的表达学习能力，再到 GPT、BERT，通过多任务大规模学习积累实战经验，战斗力爆棚。

## 优点

参数少：相比CNN、RNN

速度快：RNN无法并行计算，但Attention每一步不依赖于上一步的计算过程，因此可以和CNN一样并行处理

效果好：长距离信息不会被弱化

## 原理—带权求和

![attention原理3步分解](https://easyai.tech/wp-content/uploads/2022/08/efa5b-2019-11-13-3step.png)

第一步： query 和 key 进行相似度计算，得到权值

第二步：将权值进行softmax归一化，得到直接可用的权重/概率（和为1）

第三步：将权重和 value 进行加权求和

# 自注意力 Self-Attention

Q = K = V = 序列中的每个元素==（普通注意力中Q通常是解码器给出的，而K和V通常在编码器中）==

一般我们说Attention的时候，他的输入Source和输出Target内容是不一样的，比如在翻译的场景中，Source是一种语言，Target是另一种语言，Attention机制发生在Target元素Query和Source中所有元素之间。而**Self Attention指的**不是Target和Source之间的Attention机制，而**是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的注意力计算机制。**

Self Attention是在2017年Google机器翻译团队发表的《Attention is All You Need》中被提出来的，它完全抛弃了RNN和CNN等网络结构，而仅仅采用Attention机制来进行机器翻译任务，并且取得了很好的效果，Google最新的机器翻译模型内部大量采用了Self-Attention机制。

Self Attention可以捕获同一个句子中单词之间的一些语义特征（比如图展示的its的指代对象Law）。

[![可视化Self Attention机制](https://oss.imzhanghao.com/img/202109151007208.png)](https://oss.imzhanghao.com/img/202109151007208.png)

很明显，引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算。

但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以**远距离依赖特征之间的距离被极大缩短**，有利于有效地利用这些特征。除此外，Self Attention对于**增加计算的并行性**也有直接帮助作用。这是Self Attention逐渐被广泛使用的主要原因。

## Self-Attention的计算过程

[![Attention机制的本质思想](https://oss.imzhanghao.com/img/202109030902038.png)](https://oss.imzhanghao.com/img/202109030902038.png)

### 初始化Q，K，V

从每个编码器的输入向量（在本例中是每个单词的Embedding向量）创建三个向量。对于每个单词，我们创建一个Query向量、一个Key向量和一个Value向量。这些向量是通过将Embedding乘以我们在训练过程中训练的三个矩阵来创建的。

Q=W^Q^X

K=W^K^X

V=W^V^X

[![初始化Q，K，V](https://oss.imzhanghao.com/img/202109151102923.png)](https://oss.imzhanghao.com/img/202109151102923.png)

> 这里Thinking这个单词的Embedding向量是X1，我们用X1乘以WQ的权重矩阵，就可以得到Thinking这个词的Query，即q1。其他的q、k、v等都使用相同的计算方式。

> 这些新向量的维度比Embedding向量小。它们的维数是64，而嵌入和编码器输入/输出向量的维数是512。

### 计算Self-Attention分数

假设我们正在计算本例中第一个单词“Thinking”的自注意力。我们需要根据这个词对输入句子的每个词进行评分。当我们在某个位置对单词进行编码时，分数决定了将多少注意力放在输入句子的其他部分上。

得分是通过将查询向量与我们正在评分的各个单词的键向量进行点积来计算的。 因此，如果我们正在处理位置 #1 中单词的自注意力，第一个分数将是q1和k1的点积。第二个分数是q1和k2的点积。

[![计算Self-Attention Score](https://oss.imzhanghao.com/img/202109151123929.png)](https://oss.imzhanghao.com/img/202109151123929.png)

### 对Self-Attention 分数进行缩放和归一化,然后再进行Softmax操作

对 Step 2 中计算的分数进行缩放，这里通过除以8( 论文中维度是64，这可以让模型有更稳定的梯度)，将结果进行softmax归一化。
[![计算Softmax Socre](https://oss.imzhanghao.com/img/202109151127016.png)](https://oss.imzhanghao.com/img/202109151127016.png)

### Softmax 结果乘以Value向量，求和得到Attention Value

每个Value向量乘以softmax Score得到加权的v1和v2，对加权的v1和v2进行求和得到z1。这样，我们就计算出了第一个词Thinking的注意力值。其他的词用相同的方法进行计算。
[![Socre乘以Value向量](https://oss.imzhanghao.com/img/202109151133617.png)](https://oss.imzhanghao.com/img/202109151133617.png)

# 

# 多头注意力 Multihead-Attention

利用多个查询，来平行地计算从输入信息中选取多个信息。**每个注意力关注输入信息的不同部分（==每个注意力头都独立地处理整个输入序列==，而不是分区域进行。所以类似于使用不同卷积核来从不同的视角提取不同的信息）**，然后再进行拼接。

![image-20240910101053699](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/image-20240910101053699.png)

> W^O^ 是输出层的线性变换矩阵。

![Scaled Dot-Product Attention VS Multi-Head Attention](https://oss.imzhanghao.com/img/202109151148991.png)

# Transformer

**Transformer的三个优势**

- **模型并行度高，使得训练时间大幅度降低。** 
- **可以直接捕获序列中的长距离依赖关系。** 注意力机制允许对依赖关系进行建模，而不考虑它们在输入或输出序列中的距离。对比LSTM，Attention能够更好的解决长距离依赖问题（Long-Term Dependencies Problem）。
- **自注意力可以产生更具可解释性的模型。** 我们可以从模型中检查注意力分布。各个注意头 (attention head) 可以学会执行不同的任务。

## 模型架构

[![Transformer的架构](https://oss.imzhanghao.com/img/202109290538512.png)](https://oss.imzhanghao.com/img/202109290538512.png)

### Encoder and Decoder

### 注意力

#### Scaled Dot-Product Attention

[![Scaled Dot-Product Attention](https://oss.imzhanghao.com/img/202109290848281.png)](https://oss.imzhanghao.com/img/202109290848281.png)

#### Multi-Head Attention

[![Multi-Head Attention](https://oss.imzhanghao.com/img/202109290951436.png)](https://oss.imzhanghao.com/img/202109290951436.png)

#### Attention中的mask操作

在预测的时候防止看到下一个点的答案

### Position-wise Feed-Forward Networks

转化到对应的语义空间

![image-20240910101113904](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/image-20240910101113904.png)

### Embeddings和Softmax

Embeddings和Softmax跟在常规的序列转换模型中起到的作用是相同的。Embeddings将输入符号和输出符号转换为固定长的向量。线性变换和softmax函数将解码器输出转换为预测的下一个字符的概率。在这个模型中，两个嵌入层和pre-softmax线性变换之间共享相同的权重矩阵。

### Layer Normalization

Layer Normalization是作用于每个时序样本的归一化方法，其作用主要体现在：

- 作用于非线性激活函数前，能够将输入拉离激活函数非饱（防止梯度消失）和非线性区域（保证非线性）；
- 保证样本输入的同分布。

### Positional Encoding

由于我们的模型不包含递归和卷积，为了让模型感知到不同token的先后顺序，我们必须注入一些关于标记在序列中的相对或绝对位置的信息。