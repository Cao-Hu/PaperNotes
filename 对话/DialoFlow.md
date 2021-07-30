# 【论文阅读】Conversations Are Not Flat: Modeling the Dynamic Information Flow across Dialogue Utterances

*ACL 2021*

DialoFlow模型: 建模对话动态信息流

论文链接：[https://arxiv.org/pdf/2106.02227.pdf](https://arxiv.org/pdf/2106.02227.pdf)

## 1. 简介

现有的建模对话历史的方法主要分为两种。一种是**直接拼接对话历史**(flat pattern)，这种方法在某种程度上忽略了对话历史中跨句子的对话动态。另外一种是**多层次建模**，首先对每句话做表示，再对整个对话做表示，这种方法在对每句话做表示时忽略了其他句子的作用。

本文提出了建模对话动态信息流的方法DialoFlow，引入dynamic flow 动态流机制通过处理每个话语带来的语义影响来模拟整个对话历史中的动态信息流。

在Reddit上进行了预训练，实验表明，DialoFlow在对话生成任务（Reddit multi-reference、DailyDialog multi-reference）上明显优于DialoGPT。

此外，本文还提出了Flow score，一种基于预训练的DialoFlow的评价人机交互对话质量的有效自动度量，在DSTC9交互式对话评估数据集上的评估结果与人工评估一致性达到0.9。

## 2. 方法

### 2.1 任务

设 D = {u1, u2, ..., uN}表示整个对话。对于每一条话语$u_{k}=\left\{u_{k}^{1}, u_{k}^{2}, \ldots, u_{k}^{T}\right\}$​​​其中$u_{k}^{T}$​​​表示第k条话语中的第t个单词。定义 $u_{<k}=\left\{u_{1}, u_{2}, \ldots, u_{k-1}\right\}$​​​ 为第k条话语的历史对话，也就是上下文​​  $\mathbf{C}_{k} .$​​​  $\mathbf{C}_{k+1}$​​​ 和 $\mathbf{C}_{k}$​​​ 的差异被定义为 semantic influence $\mathbf{I}_{k}$:
$$
\mathbf{I}_{k}=\mathbf{C}_{k+1}-\mathbf{C}_{k}
$$
![img](file:///C:\Users\l\AppData\Roaming\Tencent\Users\1377330332\QQ\WinTemp\RichOle\G%L_R3AY183GQONFGWQ~2BY.png)

DialoFlow首先对对话历史进行编码, 并根据之前所有的历史上下文 $C_{1}, C_{2}, \ldots, C_{k}$ 预测未来的上下文 $C_{k+1}^{\prime}$ 然后在回复生成阶段, 模 型获取预测的目标语义影响 $I_{k}^{\prime}$, 并考虑预测语义影响和历史子句生成目标回复 $u_{k}$ 。

### 2.2 模型架构

![img](file:///C:\Users\l\AppData\Roaming\Tencent\Users\1377330332\QQ\WinTemp\RichOle\0WD(6YHCPBL}GF)2XY9$R`S.png)

## 备注

**1、捕获动态信息流**

DialoFlow中的Transformer Block模块构建话语级历史上下文的context flow, 输入给flow module模块预测第( $\mathrm{k}+1$ )条消息的上下文 $C_{k+1}^{\prime}$, $I_{k}^{\prime}=C_{k+1}^{\prime}-C_{k}$​ 表示第k个话语引起的语义影响。

当第k条消息改变时，只会引起k之后消息的上下文语义变化，而不会带来k之前的上下文语义变化（图4的可视化也证明了这点），也就是文中强调的，可以捕获动态信息流（构建话语级历史上下文的context flow+捕获第k个话语带来的语义影响）。后面提出的Flow score，也用到了这里的语义影响**I**。

**2、Flow score**

perplexity(困惑度)的基本思想是：当语言模型训练完之后，测试集中的句子都是正常的句子，那么给测试集的句子赋予较高概率值的语言模型较好。

PPL的计算过程如下：
对于一段句子(sentence)s由词构成，即:s=w1w2…wn, w代表词
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210609112811718.png)
对两边都取对数，则:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210609112909754.png)
句子概率越大，语言模型越好，迷惑度越小。

这里的flow score用相似度$s_{k}$来类比ppl中的句子概率P，机器人生成的话语带来的语义影响与真实的语义影响越相似（相似度$s_{k}$越高），对话质量越高，flow score就越小。
Flow score计算过程如下：

![公式11](https://img-blog.csdnimg.cn/20210609094953777.png)

$s_{k}$表示预测语义影响$I_{k}^{\prime}$与真实语义影响$I_{k}$​之间的相似度，

![公式12](https://img-blog.csdnimg.cn/20210609095055342.png)

M为对话的回合数，用$（s_{k}+1） / 2$​将相似度值和P一样缩放到[0,1]
相似度越高，对话质量越高，Flow score越小

**3、案例分析**

![img](file:///C:\Users\l\AppData\Roaming\Tencent\Users\1377330332\QQ\WinTemp\RichOle\X5(5HI)SKI4TQWRJX2FZ{0C.png)

上图显示了由DialoFlow编码的人机对话的语义上下文的二维T-SNE可视化。对话可以分为四个话题:问候(1 ~ 4)，谈论为什么糟糕的一天(5 ~ 13)，解释看医生的可怕经历(14 ~ 18)，讨论游泳(19 ~ 26)。

可以发现，随着话题的切换，语义语境流的可视化也发生了很大的变化，这表明DialoFlow能够捕获对话中的动态信息流，有效地度量每句话语所带来的语义影响。此外，不同的说话者保持着自己的context flows。
