# 【论文阅读】BoB: BERT Over BERT for Training Persona-based Dialogue Models from Limited Personalized Data

*ACL 2021*

BoB: 有限的个性化数据中训练基于人物角色的对话模型

论文链接：[https://aclanthology.org/2021.acl-long.14.pdf](https://arxiv.org/pdf/2106.02227.pdf)

## 1. 简介

背景：保持一致的角色对于对话agent至关重要。 PERSONA 可以定义为身份元素的组合，例如个人资料和个人背景事实。 在基于角色的对话中，生成的回复不仅取决于对话上下文，还取决于一些预定义的角色，从而呈现的个性可以更加一致。

挑战：众包数据的规模有限和大规模数据中的人物稀疏性提出了一个共同的挑战，在有限的个性化数据上训练的模型无法充分理解人物的一致性。

需要对话agent拥有以下能力：1）理解角色回复的一致性；2）根据对话上下文生成与角色相关的回复。显然，满足这两个特性的理想数据集很难注释。然而，一旦我们将基于人物角色的对话生成分解为两个子任务：**一致性理解**和**对话生成**，就很容易为它们找到168个丰富的数据资源。为了一致性理解，我们可以利用大规模非对话推理数据。至于对话生成，我们已经有了各种大规模的人物角色稀疏数据集。

这项工作的贡献有三方面：

- 我们将基于角色的对话生成任务分解为两个子任务：**一致性理解和对话生成**；

- 提出了一个基于 BERT 的生成框架 BoB，用于从有限的数据中训练基于角色的对话模型；

- 引入了一种使用非对话推理数据的可能性训练方法，以增强对角色一致性的理解。

## 2. 方法

$\mathcal{Q}=q_{1}, q_{2}, \ldots, q_{n}$​​​​​​​​​ 表示对话查询, $\mathcal{R}=r_{1}, r_{2}, \ldots, r_{m}$​​​​​​​​​ 表示目标回复, and $\mathcal{P}$​​​​​​​​​ 表示角色. 另外, $\mathcal{N}$​​​​​​​​​ 表示非对话推理数据，由前提premise，假设hypothesis和标签label组成，都是自然语句。句子 $(\mathcal{P}, \mathcal{Q}, \mathcal{R})$​，​​他们的向量表示 $\left(P, Q, R_{1}, R_{2}\right)$​​​​​​​​​​​。

模型的任务 $\mathbb{M}$​​ 是生成角色一致的回复 $\hat{\mathcal{R}}=\hat{r}_{1}, \hat{r}_{2}, \ldots, \hat{r}_{m}$​​，基于角色 $\mathcal{P}$​​ 和查询 $\mathcal{Q}$​​, i.e., $\hat{\mathcal{R}}=$​​ $\mathbb{M}(\mathcal{Q}, \mathcal{P}) .$​​ 如图所示, M由三个基于BERT的子模块组成: 编码器 $\mathbb{E}$​​, 回复解码器 $\mathbb{D}_{1}$​​, 和一致性理解解码器 $\mathbb{D}_{2} .$​​ 具体来说, $\mathbb{E}$​​ 对角色和查询的嵌入编码, i.e., $P$​​ and $Q$​​, into hidden states $H . \mathbb{D}_{1}$​​ 以典型的编码器解码器方式对 $H$​​ 执行cross attention, 并生成粗略的表示 $R_{1} . \mathbb{D}_{2}$​​ 从非对话推理数据 $\mathcal{N}$​​ 中学习一致性理解，并且进一步将 $P$​​ 和 $R_{1}$​​ 转换为最终表示 $R_{2} .$​​ 最后，一致性回复 $\hat{\mathcal{R}}$​​ 从 $R_{2}$​​​生成​.

![image-20210730203540325](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210730203540325.png)

## 备注

**1、训练过程**

+ Response Generation. Given $\mathcal{P}, \mathcal{Q}$, and $\mathcal{R}$ from personalized dialogue data, we calculate the response generation loss $\mathcal{L}_{1}=\mathcal{L}_{N L L}^{\mathbb{D}_{1}}+\alpha \mathcal{L}_{N L L}^{\mathbb{D}_{2}}$

+ Consistency Understanding. Given $\mathcal{D}^{+}$ and $\mathcal{D}^{-}$ from non-dialogue inference data, we calculate the unlikelihood loss $\mathcal{L}_{2}=\beta \mathcal{L}_{U L}^{\mathbb{D}_{2}^{+}}+(1-\beta) \mathcal{L}_{U L}^{\mathbb{D}_{2}^{-}}$

+ Optimization. Sum up $\mathcal{L}_{1}$ and $\mathcal{L}_{2} .$ Update parameters with back-propagation.

**2、案例分析**


![image-20210730203720290](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210730203720290.png)