# 【论文阅读】Learning from Perturbations: Diverse and Informative Dialogue Generation with Inverse Adversarial Training

ACL 2021

从扰动中学习：通过逆向对抗训练生成多样化和信息丰富的对话

论文链接：https://aclanthology.org/2021.acl-long.57.pdf

## 1. 简介

**背景**：由于过度简化的最大似然估计 (MLE) 训练目标和训练语料库中通用响应的高频率，它们往往会产生沉闷和通用的响应，例如“我不知道”比人类通常做的要多得多，这使得对话代理的参与度降低且效率低下。此外，关于神经对话系统是否有效使用对话历史的最新研究表明，大多数神经对话代理在生成响应时没有考虑到对话历史。这个问题使得神经对话系统倾向于产生与当前对话主题无关的反应，并且与对话历史不一致。这一问题也可能加剧一般性反应问题，因为沉闷的反应通常是离题的，与对话历史无关。

为了解决上述问题，在本文中，我们提出了用于训练神经对话系统的反向对抗训练 (IAT) 算法，以更好地避免通用响应和建模对话历史，从而生成多样化和信息丰富的响应。 传统的对抗训练方法通常使用精心设计的方法生成保留标签的对抗输入，并训练模型以生成相同的输出以增强模型的鲁棒性。 **相比之下，我们的方法会扰乱输入对话历史，因此如果输出是非通用的且与对话历史相关，那么好的对话模型不应生成相同的输出。** 我们将我们提出的方法命名为逆对抗训练，因为它与旨在提高模型对抗鲁棒性的传统对抗训练方法有关，但我们提出的目标是相反的。 

## 2. 逆对抗训练 Inverse Adversarial Training

### 2.1 扰动方法

+ **话语级别**：我们考虑以下操作：
  1. Shuf ：将对话历史中的话语序列打乱，
  2. Rev ：颠倒历史中话语的顺序（但保持每个话语中的词序）
  3. Drop ： 完全丢弃某些话语，
  4. Truncate ：对话历史以仅包含 k 个最近的话语，其中 k ≤ n，其中 n 是对话历史的长度，
  5. Repl 以30% 的概率用数据集中的另一个话语随机替换对话历史中的每个话语，类似于负抽样方法。
+ **词级别**：我们考虑类似的操作，但在每个话语中的单词级别：
  1. word-shuffle：随机打乱一句话里的单词
  2. reverse：颠倒单词的顺序
  3. word-drop：均匀地丢弃30%的单词
  4. noun-drop：丢弃所有名词
  5. verb-drop：丢弃所有动词
  6. word-repl：均匀地用词典里随机一个单词替换30%的单词

Shuf 和 Rev 扰动改变了话语的时间顺序，具有此类扰动的逆对抗训练可能有助于模型捕捉有关话语时间顺序的一些常识。 Drop 和 Repl 扰动可能有助于模型捕捉某些类型的因果效应。 最后，Truncate扰动可能有助于模型更好地捕捉长期和多轮对话历史。

### 2.2 逆对抗训练

与在给定扰动输入的情况下**最大化生成相同输出**的可能性的对抗训练目标相反，逆向对抗训练目标在输入被扰动时**最大化生成相同输出的可能性的降低** 。

![image-20210731184058941](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210731184058941.png)

Reward:

$\mathrm{NLL}_{\text {orig }}=-\sum_{i=1}^{n} \log P\left(y_{i} \mid y_{<i}, X\right)$
$\mathrm{NLL}_{a d v}=-\sum_{i=1}^{n} \log P\left(y_{i} \mid y_{<i}, X^{\prime}\right)$
$\mathcal{R}\left(Y \mid X, X^{\prime}\right)=\mathrm{NLL}_{a d v}-\mathrm{NLL}_{\text {orig }}$

Penalty:

$\mathcal{P}\left(Y \mid X, X^{\prime}\right)=\min \left(0, \mathrm{NLL}_{a d v}-\mathrm{NLL}_{\text {orig }}-\mathcal{M}\right)$

## 备注

**Case Study**

![image-20210731183923383](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210731183923383.png)