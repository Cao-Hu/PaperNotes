# Diversifying Dialog Generation via Adaptive Label Smoothing

ACL 2021

AdaLabel:通过自适应标签平滑使对话生成多样化

论文链接：https://aclanthology.org/2021.acl-long.272.pdf

## 1. 简介

**背景：**

低多样性问题和过度自信问题之间存在很强的联系。 即，过度自信的对话模型往往会产生低多样性的回复。 原因之一可以归结于监督目标。 具体来说，在硬目标（即，one-hot 分布作为真实数据）下训练具有最大似然估计（MLE）目标的对话生成模型会使模型偏向高频标记并产生过度自信的概率估计=，最终导致校准不良=，从而导致多样性低=。  Hinton et al. (2015) 和 Yang et al. (2018) 提出理想的训练目标应该是一个软目标，它为多个有效候选者分配概率（见图 1）。 有了这样一个软目标，可以缓解过度自信的问题，从而可以提高输出回复的多样性。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/HAYU~$%5BIFZ%5DAU%7DD4%5BXKWMNE.png)

不幸的是，理想的软目标很难获得。 早期的工作尝试使用标签平滑来解决这个问题，即将一个小概率统一分配给非目标词。 然而，这种方式构建的目标分布远非理想：**首先**，目标词的概率是人工选择的，并且是固定的，不能适应不同的上下文。然而，human text distribution exhibits remarkable fluctuations in the per-token perplexity。 我们认为不同的目标概率应该用于不同的上下文。 **其次**，对非目标词的概率质量的统一分配忽略了上下文和每个词之间的语义关系。 理想情况下，如果一个词与上下文更相关，它应该获得更多的概率质量。 对于图 1 所示的示例，“fun”一词比“bank”一词更有可能出现在“I make the robots seem more _”后面。

**本文工作**

为了解决上述问题，我们提出了一种自适应标签平滑（AdaLabel）方法，该方法可以针对不同的上下文在每个时间步动态估计软目标分布。 具体来说，对于训练数据中的每个目标词$y_{t}$，首先得到当前模型预测的概率分布。 该分布中的最大概率 $p_{max}$ 衡量当前预测的置信度，即更高的 $p_{max}$ 意味着对当前预测的更高置信度。 为了避免过度自信，我们在训练过程中使用 $p_{max}$ 作为目标词$y_{t}$的监督信号，以便模型在正确预测$y_{t}$时不会朝着$y_{t}$​优化。 还引入了词级因子，以方便低频词的学习。

此外，我们引入了一个新的辅助解码器模块$D_{a}$来在每个训练步骤中为这些非目标词产生监督信号。$D_{a}$只包含一个 Transformer 块，并且它被优化为基于双向上下文预测单词。 设计了一种新颖的 Target-Mask attention方案，以防止$D_{a}$在训练过程中看到目标词。 该方案还实现了$D_{a}$的并行训练和推断。

1. 我们提出了 AdaLabel，一种可以在考虑当前上下文和模型置信度的情况下生成软目标分布的方法。 具体来说，如果 $y_{t}$ 已被正确预测，AdaLabel 确保对话模型不会针对目标词 $y_{t}$ 进行优化。 这可以防止我们的模型过度自信。
2. 我们引入了一个轻量级的双向解码器，可以为非目标词产生上下文感知的监督信号。 设计了一种新颖的 Target-Mask attention方案以促进该解码器的并行训练和推理。
3. 在两个基准对话数据集上进行的大量实验以及自动和人工评估结果表明，我们的方法有助于缓解模型过度自信的问题，并显着提高了模型的多样性。

## 2. 方法

![img](https://gitee.com/cao-hu/pictures/raw/master/img/OGGK%`YV1NRQZL04F%I03UA.png)
$$
\boldsymbol{q}^{\prime}=\varepsilon \cdot \boldsymbol{q}+(1-\varepsilon) \cdot \boldsymbol{v}
$$
其中，$\varepsilon \in[0,1]$​​​​​​是自适应因子， $v$​​​​​​是依赖于当前时间步的辅助分布向量。我们约束 $\boldsymbol{v}$​​​​​​ 为目标词 $y_{t}$​​​​​​ 分配零概率，为非目标词 $\mathcal{V}_{\neq y_{t}}=\left\{y_{i} \mid y_{i} \in\right.$​​​​​​ $\left.\mathcal{V}, y_{i} \neq y_{t}\right\} $​​​分配非零概率​。这个约束允许我们显式地控制分配给 $y_{t} $的监督。​​​​​

![image-20210803111910799](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210803111910799.png)