# Semantic Representation for Dialogue Modeling

ACL 2021

[对话建模的语义表示](https://arxiv.org/abs/2105.10188)

## Abstract

​		尽管神经模型在对话系统中取得了竞争性结果，但它们在表示核心语义（例如忽略重要实体）方面显示出有限的能力。为此，我们利用抽象意义表示（AMR）来帮助进行对话建模。与文本输入相比，AMR显式提供了核心语义知识并减少了数据稀疏性。我们开发了一种从句子级AMR构造对话级AMR图的算法，并探索了两种将AMR整合到对话系统中的方法。对话理解和回复生成任务的实验结果表明了我们模型的优越性。据我们所知，我们是第一个将形式化语义表示形式运用到神经对话建模中的人。

## 1. Introduction

​		对话建模中有两个突出的子任务，即对话理解和回复生成。 前者是指对对话历史中语义和语篇细节的理解，后者是指表达流畅、新颖、连贯的话语。

​		当前sota方法采用神经网络和端到端训练进行对话建模。 例如，seq2seq模型已被用于编码对话历史，然后直接合成下一个句子（[Bao 等2020 PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable](https://arxiv.org/abs/1910.07931)）。 尽管给出了强有力的实证结果，但神经模型在其神经语义表示中可能会受到虚假特征关联的影响，这可能导致鲁棒性弱，导致不相关的对话状态并生成不准确或不相关的文本。 如图 1 所示，基线 Transformer 模型关注“lamb”这个词，但忽略了其周围的上下文，其中包含表明其真实含义的重要内容（用方块标记），从而给出与食物相关的无关回复。 直观上，这些问题可以通过语义信息的结构化表示来缓解，将实体视为节点并在节点之间建立结构关系，从而轻松找到最突出的上下文。与神经表示相比，显式结构也更具可解释性，并且已被证明可用于信息提取、摘要和机器翻译。

![image-20210928190926025](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210928190926025.png)

​		我们探索 AMR 作为对话历史的语义表示，以便更好地表示对话。 如图 2 的中央块所示，AMR 是一种句子语义表示，它使用有根有向无环图对句子进行建模，突出其主要概念（例如“错误”）和语义关系（例如“ARG0”1  )，同时抽象掉虚词。 因此，它可以潜在地提供聚合对话中的主要内容所需的核心概念和显式结构。 此外，AMR 还可用于减少具有相同含义的表面形式的方差的负面影响，这增加了数据稀疏性。  

![img](https://gitee.com/cao-hu/pictures/raw/master/img/@R__%5D6GH%5DG$HIPY%60$%7D5$0%7BB.png)

​		AMR 解析的现有工作侧重于句子级别。 然而，如图 2 的左侧块所示，对话历史的语义结构可以由丰富的交叉话语共指链接（用正方形标记）和多个说话者交互组成。 为此，我们提出了一种算法，通过添加指示说话者、相同提及和共同参考链接的交叉话语链接，自动从话语级 AMR 导出对话级 AMR。 图 2 的右侧块显示了一个示例，其中新添加的边是彩色的。 我们考虑使用这种对话级 AMR 结构的两种主要方法。 对于第一种方法，我们通过 AMR 到文本对齐将 AMR 与其相应句子中的标记合并，然后使用图转换器对结果结构进行编码。 对于第二种方法，在通过特征融合或双重注意利用这两种表示之前，我们分别对 AMR 及其相应的句子进行编码。

​		我们验证了所提出的框架在对话关系提取任务和回复生成任务上的有效性。 实验结果表明，所提出的框架优于以前的方法，在两个基准上都取得了最新的最新结果。 深度分析和人工评估表明，AMR 引入的语义信息可以帮助我们的模型更好地理解长对话并提高对话生成的连贯性。 另一个优点是 AMR 有助于增强鲁棒性，并有可能提高神经模型的可解释性。 据我们所知，这是第一次尝试将 AMR 语义表示用于神经网络以进行对话理解和生成。

## 2 Constructing Dialogue AMRs

图 2 展示了我们从多个话语级 AMR 构建对话级 AMR 图的方法。 给定包含多个话语的对话，我们采用预训练的 AMR 解析器（Cai 和 Lam，2020 [AMR parsing via graph-sequence iterative inference](https://arxiv.org/pdf/2004.05572)）来获得每个话语的 AMR 图。 对于包含多个句子的话语，我们将它们解析成多个 AMR 图，并标记它们属于同一个话语。 我们通过在话语 AMR 之间建立连接来构建每个对话 AMR 图。 特别地，我们根据说话者、相同概念和共同参考信息采取三种策略。

**Speaker**	我们添加一个虚拟节点并将其连接到话语 AMR 的所有根节点。 我们在边上添加说话者标签（例如，SPEAKER1 和 SPEAKER2）以区分不同的说话者。 虚拟节点确保所有话语 AMR 都已连接，以便在图编码期间可以交换信息。 此外，它作为全局根节点来表示整个对话。

**Identical Concept**	在不同的话语中可能有相同的提及（例如，图 2 中的第一个和第四个话语中的“possible”），导致话语 AMR 中的概念节点重复。 我们通过标有 *SAME* 的边连接对应于相同**非代词**概念的节点。 这种类型的连接可以进一步增强跨句信息交换。(与共指相比，相同的概念关系可以连接具有相同含义的不同词，例如<could, might>，<fear，afraid>。)

**Inter-sentence Co-reference**	对话理解的一个主要挑战是代词，这在对话中很常见。 我们使用现成的模型对对话文本进行共同参考解析，以识别引用同一实体的话语 AMR 中的概念节点。 例如，在图 2 中，第一个话语中的“I”和第二个话语中的“sir”指的是同一个实体 SPEAKR1。 我们在它们之间添加标有 *COREF* 的边，从较晚的节点开始到较早的节点（这里的较晚和较早是指正在进行的对话的时间顺序），以指示它们的关系。

## 3 Baseline System

我们采用标准 Transformer  进行对话历史编码。 通常，Transformer 编码器由 L 层组成，采用一系列token（即对话历史）$\mathcal{S}=\left\{w_{1}, w_{2}, \ldots, w_{N}\right\}$，其中 $w_{i}$ 是第 i 个token，N 是序列长度， 作为输入并迭代地产生向量化的词表示 $\left\{h_{1}^{l}, h_{2}^{l}, \ldots, h_{N}^{l}\right\}$，$l \in[1, \ldots, L]$。 总的来说，一个 Transformer 编码器可以写成：
$$
H=\operatorname{SeqEncoder}(\operatorname{emb}(\mathcal{S}))\tag{1}
$$
其中 H = $\left\{h_{1}^{l}, h_{2}^{l}, \ldots, h_{N}^{l}\right\} $，emb 表示将token序列映射到相应嵌入的函数。 每个 Transformer 层由两个子层组成：一个自注意力子层和一个位置前馈网络。 前者计算一组注意力分数：
$$
\alpha_{i j}=\operatorname{Attn}\left(h_{i}, h_{j}\right)\tag 2
$$
用于更新 $w_i$ 的隐藏状态:
$$
h_{i}^{l}=\sum_{j=1}^{N} \alpha_{i j}\left(W^{V} h_{j}^{l-1}\right)\tag3
$$
其中 $W^V$ 是参数矩阵。

位置前馈 (FFN) 层由两个线性变换组成：
$$
\operatorname{FFN}(h)=W_{2} \operatorname{ReLU}\left(W_{1} h+b_{1}\right)+b_{2}\tag4
$$
其中 $W_1,W_2,b_1,b_2$ 是模型参数。

### 3.1 Dialogue Understanding Task

我们以对话关系抽取任务 (Yu et al., 2020) 为例。 给定对话历史 $\mathcal{S}$ 和参数（或实体）对 $(a_1,a_2)$，目标是预测 $a_1$ 和 $a_2$ 之间的对应关系类型 $r\in\mathcal{R}$。 我们遵循先前的对话关系提取模型将 $a_1$ 和 $a_2$ 的隐藏状态（表示为 $h_{a_1},h_{a_2}$ ）输入分类器以获得每种关系类型的概率：
$$
P_{r e l}=\operatorname{softmax}\left(W_{3}\left[h_{a_{1}} ; h_{a_{2}}\right]+b_{3}\right)\tag5
$$
其中 $W_3$ 和 $b_3$ 是模型参数。  $P_{rel}$ 的第 k 个值是 $\mathcal{R}$ 中第 k 个关系的条件概率。

给定一个训练实例 <S, a1, a2, r>，局部损失为：
$$
\ell=-\log P\left(r \mid \mathcal{S}, a_{1}, a_{2} ; \theta\right)\tag6
$$
其中 θ 表示模型参数集。 在实践中，我们使用 BERT 来计算 $h_{a_1},h_{a_2}$，这可以看作是 Transformer 编码器的预训练初始化。

### 3.2 Dialogue Response Generation Task

给定对话历史 $\mathcal{S}$，我们使用标准的自回归 Transformer 解码器来生成回复 $\mathcal{Y}=\left\{y_{1}, y_{2}, \ldots, y_{|\mathcal{Y}|}\right\}$。 在时间步长 t，前一个输出词 $y_{t-1}$ 首先通过 self-attention 层转换为隐藏状态 $s_t$，如等式 2 和 3。然后应用编码器-解码器注意力机制从编码器输出隐藏状态中获得上下文向量$\left\{h_{1}^{L}, h_{2}^{L}, \ldots, h_{N}^{L}\right\}$：
$$
\begin{aligned}
\hat{\alpha}_{t i} &=\operatorname{Attn}\left(s_{t}, h_{i}^{L}\right) \\
c_{t} &=\sum_{i=1}^{N} \hat{\alpha}_{t i} h_{i}^{L}
\end{aligned}\tag7
$$
​		然后使用获得的上下文向量 $c_t$ 计算目标词汇表上下一个单词 $y_t$ 的输出概率分布：
$$
P_{v o c}=\operatorname{softmax}\left(W_{4} c_{t}+b_{4}\right)\tag8
$$
其中 $W_4,b_4$ 是可训练的模型参数。  $P_{voc}$ 的第 k 个值是给定对话的词汇表中第 k 个单词的条件概率。 

​		给定对话历史-回复对 $\{\mathcal{S},\mathcal{Y}\}$，该模型最小化交叉熵损失：
$$
\ell=-\sum_{t=1}^{|Y|} \log P_{v o c}\left(y_{t} \mid y_{t-1}, \ldots, y_{1}, \mathcal{S} ; \theta\right)\tag9
$$
其中 θ 表示所有模型参数。

## 4 Proposed Model

我们的模型将对话历史 $\mathcal{S}$ 和相应的对话 AMR 作为输入。 形式上，一个AMR 是有向无环图 $$
\mathcal{G}=\langle\mathcal{V}, \mathcal{E}\rangle
$$，其中 $\mathcal{V}$ 表示一组节点（即 AMR 概念），$\mathcal{E}$（即 AMR 关系）表示一组有标记的边。 一条边可以进一步用三元组表示$$
\left\langle n_{i}, r_{i j}, n_{j}\right\rangle
$$，这意味着边是从节点 ni 到 nj ，标签为 rij。

​		我们考虑使用对话级 AMR 的两种主要方式。 第一种方法（图 3（a））使用 AMR 语义关系来丰富对话历史的文本表示。 我们将 AMR 节点投影到相应的标记上，通过对单词之间的语义关系进行编码来扩展 Transformer。 对于第二种方法，我们分别对 AMR 及其句子进行编码，并使用特征融合（图 3（b））或双重注意（图 3（c））来合并它们的嵌入。

![image-20210928205136581](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210928205136581.png)

### 4.1 Graph Encoding

我们采用 Graph Transformer (Zhu et al., 2019[Modeling graph structure in transformer for better AMR-to-text generation.](https://aclanthology.org/D19-1548.pdf)) 来编码 AMR 图，它扩展了标准 Transformer 以对结构输入进行建模。  一个L 层的 图 Transformer 采用一组节点嵌入 $$
\left\{\boldsymbol{n}_{1}, \boldsymbol{n}_{2}, \ldots, \boldsymbol{n}_{M}\right\}
$$ 和一组边嵌入  $\left\{\boldsymbol{r}_{i j} \mid i \in[1, \ldots, M], j \in\right.$ $(1, \ldots, M]\}$ (没边记为'None') 作为 input 并迭代地产生更多抽象节点特征 $\left\{h_{1}^{l}, h_{2}^{l}, \ldots, h_{M}^{l}\right\}$，其中 l ∈ [1, ..., L]。 图 Transformer 和标准 Transformer 之间的主要区别在于图注意力层。 与 Self-attention 层（等式 2）相比，图注意力层在更新节点隐藏状态时明确考虑了图边。例如，给定一条边，计算$$
\left\langle n_{i}, r_{i j}, n_{j}\right\rangle
$$注意力分数 $\hat{\alpha}_{i j}$：
$$
\begin{aligned}
\hat{\alpha}_{i j} &=\frac{\exp \left(\hat{e}_{i j}\right)}{\sum_{m=1}^{M} \exp \left(\hat{e}_{i m}\right)} \\
\hat{e}_{i j} &=\frac{\left(W^{Q} h_{i}^{l-1}\right)^{T}\left(W^{K} h_{j}^{l-1}+W^{R}{\boldsymbol{r}}_{i j}\right)}{\sqrt{d}}
\end{aligned}\tag{10}
$$
其中 $W_R$ 是变换矩阵，$\boldsymbol{r_{i j}}$ 是关系 rij 的嵌入，d 是隐藏状态大小，$\left\{h_{1}^{0}, h_{2}^{0}, \ldots, h_{M}^{0}\right\}=\left\{\boldsymbol{n}_{1}, \boldsymbol{n}_{2}, \ldots, \boldsymbol{n}_{M}\right\}$。 然后将 ni 的隐藏状态更新为：
$$
h_{i}^{l}=\sum_{j=1}^{M} \alpha_{i j}\left(W^{V} h_{j}^{l-1}+W^{R}{r}_{i j}\right)\tag{11}
$$
其中 $W^V$ 是参数矩阵。 总的来说，给定一个输入 AMR 图，$$
\mathcal{G}=\langle\mathcal{V}, \mathcal{E}\rangle
$$，图 Transformer 编码器可以写成
$$
H=\text { GraphEncoder }(\operatorname{emb}(\mathcal{V}), \operatorname{emb}(\mathcal{E}))\tag{12}
$$
其中 $$
H=\left\{h_{1}^{L}, h_{2}^{L}, \ldots, h_{M}^{L}\right\}
$$表示顶层图编码器隐藏状态。

### 4.2 Enriching Text Representation

我们首先使用 JAMR aligner (Flanigan et al., 2014 [A Discriminative Graph-Based Parser for the Abstract Meaning Representation](https://aclanthology.org/P14-1134.pdf)) 获得节点到单词的对齐，然后采用对齐将 AMR 边缘投影到文本上，规则如下：
$$
\hat{r}_{i j}=\left\{\begin{array}{rr}
r_{i^{\prime} j^{\prime}}, & \text { if } \mathcal{A}\left(n_{i^{\prime}}\right)=w_{i}, \mathcal{A}\left(n_{j^{\prime}}\right)=w_{j} \\
\text { Self, } & \text { if } i=j \\
\text { None, } & \text { otherwise }
\end{array}\right.\tag{13}
$$
其中 A 是 one-to-K 对齐（K ∈ [0, . . , N]）。 这样，我们就得到了一个投影图 $$
\mathcal{G}^{\prime}=\left\langle\mathcal{V}^{\prime}, \mathcal{E}^{\prime}\right\rangle
$$，其中 $\mathcal{V}^{\prime}$ 表示输入词的集合 {w1, w2, ..., wN}，$\mathcal{E}^{\prime}$ 表示一组word-to-word语义关系。

​		受之前 AMR 图建模工作的启发，我们采用了一个分层编码器，该编码器堆叠了一个序列编码器和一个图编码器。 序列编码器 (SeqEncoder) 将对话历史转换为一组隐藏状态：
$$
H^{S}=\operatorname{SeqEncoder}(\operatorname{emb}(\mathcal{S}))\tag{14}
$$
​		图编码器将投影的关系特征合并到 $H^S$ 中：
$$
H^{\hat{S}}=\text { GraphEncoder }\left(H^{S}, \operatorname{emb}\left(\mathcal{E}^{\prime}\right)\right)\tag{15}
$$
​		此外，我们在图适配器和序列编码器之间添加了一个残差连接，以融合细化前后的词表示（如图 3（b）所示）：
$$
H^{F}=\operatorname{LayerNorm}\left(H^{S}+H^{\hat{S}}\right)\tag{16}
$$
​		其中 LayerNorm 表示层标准化。 我们将分层编码器命名为 *Hier*，它可用于对话理解和对话响应生成。

### 4.3 Leveraging both Text and Structure Cues

我们考虑使用双编码器网络整合文本线索和 AMR 结构线索以进行对话理解和响应生成。 

​		首先，序列编码器用于使用公式 1 将对话历史 S 转换为*text memory*（表示为 $$
H^{S}=\left\{h_{1}^{S}, h_{2}^{S}, \ldots, h_{N}^{S}\right\}
$$）。

​		其次，AMR 图 $\mathcal{G}$ 由图 Transformer 编码器使用等式 12 编码到 *graph memory*（表示为 $H^{G}=\left\{h_{1}^{G}, h_{2}^{G}, \ldots, h_{M}^{G}\right\}$）。

​		对于对话理解（图 3(b)） 和对话回复生成（图 3（c）），由于输出的不同性质，使用的特征集成方法略有不同。

**Dialogue Understanding.	**与第 4.2 节类似，我们首先使用 JAMR aligner 获得节点到词对齐 $\mathcal{A}$。然后我们将词和 AMR 节点表示融合如下：
$$
\hat{h}_{i}=\left\{\begin{array}{lr}
f\left(h_{i}^{S}, h_{j}^{G}\right), & \text { if } \exists j, \mathcal{A}\left(n_{j}\right)=w_{i} \\
f\left(h_{i}^{S}, h_{\emptyset}\right), & \text { otherwise }
\end{array}\right.\tag{17}
$$
其中 h∅ 是虚拟节点的向量表示（见图 2），f 定义为：
$$
h=\text { LayerNorm }\left(h_{1}+h_{2}\right)\tag{18}
$$
然后将融合的词表示输入到分类器中进行关系预测（等式 5）。

**Dialogue Response Generation.**	暂空

## 5 Dialogue Understanding Experiments

我们在 DialogRE 上评估我们的模型，该模型总共包含 1,788 个对话、10,168 个关系三元组和 36 个关系类型。 平均而言，DialogRE 中的对话包含 4.5 个关系三元组和 12.9 个回合。 我们报告了原始 (v1) 和更新 (v2) 英文版的实验结果。

### 5.1 Settings

我们采用与 Yu 等人相同的输入格式和超参数设置，用于提出的模型和baseline。 特别地，输入序列构造为 [CLS]d[SEP]a1[SEP]a2[SEP]，其中 d 表示对话，a1 和 a2 是两个关联的参数。 在 Yu 等人的 BERT 模型中，只有 [CLS] 标记的隐藏状态被输入分类器进行预测，而我们的基线 ($BERT_c$) 额外采用 a1 和 a2 的隐藏状态。 所有超参数都是根据验证数据集的预测精度选择的（详细超参数见表 6）。

**指标**	继之前在 DialogRE 上的工作之后，我们报告了标准 (F1) 和对话设置 (F1c; Yu 等人，2020) 中关系的宏观 F1 分数。  F1c 是在对话的前几轮中计算的，其中首先提到了两个参数。

### 5.2 Main Results

表 1 显示了不同系统在 DialogRE 上的结果。 我们将提出的模型与两种基于 BERT 的方法 BERT 和 $BERT_s$ 进行比较。 基于 BERT，$BERT_s$ 通过用特殊标记替换说话者参数来突出说话者信息。 为完整起见，我们还包括最近的方法，例如 AGGCN、LSR和 DHGAT。  $BERT_c$ 和 Hier, Dual 分别代表我们的基线和提出的模型。

​		通过结合说话者信息，$BERT_s$ 提供了之前系统中最好的性能。 我们的 $BERT_c$ 基线在很大程度上优于 $BERT_s$，因为 $BERT_c$ 还考虑了用于分类的参数表示。  Hier (p < 0.01) 在所有设置中均显着优于 $BERT_c$，F1 分数平均提高 1.4 分。 在 F1c 下观察到类似的趋势。 这表明 AMR 中的语义信息有利于对话关系提取，因为 AMR 突出了核心实体和它们之间的语义关系。  Dual 的结果略好于 Hier，显示了对语义结构进行单独编码的效果。

![image-20210929150102946](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210929150102946.png)

​		最后，Dual 和 Hier 的标准偏差值都低于基线。 这表明我们的方法在模型初始化方面更加稳健。

### 5.3 Impact of Argument Distance

我们根据两个参数之间基于话语的距离将 DialogRE (v2) 开发集的对话分为五组。 如图 4 所示，除了参数距离小于 5 时，Dual 的结果比 BERTc 更好。特别是，当参数距离大于 20 时，Dual 大大超过了 BERTc。比较表明 AMR 可以帮助模型 通过改进实体召回来更好地处理长期依赖关系。 除了话语距离之外，我们还考虑了词距离并观察到类似的趋势（如附录 7 所示）。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/M9%7D%7DJZWGCLVC6750KFIS%7DYT.png)

### 5.4 Case Study

图 5 显示了经理和可能已休假的员工之间的对话。 基线模型错误地预测两个对话者之间的关系是父子关系。 假设这是家庭成员之间的对话，它可能会受到对话中最后一句话的影响。 然而，所提出的模型成功地预测了对话者的关系，表明它可以从综合的角度提取对话中的全局语义信息。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/LKTZZU9HWTWRUUY{%BUU91A.png)

