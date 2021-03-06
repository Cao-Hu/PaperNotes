# Towards Emotional Support Dialog Systems

ACL 2021

[面向情感支持的对话系统](https://arxiv.org/abs/2106.01144)

## Abstract

​		情感支持是许多对话场景的关键能力，包括社交互动、心理健康支持和客户服务聊天。 遵循合理的程序并使用各种支持技巧有助于有效地提供支持。 然而，由于缺乏精心设计的任务和有效的情感支持对话语料库，在对话系统中构建情感支持的研究仍未触及。 在本文中，我们定义了情感支持对话 (ESC) 任务并提出了一个基于帮助技能理论的 ESC 框架 (Hill, 2009)。 我们以求助者和支持者的模式构建了一个带有丰富注释（尤其是支持策略）的情感支持对话数据集（ESConv）。 为了确保提供高质量的对话语料库，提供有效的情感支持示例，我们付出了大量努力为支持者设计培训教程和数据收集过程中的质量控制的几种机制。 最后，我们在提供情感支持的能力方面评估了最先进的对话模型。 我们的结果表明支持策略在提供有效情感支持方面的重要性以及 ESConv 在训练更多情感支持系统方面的效用 。

## 1 Introduction

​		情感支持 (ES) 旨在减少个人的情绪困扰，帮助他们理解并应对他们面临的挑战。 训练日常与用户交互的对话系统是一项关键能力（[Zhou 等人，2020 年, 小冰](https://zhuanlan.zhihu.com/p/57532328)），特别是对于包括社交互动（陪伴和鼓励用户）的设置 、心理健康支持（安慰沮丧的求助者并帮助确定问题）、客户服务聊天（安抚愤怒的客户并提供解决方案）等。最近的研究还表明，人们更喜欢可以提供更多支持性回应的对话系统（  Rains 等人，2020 年）。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/[%[_E0GMDC1]]@D[B8P4NWI.png)

<center>图 1：一个聊天示例，显示了支持者（右）向求助者（左）提供的有效情感支持（改编自 ESConv）,</center>
<center>支持者使用的支持策略（技能）在话语前的括号中标出, </center><center>虚线框中的红色粗体文本突出显示了我们提议的 ESC 框架的三个阶段（图 3）。</center>

​		为了找出求助者痛苦的原因，支持者首先探讨求助者的问题。没有探索，支持者不太可能理解求助者的经历和感受，因此如果支持者给出不相关的建议，比如“你可以去散散步放松一下”，可能会冒犯甚至有害。 在了解求助者的情况时，支持者可以通过各种技巧（如自我表露、感想等）来表达理解和同理心，以缓解求助者的挫败感。 在了解求助者的问题后，支持者可以提出建议，帮助求助者解决问题。 如果支持者只是安慰求助者而没有任何行动改变的灵感，支持者可能无法有效帮助求助者的情绪改善。 最后，在此示例对话的数据收集过程中，求助者报告其情绪强度从 5 下降到 2（情绪强度在我们的语料库中进行了标记，我们在附录 A 中给出了此对话示例的详细注释），这表明 支持者提供的 ES 的有效性。

​		尽管 ES 很重要且复杂，但由于缺乏任务设计和相关的对话语料库，因此对数据驱动的 ES 对话系统的研究是有限的。 首先，与情感聊天或共情反应相关的现有研究系统返回的信息是情感或共情的例子，因此在功能上受到限制，因为它们没有能力 经常用于提供有效 ES 的许多其他技能。 图 2 说明了三个任务之间的关系，我们在 2.1 节中提供了进一步的讨论。 其次，人们天生不擅长提供支持，因此已经制定了指导方针来训练人类如何提供更多支持。 如果没有经过培训的个人，现有的在线对话数据集不会自然地展示支持性对话的示例或元素。 因此，利用此类语料库的数据驱动模型在明确学习如何利用支持技能并因此提供 有效的 ES。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/2CUQOAA5EIME6JU22Z%VE74.png)

<center>图 2：情感支持对话（我们的工作）可以包括情感聊天（Zhou 等人，2018 年）和</center><center>共情回复（Rashkin 等人，2019 年）的元素。</center>

​		在本文中，我们定义了情感支持对话 (ESC) 的任务，旨在通过社交互动（如同龄人、朋友或家人之间的互动）而不是专业咨询来提供支持，并提出了一个基于 Helping Skills Theory (Hill, 2009) 并根据对话系统设置量身定制（图 3）。 我们通过改编希尔对话支持的帮助技能模型的相关组件，为对话系统设置精心设计了 ESC 框架。  ESC 框架提出了三个阶段（探索、安慰和行动），其中每个阶段包含几个支持策略（或技能）。 为了促进情感支持对话的研究，我们构建了一个情感支持对话数据集 ESConv，并采取多种措施来确保丰富的注释，并且所有对话都是这个特别复杂的对话任务的优质示例。  ESConv 是由众包工作者以求助者和支持者的角色聊天收集的。 我们基于 ESC 框架设计教程，对所有支持者进行培训，并设计多种手动和自动机制，以确保对话中情感支持的有效性。 最后，我们评估了最先进的模型，并观察到在使用各种支持策略时所提供的情感支持的显着改善。 对交互式评估结果的进一步分析表明，联合模型可以模拟人类支持者在策略利用中的行为。 我们相信我们的工作将促进对更多数据驱动方法的研究，以构建能够提供有效情感支持的对话系统。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/L`0YW_(XI4%[)U%R8GIJPNR.png)

<center>图 3：ESC 框架概述。 它包含三个阶段和建议的支持策略。</center> <center>情感支持程序一般遵循以下顺序：1探索→2安慰→3行动（黑色箭头所示），</center><center>但也可以根据需要和个人对话进行调整（灰色虚线箭头所示）。</center><center>  Lexical Features列显示了与使用我们数据集中每种策略的消息相关的前 5 个一元组或二元组。 </center><center>每个特征都按括号中的舍入 z 得分对数优势比 (Monroe et al., 2008) 进行排名。</center>

## 2 Related Work

### 2.1 Emotional & Empathetic Conversation

​		图 2 直观地展示了 ESC、情感对话和共情对话之间的关系。 情感已被证明对于构建更具吸引力的对话系统很重要。 作为情感对话的重要的工作，周等人(2018) 提出Emotional Chatting Machine (ECM) 以在给定预先指定的情绪的情况下生成带情绪的回复。 此任务需要在生成的回复中准确表达（指定或未指定）情绪。 虽然 ES 可能包括表达情绪，例如快乐或悲伤，但它有一个更广泛的目标，即通过利用适当的支持技能来减少用户的情绪困扰，这与情感聊天有着根本的不同。 情感聊天只是对话系统的基本素质，而ES则是对话系统应具备的更高层次、更复杂的能力。 另一个相关任务是共情回复，旨在了解用户的感受，然后做出相应的回复。 例如，拉什金等人(2019) 认为对话模型可以通过识别对话者的感受来产生更多的共情反应。 有效的 ES 自然需要根据求助者的经历和感受表达同理心，如我们提出的情感支持框架（第 3.2 节，图 3）所示。 因此，共情反应只是情感支持的必要组成部分之一。 除了共情回复，情感支持对话还需要探索用户的问题并帮助他们应对困难。

### 2.2 Related Datasets for Emotional Support

​		许多工作都考虑了社交环境中的情感支持对话，例如社交媒体或在线论坛。  Medeiros 和 Bosse（2018 年）从 Twitter 收集了与压力相关的帖子和回复对，并将回复分类为支持类别。 在 (Sharma et al., 2020b) 中，来自 TalkLife 和心理健康 subreddits 的 post-response 对用基于文本的同理心表达的通信机制进行了注释（只有 Reddit 部分的数据是公开的）。  Hosseini 和 Caragea (2021) 还从在线支持小组中收集了此类回复后配对，已被注释为需要或表达支持。这些语料库中的对话要么是单轮交互（post-response pair），要么是非常短的对话，这限制了有效 ES 的潜力，因为 ES 通常需要多次交互（Hill，2009）。

### 2.3 Emotional Support Dialog Systems

​		一些传统的对话系统应用了人类精心设计的规则来提供情感支持反应。最近的一个系统考虑了一种基于规则的算法，该算法确定回复中使用的支持行为，然后从预定义的候选列表中选择适当的回复（Medeiros和Bosse，2018）。另一个旨在为应对Covid-19提供支持的对话系统是通过识别用户提及的主题，然后通过模板的反映或预定义词汇的信息进行回复而实现的（Welch等人，2020年）。很少有研究关注生成支持性回复，而且这些研究的范围有限。例如，Shen等人（2020年）探索了如何通过反思用户输入来生成支持性回复。

## 3  Emotional Support Conversation

### 3.1 Task Definition

​		当用户处于不良情绪状态时，可能是由于特定问题，他们可能会寻求帮助以改善其情绪状态。 在此设置中，用户可以被标记为负面情绪标签 *e*、情绪强度级别 *l*（例如，范围从 1 到 5）以及用户正在经历的潜在挑战。 支持者（或系统）需要通过支持技巧在对话中安慰用户，以降低他们的情绪强度水平。 请注意，在对话之前，支持者不知道用户的状态。 在对话过程中，支持者需要识别用户面临的问题，安慰用户，然后提供一些建议或信息来帮助用户采取行动来应对他们的问题。 如果在谈话结束时降低用户的情绪强度水平，或者更具体地说，如果支持者能够有效地识别问题、安慰用户并提供解决方案或建议，则情感支持谈话是有效的。

​		ESC 任务有几个子问题：（1）支持策略选择和策略约束的回复生成。 正如我们后面的实验（第 6.4 节）所示，应用策略的时机与 ES 的有效性有关。 因此，生成的回复符合指定的策略非常重要。  (2)情绪状态建模。 对于动态策略选择和衡量 ESC 的有效性，动态建模和跟踪用户的情绪状态非常重要。  (3) 支持效果评估。 除了评估对话的相关性、连贯性和用户参与度的传统维度外，ESC 提出了评估 ES 有效性的新维度。

### 3.2 ESC Framework

​		我们提出了一个 ESC 框架，它将情感支持的过程分为三个阶段，每个阶段都有几个建议的支持策略。 我们将 ESC 框架建立在 Hill 的帮助技能理论的基础上，并使其更适合对话系统设置，旨在通过社交互动（如同龄人、朋友或家人之间的互动）提供支持，而不仅仅是专业咨询 . 图 3 显示了 ESC 框架中对话阶段和策略的概述。 

**Stages：** Hill (2009) 提出了支持人们的三个阶段：*exploration*（探索以帮助求助者识别问题）、*insight*（帮助求助者进入自我理解的新深度）和 *action*（帮助求助者做出应对问题的行动决定）。 然而，我们注意到 *insight* 通常需要重新解释用户的行为和感受，这对于没有足够支持经验的支持者来说既困难又冒险。 因此，我们将 *insight* 转化为 *comforting*（定义为通过同理心和理解提供支持）。 虽然建议情感支持对话针对这三个有序阶段，但实际上对话不能遵循固定或线性顺序，必须适当调整。 正如 (Hill, 2009) 所建议的，这三个阶段可以灵活调整以满足求助者的需要。  

**Strategies：** Hill (2009) 还为每个阶段提供了几种推荐的会话技巧。 在没有专业监督和经验的对话系统设置中，某些描述的技能是不合适的。 为了使这些技能适合对话系统设置，我们从这些技能中提取了七种方法（以及“其他”方法），我们在我们的任务中和以后将其称为策略。 我们在附录 B 中提供了每个策略的详细定义。

## 4 Data Collection

​		为了促进对话系统中情感支持技能的研究，我们引入了情感支持对话数据集 ESConv，该数据集以求助者和支持者模式与众包工作人员一起收集。 由于这项复杂的任务需要高质量的对话示例，因此我们付出了巨大的努力来确保 ES 在对话中的有效性。 我们的努力包括以下主要方面： (1) 因为提供对话支持是支持者必须经过培训才能有效的技能 (Burleson, 2003)，我们使用 ESC 框架设计了一个教程，并培训众包工作者成为支持者。 只有通过考试的人才能进入任务。  (2) 我们要求求助者完成一个关于他们的问题和情绪的聊天前调查，并在谈话期间和之后提供反馈。  (3) 在收集原始对话数据后，我们设计并使用多种手动或自动机制来过滤掉低质量的对话。

### 4.1 Supporter-specific Tasks

**Training and Examination**	为了教会众包工作者如何提供有效的情感支持，我们设计了一个带有 ESC 框架的教程。 受 7cups (7cups.com)的启发，我们开发了 11 个子任务 (3 + 8) 来帮助员工学习三个阶段的定义和八个支持策略。 每个子任务包括一个示例对话摘录和一个相应的测验问题。 如第 3.2 节所述，我们还告知参与者可能无法遵循固定顺序，可能需要灵活调整阶段过渡。

**Strategy Annotation**	为了鼓励支持者在对话期间使用 ESC 支持策略并构建结果数据集，我们要求支持者首先根据对话上下文选择他们想要使用的适当策略。 然后他们能够写出反映他们选择的策略的话语。 如果支持者想使用多种策略来提供支持，我们鼓励他们发送多条消息。

**Post-chat Survey**	在每次谈话之后，支持者被要求评价求助者在李克特LIKERT五分量表上详细讨论他们的问题的程度。

### 4.2 Seeker-specific Tasks

**Pre-chat Survey**	在每次谈话之前，求助者被要求完成以下调查： (1) 问题和情感类别：求助者应从 5 个选项中选择一个问题，并从 7 个选项中选择一个情感（选项基于试点数据中收集的对话）。  (2)情绪强度：1-5分（数字越大表示情绪越强烈）。  (3) 情境：描述情绪问题原因的公开文本。  (4) 经验来源：所描述的情况是求助者目前的经验还是基于以前的生活情况。 我们发现 75.2% 的对话源自求助者当前的经历。

**Feedback**	在对话过程中，求助者每收到两次来自支持者的新话语，就被要求提供反馈。 他们的反馈对支持者信息的帮助程度进行了 5 星评分。 我们将每个对话分为三个阶段，并计算每个阶段的平均反馈分数。 三个阶段的得分分别为4.03、4.30和4.44，表明支持者得到了充分的培训，可以有效地帮助求助者感觉更好。

**Post-chat Survey**	每次谈话后，求助者都被要求根据以下李克特LIKERT五分量表对他们的情绪和支持者的表现进行评分：（1）他们在情感支持谈话后的情绪强度（比谈话前的强度降低反映情绪的改善 )，(2) 支持者对求助者的经历和感受的同理心和理解，以及 (3) 支持者对对话主题的回复的相关性。

### 4.3 Quality Control

我们使用多种方法来确保语料库包含有效情感支持对话的高质量示例。

**Preliminary Filtering Mechanisms**	在招募支持者角色的参与者时，我们最初收到了 5,449 名申请者，但只有 425 人 (7.8%) 通过了培训教程。 从我们最初收集的 2,472 条对话中，我们过滤掉了那些求助者未完成或少于 16 条话语的对话。 此过滤留下 1,342 个对话 (54.3%) 以供考虑。

**Auto-approval Program for Qualified Conversations**	我们精心设计了自动审批程序，这是数据质量控制中最重要的部分。 该程序使用基于角色和话语长度的Post-chat Survey回复的标准，表 1 总结了这些标准。这些标准基于最初的人工审查结果。 我们在附录 D 中展示了如何选择这些自动批准标准。计算出的对话前的平均情绪强度为 4.04，之后为 2.14。 这种改善证明了支持者提供的情感支持的有效性。 在少数对话中，求助者没有完成Post-chat Survey，因此我们为这些对话添加了另一个标准，要求求助者的最后两个反馈得分均大于 4。因此，其中 没有Post-chat Survey的对话，只有同时满足（2）和（3）的人才有资格。 使用这些质量标准，1,053 个（1,342 个的 78.5%）收集到的对话是合格的。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/H$TUD~8$0QTXO%5BQZJ%7B5RO%5BP.png)

<center>表 1：高质量对话的标准。  * 表示支持者必须至少满足三个标准中的两个。</center><center> 在**中，求助者情绪强度的改善是通过从谈话前的强度减去之后的强度来计算的。</center>

**Annotation Correction**	为了进一步确保数据质量，我们审查并修改了支持策略和寻求者情绪强度的错误注释。  （1）对于策略注释修正，我们要求新的合格支持者根据需要审查和修改先前收集的对话的注释，导致审查了2,545条话语（17.1％）。 我们手动审查了超过 75% 的评论者不同意的注释，并修改了其中的 139 个。  (2) 根据自动认可标准（表 7），当寻求者情绪改善的分数小于 1，但满足其他三个标准时，对话才合格。 经过审查，我们发现这通常是由于寻求者将消极情绪的强度误认为他们情绪的积极性所致。 我们通过使用其他有用信息手动重新检查和修改这些对话的情绪强度，例如对聊天后调查开放问题的回答和聊天过程中寻求者的反馈分数。 在 130 个这样的对话中，92% 被修改并包含在语料库中。

## 5 Data Characteristics

### 5.1 Statistics

​		表 2 显示了 1,053 个 ESConv 示例的总体统计数据。相对较长的对话（平均 29.8 次话语）表明，提供有效的 ES 通常需要多次交互，并且比以前的情感聊天的典型轮次要多得多（Zhou 等人，  2018) 或移情对话 (Rashkin et al., 2019) 数据集。 我们还在表 3 中提供了其他注释的统计数据。 也许由于当前 COVID-19 的爆发，持续的抑郁和工作危机是寻求帮助者最常见的问题，而抑郁和焦虑是最常见的情绪。 从求助者的反馈中，我们发现他们通常对情感支持非常满意，这进一步表明基于ESC框架的培训教程确实帮助支持者学习提供有效的ES。 我们发布所有这些注释以促进进一步的研究。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/L@CJ58}T9%`SI1KJ`GU3L2I.png)

![image-20210928164928056](https://gitee.com/cao-hu/pictures/raw/master/img/image-20210928164928056.png)

### 5.2 Strategy Analysis

**Lexical Features**	我们通过计算每个策略与所有其他策略对比的所有 unigrams 和 bigrams 的对数优势比、信息性狄利克雷先验来提取每个策略的词汇特征。 我们在图 3 中列出了每个策略的前 5 个短语。这些策略都与某些短语（例如，带有“are you”的Question，带有“I”的Self-disclosure）显着相关（z-score > 3）。

**Strategy Distribution**	我们计算了对话不同阶段的策略分布。 对于总共有L个句子的对话，第k（1≤k≤L）句来自支持者并采用策略st，我们说它位于对话进度k/L处。 具体来说，我们将对话进程分为六个区间。 然后，对于 ESConv 中的所有对话，我们统计了六个区间内不同策略的比例。 我们将对话进程分为六个区间并在六个点处绘制六个区间的分布并将它们连接起来，最终得到图 4。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/%7DE(MEV%5BOOZ$VH)HE5K00WJJ.png)

​		支持者一般遵循ESC框架建议的阶段顺序（图3），但也有阶段的灵活调整和策略的采用。 例如，在谈话的早期阶段，支持者通常采用诸如问题等探索性策略。 支持者在了解求助者的情况后，倾向于提出自己的意见（如提供建议）。 在整个对话过程中，使用安慰策略（例如肯定和再保证）并标记相对恒定比例的消息。

**Strategy Transition**	我们在附录（表 6）中展示了具有 3 或 4 跳的前 5 名最频繁的策略转换。 这些转变表明，在 ESC 框架教程的培训中，支持者通常会在安慰求助者之前提出问题并探索求助者的情况。

## 6 Experiments

​		我们的实验集中在两个关键问题上：（1）带有策略注释的 ESConv 可以在多大程度上改进最先进的生成对话模型？  (2) 这些模型能否学会从 ESConv 提供有效的情感支持？

### 6.1 Backbone Models

我们使用两个最先进的预训练模型作为比较变体模型的支柱：

**BlenderBot** 	BlenderBot（Roller 等人，2020 年）是一种开放域对话代理，经过多种沟通技巧训练，包括共情回复。 因此，BlenderBot 应该能够在一定程度上为用户提供 ES。 我们在实验中使用了 BlenderBot 的小版本 ，因为大版本有最大上下文长度 128 的限制，我们发现这会损害模型性能和回复一致性。

**DialoGPT**	 我们还评估了 DialoGPT (Zhang et al., 2020)，这是一种在大规模对话语料库上预训练的基于 GPT-2 的模型。 我们使用了小版本。

### 6.2 Variant Models

以上述每个预训练模型为骨干，我们构建了以下变体模型：

**Vanilla**	直接微调 ESConv 上的主干模型，无法访问策略注释。 形式上，假设展平的对话历史是 x 并且要生成的响应是 y，我们最大化条件概率：$\mathbb{P}(\mathbf{y} \mid \mathbf{x})=\prod_{i=1}^{|\mathbf{y}|} \mathbb{P}\left(y_{i} \mid \mathbf{x}, \mathbf{y} \leq i\right)$

**Variants with strategy**	为了将策略注释合并到主干模型中，我们使用了一个特殊的标记来表示每个策略。 对于来自支持者的每个句子 y，我们在该句子之前附加相应的策略标记：$\tilde{\mathbf{y}}=[\mathrm{st}] \oplus \mathbf{y}$，其中 $[\mathrm{st}]$ 表示所使用策略的特殊标记。 然后，将扁平化的对话历史 x 作为输入，模型生成以第一个预测（或指定）策略标记为条件的响应：$\mathbb{P}(\tilde{\mathbf{y}} \mid \mathbf{x})=$ $\mathbb{P}([\mathrm{st}] \mid \mathbf{x}) \prod_{i=1}^{|\mathbf{y}|} \mathbb{P}\left(y_{i} \mid \mathbf{x},[\mathrm{st}], \mathbf{y}_{<i}\right)$

​		我们在后面的实验中研究了三种使用策略注释的变体。  (1) **Oracle**：根据gold reference strategy tokens生成响应。  (2) **Joint**：根据预测（采样）策略标记生成响应。  (3) **Random**：根据随机选择的策略生成响应。 实施细节在附录 C 中。

### 6.3 Automatic Evaluation

为了研究使用支持策略对以 BlenderBot 或 DialoGPT 作为主干的模型性能的影响，我们比较了上述 Vanilla、Joint 和 Oracle 变体的性能。 我们采用的自动度量包括 **perplexity (PPL)**、**BLEU-2 (B2)** 、**ROUGE-L (RL)**  和 BOW Embedding-based **Extrema** 匹配分数。 除 PPL 之外的指标是使用 NLG 评估工具包计算的，回复由 NLTK 标记。

​		实验有三个主要发现（表 4）。  (1) Oracle 模型在所有指标上都显着优于 Vanilla 模型，表明支持策略的巨大效用。  (2) Joint 模型的得分略低于 Vanilla 模型，因为如果预测策略与真实情况不同，生成的回复将与参考回复有很大不同。 然而，当没有提供真实标签时，学习预测策略很重要，我们将进一步研究联合模型在人类交互评估中的性能（第 6.4 节）。  (3) BlenderBot 变体始终比 DialoGPT 变体表现更好，表明 BlenderBot 更适合 ESC 任务。因此，在随后的人工评估中，我们将重点评估 Blenderbot 变体。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/V%60~Z8%60P7L71$Q_F92EATJPF.png)

### 6.4 Human Interactive Evaluation

我们招募了来自 Amazon Mechanical Turk 的参与者与模型聊天。 在线测试与我们的数据收集在同一平台上进行，但模型承担了支持者的角色。 每个参与者都与两个不同的模型聊天，这些模型被随机排序以避免暴露偏差。 参与者被要求根据以下问题比较这两个模型：（1）**Frequency**：哪个机器人的反应更流畅和更易理解？  （2）**Identification**：哪个机器人更深入地探索了你的情况，更有助于识别你的问题？  （3）**Comforting**：哪个机器人在安慰你方面更熟练？  (4) **Suggestion**：哪个机器人给你的问题提供了更有帮助的建议？  (5) **Overall**：一般来说，你更喜欢哪种机器人的情感支持？  (2)、(3) 和 (4) 中的指标对应于 ESC 框架中的三个阶段。

​		我们比较了三对模型：（a）Joint 与 BlenderBot（没有对 ESConv 进行微调），（b）Joint 与 Vanilla，以及（c）Joint 与 Random（使用随机选择的策略）。 为了更好地模拟真实的策略发生，Random 模型按照 ESConv 中的策略分布随机选择一个策略（表 3）。

​		每对模型通过与人类参与者的 100 次对话进行比较（表 5）。 比较（a）的结果表明，BlenderBot 在 ESConv 上进行微调后，提供 ES 的能力在所有指标上都有显着提高。 从比较（b）中，我们发现利用策略可以更好地安慰用户。 比较 (c) 的结果还表明，适当的策略时机对于帮助用户识别他们的问题并提供有效建议至关重要。 总的来说，通过在 ESConv 上策略预测的监督下进行微调，预训练的模型成为用户的首选，这证明了 ESConv 的高质量和实用性。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/~YBW4AG67H6%7DO5Y8H0%5D9Y8L.png)

### 6.5 Further Analysis of Human Interactive Evaluation

在本节中，我们将探索对话模型从 ESConv 中学到了什么。 **首先**，我们基于人类交互实验中用户和联合模型之间的 300 次对话分析了策略分布。 我们可以在图 5 中看到（计算与图 4 一致），联合模型采用的策略与 ESConv 中的真值分布（图 4）具有非常相似的分布。 它提供了重要的证据，证明模型模拟了人类支持者为实现更有效的 ES 所做的策略选择和利用。 **其次**，我们在图 7 中展示了一个案例研究。我们在案例中看到 Joint 模型提供了更多的支持性回复并在对话中使用了更多的技巧，而没有微调的 BlenderBot 似乎不太了解用户的苦恼，更喜欢多说关于自己的话。 这可能意味着拥有更多支持性回复和多样化的支持策略对于有效的情感支持至关重要。

![img](https://gitee.com/cao-hu/pictures/raw/master/img/NDAU_L1%60XEFOM81MQVT3%60$W.png)

## 7 Conclusion

在这项工作中，我们定义了情感支持对话的任务并提出了一个 ESC 框架。  ESC 框架从帮助技能理论改编为对话系统设置，其特征在于三个阶段，每个阶段都有相应的支持策略。 然后我们构建了一个情感支持对话数据集 ESConv。 我们精心设计了数据收集的流程，并设计了多种机制来确保 ES 在对话中的有效性。 最后，我们使用最先进的对话模型评估 ES 能力。 实验结果表明 ESConv 在提高对话系统提供有效 ES 的能力方面的潜在效用。 我们的工作可以促进 ES 对话系统的未来研究，并改进情感支持发挥重要作用的其他对话场景的模型。 策略选择与实现、用户状态建模和任务评估是进一步研究的重要方向。