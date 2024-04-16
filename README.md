# Awesome SafeRL and RLHF
A collection of some awesome public projects about Safe RL and RLHF for LLM.


## Table of Contents

- [Awesome SafeRL and RLHF](#awesome-saferl-and-rlhf)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Details of Safe RLHF ](#details-of-safe-rlhf)
  - [Papers](#papers)
    - [2024](#2024)
    - [2023](#2023)
    - [2022](#2022)
    - [2021](#2021)
    - [2020 and before](#2020-and-before)
  - [Dataset](#dataset)
  - [Blogs](#blogs)

## Overview

OpenAI团队开发的[InstructGPT](https://arxiv.org/abs/2203.02155)首先提出了RLHF的基本框架，它使用人类反馈的偏好信号训练奖励模型，并使用强化学习中的PPO算法微调大模型，使得其输出更符合人类偏好。后续的大模型基本都采用了这种训练方法来提高大模型的性能。但是这种方法使用人类对于两种回答的偏好作为奖励信号，只关注模型输出的有用性，并没有对模型输出的有害内容做限制。

- RLHF from InstrucGPT

![image info](./images/instructGPT.png)

北京大学团队近期将安全强化学习的框架引入RLHF，开发了[Safe RLHF](https://arxiv.org/abs/2310.12773)框架，对模型的有用性和无害性做了平衡，他们对强化学习的训练过程进行了优化，引入了约束（Constrained），并将受约束的原始问题转换为无约束拉格朗日对偶形式，通过迭代求解方程，交替更新LLM的参数和拉格朗日乘数，以平衡LLM在有用和无害的训练目标之间的冲突。

- Safe RLHF

![image info](./images/safeRLHF.png)

本仓库旨在收集确保LLM训练和推理安全性的RLHF框架及其相关改进的论文，同时包括对Safe RL算法的相关改进的论文，以期将其与RLHF框架结合，训练出更为安全的LLM。

### Details of Safe RLHF 

通常，安全强化学习的过程被建模为约束马尔可夫决策过程（Constrained Markov Decision Process，CMDP），即在标准马尔科夫决策过程$M={S,A,P,r,\gamma}$的基础上添加了关于成本函数的约束项$C = { (c_i, b_i) }_{i=1}^m$，其中，$c_i$为成本函数，$b_i$表示成本阈值，$i=1,…,m$。成本回报定义为$J^{(c_i)}(\pi_\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t c_i(s_{t+1} | s_t, a_t) \right]$，可行的策略集为$\Pi_C = \bigcap_{i=1}^m \left\{ \pi_\theta \in \Pi_\Theta \, \middle| \, J^{(c_i)}(\pi_\theta) \leq b_i \right\}$。安全强化学习算法的目标是找到最优的可行策略：$\pi^* = \arg\max_{\pi_\theta \in \Pi_C} J(\pi_\theta)$。
通用的CMDP优化问题可以通过下式表达：
$$
\underset{\pi \in \Pi_\Theta}{\max} J^R(\pi) \quad \text{s.t.} \quad J^C(\pi) \leq d
$$
其中，$\Pi_\Theta$表示具有参数$\theta$的参数化策略集合。

安全强化学习算法当中增加的约束项保证了强化学习方法在满足安全约束的情况下去求解使期望回报最大化，Safe RLHF算法中，设计了一个安全折扣累积成本函数，用于衡量智能体在执行各种动作时所产生的潜在风险。这个成本函数考虑了与系统安全相关的各种因素，包括可能引发危险的环境状态和潜在的损害程度：
$$
J_C(\theta) \triangleq \mathbb{E}_{(x \sim D, y \sim \pi_\theta(\cdot|x))} [C_\psi(y,x)] + d
$$
其中，$C_\psi$是代价模型，它通过附加有关无害性信息的数据进行训练。训练的目标是最小化这个安全折扣累积成本的期望，以确保智能体在执行动作时考虑到无害性，保证大模型输出结果的安全性。

在求解安全约束最优化问题这一步骤中使用拉格朗日方法对CMDP优化求解。其基本思想为将约束马尔科夫决策过程问题转化为无约束对偶问题，将约束作为惩罚信号引入回报函数，然后交替地应用策略优化对其对偶变量进行更新。具体来说，运用拉格朗日方法，上述优化目标的无约束的形式可写为： 
$$
\underset{\lambda \geq 0}{\min} \, \underset{\theta}{\max} \, G(\lambda,\theta) = \underset{\lambda \geq 0}{\min} \, \underset{\theta}{\max} \left[ J^R(\pi) - \lambda J^C(\pi) \right]
$$
其中，G 是拉格朗日函数，$\lambda \geq 0 $ 是拉格朗日乘子（一个惩罚系数）。
在RLHF这一技术范式中，强大的近端策略优化（PPO）是其实行稳健高效策略优化的关键。在考虑安全约束的前提下基于近端策略优化算法，相当于在上式中增加了旧策略和新策略之间差异的约束，用$D(\pi,\pi_k)$表示，最终的优化目标如下：
$$
\pi_{k+1} = \underset{\pi \in \Pi_\Theta}{\arg\max} \, J^R(\pi) \\
\quad \text{s.t.} \quad J^C(\pi) \leq d \quad \\
D(\pi,\pi_k) \leq \delta
$$

其中，D表示距离度量方法， $\delta$ 表示步长。


## Papers

```
format:
- [title](paper link) [links]
  - author1, author2, and author3...
  - publisher
  - keyword
  - code
  - experiment environments and datasets
```



## Dataset
```
format:
- [title](dataset link) [links]
  - author1, author2, and author3...
  - keyword
  - experiment environments or tasks
```




## Blogs



