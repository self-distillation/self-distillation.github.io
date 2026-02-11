---
title: Self-Distillation
authors: <a href="https://idanshen.github.io/">Idan Shenfeld</a><sup>1</sup>, <a href="https://jonhue.github.io/">Jonas Hübotter</a><sup>2</sup>, <a href="https://thomasklbg.github.io/">Thomas Kleine Buening</a><sup>2</sup> and others
affiliations: <sup>1</sup>MIT <sup>2</sup>ETH Zurich
date: February 2026
---

**tl;dr:** Self-distillation is a new learning paradigm enabling continual in-weight learning from arbitrary data. We show how it can be used to learn from [expert demonstrations](#1-learning-from-demonstrations), from [scalar rewards](#2-learning-from-scalar-rewards), and from arbitrary textual environment feedback such as [errors in a code environment](#3-learning-from-environment-feedback) or [raw user conversations](#4-learning-from-raw-user-interactions).

## Introduction

One of the most remarkable aspects of human intelligence is our ability to learn from virtually any kind of signal. We learn by watching others succeed or fail, by receiving explicit feedback from teachers, and even by reflecting on our own mistakes. Each learning experience, no matter how different in form, changes what we know and how we behave.

For decades, machine learning couldn't match this flexibility. Different types of learning signals require entirely different algorithms: supervised learning for labeled data, reinforcement learning for scalar rewards, and so on. Each approach came with its own mathematical framework, training procedures, and limitations. Building AI systems that could learn flexibly and cotinually, the way humans do, seemed like a distant goal.

Then something changed. The emergence of large-scale pretraining gave rise to a surprising capability called in-context learning (ICL). Large language models, it turned out, could adapt their behavior simply by being shown examples or instructions in their input. Show a model a few examples of translating English to French, and it starts translating. Give it a new task description, and it adjusts accordingly. All of this happens at inference time, just by conditioning the model on the right context.
The research community has extensively studied ICL and discovered just how flexible it can be. Models can digest an impressive variety of learning signals through their context window—from demonstrations [@brown2020language] to verbal feedback [@shinn2023reflexion] and self-reflection [@madaan2023self].

But ICL has a fundamental limitation: it's transient. The moment you remove the context, the learned behavior disappears. The model reverts to its base behavior, unable to retain what it just learned.
However, context windows are inherently bounded,[^boundedcontext] and hence, long-term learning requires some form of compression.

A natural form of compression is compression into model weights. 
Most common methods for in-weight learning perform gradient descent on an external signal: either imitating demonstrations (e.g., supervised fine-tuning (SFT)) [@ouyang2022training], mimicking another model through distillation [@hinton2015distilling], or following an external reward signal (e.g., reinforcement learning with verifiable rewards (RLVR)) [@lambert2024tulu].
These existing methods for in-weight learning exhibit fundamentally different behavior to ICL: they *force* the model to change its behavior based on an external signal, whereas ICL enables the model to decide itself how its behavior should change given its context.

For example, consider giving the model a complex math problem. After submitting its attempt, you provide it with a sample solution. SFT would change the model's weights to imitate that sample solution. RLVR would check whether the attempt was correct and then reinforce or discourage the full attempt.
In contrast, through ICL the model can retrospectively adjust its initial attempt based on the sample solution.
That is, the model *decides for itself* how its answer should change in response to the additional context.

We propose a learning paradigm that bridges this gap: **Self-Distillation**. The key insight is that we can harness the powerful generalization capabilities and flexibility of in-context learning to permanently change the model's parameters.[^contextdistillation]

![> Pretraining -> ICL -> self-distillation](figures/main.png)
***Figure 1:** Whereas the ability for in-context learning emerged from pre-training, self-distillation emerges as a consequence of in-context learning.*

## The Self-Distillation Framework

The mechanism of self-distillation is simple: we perform teacher-student distillation where both the teacher and student are the same model. The only difference is that the teacher sees the learning signal in its context (demonstrations, feedback, or corrections), while the student does not. By training the student to match the teacher's outputs, we effectively compress the context-dependent behavior into the model's weights.
What makes self-distillation particularly significant is a subtle but profound shift in how learning happens. Unlike traditional training algorithms that impose changes on a model from the outside—pushing parameters toward predefined targets—self-distillation lets the model determine what to learn. The behavioral changes emerge from the model's own in-context understanding of the learning signal. In a sense, we're not teaching the model; we're giving it the tools to teach itself.

The core mechanism of self-distillation is straightforward: we use the same model in two roles simultaneously. The teacher sees the learning signal in its context and generates informed predictions, while the student makes predictions without that context. By training the student to match the teacher, we transfer the context-dependent knowledge into the model's parameters.

Here's how it works in practice. Given a prompt $x$ and a context $c$ (which could be demonstrations, feedback, or any other learning signal), we first sample a complete response from the student: $y \sim \pi(\cdot|x)$. This is the current model operating without the benefit of the context. Then, for this same response $y$, we evaluate what the teacher---the model \emph{with} the context---would have predicted at each token position.

![> Illustration of on-policy self-distillation](figures/main.png)
***Figure 2:** ...*

Formally, we train the policy $\pi$ by minimizing the reverse KL divergence between the student and the self-teacher:

$$
\mathcal{L}(\theta) = \mathbb E_{y \sim \pi(\cdot|x)} \left[ \sum_{t=1}^{|y|} D_{\text{KL}}\large(\underbrace{\pi_\theta(\cdot|x, y_{<t})}_{\text{student}} \,\|\, \underbrace{\pi_\theta(\cdot|x, c, y_{<t})}_{\text{self-teacher}}\large) \right]
$$
<!-- Jonas: I dropped \theta from the outer expectation so that we have the simpler token-level gradient -->

Jonas: use general divergence?

where $\pi_\theta(\cdot|x, y_{<t})$ is the student's next-token distribution given the prompt and previous tokens, and $\pi_\theta(\cdot|x, c, y_{<t})$ is the self-teacher's distribution when additionally conditioned on the context $c$.

Taking the gradient of this objective through the student (while keeping the self-teacher fixed) gives us the following update rule:

$$
\nabla_{\!\theta}\, \mathcal{L}(\theta) = \mathbb E_{y \sim \pi_\theta(\cdot|x)} \!\left[ \sum_{t=1}^{|y|} \mathbb E_{\hat{y}_t \sim \pi_\theta(\cdot|x,y_{<t})} \!\left[ \nabla_\theta \log \pi_\theta(\hat{y}_t|x, y_{<t}) \cdot \log \frac{\pi_\theta(\hat{y}_t|x, y_{<t})}{\pi_\theta(\hat{y}_t|x, c, y_{<t})} \right] \right]
$$

TODO: describe intuition / relate to policy gradient

## Use Cases

TODO: overview

### 1) Learning from demonstrations

### 2) Learning from scalar rewards

### 3) Learning from environment feedback

### 4) Learning from raw user interactions

<!-- ## Does RL scale?

Over the past few years, we have seen several objective families scale smoothly with data and compute. Reinforcement learning also scales, but some formulations scale less gracefully than others [@silver2016go] [@schulman2017ppo].[^scale]

A common issue is temporal credit assignment. In tabular notation, we still optimize quantities like $Q^\pi(s,a)$ and $V^\pi(s)$, but learning can become unstable when trajectories are long and noisy.

## Divide and conquer updates

One practical strategy is to decompose a long return into shorter chunks, then recursively combine local value estimates. Intuitively, this replaces one fragile $T$-step update with a tree of simpler updates [@bertsekas1995dp].[^dp]

$$
\hat{G}_{1:T} = \hat{G}_{1:T/2} + \gamma^{T/2}\hat{G}_{T/2+1:T}
$$

In practice, the recursion depth is $\mathcal{O}(\log T)$ for balanced splits, which can substantially reduce optimization pressure compared to naively chaining $T$ terms. -->

## Understanding / Interpretation

### Intuition behind positive / negative advantages

### Sparse advantages

### Why on-policy learning?

TODO: define on-policy vs off-policy

But there's a crucial design choice that makes self-distillation effective: we use \emph{on-policy distillation}. This means that instead of distilling over some fixed dataset, we generate fresh responses from the current student model at each training step and then distill the teacher's knowledge on those specific responses.

Why does this matter? On-policy learning has been shown to provide several key benefits. It improves in-distribution performance by avoiding the distributional mismatch between training and inference [CITE DAGGER]. It substantially reduces catastrophic forgetting when learning new tasks, as the model continues to practice its existing capabilities [CITE RL's RAZOR]. And perhaps most surprisingly, it improves out-of-distribution generalization compared to standard supervised learning [CITE SFT MEMORIZE RL GENERALIZE]. The intuition is simple: by training on the model's own outputs, we ensure that the model learns to handle the kinds of mistakes and patterns it actually produces, rather than only learning from a static dataset that may not reflect its current behavior.

Jonas: often $c$ depends causally on $y$, in which case on-policy learning is more compute efficient. We should also mention how this improves performance over off-policy training on the self-teacher.

### Combining on-policy and off-policy learning

### Continual learning

## Conclusion

...

## Citation

Please cite this work as:

```
Shenfeld, Idan and Damani, Mehul and Hübotter, Jonas and Agrawal, Pulkit.
"Self-Distillation Enables Continual Learning". Jan 2026,

Hübotter, Jonas and Lübeck, Frederike and Behric, Lejs and Baumann, Anton and
Bagatella, Marco and Marta, Daniel and Hakimi, Ido and Shenfeld, Idan and
Kleine Buening, Thomas and Guestrin, Carlos and Krause, Andreas.
"Reinforcement Learning via Self-Distillation". Jan 2026.

Kleine Buening, Thomas and Hübotter, Jonas and Pasztor, Barna and
Shenfeld, Idanand Ramponi, Giorgia and Krause, Andreas.
"Aligning Language Models from User Interactions". Feb 2026.
```

Or use the BibTeX citations:

```
@article{shenfeld2026self,
  title = {Self-Distillation Enables Continual Learning},
  author = {Shenfeld, Idan and Damani, Mehul and Hübotter, Jonas and Agrawal, Pulkit},
  year = {2026},
  month = {Jan},
  journal = {arXiv preprint arXiv:2601.19897},
}

@article{hubotter2026reinforcement,
  title = {Reinforcement Learning via Self-Distillation},
  author = {Hübotter, Jonas and Lübeck, Frederike and Behric, Lejs and Baumann, Anton and Bagatella, Marco and Marta, Daniel and Hakimi, Ido and Shenfeld, Idan and Kleine Buening, Thomas and Guestrin, Carlos and Krause, Andreas},
  year = {2026},
  month = {Jan},
  journal = {arXiv preprint arXiv:2601.20802},
}

@article{buening2026aligning,
  title = {Aligning Language Models from User Interactions},
  author = {Kleine Buening, Thomas and Hübotter, Jonas and Pasztor, Barna and Shenfeld, Idan and Ramponi, Giorgia and Krause, Andreas},
  year = {2026},
  month = {Feb},
  journal = {arXiv preprint arXiv:XXX},
}
```

[^boundedcontext]: Attending to all tokens in-context requires linearly growing memory, preventing learning from long streams of experience.
[^contextdistillation]: Self-distillation is a natural descendant of "context distillation", where a fixed context (e.g., a system prompt or long document) is compressed into model weights [@bai2022constitutional] [@eyuboglu2025cartridges].

[@brown2020language]: Language Models are Few-Shot Learners (Brown et al, 2020) || https://arxiv.org/abs/2005.14165
[@shinn2023reflexion]: Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al, 2023) || https://arxiv.org/abs/2303.11366
[@madaan2023self]: Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al, 2023) || https://arxiv.org/abs/2303.17651
[@agrawal2025gepa]: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (Agrawal et al, 2025) || https://arxiv.org/abs/2507.19457
[@ouyang2022training]: Training Language Models to Follow Instructions with Human Feedback (Ouyang et al, 2022) || https://arxiv.org/abs/2203.02155
[@hinton2015distilling]: Distilling the Knowledge in a Neural Network (Hinton et al, 2015) || https://arxiv.org/abs/1503.02531
[@lambert2024tulu]: Tulu 3: Pushing Frontiers in Open Language Model Post-Training (Lambert et al, 2024) || https://arxiv.org/abs/2411.15124
[@bai2022constitutional]: Constitutional AI: Harmlessness from AI Feedback (Bai et al, 2022) || https://arxiv.org/abs/2212.08073
[@eyuboglu2025cartridges]: Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study (Eyuboglu et al, 2025) || https://arxiv.org/abs/2506.06266
