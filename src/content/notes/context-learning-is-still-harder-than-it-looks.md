---
title: "Context Learning Is Still Harder Than It Looks"
description: "CL-bench shows that frontier models still struggle to genuinely learn and apply novel knowledge that was not part of pre-training."
pubDatetime: 2026-02-10T09:00:00+01:00
tags:
  - llms
  - context-learning
  - evaluation
links:
  - name: Learning from context is harder than we thought
    url: "https://arxiv.org/abs/2602.03587"
---

How much can LLMs actually learn and use new knowledge that was not part of their pre-training, i.e., Context Learning (CL)?

A recent paper, [Learning from context is harder than we thought](https://arxiv.org/abs/2602.03587), introduced CL-bench to test exactly this. Unlike benchmarks that rely on context retrieval, CL-bench tasks can only be solved by learning genuinely new knowledge from the provided context, which include new rules, procedures, and domain-specific concepts crafted by domain experts and are absent from pre-training. In total it contains 500 contexts, 1899 tasks and more than 30k verifications. The evaluation result shows that the frontier LLMs (include GPT 5.x, Opus 4.5 and several open weights models) can solve only 13.2% to 23.7% of the tasks.

This shows that the LLMs' ability to truly learn and apply new knowledge within context is still very limited.

For many enterprise use cases, the private domain data may not have been directly used in pre-training, but equivalent knowledge are probably represented in the training data. This is why we see current LLM agents perform well on many domain specific tasks.

For truly novel and unseen knowledge / rules / procedures, what can we do before LLMs got improved their context learning ability? Are there more approaches than enriching and engineering the context, like adding connections to existing knowledge, or providing self-contained task context, etc.?

Curious to hear how others are tackling this in practice.
