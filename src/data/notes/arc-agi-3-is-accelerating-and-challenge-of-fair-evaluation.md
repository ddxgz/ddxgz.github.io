---
title: "ARC-AGI-3 Is Accelerating and the Challenge of Fair Evaluation"
description: "Big improvement from frontier models shows that ARC-AGI-3 performance depends on both model capability and evaluation harness, raising questions about how to compare models fairly while still letting them perform at their best."
pubDatetime: 2026-08-05T22:15:00+02:00
tags:
  - arc-agi-3
  - benchmarks
  - evaluation-harness
---

About three months ago when I posted about **ARC-AGI-3**, GPT-5.5 and Claude Opus 4.7 both scored below **1%**.

Now there is a lot of progress.

Claude Opus 5 has reached [**30.2%** on ARC-AGI-3’s verified semi-private evaluation set](https://arcprize.org/results/anthropic-claude-opus-5). It's a huge jump from GPT-5.6 Sol’s previous verified score of **7.8%** on the same evaluation set.

At the same time, OpenAI showed another interesting [report](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/). On the ARC-AGI-3 public set, GPT-5.6 Sol improved from **13.3% to 38.3%**  by using its own harness. The ARC-AGI official harness manages conversation state client-side and uses rolling truncation, while OpenAI's Responses API harness preserves reasoning across turns and compacts long conversations.

The 30.2% and 38.3% scores are not comparable as they were measured on different sets and with different harnesses. But they show both how quickly the models are improving and how much the evaluation setup can affect long-horizon performance.

This raises an important question for agent benchmarks:

**Where should we draw the boundary between the model and the agent system around it?**

For long-horizon tasks like ARC-AGI-3, memory, state management, and context handling can have a significant impact on performance. These capabilities may be essential parts of practical AI systems, but they also make evaluation more complicated.

Benchmark evaluation should reflect improvements in general intelligence, not just improvements from optimising for a particular evaluation. But the models still need the tools to manage their own context.

How do we ensure fair comparisons while also allowing each model to perform with a harness that reflects how it would be used in practice?

[ARC Prize says](https://x.com/arcprize/status/2082672003765670160?utm_source=chatgpt.com) they are working with several labs to incorporate these server-side state-management findings into its verified testing while preserving consistency across model providers.
