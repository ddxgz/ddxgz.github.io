---
title: "Frontier Model Benchmarks Need Maintenance, Audits, and Rolling Updates"
description: "HLE shows why frontier model scores need to be read alongside audits, verified answer sets, and maintenance, especially when questions sit at the edge of expert knowledge."
pubDatetime: 2026-06-01T21:35:00+02:00
tags:
  - benchmarks
  - evaluation
  - llm
---

You may notice an LLM benchmark called **HLE (Humanity's Last Exam)** if you have been following frontier model releases. Opus 4.8, released last week, scored 49.8% without tools and 57.9% with tools respectively.

The number looks good as for a benchmark, not as saturated as some old benchmarks that are 90+%, there is still some head room for the models to catch up. Also not as low as ARC-AGI-3 that are close to 0.

But an interesting question came to my mind when I look closer at HLE. How do we reliably evaluate frontier models when the benchmark itself is built at the **edge of human expert knowledge**?

HLE was introduced in 2025 by Center for AI Safety and Scale AI. I don't like the name, "Humanity’s Last Exam" sounds like an AGI threshold, though it was an attempt to level up the difficulty of the benchmarks. It consists of 2,500 expert-curated, closed-ended questions across fields like mathematics, humanities, and natural sciences and other domains, which are near the edge of knowledge of the experts. It also has a private held-out question set.

Now let's talk about the challenges come with these frontier benchmarks. To create questions hard enough to challenge frontier models, the benchmark builders need the questions that are specialised, assumption sensitive, or dependent on narrow literature. These are the kinds of questions where **ground truth becomes difficult to establish**.

FutureHouse audited the biology and chemistry subset of HLE and found that about **29%** of the questions had answers with directly conflicting evidence in peer-reviewed literature ([FutureHouse research post](https://www.futurehouse.org/research-announcements/hle-exam)).

Alibaba's Qwen team introduced [HLE-Verified](https://arxiv.org/html/2602.13964v2), it splitted the original HLE questions into a revised set of HLE with 641 verified items, 1,170 revised-and-certified items, and 689 uncertain items. The paper reports that 7 frontier models by then increased the their score 7 to 10 points on full HLE-Verified, and 30 to 40 points on revised items, where the original problems or answers had issues.

Another challenge is that, some benchmark answers can be stable, like mathematics and history. But in biology, medicine, chemistry, social science, and parts of engineering, today's correct answer can be revised by new evidence, better measurement, or a change in expert consensus.

These challenges do not make the benchmark useless, they just need **maintenance, audits and rolling updates**.

Another point is the next time when we read those benchmark scores, we should also ask how reliable are the benchmarks themselves.
