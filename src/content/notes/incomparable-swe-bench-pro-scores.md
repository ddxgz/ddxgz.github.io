---
title: "Incomparable SWE-bench Pro Scores"
description: "Claude Opus 4.7 and GPT-5.5 SWE-bench Pro scores are useful directional signals, but their reported margins are not directly comparable."
pubDatetime: 2026-04-27T19:37:46+02:00
tags:
  - swe-bench-pro
  - coding-agents
  - benchmarks
---

The scores of Claude Opus 4.7 at 64.3% and GPT-5.5 at 58.6% on SWE-bench Pro are NOT directly comparable!

It looks like Opus 4.7's 64.3% wins over GPT-5.5's 58.6% with a good margin. However, the margin is NOT real!

Both the models were released in the past 2 weeks. Both emphasise on agentic coding, SWE-bench Pro is one of the important benchmarks they use to evaluate the model performance.

Anthropic's footnote on the [Opus 4.7 announcement](https://www.anthropic.com/news/claude-opus-4-7) says:

> Our memorization screens flag a subset of problems in these SWE-bench evals. Excluding any problems that show signs of memorization, Opus 4.7's margin of improvement over Opus 4.6 holds.

But they reported 64.3% anyway, without telling which problems were flagged or what the score would be after the exclusion (still not comparable with that score).

This problem was flagged by OpenAI in their [GPT 5.5's release note](https://openai.com/index/introducing-gpt-5-5/), but it does not mean that we can assume GPT 5.5's score is exempted from contamination, as they didn't disclose whether GPT-5.5 itself shows the same signs.

Benchmarks are just a proxy for model performance, a useful but noisy, playable proxy, we can only treat the numbers as a directional signal.
