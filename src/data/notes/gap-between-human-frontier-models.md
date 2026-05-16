---
title: "ARC-AGI-3 shows there is still a huge gap between frontier models and humans on agentic intelligence"
description: "GPT-5.5 and Opus 4.7 scored below 1% while human baseline is 100% on ARC-AGI-3, where models need to explore and learn in the novel game-style environments."
pubDatetime: 2026-05-16T19:37:46+02:00
tags:
  - arc-agi-3
  - benchmarks
  - ai-human-gap
---

After the recent releases of Claude Opus 4.7 and GPT-5.5, as usual, both showed improvements on many benchmarks. After a few weeks, ARC Prize Foundation published a blog analysing the scores of these two frontier models on the semi-private ARC-AGI-3 dataset. The result was a surprise to me. I knew the scores would be low, but I didn’t expect them to be that low.

GPT-5.5 got **0.43%**, and Opus 4.7 got **0.18%**. Yes, they are both below 1%, while the human baseline is 100%.

ARC-AGI-3 was released in March 2026 as the successor of the almost saturated ARC-AGI-2. It contains 135 novel interactive game-style environments, split into public, semi-private, and fully private sets, which makes it much harder to exploit the score.

Unlike the tasks in many other benchmarks that have clear instruction and goals, the ARC-AGI-3 environments have no task specific instructions, no stated rules, and no stated goals. It's fully interactive, an agent must explore itself to know how it works, what are the rules, and how to win.

To win, the models need to really learn the new games and build a world model of the environment. They cannot simply rely on familiar game patterns from their training data. In the ARC Prize team’s analysis, one common failure mode was exactly this: models sometimes understood a local effect, but failed to turn it into a correct global rule. Another failure mode was using the wrong abstraction from games in the training data.

It’s exciting to see new benchmarks bring challenges to frontier models when many existing ones are becoming saturated. Frontier models are clearly getting stronger, but there is still a huge gap between on exploring, learning, and adapting in a novel environment.

Really looking forward to seeing how much the models can climb on ARC-AGI-3 within 2026.
