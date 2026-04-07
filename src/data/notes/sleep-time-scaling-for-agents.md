---
title: "About Scaling of Sleep-time Compute for Agents"
description: "You probably often hear scaling law, pre-training scaling, post-training scaling or test-time scaling, but you may not often hear sleep-time scaling. It's worth a few minutes to talk about it and how it would scale on the volume of context/memory, latency, and proactiveness."
pubDatetime: 2026-04-07T21:00:00+01:00
tags:
  - sleep-time-compute
  - agents
  - sleep-time-scaling
links:
  - name: Sleep-time Compute
    url: "https://www.letta.com/blog/sleep-time-compute"
---

You probably often hear scaling law, pre-training scaling, post-training scaling or test-time scaling, but you may not often hear sleep-time scaling.

It's worth a few minutes to talk about it and how it would scale on the volume of context/memory, latency, and proactiveness.

Sleep-time here means the time when the user is not actively interacting with the AI agents. The first time I heard about the term "sleep-time compute" is from [Letta's post](https://www.letta.com/blog/sleep-time-compute).

Sleep-time compute is the idea that AI agents shouldn't sit idle between your interactions. Instead, one or several background agents should continuously processes context: conversation history, all types of data the agents have access to, include databases, event steams. With the timestamps and relationships of information, the agents can refine memory, identify facts, consolidating insights, and discover patterns before the user assign the next task. AI agents use their "sleep" time to process information and form new connections by writing their memory state.

Sleep-time compute have actually been widely implemented in many applications in different forms.

- ChatGPT is consolidating memories about your conversations in the background.
- Claude Code will have a background agent to extract different information as memory files, which includes user preferences, feedback, project, and sessions memory, etc.
- In recent months, there is a wave of work in this space, include Andrej Karpathy's idea about [LLM wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

If we expand the definition of sleep-time compute, even the scheduled or event based data pipelines, data metrics, BI reports are one form of sleep time compute. Though it's non-agentic (they are predefined), they can also be the input to background agents to keep discovering and write non-predefined insights as out to memory store, or trigger notifications.

**So what does it actually unlock?**

First, it shifts compute from interact-time/task-time to sleep-time. The overall token cost can possibly get lower, depend on the form of implementation. Sleep-time agents can burn a lot of tokens in the background, but it's amortised. When you actually query the agent, it's faster, cheaper and more accurate.

Of course, when sleep-time compute scales it will burn way more significant tokens, but it gives agents access to far larger volume of context, far deeper insights, while keep the span of interact/task-time reasonable.

Most importantly, it unlocks proactiveness. An agent that only thinks when prompted will always feel like a tool. An agent that thinks while you sleep, and surfaces a pattern you didn't ask about, flags a risk before you noticed, prepares context before you open the chat, that starts to feel like a colleague who never sleeps.
