---
title: "The Missing Layer in Agent Systems"
description: "Explicit context is only part of the problem. The harder part is capturing the reasoning that links information, decisions, and actions."
pubDatetime: 2026-01-13T09:00:00+01:00
tags:
  - agents
  - context
  - reasoning
---


One point from our [recent post](/posts/beyond-answers-below-autonomy) is about explicit and implicit context in agent systems, and the big gap that the reasoning that connects `information` -> `decision` -> `action` was not captured, as data are scattered in different systems and in humans' brains.

A recent [article posted on X](https://x.com/jayagup10/status/2003525933534179480?s=46) talked about the same issue from another angle: context graphs as a way to represent the reasoning chain behind actions (decision traces, exceptions, overrides, cross-system nuance).

Though I think graph may not be the best data representation in this agent era, but the core questions remain for exploration:
- How do we capture the implicit contexts with low friction (as work happens)?
- How do we keep them so it stays correct as systems and policies change?
- How do we use let agents use them effectively?
