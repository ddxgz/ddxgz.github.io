---
title: "Human-Written AGENTS.md Still Wins"
description: "LLM-generated repo context files can be redundant and even hurt coding-agent performance, while human-written ones are more likely to close the real context gap."
pubDatetime: 2026-03-02T09:00:00+01:00
tags:
  - coding-agents
  - agents.md
  - context
links:
  - name: "Evaluating AGENTS.md"
    url: "https://arxiv.org/abs/2602.11988"
---

Hi fellow developers, do you let your coding agent generate `AGENTS.md`? You may be actually hurting its performance by doing so.

A recent paper by [Gloaguen et al.](https://arxiv.org/abs/2602.11988) evaluated whether repo context files help coding agents. They found that:

if repo context files help coding agents. They found that

- LLM-generated context files reduced task success rate by a small percentage (-0.5% on SWE-bench Lite and -2% on AGENTbench).
- While human-written group improved by ~4%.

![Bar chart comparing repo-context strategies across Sonnet-4.5, GPT-5.2, GPT-5.1 Mini, and Qwen3-30B. Human-written context outperforms LLM-generated context in most cases.](/notes/agents-md-context-files-eval.png)

The main reason is that LLM-generated context files are mostly redundant information already present in README, docs, or other files in the repo. That's obvious, agents can only know what they can discover, they won't know what is not told. It just burn more tokens for worse results.

In contrast, human-written context files normally include information that is not present in the repo or not obvious from the code, which are things like tooling and conventions.

So fellow developers, let's don't be lazy enough to even offload the context file writing to agents. We should provide them necessary important context, keep it minimal and specific to fill the context gap.
