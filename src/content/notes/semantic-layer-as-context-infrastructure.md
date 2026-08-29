---
title: "Semantic Layer as Context Infrastructure for Agents"
description: "A semantic layer can turn scattered business definitions and local knowledge into explicit context that both humans and agents can use reliably."
pubDatetime: 2026-01-28T09:00:00+01:00
tags:
  - agents
  - context
  - semantic-layer
---

Continuing some thoughts on context for AI agents. Lately I've been revisiting the semantic layer for structured data.

In a [previous post](/posts/beyond-answers-below-autonomy), I touched on the gap between explicit and implicit context and the problem of capturing the reasoning and trace that connects information to historical decisions and actions. The core problem is that this context is often scattered in different systems and people's minds.

In the context of data and BI world, take Databricks Unity Catalog as example, a well-designed semantic layer points a potentially good solution to address some of the problems. By creating a shared business language (definitions, metrics, etc.), it provides a structured and governed foundation for both human and agent to work with. This turns ambiguous, scattered, or implicit knowledge into explicit and reusable definitions in one place.

The concept of "core" and "edge" semantics is simple but powerful model here. The "edge" acts as a space to capture new insights and local context from user interactions, which can then be promoted to the "core" and become part of the trusted and shared information. This creates a continuous loop of learning and transforming implicit knowledge into explicit and evolving assets.

This approach doesn't just solve for consistent metrics for human, it also builds the foundational explicit context store that will be critical for agents to operate effectively and reliably within an organization.
