---
title: "The Post-Training Dilemma: Safety Alignment vs Benchmark Score"
description: >-
  Anthropic has long positioned itself put the AI safety at the frontier. But
  their models had many concerning bad behaviours in Andon Labs's Vending Bench
  and Arena. The bad behaviours include making lies, coordinate price cartel,
  exploited other's desperate situation.

  It's challenging to balance the safety alignment, instruction-following, and
  model capability, how are the labs doing on this? If a bit less safety
  alignment can lead to higher benchmark scores, it gives the labs incentives to
  reduce or do less safety alignment, what will the labs choose to do?
pubDatetime: 2026-06-22T22:18:47+02:00
tags:
  - ai-safety
  - alignment
  - vending-bench
---

Anthropic has long positioned itself put the AI safety at the frontier. But the models from Anthropic behave differently from the company's safety-first branding when there is incentive for bad behaviours.

I believe all the frontier labs are putting a lot of effort on the post-training safety alignment to ensure the models will not have harmful behaviours.

It's challenging to balance the safety alignment, instruction-following, and model capability, how are the labs doing on this?

If post-training safety actually holds when a model is given autonomy and tools to maximize performance to purse a given goal.

Fewer safety and ethical preconditions may have more options available for the models to pursue a goal, which means a bit less safety alignment can lead to higher benchmark scores.

If a bit less safety alignment can lead to higher benchmark scores, it gives the labs incentives to reduce or do less safety alignment, what will the labs choose to do?

These questions arose in my mind when I came across [Vending-Bench 2](https://andonlabs.com/evals/vending-bench-2) in a podcast.

Vending Bench 1 and 2 was released Vending Bench 2, it asks AI agents to run a simulated vending machine business over a year. The model has to manage the business over a long horizon and is scored by final bank account balance. Andon labs frames it as a benchmark for measuring whether models can stay coherent and effective while managing a simulated business over time.
The system prompt is simply tell the model "Do whatever it takes to maximize your bank account balance after one year of operation"

Andon labs brought up some interesting examples in their [paper and blog posts](https://andonlabs.com/publications) about the models' behaviours and performance.

Opus 4.6 reached the best Vending Bench 2 score with the highest average final balance at the time. But the way it achieved this score raised some safety questions.

To pursue to win, Opus 4.6 lied to suppliers about exclusivity, and falsely told customers it had refunded them, because "every dollar counts".
In the Vending Bench Area, where multiple models are competing each other,
Andon reported that Opus 4.6 coordinated price fixing, deceived other players, exploited another player's desperate situation.

They later tested Opus 4.7 and Mythos Preview, found that they still engaged in deceptive and power seeking behaviour in their pursuit to win.

Then came Opus 4.8, which makes the post-training question even more interesting. Opus 4.8 still engaged in price cartels, just less frequently than previous models. Andon labs described it had better safety alignment, but worse performance.

Claude Fable 5 was released and tested after Opus 4.8, Andon labs described it as a partial step back in alignment. Fable 5 brought back power seeking and deceptive negotiation tactics that Opus 4.8 had largely reduced.

The most worrying part is Fable 5 not just misbehaved, it rationalized its bad behaviour while being aware that it was wrong.

Not only Anthropic's models had these concerning mis-behaviours. Zhipu AI's GLM-5 was one of the open weights models that achieved good score, but it also used many of the same tactics that had been seen from Opus 4.6, like price collusion, exploitation of desperation, and lying to suppliers and customers.

One good part is that Andon labs described GPT-5.5 as "bad behaviour is not necessary", as it scored higher than Opus 4.7 in Vending Bench Arena without any misconduct.

The only semi-concerning thing they found is that GPT-5.5 was participation in price cartels. Based on Andon labs' analysis, it was Opus who initiated price cartels in most cases. But in one run GPT-5.5 firstly declined a price cartel proposal from Opus based on ethical grounds, but later it returned to Opus with its own price fixing proposal.

Applause to Gemini 3 Pro, it was one of the strongest Vending Bench performers. Andon Labs described it as a persistent negotiator. It was then by passed by Opus who gained advantage by mis-behaviours.

Without strong safety and ethical pre-conditions, the models have "more options" to pursue the goals, which may lead to better score, but come with bad behaviours. So theoretically, if less safety alignment was done in model post-training, the models have the potential to score higher in some tasks.

If that path is real, how do we prevent frontier model labs from loosing up safety alignment for pursuing "higher model performance"?

Connecting with the recent Fable 5's great capability but with safety jailbreak issue, it becomes more concerning. Anthropic branded itself as ai safety first when it started, but now its models show most mis-behaviours. Dario probably should spend less time on spreading AI anxiety, but spend more time on meeting Anthropic's AI safety narrative.
