# Distribution-based strategies

Distribution-based evolution strategies maintain a probability distribution over the search space and adapt its parameters using the fitness feedback collected during the ask-eval-tell loop. These methods shine when gradients are unavailable but sampling-based updates remain efficient.

## Covariance matrix adaptation

- [`CMA_ES`](../api/algorithms.md#evosax.algorithms.CMA_ES) – classical CMA-ES with diagonal and full covariance adaptation implemented in JAX.
- [`Sep_CMA_ES`](../api/algorithms.md#evosax.algorithms.Sep_CMA_ES) – separable CMA-ES for high-dimensional problems using a diagonal covariance matrix.
- [`SV_CMA_ES`](../api/algorithms.md#evosax.algorithms.SV_CMA_ES) – Stein variational CMA-ES with particle-based updates.

Use these variants when you want powerful adaptation of both the mean and covariance of the search distribution.

## Natural evolution strategies

- [`xNES`](../api/algorithms.md#evosax.algorithms.xNES) and [`SNES`](../api/algorithms.md#evosax.algorithms.SNES) implement exponential-family natural gradient updates with log-likelihood gradients.
- [`MA_ES`](../api/algorithms.md#evosax.algorithms.MA_ES) and [`LM_MA_ES`](../api/algorithms.md#evosax.algorithms.LM_MA_ES) extend the natural gradient view with mirrored sampling and learning-rate control.
- [`CR_FM_NES`](../api/algorithms.md#evosax.algorithms.CR_FM_NES) combines covariance reshaping with factored matrix parameterizations for stability.

These strategies are easy to jit-compile and vectorize, making them a strong default for accelerator-backed experiments.

## Gradient-guided methods

- [`Open_ES`](../api/algorithms.md#evosax.algorithms.Open_ES) – scalable evolution strategy popularized by OpenAI, with utilities for fitness shaping and antithetic sampling.
- [`GuidedES`](../api/algorithms.md#evosax.algorithms.GuidedES) – blends low-variance gradient estimates with search directions learned from data.
- [`NoiseReuseES`](../api/algorithms.md#evosax.algorithms.NoiseReuseES) and [`PersistentES`](../api/algorithms.md#evosax.algorithms.PersistentES) reuse perturbations across iterations for lower-variance updates.
- [`ASEBO`](../api/algorithms.md#evosax.algorithms.ASEBO) adapts the sampling distribution using principal subspaces discovered from gradients.

These approaches are well-suited for reinforcement learning tasks where low-variance gradient estimators are crucial.

## Lightweight & restart-friendly methods

- [`SimpleES`](../api/algorithms.md#evosax.algorithms.SimpleES) and [`HillClimbing`](../api/algorithms.md#evosax.algorithms.HillClimbing) offer minimal-state optimizers for quick experimentation.
- [`SimulatedAnnealing`](../api/algorithms.md#evosax.algorithms.SimulatedAnnealing) and [`RandomSearch`](../api/algorithms.md#evosax.algorithms.RandomSearch) provide baseline heuristics.
- Restart-aware strategies integrate seamlessly with the [core restart schedules](../api/core.md#evosax.core.restart).

## Learned and hybrid strategies

- [`LearnedES`](../api/algorithms.md#evosax.algorithms.LearnedES) wraps neural controllers trained to steer evolution strategies.
- [`DiscoveredES`](../api/algorithms.md#evosax.algorithms.DiscoveredES) and [`EvoTF_ES`](../api/algorithms.md#evosax.algorithms.EvoTF_ES) showcase program-search and tensor factorization hybrids.
- [`GradientlessDescent`](../api/algorithms.md#evosax.algorithms.GradientlessDescent) implements a learned update rule for gradient-free optimization.

For end-to-end differentiable pipelines, experiment with [`PGPE`](../api/algorithms.md#evosax.algorithms.PGPE) and [`ARS`](../api/algorithms.md#evosax.algorithms.ARS) to combine policy gradients with evolution strategies.

## Example notebooks

| Notebook | Highlights |
| --- | --- |
| [00_getting_started.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/00_getting_started.ipynb) | Run CMA-ES and NES on toy objectives. |
| [02_rl.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/02_rl.ipynb) | Reinforcement learning with OpenAI-ES, PGPE, and ARS. |
| [07_persistent_es.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/07_persistent_es.ipynb) | Persistent and noise-reuse strategies for meta-learning. |
