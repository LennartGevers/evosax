# Distribution-based strategies

Distribution-based evolution strategies maintain a probability distribution over the search space and adapt its parameters using the fitness feedback collected during the ask-eval-tell loop. These methods shine when gradients are unavailable but sampling-based updates remain efficient.

## Covariance matrix adaptation

- [`CMA_ES`](../api/algorithms.md#evosax.algorithms.distribution_based.cma_es.CMA_ES) – classical CMA-ES with diagonal and full covariance adaptation implemented in JAX.
- [`Sep_CMA_ES`](../api/algorithms.md#evosax.algorithms.distribution_based.sep_cma_es.Sep_CMA_ES) – separable CMA-ES for high-dimensional problems using a diagonal covariance matrix.
- [`SV_CMA_ES`](../api/algorithms.md#evosax.algorithms.distribution_based.sv.sv_cma_es.SV_CMA_ES) – Stein variational CMA-ES with particle-based updates.

Use these variants when you want powerful adaptation of both the mean and covariance of the search distribution.

## Natural evolution strategies

- [`xNES`](../api/algorithms.md#evosax.algorithms.distribution_based.xnes.xNES) and [`SNES`](../api/algorithms.md#evosax.algorithms.distribution_based.snes.SNES) implement exponential-family natural gradient updates with log-likelihood gradients.
- [`MA_ES`](../api/algorithms.md#evosax.algorithms.distribution_based.ma_es.MA_ES) and [`LM_MA_ES`](../api/algorithms.md#evosax.algorithms.distribution_based.lm_ma_es.LM_MA_ES) extend the natural gradient view with mirrored sampling and learning-rate control.
- [`CR_FM_NES`](../api/algorithms.md#evosax.algorithms.distribution_based.cr_fm_nes.CR_FM_NES) combines covariance reshaping with factored matrix parameterizations for stability.

These strategies are easy to jit-compile and vectorize, making them a strong default for accelerator-backed experiments.

## Gradient-guided methods

- [`Open_ES`](../api/algorithms.md#evosax.algorithms.distribution_based.open_es.Open_ES) – scalable evolution strategy popularized by OpenAI, with utilities for fitness shaping and antithetic sampling.
- [`GuidedES`](../api/algorithms.md#evosax.algorithms.distribution_based.guided_es.GuidedES) – blends low-variance gradient estimates with search directions learned from data.
- [`NoiseReuseES`](../api/algorithms.md#evosax.algorithms.distribution_based.noise_reuse_es.NoiseReuseES) and [`PersistentES`](../api/algorithms.md#evosax.algorithms.distribution_based.persistent_es.PersistentES) reuse perturbations across iterations for lower-variance updates.
- [`ASEBO`](../api/algorithms.md#evosax.algorithms.distribution_based.asebo.ASEBO) adapts the sampling distribution using principal subspaces discovered from gradients.

These approaches are well-suited for reinforcement learning tasks where low-variance gradient estimators are crucial.

## Lightweight & restart-friendly methods

- [`SimpleES`](../api/algorithms.md#evosax.algorithms.distribution_based.simple_es.SimpleES) and [`HillClimbing`](../api/algorithms.md#evosax.algorithms.distribution_based.hill_climbing.HillClimbing) offer minimal-state optimizers for quick experimentation.
- [`SimulatedAnnealing`](../api/algorithms.md#evosax.algorithms.distribution_based.simulated_annealing.SimulatedAnnealing) and [`RandomSearch`](../api/algorithms.md#evosax.algorithms.distribution_based.random_search.RandomSearch) provide baseline heuristics.
- Restart-aware strategies integrate seamlessly with the [core restart schedules](../api/core.md#evosax.core.restart).

## Learned and hybrid strategies

- [`LearnedES`](../api/algorithms.md#evosax.algorithms.distribution_based.learned_es.LearnedES) wraps neural controllers trained to steer evolution strategies.
- [`DiscoveredES`](../api/algorithms.md#evosax.algorithms.distribution_based.discovered_es.DiscoveredES) and [`EvoTF_ES`](../api/algorithms.md#evosax.algorithms.distribution_based.evotf_es.EvoTF_ES) showcase program-search and tensor factorization hybrids.
- [`GradientlessDescent`](../api/algorithms.md#evosax.algorithms.distribution_based.gradientless_descent.GradientlessDescent) implements a learned update rule for gradient-free optimization.

For end-to-end differentiable pipelines, experiment with [`PGPE`](../api/algorithms.md#evosax.algorithms.distribution_based.pgpe.PGPE) and [`ARS`](../api/algorithms.md#evosax.algorithms.distribution_based.ars.ARS) to combine policy gradients with evolution strategies.

## Example notebooks

| Notebook | Highlights |
| --- | --- |
| [00_getting_started.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/00_getting_started.ipynb) | Run CMA-ES and NES on toy objectives. |
| [02_rl.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/02_rl.ipynb) | Reinforcement learning with OpenAI-ES, PGPE, and ARS. |
| [07_persistent_es.ipynb](https://github.com/RobertTLange/evosax/blob/main/examples/07_persistent_es.ipynb) | Persistent and noise-reuse strategies for meta-learning. |
