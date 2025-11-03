# Core utilities API

The `evosax.core` package provides shared utilities such as restart schedules, fitness shaping, optimizers, and kernel helpers used across algorithms.

## Optimizers

::: evosax.core.optimizer.ScaleByClipUpState
    options:
        members: true

::: evosax.core.optimizer.scale_by_clipup
    options:
        members: true

::: evosax.core.optimizer.clipup
    options:
        members: true

## Fitness shaping

::: evosax.core.fitness_shaping.normalize
    options:
        members: true

::: evosax.core.fitness_shaping.add_weight_decay
    options:
        members: true
::: evosax.core.fitness_shaping.identity_fitness_shaping_fn
    options:
        members: true
::: evosax.core.fitness_shaping.normalize_fitness_shaping_fn
    options:
        members: true
::: evosax.core.fitness_shaping.centered_rank_fitness_shaping_fn
    options:
        members: true
::: evosax.core.fitness_shaping.weights_fitness_shaping_fn
    options:
        members: true
## Restart strategies

::: evosax.core.restart.RestartState
    options:
        members: true

::: evosax.core.restart.RestartParams
    options:
        members: true

::: evosax.core.restart.IPORestartState
    options:
        members: true

::: evosax.core.restart.IPORestartParams
    options:
        members: true

::: evosax.core.restart.BIPORestartState
    options:
        members: true

::: evosax.core.restart.BIPORestartParams
    options:
        members: true

::: evosax.core.restart.generation_cond
    options:
        members: true

::: evosax.core.restart.spread_cond
    options:
        members: true

::: evosax.core.restart.cma_cond
    options:
        members: true

::: evosax.core.restart.amalgam_cond
    options:
        members: true

## Kernels

::: evosax.core.kernel.kernel_rbf
    options:
        members: true
