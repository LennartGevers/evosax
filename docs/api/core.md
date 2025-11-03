# Core utilities API

The `evosax.core` package provides shared utilities such as restart schedules, fitness shaping, optimizers, and kernel helpers used across algorithms.

::: evosax.core.optimizer
    options:
        members:
            - ScaleByClipUpState
            - scale_by_clipup
            - clipup


::: evosax.core.fitness_shaping
    options:
        members:
            - normalize
            - add_weight_decay
            - identity_fitness_shaping_fn
            - standardize_fitness_shaping_fn
            - normalize_fitness_shaping_fn
            - centered_rank_fitness_shaping_fn
            - weights_fitness_shaping_fn

::: evosax.core.restart
    options:
        members:
            - RestartState
            - RestartParams
            - IPOPRestartState
            - IPOPRestartParams
            - BIPOPRestartState
            - BIPOPRestartParams
            - generation_cond
            - spread_cond
            - cma_cond
            - amalgam_cond

::: evosax.core.kernel
    options:
        members:
            - kernel_rbf
