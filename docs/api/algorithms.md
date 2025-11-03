# Algorithms API

`evosax.algorithms` exposes both distribution-based and population-based evolution strategies behind a unified ask-eval-tell interface. All classes are JAX pytrees with immutable state updates to ensure compatibility with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

::: evosax.algorithms
    options:
        members: true
        filters:
            - '!algorithms'
