# Types and dataclasses

Strategy states and parameter trees are documented in `evosax.types`. Inspect these definitions when you need to log internal statistics or integrate with custom training loops.

## Type aliases

| Alias | Definition | Description |
| --- | --- | --- |
| `PyTree` | `typing.Any` | JAX-compatible tree structure used throughout the library. |
| `Solution` | `PyTree` | Individual candidate sampled by an algorithm. |
| `Population` | `PyTree` | Batched collection of solutions passed between `ask` and `tell`. |
| `Fitness` | `jax.Array` | Fitness values returned by objective functions. |
| `Metrics` | `PyTree` | Auxiliary statistics emitted during `tell` for logging or analysis. |

## Base dataclasses

::: evosax.types.State

::: evosax.types.Params
