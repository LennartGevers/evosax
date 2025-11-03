# evosax Documentation

`evosax` is a high-performance library of evolution strategies built on top of JAX. It is designed to make modern black-box optimization workflows easy to prototype and scale, from classic CMA-ES variants to recent differentiable and learned approaches. With full support for JAX transformations, you can jit-compile, vectorize, and parallelize evolutionary algorithms on CPUs, GPUs, and TPUs.

!!! tip "New to evolution strategies?"
    Start with the [Getting started](getting-started.md) guide for installation instructions and the core ask-eval-tell workflow.

## Key capabilities

- **Comprehensive algorithm collection** – more than 30 distribution- and population-based algorithms unified behind a consistent API.
- **JAX-native performance** – compatible with `jax.jit`, `jax.vmap`, and `jax.lax.scan` to leverage accelerators efficiently.
- **Production ready** – includes restart policies, fitness shaping, and learned evolution components used in research environments.

## Quick example

```python
import jax
from evosax.algorithms import CMA_ES

# Instantiate the search strategy
es = CMA_ES(population_size=32, solution=dummy_solution)
params = es.default_params

# Initialize state
key = jax.random.key(0)
state = es.init(key, params)

for generation in range(num_generations):
    key, key_ask, key_eval = jax.random.split(key, 3)
    population, state = es.ask(key_ask, state, params)
    fitness = evaluate(population)
    state = es.tell(population, fitness, state, params)

best_solution = state.best_solution
best_fitness = state.best_fitness
```

## Learning resources

- Explore interactive examples in the [`examples/`](https://github.com/RobertTLange/evosax/tree/main/examples) directory, including reinforcement learning, black-box optimization benchmarks, and diffusion evolution.
- Watch the [MLC Research Jam talk](https://www.youtube.com/watch?v=Wn6Lq2bexlA&t=51s) for an overview of evolution strategies in JAX.
- Read the [arXiv paper](https://arxiv.org/abs/2212.04180) for the full description of the library and its design motivations.

## Citing evosax

If you use `evosax` in academic work, please cite the accompanying paper:

```bibtex
@article{evosax2022github,
    author  = {Robert Tjarko Lange},
    title   = {evosax: JAX-based Evolution Strategies},
    journal = {arXiv preprint arXiv:2212.04180},
    year    = {2022},
}
```
