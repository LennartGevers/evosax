# `evosax`: Evolution Strategies in JAX 🦎

[![Pyversions](https://img.shields.io/pypi/pyversions/evosax.svg?style=flat)](https://pypi.python.org/pypi/evosax) [![PyPI version](https://badge.fury.io/py/evosax.svg)](https://badge.fury.io/py/evosax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/RobertTLange/evosax/branch/main/graph/badge.svg?token=5FUSX35KWO)](https://codecov.io/gh/RobertTLange/evosax)
[![Paper](http://img.shields.io/badge/paper-arxiv.2212.04180-B31B1B.svg)](http://arxiv.org/abs/2212.04180)
![Funding provided by DFG Project ID 390523135 - EXC 2002/1](https://img.shields.io/badge/DFG%20funded-Project%20ID%390523135%20--%20EXC%2002%20-blue)
<a href="assets/logo.png"><img src="assets/logo.png" width="170" align="right" /></a>

`evosax` is a comprehensive, high-performance library of evolution strategies (ES) implemented in JAX. It embraces XLA compilation and JAX transformations to make the classical ask–eval–tell workflow trivial to jit-compile, vectorize, and parallelize across CPUs, GPUs, and TPUs. With 30+ algorithms ranging from CMA-ES and OpenAI-ES to diffusion- and learning-based variants, `evosax` scales neuro-evolution without the usual orchestration overhead.

**Try it now** 👉 [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/00_getting_started.ipynb)

!!! tip "New to evolution strategies?"
    Head to the [Getting started](getting-started.md) guide for installation, the basic ask–eval–tell loop, and guidance on choosing an algorithm.

## Highlights

- **Unified API** – Both distribution-based and population-based strategies expose the same JAX-compatible interface.
- **Accelerator friendly** – Designed for `jax.jit`, `jax.vmap`, `jax.lax.scan`, and batched evaluation on modern hardware.
- **Battle tested** – Includes restart schedules, fitness shaping, learned evolution components, and metrics used in research projects.
- **Extensible** – Build custom strategies by subclassing [`EvolutionaryAlgorithm`](api/algorithms.md#evosax.algorithms.EvolutionaryAlgorithm) and reusing the provided utilities.

## Quick start 🍲

```python
import jax
from evosax.algorithms import CMA_ES

# Instantiate the search strategy
es = CMA_ES(population_size=32, solution=dummy_solution)
params = es.default_params

# Initialize state
key = jax.random.key(0)
state = es.init(key, params)

# Ask–Eval–Tell loop
for i in range(num_generations):
    key, key_ask, key_eval = jax.random.split(key, 3)
    population, state = es.ask(key_ask, state, params)
    fitness = evaluate(population)
    state = es.tell(population, fitness, state, params)

# Monitor best result
state.best_solution, state.best_fitness
```

## Algorithm zoo 🦎

| Strategy | Paper | Module | Notebook |
| --- | --- | --- | --- |
| Simple Evolution Strategy | [Rechenberg (1978)](https://link.springer.com/chapter/10.1007/978-3-642-81283-5_8) | [`SimpleES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/simple_es.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb) |
| OpenAI-ES | [Salimans et al. (2017)](https://arxiv.org/abs/1703.03864) | [`Open_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/open_es.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/02_rl.ipynb) |
| CMA-ES | [Hansen & Ostermeier (2001)](https://arxiv.org/abs/1604.00772) | [`CMA_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/cma_es.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb) |
| Diffusion Evolution | [Mordatch et al. (2023)](https://arxiv.org/abs/2305.15784) | [`DiffusionEvolution`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/diffusion_evolution.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/05_diffusion_evolution.ipynb) |

See the [Algorithms](algorithms/distribution.md) section for the full catalogue, parameter defaults, and implementation notes.

## Example notebooks

- [00 – Getting started](examples/00_getting_started.ipynb): ask–eval–tell essentials.
- [02 – Reinforcement learning](examples/02_rl.ipynb): train policies with OpenAI-ES and PGPE.
- [05 – Diffusion evolution](examples/05_diffusion_evolution.ipynb): hybrid diffusion-based search.
- [08 – Parallelisation](examples/08_parallelization.ipynb): scale ES across devices.

The [Examples navigation](examples/00_getting_started.ipynb) links to the full tutorial suite hosted in the repository.

## Further resources

- Watch the [MLC Research Jam talk](https://www.youtube.com/watch?v=Wn6Lq2bexlA&t=51s) for a guided walkthrough of ES in JAX.
- Read the [arXiv paper](https://arxiv.org/abs/2212.04180) for architectural details and benchmarks.
- Explore companion projects such as [QDax](https://github.com/adaptive-intelligent-robotics/QDax) for quality-diversity algorithms in JAX.

## Citing `evosax` ✏️

If you use `evosax` in academic work, please cite the accompanying paper:

```bibtex
@article{evosax2022github,
    author  = {Robert Tjarko Lange},
    title   = {evosax: JAX-based Evolution Strategies},
    journal = {arXiv preprint arXiv:2212.04180},
    year    = {2022},
}
```

## Contributing & acknowledgements 🙏

- Contributions are welcome—open an [issue](https://github.com/RobertTLange/evosax/issues) or submit a pull request after reading the [guidelines](https://github.com/RobertTLange/evosax/blob/main/CONTRIBUTING.md).
- `evosax` is supported in part by the [Google TRC](https://sites.research.google/trc/about/) and the DFG Cluster of Excellence “Science of Intelligence” (EXC 2002/1, project number 390523135).
