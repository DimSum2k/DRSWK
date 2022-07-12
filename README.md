# Code to reproduce experiments of the ICML 2022 Paper Distribution Regression with Sliced Wasserstein Kernels

ArXiv link: https://arxiv.org/abs/2202.03926. Proceeding link: https://proceedings.mlr.press/v162/meunier22b.html.

## Installation

Using `conda`: clone the repository, and once inside it,
execute the following bash commands

### CPU install

```bash
conda create -n drswk && conda activate drswk

# To run experiments
conda install pytorch numpy scipy pot pillow

# To plot experiment resulst
conda install matplotlib ipympl pandas
```

