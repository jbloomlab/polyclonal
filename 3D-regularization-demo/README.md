3D regularization demo
==

[This notebook](penalties.ipynb) is a prototype for regularizing on structural info to identify an epitope. See discussion and notes in issue [#24](https://github.com/jbloomlab/polyclonal/issues/24).

To use the notebook interactively:
1. Create conda environment from the provided spec:
  ```bash
  conda env create -f env.yml
  ```
  **Note:** python 3.7 is currently required by the [`mushi.optimization`](https://harrispopgen.github.io/mushi/stubs/mushi.optimization.html#module-mushi.optimization) dependency, but there are [plans](https://github.com/harrispopgen/mushi/issues/59) to fix that.

2. Activate the new environment:
  ```bash
  conda activate 3dreg
  ```
3. Open the notebook in jupyter:
  ```bash
  jupyter notebook penalties.ipynb
  ```
