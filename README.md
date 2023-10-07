# transmon

Code for the paper [Sample-efficient Model-based Reinforcement Learning for Quantum Control](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043002).


## Installation
To setup, run on the command line:
```bash
pip install -e .
```

## Basic Usage
To train a model-based policy optimization agent, an example command to train the model on the 2-qubit transmon would look like:
```bash
python3 mbsac.py --epochs 2000 --system "transmon" --verbose True --experiment_name "transmon_model_no_shots_1" --buffer_size 2000 --respawn True --use_rl_model True --use_ham_model True --use_totally_random_ham True
```

## Credit
The model-based policy optimization algorithm in the repo that was introduced in [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253) is based upon the starter code from [this repo](https://github.com/Xingyu-Lin/mbpo_pytorch/).

## Cite
If you use this code, please cite the following paper:
```
@article{PhysRevResearch.5.043002,
  title = {Sample-efficient model-based reinforcement learning for quantum control},
  author = {Khalid, Irtaza and Weidner, Carrie A. and Jonckheere, Edmond A. and Schirmer, Sophie G. and Langbein, Frank C.},
  journal = {Phys. Rev. Res.},
  volume = {5},
  issue = {4},
  pages = {043002},
  numpages = {21},
  year = {2023},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.5.043002},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.5.043002}
}

```
