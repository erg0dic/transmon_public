# transmon

Code for the paper [Sample-efficient Model-based Reinforcement Learning for Quantum Control](https://arxiv.org/pdf/2304.09718.pdf).


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
@article{khalid2023sample,
  title={Sample-efficient Model-based Reinforcement Learning for Quantum Control},
  author={Irtaza Khalid and Carrie A. Weidner and Edmond A. Jonckheere and Sophie G. Shermer and Frank C. Langbein},
  journal={arXiv preprint arXiv:2304.09718},
  year={2023}
}
```
