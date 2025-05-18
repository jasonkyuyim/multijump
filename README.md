# MultiJump

Implementation of protein generation experiments in [Generator Matching: Generative modeling with arbitrary Markov processes](https://arxiv.org/abs/2410.20587).
This codebase is a fork of [MultiFlow](https://github.com/jasonkyuyim/multiflow).
We don't perform any training. Only the inference code of MultiFlow is modified to utilize jumps in the SO(3) flow.

If you use this codebase, then please cite

```bibtex
@article{holderrieth2024generator,
  title={Generator Matching: Generative modeling with arbitrary Markov processes},
  author={Holderrieth, Peter and Havasi, Marton and Yim, Jason and Shaul, Neta and Gat, Itai and Jaakkola, Tommi and Karrer, Brian and Chen, Ricky TQ and Lipman, Yaron},
  journal={arXiv preprint arXiv:2410.20587},
  year={2024}
}
```

LICENSE: MIT

## Installation

We recommend using [mamba](https://mamba.readthedocs.io/en/latest/).
If using mamba then use `mamba` in place of `conda`.

```bash
# Install environment with dependencies.
conda env create -f multijump.yml

# Activate environment
conda activate multijump

# Install local package.
# Current directory should have setup.py.
pip install -e .
```

Next you need to install torch-scatter manually depending on your torch version.
(Unfortunately torch-scatter has some oddity that it can't be installed with the environment.)
We use torch 2.0.1 and cuda 11.7 so we install the following

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

If you use a different torch then that can be found with the following.
```bash
# Find your installed version of torch
python
>>> import torch
>>> torch.__version__
# Example: torch 2.0.1+cu117
```

> [!WARNING]  
> You will likely run into the follow error from DeepSpeed
```bash
ModuleNotFoundError: No module named 'torch._six'
```
If so, replace `from torch._six import inf` with `from torch import inf`.
* `/path/to/envs/site-packages/deepspeed/runtime/utils.py`
* `/path/to/envs/site-packages/deepspeed/runtime/zero/stage_1_and_2.py`

where `/path/to/envs` is replaced with your path. We would appreciate a pull request to avoid this monkey patch!

## Wandb

Our training relies on logging with wandb. Log in to Wandb and make an account.
Authorize Wandb [here](https://wandb.ai/authorize).

## Inference

We provide pre-trained model weights at this [Zenodo link](https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTk2YjlmLTM4OTUtNGVhYi1hODcxLWE1ZjExOTczY2IzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDY5ZDUzYjVjMTNhNDllMDYxNmI3Yjc2NjcwYjYxZiJ9.C2eZZmRu-nu7H330G-DkV5kttfjYB3ANozdOMNm19uPahvtLrDRvd_4Eqlyb7lp24m06e4OHhHQ4zlj68S1O_A).

Run the following to unpack the weights
```bash
tar -xzvf weights.tar.gz
```

To run inference with jumps,
```bash
# Unconditional Co-Design
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_jumps
```

The SO(3) jump parameters can be found in `multiflow/configs/inference_jumps.yaml`.
Specifically, see `interpolant.interpolant.rots` and `interpolant.interpolant.vmf`.
By default, we use the best parameters for the results reported in the paper.
Some notes:
* If `jump_weight` > 0.0 and `flow_weight` == 0.0 then jumps are performed.
* If `jump_weight` == 0.0 and `flow_weight` > 0.0 then jumps are performed.
* If `jump_weight` > 0.0 and `flow_weight` > 0.0 then a markov superposition is performed.

```yaml
inference:
    interpolant:
        rots:
            corrupt: True
            sample_schedule: exp
            exp_rate: 10
            num_jump_bins: 2056
            jump_weight: 1.0
            jump_schedule: fixed
            jump_exp_weight: 1.0
            jump_temp: 1.0
            flow_weight: 0.0
            flow_schedule: fixed
            flow_exp_weight: 1.0

        vmf:
            path_name: vmf
            kappa_max: 200.0
            kappa_min: 0.01
            t_min: 0.01
            logive_approx: False
            upper_half: True
            kappa_alpha: 4.0
```
