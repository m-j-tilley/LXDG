# LXDG: Learned Context-Dependent Gating for Continual Learning

LXDG is a continual learning approach that uses learned context-dependent gating to handle catastrophic forgetting in neural networks. In this code it is tested on the benchmarks permuted MNIST and rotated MNIST.

ArXiv:  https://arxiv.org/abs/2301.07187
openreview: https://openreview.net/pdf?id=dBk3hsg-n6

Published as a conference paper at ICLR 2023

## Requirements
- Python 3.x
- PyTorch
- torchvision
- tqdm
- numpy
- SciPy

## Usage

To run the LXDG experiments, simply run

```python
python run.py
```

Comment/uncomment the relevent lines at the end of the file to run the different experiments.

Here is an example configuration for running LXDG+EWC over 50 tasks:

### Example

```python
config = {
    "TRAIN_BATCH": 256,
    "output_size": 10,
    "device": 0,
    "rndseed": 0,
    "ntasks": 50,
}
config_perm_LXDG_EWC = {
    "task_type": "permuted_MNIST",
    "name": f"perm_LXDG_EWC_{config['rndseed']}",
    "use_ewc": True,
    "use_sparse": True,
    "use_keepchange": True,
    "include_gating_layers": True,
    "random_xdg": False,
    "dump_gates": True,
}

config_perm_LXDG_EWC.update(config)
run_lxdg(config_perm_LXDG_EWC)
```