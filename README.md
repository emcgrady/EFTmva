# EFTmva: multivariate discriminants for EFT

## Set-up instructions

```
conda create -n eft_mva python=3.10.8
conda activate eft_mva

# get pytorch https://pytorch.org/get-started/locally/

pip install tqdm
pip install matplotlib
pip install pyyaml
pip install uproot awkward
pip install torch

```

## Instructions to run

`train.py` allows the user to train a neural network to discriminate between the SM and either (a) a specific BSM point (specified with the option `--bsm-point`) or (b) events reweighed by a given term of the quadratic expansion (specified with the option `--term`)

Examples of usage would be 

```
python train.py --term ctq8_ctq8 --epochs 50 --name ctq8_ctq8
```

to train a NN discriminating SM against the `ctq8^2` term, or 

```
python train.py --bsm-point ctq8=16 --epochs 50 --name ctq8_16
python train.py --bsm-point ctq8=12 --epochs 50 --name ctq8_12
python train.py --bsm-point ctq8=8 --epochs 50 --name ctq8_8
python train.py --bsm-point ctq8=4 --epochs 50 --name ctq8_4
```

to train different NNs discriminating the SM against `ctq8=4,8,12,16` scenarios.

The data is provided through `rootfiles` that are converted to `pytorch` tensors. The tensors are stored in separate files for the input features, the SM weights and the BSM weights, the latter with a name specific to the BSM point / term chosen. If any of these need to be reproduced, the associated file can be erased or pass the argument `--forceRebuild`


The example above allows to parametrize the likelihood as a function of `ctq8`. The procedure to do so is implemented in `utils/buildLikelihood.py` (modulo cross-terms in the likelihood, to do), which takes a `yaml` file describing the networks trained in the previous step.

A ROC curve comparing a dedicated discriminator (the one trained against `ctq8=16`) against the optimal observable given by the regressed likelihood ratio. 

```
python do_roc.py --likelihood examples/ctq8_regression.yaml  --bsm-point ctq8=16   --dedicated ctq8_16/network_bsm_weight_ctq8_16_last.p --name regressed_likelihood_16
```

## To do

* Implement crossed terms in `buildLikelihood.py`.
* Update dataloader to `IterableDataset` for when we need to handle larger datasets.
* Fetch list of WC from sample directly, if possible.
* Read list of input variables, network architecture, regularization from configuration files.
