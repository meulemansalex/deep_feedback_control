# Credit Assignment in Neural Networks through Deep Feedback Control
This repository is the official implementation of the NeurIPS 2021 submission 
"Credit Assignment in Neural Networks through Deep Feedback Control".

## Install Python packages
All the needed Python libraries can be installed with conda by running:
```
$ conda env create -f DFC_environment.yml
```

## Running the methods
One can run the various methods on any feedforward fully connected neural network architecture by calling the
`main.py` script and specifying the needed
command-line arguments. 
Run `python3 main.py --help` for a documentation on all command-line arguments.

To avoid typing all hyperparameters into the command window, one can also define
a config file and use the `run_config.py` script with the specified config file.
```
$ python3 run_config.py --config_module=configs.configs_mnist.mnist_dfc_ssa
```
We provide the hyperparameter configurations used for all results in Table 1 of 
the NeurIPS 2021 submission in the directory `configs`.


## Generating the computer vision results
We used 5 random weight initializations for computing the results in Table 1. 
To automatically run all 5 random seeds and save the results (together with 
mean and std) in a `.csv` file, run the `seed_robustness.py` script provided 
with the correct config file.
```
$ python3 run_seed_robustness.py --config_module=configs.configs_mnist.mnist_dfc_ssa --name=mnist_dfc_ssa.csv
```
## Creating figure 1D
For creating figure 1D, which provides an example of the DFC dynamics, run:
```
$ python3 run_toy_exp_fig1D.py
```
This figure will be saved in `logs/toy_experiments/fig1D`.
## Creating figure 2
For creating figure 2, run:
```
$ python3 run_toy_exp_fig2.py
```
This figure will be saved in `logs/toy_experiments/fig2`.

## Creating figure 3
For creating figure 3, we need to run all methods separately in pipelines and
then run a plotting script. To run the pipelines, run:
```
$ python3 run_pipeline.py --config_module=configs.toy_experiments.configs_fig3.config_dfc --experiment_name=fig3_dfc --out_dir=logs/toy_experiments/fig3
```
for all configs in `configs/toy_experiments/configs_fig3` (remember to change the config_module and experiment_name accordingly).

Note that the pipeline of DFC also generates Figure S1 in `logs/toy_experiments/fig3/figS1`. Next, to create Figure 3, which will be saved in `logs/toy_experiments/fig3`, run: 
```
$ python3 run_toy_exp_fig3.py
```
## Creating figures of Section C.2 and C.3
For creating the feedback learning figures of Section C.2 and C.3,
run the following pipeline script:
```
$ python3 run_pipeline.py --config_module=configs.toy_experiments.configs_figC2C3.fig_S2_linear --experiment_name=fig_S2_linear --out_dir=logs/toy_experiments/figC2C3
```
for all configs in `configs/toy_experiments/configs_figC2C3` (remember to change the config_module and experiment_name accordingly).

## Creating the alignment figures of Section E.5
For creating the alignment figures of Sections S5-S12 in the appendices, 
run the following script with the desired config file (e.g., here is shown 
for MNIST with trained feedback weights):
```
$ python3 run_pipeline.py --config_module=configs.pipeline_configs.mnist --experiment_name=mnist
```

## Method names
For historical reasons, the naming of the methods in the code base does not 
always correspond to the naming in the NeurIPS submission. The table below 
provides a translation sheet.

| NeurIPS Submission  | Code Base |
| ------------- | ------------- |
| DFC  | cont  |
| DFC-SS  | ss |
| DFC-SSA  | ndi|
| MN | GNT  |
| DFC | tpdi|

To use DFC, provide `--network_type=DFC` and `--grad_deltav_cont`. To use 
DFC-SS, provide `--network_type=DFC` without `--grad_deltav_cont`. To use 
DFC-SSA, provide `--network_type=DFC` and `--ndi`.

## References
This code base is built further upon the code base of 
Meulemans et al. 2020, 'A Theoretical Framework for Target Propagation', with the
Apache 2.0 license:
https://www.apache.org/licenses/LICENSE-2.0