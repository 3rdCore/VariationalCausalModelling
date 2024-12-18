# Autoregressive Variational Autoencoder for Causal Modelling

## Description

This repository contains the implementation of an autoregressive VAE with discrete latent variables.
## Installation

1. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. That's it. You're ready to go!

You can launch the experiments super easily thanks to Lightning and Hydra. The configuration used in the report in the `src/configs` folder. To launch an experiment, you can use the following command:

```bash
python src/train.py
```
You can overwrite any parameter in the config file by specifying it in the command line. For example, to change the number of nodes in the ANM, you can run:

```bash
python src/train.py SCM.n_variables=50
```

To reproduce exactly the same results as in the report, you will need to overwrite the logger with your own wand credentials in the config file `src/configs/train.yaml`.

Then, you can just run the following command:

```bash
./run.sh
```

## Contribution

All kinds of contributions are welcome, e.g. adding more tools, better practices, discussions on trade-offs. Make sure to add any external dependencies to the `requirements.txt` file. If you add a new dependency, please make sure to add it to the `requirements.txt` file.

Set up the pre-commit hooks:

    ```bash
    pre-commit install
    ```

    Make sure you have the `.pre-commit-config.yaml` file set up in your project root.
