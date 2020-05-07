# Requirements

The codes were written and tested under python 3.7.3 and pytorch 1.0.1


# Execution - Anomaly Detection

```bash
python main_pgn.py
```

- Tested on CIFAR10, FMNIST, MNIST Datasets
- one could find some options in `utils.py`
- other methods (GPND, DSVDD, AE, VAE, AAE) can be executed in a similar way with execution files `main_dsvdd.py`, `main_gpnd.py` and so on.


# Execution - beta-VAE for rate-distortion analysis

```bash
python beta_vae_test.py
```

# Citation 
- Please refer to the following bib 