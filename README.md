# Predicting single-cell perturbation responses for unseen drugs
Code accompanying the ICLR 2022 MLDD paper.
Main authors: Leon Hetzel, Simon Boehm (equal contribution).

![architecture of CCPA](docs/chemical_CPA.png)

## Codebase overview
Disclaimer: Various people are working on more maintainable implementations of
(chem)CPA, and it may be worth waiting until CPA lands in [scvi-tools](https://github.com/scverse/scvi-tools).
However, this current version implements many performance optimizations (careful data-movement, disentanglement on GPU,
numerically stable losses, ...) that allow for training on large datasets, like L1000.

For the final models, we provide [weight checkpoints](https://f003.backblazeb2.com/file/chemCPA-models/chemCPA_models.zip) as well as the [hyperparameter configuration & loss curves](https://f003.backblazeb2.com/file/chemCPA-models/finetuning_num_genes.json).
The raw datasets can be downloaded from a [FAIR server](https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar).

To setup the environment, install conda and run:
```python
conda env create -f environment.yml
python setup.py install -e .
```

- `chemCPA/`: contains the code for the model, the data, and the training loop.
- `embeddings`: There is one folder for each molecular embedding model we benchmarked. Each contains an `environment.yml` with dependencies. We generated the embeddings using the provided notebooks and saved them to disk, to load them during the main training loop.
- `experiments`: Each folder contains a `README.md` with the experiment description, a `.yaml` file with the seml configuration, and a notebook to analyze the results.
- `notebooks`: Example analysis notebooks.
- `preprocessing`: Notebooks for processing the data. For each dataset there is one notebook that loads the raw data.
- `tests`: A few very basic tests.
 
All experiments where run through [seml](https://github.com/TUM-DAML/seml).
The entry function is `ExperimentWrapper.__init__` in `chemCPA/seml_sweep_icb.py`.
For convenience, we provide a script to run experiments manually for debugging purposes at `chemCPA/manual_seml_sweep.py`.
The script expects a `config_sciplex3_interactive.yaml` file containing the experiment configuration.

All notebooks also exist as Python scripts (converted through [jupytext](https://github.com/mwouts/jupytext)) to make them easier to review.

## Citation
You can cite our work as:
```
@inproceedings{hetzel2022predicting,
  title={Predicting single-cell perturbation responses for unseen drugs},
  author={Hetzel, Leon and Böhm, Simon and Kilbertus, Niki and Günnemann, Stephan and Lotfollahi, Mohammad and Theis, Fabian J},
  booktitle={ICLR2022 Machine Learning for Drug Discovery},
  year={2022}
}
```
