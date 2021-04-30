import logging
from typing import OrderedDict
from sacred import Experiment
from collections import defaultdict
import os
import json
import time
import torch
import seml
import numpy as np
from compert.train import custom_collate, evaluate
from compert.data import load_dataset_splits
from compert.model import ComPert

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


# class ModelVariant1:
#     """
#     A dummy model variant 1, which could, e.g., be a certain model or baseline in practice.
#     """

#     def __init__(self, hidden_sizes, dropout):
#         self.hidden_sizes = hidden_sizes
#         self.dropout = dropout


# class ModelVariant2:
#     """
#     A dummy model variant 2, which could, e.g., be a certain model or baseline in practice.
#     """

#     def __init__(self, hidden_sizes, dropout):
#         self.hidden_sizes = hidden_sizes
#         self.dropout = dropout
def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="dataset")
    def init_dataset(self, dataset_type: str, data_params: dict):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        if dataset_type == "kang":
            self.datasets, self.dataset = load_dataset_splits(
                **data_params, return_dataset=True
            )

    @ex.capture(prefix="model")
    def init_model(self, hparams: dict, additonal_params: dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # if training.scorer_type == "linear":
        # Here we can pass the "model_params" dict to the constructor directly, which can be very useful in
        # practice, since we don't have to do any model-specific processing of the config dictionary.
        self.autoencoder = ComPert(
            self.datasets["training"].num_genes,
            self.datasets["training"].num_drugs,
            self.datasets["training"].num_cell_types,
            self.datasets["training"].num_gene_sets,
            device=device,
            hparams=hparams,
            **additonal_params,
        )
        if additonal_params["scores_discretizer"] == "kbins":
            print("discretizer is not none")
            self.autoencoder.scores_discretizer.fit(self.dataset.scores)

    def update_datasets(self):
        self.datasets.update(
            {
                "loader_tr": torch.utils.data.DataLoader(
                    self.datasets["training"],
                    batch_size=self.autoencoder.hparams["batch_size"],
                    collate_fn=custom_collate,
                    shuffle=True,
                )
            }
        )
        # pjson({"training_args": args})
        pjson({"autoencoder_params": self.autoencoder.hparams})

    # @ex.capture(prefix="optimization")
    # def init_optimizer(self, regularization: dict, optimizer_type: str):
    #     weight_decay = regularization["weight_decay"]
    #     self.optimizer = optimizer_type  # initialize optimizer

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.update_datasets()

    @ex.capture(prefix="training")
    def train(
        self,
        num_epochs: int,
        max_minutes: int,
        checkpoint_freq: int,
        ignore_evaluation: bool,
    ):
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_training_stats = defaultdict(float)

            for genes, drugs, cell_types, scores in self.datasets["loader_tr"]:
                minibatch_training_stats = self.autoencoder.update(
                    genes, drugs, cell_types, scores
                )

                for key, val in minibatch_training_stats.items():
                    epoch_training_stats[key] += val

            for key, val in epoch_training_stats.items():
                epoch_training_stats[key] = val / len(self.datasets["loader_tr"])
                if not (key in self.autoencoder.history.keys()):
                    self.autoencoder.history[key] = []
                self.autoencoder.history[key].append(val)
            self.autoencoder.history["epoch"].append(epoch)

            ellapsed_minutes = (time.time() - start_time) / 60
            self.autoencoder.history["elapsed_time_min"] = ellapsed_minutes

            # decay learning rate if necessary
            # also check stopping condition: patience ran out OR
            # time ran out OR max epochs achieved
            stop = ellapsed_minutes > max_minutes or (epoch == num_epochs - 1)

            if (epoch % checkpoint_freq) == 0 or stop:
                if not ignore_evaluation:
                    evaluation_stats = evaluate(self.autoencoder, self.datasets)
                    for key, val in evaluation_stats.items():
                        if not (key in self.autoencoder.history.keys()):
                            self.autoencoder.history[key] = []
                        self.autoencoder.history[key].append(val)
                    self.autoencoder.history["stats_epoch"].append(epoch)
                else:
                    evaluation_stats = {}

                pjson(
                    {
                        "epoch": epoch,
                        "training_stats": epoch_training_stats,
                        "evaluation_stats": evaluation_stats,
                        "ellapsed_minutes": ellapsed_minutes,
                    }
                )

                # torch.save(
                #     (autoencoder.state_dict(), args, autoencoder.history),
                #     os.path.join(
                #         args["save_dir"],
                #         "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                #     ),
                # )

                # pjson(
                #     {
                #         "model_saved": "model_seed={}_epoch={}.pt\n".format(
                #             args["seed"], epoch
                #         )
                #     }
                # )
                if not ignore_evaluation:
                    stop = stop or self.autoencoder.early_stopping(
                        np.mean(evaluation_stats["test"])
                    )
                if stop:
                    pjson({"early_stop": epoch})
                    break

        results = self.autoencoder.history
        results["total_epochs"] = epoch
        return results

        # # everything is set up
        # for e in range(num_epochs):
        #     # simulate training
        #     continue
        # results = {
        #     "test_acc": 0.5 + 0.3 * np.random.randn(),
        #     "test_loss": np.random.uniform(0, 10),
        #     # ...
        # }
        # return results


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print("get_experiment")
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.train()
