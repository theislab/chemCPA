import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from pprint import pformat

import numpy as np
import seml
import torch
from sacred import Experiment

from chemCPA.data import load_dataset_splits
from chemCPA.embedding import get_chemical_representation
from chemCPA.model import ComPert
from chemCPA.profiling import Profiler
from chemCPA.train import custom_collate, evaluate, evaluate_r2, evaluate_r2_sc

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


profiler = None


@ex.pre_run_hook(prefix="profiling")
def init_profiler(run_profiler: bool, outdir: str):
    if run_profiler:
        if not Path(outdir).exists():
            Path(outdir).mkdir(parents=True)
        global profiler
        profiler = Profiler(
            str(seml.utils.make_hash(ex.current_run.config)),
            outdir,
        )
        profiler.start()


@ex.post_run_hook
def stop_profiler():
    if profiler:
        profiler.stop(experiment=ex)


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

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "dataset".
    @ex.capture(prefix="dataset")
    def init_dataset(self, dataset_type: str, data_params: dict):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="dataset ", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """

        if dataset_type in ("kang", "trapnell", "lincs"):
            self.datasets, self.dataset = load_dataset_splits(
                **data_params, return_dataset=True
            )

    @ex.capture(prefix="model")
    def init_drug_embedding(self, embedding: dict):
        self.embedding_model_type = embedding["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if embedding["model"] != "vanilla":
            # ComPert will use the provided embedding, which is frozen during training
            self.drug_embeddings = get_chemical_representation(
                smiles=self.dataset.canon_smiles_unique_sorted,
                embedding_model=embedding["model"],
                data_dir=embedding["directory"],
                device=device,
            )
        else:
            assert embedding["model"] == "vanilla"
            # ComPert will initialize a new embedding, which is updated during training
            self.drug_embeddings = None

    @ex.capture(prefix="model")
    def init_model(
        self,
        hparams: dict,
        additional_params: dict,
        load_pretrained: bool,
        append_ae_layer: bool,
        enable_cpa_mode: bool,
        pretrained_model_path: str,
        pretrained_model_hashes: dict,
    ):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if load_pretrained:
            (
                state_dict,
                cov_embeddings_state_dicts,
                model_config,
                COVARIATE_AVAILABLE,
            ) = self.load_state_dict(
                pretrained_model_hashes, pretrained_model_path, append_ae_layer
            )
            append_layer_width = (
                self.datasets["training"].num_genes if append_ae_layer else None
            )
            in_out_size = (
                model_config["num_genes"]
                if append_ae_layer
                else self.datasets["training"].num_genes
            )
            # idea: Reconstruct the ComPert model as pretrained (hence the "old" in_out_size)
            # then add the append_layer (the "new" in_out_size)
            self.autoencoder = ComPert(
                in_out_size,
                self.datasets["training"].num_drugs,
                self.datasets["training"].num_covariates,
                device=device,
                **additional_params,
                hparams=hparams,
                drug_embeddings=self.drug_embeddings,
                use_drugs_idx=self.dataset.use_drugs_idx,
                append_layer_width=append_layer_width,
                enable_cpa_mode=enable_cpa_mode,
            )
            incomp_keys = self.autoencoder.load_state_dict(state_dict, strict=False)
            if COVARIATE_AVAILABLE:
                for embedding, state_dict in zip(
                    self.autoencoder.covariates_embeddings, cov_embeddings_state_dicts
                ):
                    embedding.load_state_dict(state_dict)
            incomp_keys_info = {
                "Missing keys": incomp_keys.missing_keys,
                "Unexpected_keys": incomp_keys.unexpected_keys,
            }
            logging.info(
                "INCOMP_KEYS (make sure these contain what you expected):\n%s",
                pformat(incomp_keys_info, indent=4, width=1),
            )
        else:
            self.autoencoder = ComPert(
                self.datasets["training"].num_genes,
                self.datasets["training"].num_drugs,
                self.datasets["training"].num_covariates,
                device=device,
                **additional_params,
                hparams=hparams,
                drug_embeddings=self.drug_embeddings,
                use_drugs_idx=self.dataset.use_drugs_idx,
                append_layer_width=None,
                enable_cpa_mode=enable_cpa_mode,
            )

    def load_state_dict(
        self, pretrained_model_hashes, pretrained_model_path, append_ae_layer
    ):
        filename = pretrained_model_hashes[self.embedding_model_type] + ".pt"
        filepath = Path(pretrained_model_path) / filename
        logging.info(
            f"Loading pretrained {self.embedding_model_type} model from: {filepath}"
        )
        dumped_model = torch.load(filepath)
        COVARIATE_AVAILABLE = False
        if len(dumped_model) == 3:
            # old version
            state_dict, model_config, history = dumped_model
        elif len(dumped_model) == 4:
            logging.info(f"Loading model without covariate embedding.")
            (
                state_dict,
                adversary_cov_state_dicts,
                model_config,
                history,
            ) = dumped_model
            cov_embeddings_state_dicts = []
        else:
            # new version
            assert len(dumped_model) == 5
            COVARIATE_AVAILABLE = True
            (
                state_dict,
                adversary_cov_state_dicts,
                cov_embeddings_state_dicts,
                model_config,
                history,
            ) = dumped_model
            assert len(cov_embeddings_state_dicts) == 1
        # sanity check
        if append_ae_layer:
            assert model_config["num_genes"] < self.datasets["training"].num_genes
        else:
            assert model_config["num_genes"] == self.datasets["training"].num_genes
        assert model_config["use_drugs_idx"]
        keys = list(state_dict.keys())
        for key in keys:
            # remove all components which we will train from scratch
            # the drug embedding is saved in the state_dict for some reason, but we don't need it
            if key.startswith("adversary_drugs") or key == "drug_embeddings.weight":
                state_dict.pop(key)
        if self.embedding_model_type == "vanilla":
            # for Vanilla CPA, we also train the amortized doser & drug_embedding_encoder anew
            keys = list(state_dict.keys())
            for key in keys:
                if key.startswith("dosers") or key.startswith("drug_embedding_encoder"):
                    state_dict.pop(key)

        return state_dict, cov_embeddings_state_dicts, model_config, COVARIATE_AVAILABLE

    def update_datasets(self):
        """
        Instantiates a torch DataLoader for the given batchsize
        """

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
        # pjson({"autoencoder_params": self.autoencoder.hparams})

    @ex.capture
    def init_all(self, seed: int):
        """
        Sequentially run the sub-initializers of the experiment.
        """

        self.seed = seed
        self.init_dataset()
        self.init_drug_embedding()
        self.init_model()
        self.update_datasets()

    @ex.capture(prefix="training")
    def train(
        self,
        num_epochs: int,
        max_minutes: int,
        checkpoint_freq: int,
        full_eval_during_train: bool,
        run_eval_disentangle: bool,
        save_checkpoints: bool,
        save_dir: str,
        run_eval_r2: bool = True,
        run_eval_r2_sc: bool = True,
        run_eval_logfold: bool = True,
    ):

        print(f"CWD: {os.getcwd()}")
        print(f"Save dir: {save_dir}")
        if save_checkpoints:
            assert save_dir is not None
            if not os.path.exists(save_dir):
                Path(save_dir).mkdir()
                print(f"Created savedir for checkpoints: {save_dir}")

        start_time = time.time()
        for epoch in range(num_epochs):
            # all items are initialized to 0.0
            epoch_training_stats = defaultdict(float)

            for data in self.datasets["loader_tr"]:
                if self.dataset.use_drugs_idx:
                    genes, drugs_idx, dosages, degs, covariates = (
                        data[0],
                        data[1],
                        data[2],
                        data[3],
                        data[4:],
                    )
                    training_stats = self.autoencoder.update(
                        genes=genes,
                        drugs_idx=drugs_idx,
                        dosages=dosages,
                        degs=degs,
                        covariates=covariates,
                    )
                else:
                    genes, drugs, covariates = data[0], data[1], data[2:]
                    training_stats = self.autoencoder.update(
                        genes=genes,
                        drugs=drugs,
                        degs=degs,
                        covariates=covariates,
                    )

                for key, val in training_stats.items():
                    epoch_training_stats[key] += val

            self.autoencoder.scheduler_autoencoder.step()
            self.autoencoder.scheduler_adversary.step()
            if self.autoencoder.num_drugs > 0:
                self.autoencoder.scheduler_dosers.step()

            for key, val in epoch_training_stats.items():
                epoch_training_stats[key] = val / len(self.datasets["loader_tr"])
                if key not in self.autoencoder.history.keys():
                    self.autoencoder.history[key] = []
                self.autoencoder.history[key].append(val)
            self.autoencoder.history["epoch"].append(epoch)

            # print some stats for each epoch
            epoch_training_stats["epoch"] = epoch
            logging.info("\n%s", pformat(dict(epoch_training_stats), indent=4, width=1))

            ellapsed_minutes = (time.time() - start_time) / 60
            self.autoencoder.history["elapsed_time_min"] = ellapsed_minutes
            reconst_loss_is_nan = math.isnan(
                epoch_training_stats["loss_reconstruction"]
            )

            stop = (
                ellapsed_minutes > max_minutes
                or (epoch == num_epochs - 1)
                or reconst_loss_is_nan
            )

            # we always run the evaluation when training has stopped
            if ((epoch % checkpoint_freq) == 0 and epoch > 0) or stop:
                evaluation_stats = {}
                with torch.no_grad():
                    self.autoencoder.eval()
                    evaluation_stats["test"] = evaluate_r2(
                        self.autoencoder,
                        self.datasets["test_treated"],
                        self.datasets["test_control"].genes,
                    )
                    # evaluation_stats["test_sc"] = evaluate_r2_sc(
                    #     self.autoencoder,
                    #     self.datasets["test_treated"],
                    # )
                    self.autoencoder.train()
                test_score = (
                    np.mean(evaluation_stats["test"])
                    if evaluation_stats["test"]
                    else None
                )

                # test_score = (
                #     evaluation_stats["test"][1]  # DE genes
                #     if evaluation_stats["test"]
                #     else None
                # )

                test_score_is_nan = test_score is not None and math.isnan(test_score)
                autoenc_early_stop = self.autoencoder.early_stopping(test_score)
                stop = stop or autoenc_early_stop or test_score_is_nan
                # we don't do disentanglement if the loss was NaN
                # run_full_eval determines whether we run the full evaluate also during training, or only at the end
                if (
                    (full_eval_during_train or stop)
                    and not reconst_loss_is_nan
                    and not test_score_is_nan
                ):
                    logging.info(f"Running the full evaluation (Epoch:{epoch})")
                    evaluation_stats = evaluate(
                        self.autoencoder,
                        self.datasets,
                        eval_stats=evaluation_stats,
                        run_disentangle=run_eval_disentangle,
                        run_r2=run_eval_r2,
                        run_r2_sc=run_eval_r2_sc,
                        run_logfold=run_eval_logfold,
                    )
                    for key, val in evaluation_stats.items():
                        if key not in self.autoencoder.history:
                            self.autoencoder.history[key] = []
                        self.autoencoder.history[key].append(val)
                    self.autoencoder.history["stats_epoch"].append(epoch)

                # print some stats for the evaluation
                stats = {
                    "epoch": epoch,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                    "test_score_is_nan": test_score_is_nan,
                    "reconst_loss_is_nan": reconst_loss_is_nan,
                    "autoenc_early_stop": autoenc_early_stop,
                    "max_minutes_reached": ellapsed_minutes > max_minutes,
                    "max_epochs_reached": epoch == num_epochs - 1,
                }

                logging.info("\n%s", pformat(stats, indent=4, width=1))

                # Cmp using == is fine, since if we improve we'll have updated this in `early_stopping`
                improved_model = self.autoencoder.best_score == test_score
                # Ignore early stopping and save results at the end -> match data in mongoDB
                if save_checkpoints and stop:
                    logging.info(f"Updating checkpoint at epoch {epoch}")
                    file_name = f"{ex.observers[0].run_entry['config_hash']}.pt"
                    torch.save(
                        (
                            self.autoencoder.state_dict(),
                            # adversary covariates are saved as a list attr on the autoencoder
                            # which PyTorch doesn't include in the autoencoder's state dict
                            [
                                adversary_covariates.state_dict()
                                for adversary_covariates in self.autoencoder.adversary_covariates
                            ],
                            [
                                covariate_embedding.state_dict()
                                for covariate_embedding in self.autoencoder.covariates_embeddings
                            ],
                            self.autoencoder.init_args,
                            self.autoencoder.history,
                        ),
                        os.path.join(save_dir, file_name),
                    )
                    logging.info(f"model_saved: {file_name}")

                if stop:
                    logging.info(f"early_stop: {epoch}")
                    break

        results = self.autoencoder.history
        results["total_epochs"] = epoch
        return results


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
