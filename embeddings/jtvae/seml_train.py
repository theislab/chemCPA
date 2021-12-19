import argparse
import resource
from pathlib import Path

import pretrain
import seml
import torch
from sacred import Experiment
from seml.utils import make_hash

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


class ExperimentWrapper:
    def __init__(self, init_all=True):
        pass

    @ex.capture(prefix="training")
    def train(
        self,
        training_path,
        incl_zinc,
        save_path,
        batch_size,
        hidden_size,
        latent_size,
        depth,
        lr,
        gamma,
        max_epoch,
        num_workers,
        print_iter,
        save_iter,
        subsample_zinc_percent,
        multip_share_strategy=None,
    ):
        if multip_share_strategy:
            torch.multiprocessing.set_sharing_strategy(multip_share_strategy)

        # allow for more file descriptors open in parallel
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

        # Construct the training file. If requested, also add all SMILES from ZINC
        outpath = (
            Path().cwd()
            / "data"
            / f"train_{seml.utils.make_hash(ex.current_run.config)}.txt"
        )
        zinc_f = Path().home() / ".dgl" / "jtvae" / "train.txt"
        assert zinc_f.exists()

        # truncates the outfile if it already exists
        with open(outpath, "w") as outfile:
            n_total_smiles = 0
            with open(zinc_f) as infile:
                for i, line in enumerate(infile):
                    # subsampling the file
                    if i >= int(subsample_zinc_percent * 220011):
                        break
                    line = line.strip()
                    # skip the header
                    if line != "smiles":
                        n_total_smiles += 1
                        outfile.write(line + "\n")

            with open(training_path) as infile:
                for line in infile:
                    line = line.strip()
                    # skip the header
                    if line != "smiles":
                        n_total_smiles += 1
                        outfile.write(line + "\n")
        print(f"Total SMILES: {n_total_smiles}, stored at {outpath.resolve()}")

        if training_path:
            assert Path(training_path).exists(), training_path
        args = argparse.Namespace(
            **{
                "train_path": str(outpath),
                "save_path": save_path,
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "latent_size": latent_size,
                "depth": depth,
                "lr": lr,
                "gamma": gamma,
                "max_epoch": max_epoch,
                "num_workers": num_workers,
                "print_iter": print_iter,
                "save_iter": save_iter,
                "use_cpu": False,
            }
        )
        results = pretrain.main(args)
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
