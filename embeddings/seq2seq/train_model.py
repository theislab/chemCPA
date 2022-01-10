from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
from deepchem.models.optimizers import ExponentialDecay
from rdkit import Chem

# I generated these on 15.11.2021. If we update the smiles or add new drug
# then these should be stored anew.
TOKENS = [
    "#",
    "(",
    ")",
    "*",
    "+",
    "-",
    ".",
    "/",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "=",
    "@",
    "A",
    "B",
    "C",
    "F",
    "H",
    "I",
    "L",
    "M",
    "N",
    "O",
    "P",
    "S",
    "[",
    "\\",
    "]",
    "a",
    "c",
    "d",
    "e",
    "g",
    "i",
    "l",
    "n",
    "o",
    "r",
    "s",
    "t",
    "u",
]
MAX_LENGTH = 461


def load_train_val(datasets_fpath="../../datasets"):
    datasets_fpath = Path(datasets_fpath)

    # read in all relevant smiles
    train_smiles = []
    for f in ["all_smiles_lincs_trapnell.csv", "train_smiles_muv.csv"]:
        x = pd.read_csv(datasets_fpath / f, header=None)
        train_smiles += list(x[0])

    val_smiles = list(
        pd.read_csv(datasets_fpath / "validation_smiles_muv.csv", header=None)[0]
    )

    # get canoncialized train / val split
    canonicalize = lambda smile: Chem.MolToSmiles(Chem.MolFromSmiles(smile))
    train_smiles = np.array([canonicalize(smile) for smile in list(train_smiles)])
    val_smiles = np.array([canonicalize(smile) for smile in list(val_smiles)])
    return train_smiles, val_smiles


def get_model(
    train_smiles, model_dir="data/small_56", encoder_layers=2, decoder_layers=2
):
    tokens = set()
    for s in train_smiles:
        tokens = tokens.union(set(c for c in s))
    tokens = sorted(list(tokens))

    max_length = max(len(s) for s in train_smiles)
    batch_size = 100
    batches_per_epoch = int(len(train_smiles) / batch_size)
    model = dc.models.SeqToSeq(
        TOKENS,
        TOKENS,
        MAX_LENGTH,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        embedding_dimension=56,
        model_dir=model_dir,
        batch_size=batch_size,
        learning_rate=ExponentialDecay(0.001, 0.9, batches_per_epoch),
    )
    return model


def train_model(model, train_smiles):
    def generate_sequences(epochs):
        for i in range(epochs):
            print("Epoch:", i)
            for s in train_smiles:
                yield (s, s)

    # there are ~92K molecules, batchsize is 100 -> ~920 train steps per epoch
    model.fit_sequences(
        generate_sequences(200),
    )


if __name__ == "__main__":
    print(Path().cwd())
    # make sure GPU is available
    assert len(tf.config.list_physical_devices("GPU")) > 0
    train_smiles, val_smiles = load_train_val()
    model = get_model(train_smiles)
    train_model(model, train_smiles)

    # load the most recent checkpoint
    model.restore()
    pred = model.predict_from_sequences(val_smiles)
    n_restored_smiles = 0
    for s_pred, s_real in zip(pred, val_smiles):
        s_pred = "".join(s_pred)
        if s_pred == s_real:
            n_restored_smiles += 1
    print(f"Acc: {n_restored_smiles / len(val_smiles)}")
