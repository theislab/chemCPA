# Imports / Arg Parser / Functions
# Copied from generate.py preprocess.py and / or hgraph/hgnn.py
import argparse
import json

import pandas as pd
import torch

from hgraph import *
from hgraph.hgnn import make_cuda
from preprocess import tensorize

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", default="data/chembl/vocab.txt")
parser.add_argument("--atom_vocab", default=common_atom_vocab)
parser.add_argument("--model", default="ckpt/chembl-pretrained/model.ckpt")

parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--nsample", type=int, default=10000)

parser.add_argument("--rnn_type", type=str, default="LSTM")
parser.add_argument("--hidden_size", type=int, default=250)
parser.add_argument("--embed_size", type=int, default=250)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--latent_size", type=int, default=32)
parser.add_argument("--depthT", type=int, default=15)
parser.add_argument("--depthG", type=int, default=15)
parser.add_argument("--diterT", type=int, default=1)
parser.add_argument("--diterG", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.0)

args = parser.parse_args()

# Parse Vocabuary File
vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
args.vocab = PairVocab(vocab)

# these keys currently don't work (RDKit throws some error, need to fix by getting new vocabulary)
# also it seems not all of them are actually present in Sciplex3_new.hdf5 dataset
non_working_keys = [
    "Taselisib",
    "Everolimus",
    "AZD5591",
    "Temsirolimus",
    "Epothilone",
    "AG-490",
    "Divalproex",
    "SNS-314",
    "Mesna",
    "Tanespimycin",
    "Rigosertib",
    "Patupilone",
    "G007-LK",
    "Alvespimycin",
]
# Test Compound to reconstruct
keys = []
smiles = []

with open("data/sciplex3/drugs_oksana.json") as f:
    oksana_drugs_smiles = json.load(f)

for drugname, smile in oksana_drugs_smiles.items():
    # we filter out the drugs that we currently cannot encode
    if drugname not in non_working_keys:
        keys.append(drugname)
        smiles.append(smile)
print(f"Encoding {len(smiles)} SMILES (out of {len(oksana_drugs_smiles)})")

# Convert SMILES String into MolGraph Tree / Graph Tensors
# (See preprocess.py)
o = tensorize(smiles, args.vocab)
batches, tensors, all_orders = o

# Extract pieces we need
tree_tensors, graph_tensors = make_cuda(tensors)

# Load Checkpoint model
model = HierVAE(args).cuda()
model.load_state_dict(torch.load(args.model)[0])
model.eval()

# Encode compound
with torch.no_grad():
    root_vecs, tree_vecs, _, graph_vecs = model.encoder(tree_tensors, graph_tensors)
    # TODO What's the difference between the first and second root_vecs?
    root_vecs, root_kl = model.rsample(
        root_vecs, model.R_mean, model.R_var, perturb=False
    )

# save the result
df = pd.DataFrame.from_dict({"drug": keys, "embedding": list(root_vecs.cpu().numpy())})
df.to_parquet("data/sciplex3/oksana_drugs_embedding_incomplete.parquet")

# Decode compound
# decoded_smiles = model.decoder.decode(
#     (root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150
# )


# The decoded and original smiles / compound do not match
# Not sure if this is because something is done wrong or just
# because this compound is one that couldn't be reconstructed
# accurately
# print(f'ORIGINA SMILES: {"".join(smiles)}')
# print("DECODED SMILES: {0}".format("".join(decoded_smiles)))
