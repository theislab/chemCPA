# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
# ---

# %% jupyter={"outputs_hidden": true} pycharm={"name": "#%%\n"}
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.pyplot.rcParams["savefig.facecolor"] = "white"
sn.set_context("poster")
IPythonConsole.ipython_useSVG = False


# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
trapnell_df = pd.read_csv(
    "../embeddings/trapnell_drugs_smiles.csv", names=["drug", "smiles", "pathway"]
)
trapnell_df["smiles"] = trapnell_df.smiles.str.strip()
lincs_df = pd.read_csv("../embeddings/lincs_drugs_smiles.csv", names=["drug", "smiles"])
lincs_df["smiles"] = lincs_df.smiles.str.strip()


# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
def tanimoto_score(input_smiles, target_smiles):
    input_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(input_smiles))
    target_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(target_smiles))
    return DataStructs.TanimotoSimilarity(input_fp, target_fp)


# %% [markdown]
# ## Checking 3 hold out drugs
# Looking for the most similar drugs in LINCS to our 3 hold out drug in Trapnell

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
loo_drugs = trapnell_df[
    trapnell_df.drug.isin(["Quisinostat", "Flavopiridol", "BMS-754807"])
]
loo_drugs

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
smiles_orig = []
smiles_lincs = []
for i, (drug, smiles, pathway) in loo_drugs.iterrows():
    tanimoto_sim_col = f"tanimoto_sim_{drug}"
    lincs_df[tanimoto_sim_col] = lincs_df.smiles.apply(
        lambda lincs_smiles: tanimoto_score(lincs_smiles, smiles)
    )
    most_similar = lincs_df.sort_values(tanimoto_sim_col, ascending=False).head(1)
    smiles_orig.append(smiles)
    smiles_lincs.append(most_similar["smiles"].item())
    print(
        drug,
        any(lincs_df.smiles.isin([smiles])),
        most_similar[tanimoto_sim_col].item(),
        most_similar["drug"].item(),
    )
    print(
        lincs_df.sort_values(tanimoto_sim_col, ascending=False).head(5)[
            ["drug", tanimoto_sim_col]
        ]
    )

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
for orig, lincs in zip(smiles_orig, smiles_lincs):
    im = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(orig), Chem.MolFromSmiles(lincs)],
        subImgSize=(600, 400),
        legends=[orig, lincs],
    )
    plt.tight_layout()
    display(im)
