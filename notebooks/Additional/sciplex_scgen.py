# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scgen

from chemCPA.paths import CHECKPOINT_DIR, DATA_DIR

# sc.set_figure_params(dpi=300, frameon=False)
# sc.logging.print_header()
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2

# %%
dose = 1.0

if dose == 0.1:
    suffix = ""
elif dose == 1.0:
    suffix = "_high_dose"
adata = sc.read(DATA_DIR / f"adata_baseline{suffix}.h5ad")

# %%
split = "split_baseline_A549"
df_ood = adata.obs.loc[
    adata.obs[split] == "ood", ["cell_type", "condition"]
].drop_duplicates()

df_ood

# %%
splits = [c for c in adata.obs.columns if "baseline" in c]

splits

# %%
split_model_dict = dict(
    split_baseline_A549=CHECKPOINT_DIR / f"scgen_sciplex_A549{suffix}.pt",
    split_baseline_K562=CHECKPOINT_DIR / f"scgen_sciplex_K562{suffix}.pt",
    split_baseline_MCF7=CHECKPOINT_DIR / f"scgen_sciplex_MCF7{suffix}.pt",
)


# %% [markdown]
# ### Train scGen

# %%
def train_scgen(split, path=None):
    adata_train = adata[adata.obs[split] == "train"].copy()

    scgen.SCGEN.setup_anndata(
        adata_train, batch_key="condition", labels_key="cell_type"
    )

    model = scgen.SCGEN(adata_train)

    model.train(
        max_epochs=50,
        batch_size=128,
        early_stopping=True,
        early_stopping_patience=25,
        plan_kwargs=dict(n_epochs_kl_warmup=45),
    )
    if path:
        # fname = CHECKPOINT_DIR/f"scgen_sciplex_{split.split('_')[-1]}.pt"
        fname = path
        model.save(fname, overwrite=True)
    print(f"Model saved at \n\t f{fname}")
    del model


# %%
retrain = False

for split, model_path in split_model_dict.items():
    if not model_path.exists() or retrain:
        train_scgen(split, model_path)
    else:
        print(f"Model for {split} already exists at:\n\t {model_path}")

# %% [markdown]
# ### Compute predictions

# %%
import torch

from chemCPA.train import compute_r2


def compute_prediction(
    split, model, adata, use_DEGs=False, degs_key="lincs_DEGs", dose=0.1
):
    drug_r2 = {}
    ood_idx = adata.obs[split] == "ood"
    df_ood = adata.obs.loc[ood_idx, ["cell_type", "condition"]].drop_duplicates()
    for _, (ct, condition) in df_ood.iterrows():
        cell_drug_dose_comb = f"{ct}_{condition}_{dose}"
        ctrl_idx = (
            adata.obs[[split, "condition", "cell_type"]]
            .isin(["test", "control", ct])
            .prod(1)
            .astype(bool)
        )
        y_idx = (
            adata.obs[[split, "cell_type", "condition"]]
            .isin(["ood", ct, condition])
            .prod(1)
            .astype(bool)
        )
        y_true = adata[y_idx].X.A

        # adata_pred, _ = model.predict(
        #     ctrl_key='control',
        #     stim_key=condition,
        #     celltype_to_predict=ct,
        #     )
        adata_pred, _ = model.predict(
            ctrl_key="control",
            stim_key=condition,
            adata_to_predict=adata[ctrl_idx].copy(),
        )

        y_pred = adata_pred.X

        y_pred = torch.Tensor(y_pred).mean(0)
        y_true = torch.Tensor(y_true).mean(0)

        if use_DEGs:
            degs = adata.uns[degs_key][f"{ct}_{condition}_{dose}"]
            idx_de = adata.var_names.isin(degs)
            r2_m_de = compute_r2(y_true[idx_de].cuda(), y_pred[idx_de].cuda())
            drug_r2[cell_drug_dose_comb] = max(r2_m_de, 0.0)
        else:
            r2_m = compute_r2(y_true.cuda(), y_pred.cuda())
            drug_r2[cell_drug_dose_comb] = max(r2_m, 0.0)

    return drug_r2


# %%
scgen.SCGEN.setup_anndata(adata)
predictions = []
for split, model_path in split_model_dict.items():
    _adata = adata[adata.obs[split] == "train"].copy()
    model = scgen.SCGEN.load(model_path, _adata)
    for use_DEGs in [False, True]:
        preds = compute_prediction(
            split=split,
            model=model,
            adata=adata,
            use_DEGs=use_DEGs,
        )
        preds = pd.DataFrame.from_dict(preds, orient="index", columns=["R2"])

        preds["model"] = f"scGen_{split.split('_')[-1]}_{dose}"
        preds["genes"] = "degs" if use_DEGs else "all"
        predictions.append(preds)

# %%
predictions = pd.concat(predictions)
predictions.reset_index(inplace=True)
predictions["cell_type"] = predictions["index"].apply(lambda s: s.split("_")[0])
predictions["condition"] = predictions["index"].apply(lambda s: s.split("_")[1])
predictions["dose"] = f"{dose}"
predictions["model_ct"] = predictions["model"]
predictions["model"] = predictions["model"].apply(lambda s: s.split("_")[0])

# %%
predictions

# %%
predictions.groupby(["model", "genes"]).mean()

# %%
predictions.to_parquet(f"scgen_predictions{suffix}.parquet")

# %%
