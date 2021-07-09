import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from dgl import DGLGraph
from compert.graph_model.helper_gnn import (
    AttentiveFPPredictor,
    GATPredictor,
    GCNPredictor,
    MPNNPredictor,
    Weave_GNN,
)


use_cuda = lambda model: model.cuda() if torch.cuda.is_available() else model


class Drugemb(nn.Module):
    def __init__(
        self,
        dim: int,
        gnn_model: str,
        graph_feats_shape: tuple,
        idx_wo_smiles: list,
        batched_graph_collection: DGLGraph,
        hparams: Union[dict, None] = None,
        device: str = "cpu",
    ):

        super().__init__()
        self.device = device
        self.dim = dim

        gnn_names = ["AttentiveFP", "GAT", "GCN", "MPNN", "weave"]
        gnn_models = [
            AttentiveFPPredictor,
            GATPredictor,
            GCNPredictor,
            MPNNPredictor,
            Weave_GNN,
        ]
        available_gnns = dict(zip(gnn_names, gnn_models))

        if gnn_model not in list(available_gnns):
            raise ValueError(
                f"Got unkonown gnn_model: '{gnn_model}'. Choose from {list(available_gnns)}"
            )

        self.gnn_model = gnn_model
        self.graph_feats_shape = graph_feats_shape
        self.graph = batched_graph_collection
        self.idx_wo_smiles = idx_wo_smiles

        # Set GNN model
        model = available_gnns[gnn_model]
        # Set default hparams
        params = self.set_params(hparams, model)
        # Initialise model
        self.graph_embedding = use_cuda(model(**params))

    def forward(self):
        # drug embedding matrix of valid drugs
        graph = self.graph
        if graph.device.type != self.device:
            graph = graph.to(self.device)

        latent_drugs = self.graph_embedding(graph, graph.ndata["h"], graph.edata["h"])
        # zero tensor for control cells
        latent_control = torch.zeros(self.dim, device=self.device).view(1, -1)
        # insert zero tensors at positions with invalid graphs
        for idx in self.idx_wo_smiles:
            latent_drugs = torch.cat(
                [latent_drugs[:idx], latent_control, latent_drugs[idx:]], 0
            )
        return latent_drugs

    def set_params(self, hparams, model):
        if hparams is None:
            hparams = {}
        graph_feats_shape = self.graph_feats_shape
        dim = self.dim
        n_layers = 2 if "n_layers" not in list(hparams) else hparams["n_layers"]
        _hparams = {
            "node_in_feats": graph_feats_shape[0],
            "edge_in_feats": graph_feats_shape[1],
            "hidden_feats": int(dim / 2),
            "edge_hidden_feats": int(dim / 2),
            "graph_feats": dim,
            "num_layers": 2,
            "dropout": 0.0,
            "activation": F.relu,
            "activations": [F.relu] * n_layers,
            "readout_activation": nn.Tanh(),
            "residual": True,
            "residuals": [True] * n_layers,
            # AttentiveFP
            "num_timesteps": 6,
            # GAT
            "num_heads": 4,
            "feat_drops": 0.0,
            "attn_drops": 0.0,
            "alphas": None,
            "agg_modes": None,
            # GCN
            "batchnorm": None,
            # MPNN
            "num_step_message_passing": 6,
            "num_step_set2set": 6,
            "num_layer_set2set": 3,
            # WEAVE
            "gaussian_expand": True,
            "gaussian_memberships": None,
        }
        # Update all hparams by user specification
        for param, value in hparams.items():
            _hparams[param] = value
        # Subset hparams to relevant params for model
        _model_signature = inspect.signature(model.__init__)
        _model_params = _model_signature.parameters
        params = {}
        for param in set(_hparams).intersection(_model_params):
            params[param] = _hparams[param]
        print(f"\nArgs for Drugemb({self.gnn_model}): {params}\n")
        return params


if __name__ == "__main__":
    for model in ["AttentiveFP", "GAT", "GCN", "MPNN", "weave"]:
        Drugemb(
            dim=32,
            gnn_model=model,
            graph_feats_shape=[12, 12],
            idx_wo_smiles=1,
            batched_graph_collection=1,
        )
