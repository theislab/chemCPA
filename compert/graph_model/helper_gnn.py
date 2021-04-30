import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from dgl import function as fn
from dgl.nn.pytorch import Set2Set
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

from dgllife.model.gnn import AttentiveFPGNN, gat, gcn, weave
from dgllife.model.readout import (
    AttentiveFPReadout,
    weave_readout,
    weighted_sum_and_max,
)

GCN = gcn.GCN
GAT = gat.GAT
WeaveGNN = weave.WeaveGNN
WeaveGather = weave_readout.WeaveGather
WeightedSumAndMax = weighted_sum_and_max.WeightedSumAndMax


def reduce_func_sum(nodes):
    return {"neigh": torch.sum(nodes.mailbox["m"], dim=1)}


def reduce_func_mean(nodes):
    return {"neigh": torch.mean(nodes.mailbox["m"], dim=1)}


def reduce_func_max(nodes):
    return {"neigh": torch.max(nodes.mailbox["m"], dim=1)}


class NNConv(nn.Module):
    """
    Description
    -----------
    Graph Convolution layer introduced in `Neural Message Passing
    for Quantum Chemistry <https://arxiv.org/pdf/1704.01212.pdf>`__.
    .. math::
        h_{i}^{l+1} = h_{i}^{l} + \mathrm{aggregate}\left(\left\{
        f_\Theta (e_{ij}) \cdot h_j^{l}, j\in \mathcal{N}(i) \right\}\right)
    where :math:`e_{ij}` is the edge feature, :math:`f_\Theta` is a function
    with learnable parameters.
    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
        NNConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    edge_func : callable activation function/layer
        Maps each edge feature to a vector of shape
        ``(in_feats * out_feats)`` as weight to compute
        messages.
        Also is the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``mean`` or ``max``).
    residual : bool, optional
        If True, use residual connection. Default: ``False``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import NNConv
    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> lin = th.nn.Linear(5, 20)
    >>> def edge_func(efeat):
    ...     return lin(efeat)
    >>> efeat = th.ones(6+6, 5)
    >>> conv = NNConv(10, 2, edge_func, 'mean')
    >>> res = conv(g, feat, efeat)
    >>> res
    tensor([[-1.5243, -0.2719],
            [-1.5243, -0.2719],
            [-1.5243, -0.2719],
            [-1.5243, -0.2719],
            [-1.5243, -0.2719],
            [-1.5243, -0.2719]], grad_fn=<AddBackward0>)
    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_feat = th.tensor(np.random.rand(2, 10).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> conv = NNConv(10, 2, edge_func, 'mean')
    >>> efeat = th.ones(5, 5)
    >>> res = conv(g, (u_feat, v_feat), efeat)
    >>> res
    tensor([[-0.6568,  0.5042],
            [ 0.9089, -0.5352],
            [ 0.1261, -0.0155],
            [-0.6568,  0.5042]], grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        edge_func,
        aggregator_type="mean",
        residual=False,
        bias=True,
    ):
        super(NNConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.edge_func = edge_func
        if aggregator_type == "sum":
            # self.reducer = fn.sum
            self.reducer = "sum"
        elif aggregator_type == "mean":
            # self.reducer = fn.mean
            self.reducer = "mean"
        elif aggregator_type == "max":
            # self.reducer = fn.max
            self.reducer = "max"
        else:
            raise KeyError(
                "Aggregator type {} not recognized: ".format(aggregator_type)
            )
        self._aggre_type = aggregator_type
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = init.calculate_gain("relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, efeat):
        """Compute MPNN Graph Convolution layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        efeat : torch.Tensor
            The edge feature of shape :math:`(E, *)`, which should fit the input
            shape requirement of ``edge_func``. :math:`E` is the number of edges
            of the graph.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # (n, d_in, 1)
            graph.srcdata["h"] = feat_src.unsqueeze(-1)
            # (n, d_in, d_out)
            graph.edata["w"] = self.edge_func(efeat).view(
                -1, self._in_src_feats, self._out_feats
            )
            # (n, d_in, d_out)
            # graph.update_all(fn.u_mul_e('h', 'w', 'm'), self.reducer('m', 'neigh'))
            graph.update_all(
                fn.u_mul_e("h", "w", "m"),
                fn.reducer.SimpleReduceFunction(self.reducer, "m", "neigh"),
            )
            rst = graph.dstdata["neigh"].sum(dim=1)  # (n, d_out)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst


class Weave_GNN(nn.Module):

    """Weave for regression and classification on graphs.
    Weave is introduced in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    num_layers : int
        Number of GNN (Weave) layers to use. Default to 2.
    hidden_feats : int
        Size for the hidden node and edge representations.
        Default to 50.
    gnn_activation : callable
        Activation function to be used in GNN (Weave) layers.
        Default to ReLU.
    graph_feats : int
        Size for the hidden graph representations. Default to 50.
    gaussian_expand : bool
        Whether to expand each dimension of node features by
        gaussian histogram in computing graph representations.
        Default to True.
    gaussian_memberships : list of 2-tuples
        For each tuple, the first and second element separately
        specifies the mean and std for constructing a normal
        distribution. This argument comes into effect only when
        ``gaussian_expand==True``. By default, we set this to be
        ``[(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
        (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
        (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
        (1.080, 0.170), (1.645, 0.283)]``.
    readout_activation : callable
        Activation function to be used in computing graph
        representations out of node representations. Default to Tanh.

    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """

    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        num_layers=2,
        hidden_feats=50,
        activation=F.relu,
        graph_feats=128,
        gaussian_expand=True,
        gaussian_memberships=None,
        readout_activation=F.tanh,
        # n_tasks=1
    ):
        super(Weave_GNN, self).__init__()

        self.gnn = WeaveGNN(
            node_in_feats=node_in_feats,
            edge_in_feats=edge_in_feats,
            num_layers=num_layers,
            hidden_feats=hidden_feats,
            activation=activation,
        )
        self.node_to_graph = nn.Sequential(
            nn.Linear(hidden_feats, graph_feats),
            readout_activation,
            nn.BatchNorm1d(graph_feats),
        )
        self.readout = WeaveGather(
            node_in_feats=graph_feats,
            gaussian_expand=gaussian_expand,
            gaussian_memberships=gaussian_memberships,
            activation=readout_activation,
        )
        # self.predict = nn.Linear(graph_feats, n_tasks)

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges.
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats, node_only=True)
        node_feats = self.node_to_graph(node_feats)
        g_feats = self.readout(g, node_feats)

        return g_feats


# pylint: disable=W0221
class GATPredictor(nn.Module):
    """GAT-based model for regression and classification on graphs.
    GAT is introduced in `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__.
    This model is based on GAT and can be used for regression and classification on graphs.
    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.
    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.
    Parameters
    ----------
    node_in_feats : int
        Number of input node features
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the output size of an attention head in the i-th GAT layer.
        ``len(hidden_feats)`` equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
        for each GAT layer.
    feat_drops : list of float
        ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
        ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
        all GAT layers.
    attn_drops : list of float
        ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
        layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
        for all GAT layers.
    alphas : list of float
        Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
        gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
        number of GAT layers. By default, this will be 0.2 for all GAT layers.
    residuals : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
        ``len(residual)`` equals the number of GAT layers. By default, residual connection
        is performed for each GAT layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
        multi-head results for intermediate GAT layers and compute mean of multi-head results
        for the last GAT layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
        By default, ELU is applied for intermediate GAT layers and no activation is applied
        for the last GAT layer.

    -----------------------------------------------------------------------------------------
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """

    def __init__(
        self,
        node_in_feats,
        hidden_feats=None,
        num_layers=None,
        num_heads=None,
        feat_drops=None,
        attn_drops=None,
        alphas=None,
        residuals=None,
        agg_modes=None,
        activations=None,
        # classifier_hidden_feats=128,
        # classifier_dropout=0.,
        # n_tasks=1,
        # predictor_hidden_feats=128,
        # predictor_dropout=0.
    ):
        super(GATPredictor, self).__init__()

        #         if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
        #             print('classifier_hidden_feats is deprecated and will be removed in the future, '
        #                   'use predictor_hidden_feats instead')
        #             predictor_hidden_feats = classifier_hidden_feats

        #         if predictor_dropout == 0. and classifier_dropout != 0.:
        #             print('classifier_dropout is deprecated and will be removed in the future, '
        #                   'use predictor_dropout instead')
        #             predictor_dropout = classifier_dropout

        if not isinstance(hidden_feats, list):
            assert isinstance(num_layers, int)
            hidden_feats = [hidden_feats] * num_layers
        if not isinstance(attn_drops, list):
            assert isinstance(num_layers, int)
            attn_drops = [attn_drops] * num_layers
        if not isinstance(feat_drops, list):
            assert isinstance(num_layers, int)
            feat_drops = [feat_drops] * num_layers
        if not isinstance(num_heads, list):
            assert isinstance(num_layers, int)
            num_heads = [num_heads] * num_layers

        self.gnn = GAT(
            in_feats=node_in_feats,
            hidden_feats=hidden_feats,
            num_heads=num_heads,
            feat_drops=feat_drops,
            attn_drops=attn_drops,
            alphas=alphas,
            residuals=residuals,
            agg_modes=agg_modes,
            activations=activations,
        )

        if self.gnn.agg_modes[-1] == "flatten":
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        # self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,

    #                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats, edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization
        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return graph_feats
        # return self.predict(graph_feats)


# pylint: disable=W0221
class MPNNPredictor(nn.Module):
    """MPNN for regression and classification on graphs.
    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.

    n_tasks : int
    Number of tasks, which is also the output size. Default to 1.
    """

    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        graph_feats=1,
        hidden_feats=64,
        edge_hidden_feats=128,
        num_step_message_passing=6,
        num_step_set2set=6,
        num_layer_set2set=3,
    ):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(
            node_in_feats=node_in_feats,
            node_out_feats=hidden_feats,
            edge_in_feats=edge_in_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
        )
        self.readout = Set2Set(
            input_dim=hidden_feats,
            n_iters=num_step_set2set,
            n_layers=num_layer_set2set,
        )
        # Leon: Added this back
        self.predict = nn.Sequential(
            nn.Linear(2 * hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, graph_feats),
        )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return graph_feats
        # return self.predict(graph_feats)


class AttentiveFPPredictor(nn.Module):
    """AttentiveFP for regression and classification on graphs.
    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feats : int
        Size for the learned graph representations. Default to 200.
    dropout : float
        Probability for performing the dropout. Default to 0.

    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """

    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        num_layers=2,
        num_timesteps=2,
        graph_feats=200,
        # n_tasks=1,
        dropout=0.0,
    ):
        super(AttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_in_feats,
            edge_feat_size=edge_in_feats,
            num_layers=num_layers,
            graph_feat_size=graph_feats,
            dropout=dropout,
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feats, num_timesteps=num_timesteps, dropout=dropout
        )

    #         self.predict = nn.Sequential(
    #             nn.Dropout(dropout),
    #             nn.Linear(graph_feat_size, n_tasks)
    #         )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        get_node_weight : bool
            Whether to get the weights of atoms during readout. Default to False.
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats, get_node_weight=False)
        return g_feats
        # return self.predict(g_feats)


class GCNPredictor(nn.Module):
    """GCN-based model for regression and classification on graphs.
    GCN is introduced in `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__. This model is based on GCN and can be used
    for regression and classification on graphs.
    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.
    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.
    Parameters
    ----------
    node_in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers. By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If None, no activation will be applied. If not None, ``activation[i]`` gives the
        activation function to be used for the i-th GCN layer. ``len(activation)`` equals
        the number of GCN layers. By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """

    def __init__(
        self,
        node_in_feats,
        # gnn_norm=None,
        num_layers=None,
        hidden_feats=None,
        activation=None,
        residual=None,
        batchnorm=None,
        dropout=None,
        # classifier_hidden_feats=128,
        # classifier_dropout=0.,
        # n_tasks=1,
        # predictor_hidden_feats=128,
        # predictor_dropout=0.
    ):
        super(GCNPredictor, self).__init__()

        #         if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
        #             print('classifier_hidden_feats is deprecated and will be removed in the future, '
        #                   'use predictor_hidden_feats instead')
        #             predictor_hidden_feats = classifier_hidden_feats

        #         if predictor_dropout == 0. and classifier_dropout != 0.:
        #             print('classifier_dropout is deprecated and will be removed in the future, '
        #                   'use predictor_dropout instead')
        #             predictor_dropout = classifier_dropout
        if not isinstance(hidden_feats, list):
            assert isinstance(num_layers, int)
            hidden_feats = [hidden_feats] * num_layers
        if not isinstance(activation, list):
            assert isinstance(num_layers, int)
            activation = [activation] * num_layers
        if not isinstance(residual, list):
            assert isinstance(num_layers, int)
            residual = [residual] * num_layers
        if not isinstance(dropout, list):
            assert isinstance(num_layers, int)
            dropout = [dropout] * num_layers
        if not isinstance(batchnorm, list):
            assert isinstance(num_layers, int)
            batchnorm = [batchnorm] * num_layers

        self.gnn = GCN(
            in_feats=node_in_feats,
            hidden_feats=hidden_feats,
            # gnn_norm=gnn_norm,
            activation=activation,
            residual=residual,
            batchnorm=batchnorm,
            dropout=dropout,
        )
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)

    #         self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
    #                                     n_tasks, predictor_dropout)

    def forward(self, bg, feats, edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization
        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return graph_feats
        # return self.predict(graph_feats)


# --------------------------------------------------------------------------------------------------------------------------
# pylint: disable=W0221
class MPNNGNN(nn.Module):
    """MPNN.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    This class performs message passing in MPNN and returns the updated node representations.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_in_feats : int
        Size for the input edge features. Default to 128.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    """

    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        node_out_feats=64,
        edge_hidden_feats=128,
        num_step_message_passing=6,
    ):
        super(MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats), nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type="sum",
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        """
        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats
