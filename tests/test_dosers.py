import pytest
import torch.nn
import torch.nn.functional as F

from chemCPA.model import ComPert, GeneralizedSigmoid


@pytest.mark.parametrize("nonlin", ["sigm", "logsigm", None])
def test_sigm_ohe_idx(nonlin):
    # test to make sure the generalized sigmoid doser has the same outputs
    # with indices and OHE
    sigm = GeneralizedSigmoid(10, "cpu", nonlin=nonlin)

    beta = torch.nn.Parameter(
        torch.tensor(
            [[x / 10 for x in range(0, 10)]], dtype=torch.float32, device="cpu"
        )
    )
    assert sigm.beta.shape == beta.shape
    sigm.beta = beta
    bias = torch.nn.Parameter(
        torch.tensor([[x / 5 for x in range(-5, 5)]], dtype=torch.float32, device="cpu")
    )
    assert sigm.bias.shape == bias.shape
    sigm.bias = bias

    dosages = torch.tensor([1.0, 1.0, 1.0, 1.0], device="cpu", dtype=torch.float32)
    x = torch.tensor([0, 2, 9, 2], dtype=torch.long)
    ohe = F.one_hot(x, num_classes=10)
    ohe_scaled = torch.einsum("a,ab->ab", [dosages, ohe])
    ohe_s = sigm(ohe_scaled)
    idx_s = sigm(dosages, idx=x)
    assert ohe_s[0][0] == idx_s[0]
    assert ohe_s[1][2] == idx_s[1]
    assert ohe_s[2][9] == idx_s[2]
    assert ohe_s[3][2] == idx_s[3]


@pytest.mark.parametrize("doser_type", ["logsigm", "sigm", "mlp", None])
def test_drug_embedding(doser_type):
    drug_emb = torch.nn.Embedding.from_pretrained(
        torch.tensor(list(range(10 * 10)), dtype=torch.float32, device="cpu").view(
            10, 10
        )
    )
    model = ComPert(
        num_genes=50,
        num_drugs=10,
        num_covariates=[1],
        doser_type=doser_type,
        device="cpu",
        drug_embeddings=drug_emb,
        use_drugs_idx=False,
    )
    idx = torch.tensor([0, 2, 9, 2], dtype=torch.long, device="cpu")
    dosages = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device="cpu")
    ohe = F.one_hot(idx, num_classes=10).to(dtype=torch.float32, device="cpu")
    ohe_scaled = torch.einsum("a,ab->ab", [dosages, ohe])

    emb_ohe = model.compute_drug_embeddings_(drugs=ohe_scaled)
    model.use_drugs_idx = True
    emb_idx = model.compute_drug_embeddings_(drugs_idx=idx, dosages=dosages)

    torch.testing.assert_close(emb_ohe, emb_idx)
