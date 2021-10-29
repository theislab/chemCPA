import torch.nn
import torch.nn.functional as F

from compert.model import GeneralizedSigmoid


def test_sigm():
    sigm = GeneralizedSigmoid(10, "cpu")

    beta = torch.nn.Parameter(
        torch.tensor([[x / 10 for x in range(0, 10)]], dtype=torch.float32, device="cpu")
    )
    assert sigm.beta.shape == beta.shape
    sigm.beta = beta
    bias = torch.nn.Parameter(
        torch.tensor([[x / 5 for x in range(-5, 5)]], dtype=torch.float32, device="cpu")
    )
    assert sigm.bias.shape == bias.shape
    sigm.bias = bias

    x = torch.tensor([2, 9], dtype=torch.long)
    ohe = F.one_hot(x, num_classes=10)
    ohe_s = sigm(ohe)
    idx_s = sigm(torch.tensor([1.0, 1.0]), idx=x)
    assert ohe_s[0][2] == idx_s[0]
    assert ohe_s[1][9] == idx_s[1]
