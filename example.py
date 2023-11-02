import torch
from ge.main import GradientEquilibrium
from einops import rearrange

x = torch.tensor([1.0, 2.0, 3.0])


def sample_function(x):
    y = rearrange(x, "b -> b")
    return torch.einsum("i,i->", y, y)


ge = GradientEquilibrium(sample_function)
print(ge.find_equilibrium())
