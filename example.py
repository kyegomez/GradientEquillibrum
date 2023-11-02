import torch
from ge.main import GradientEquilibrum

# Example usage
model = torch.nn.Linear(10, 10)
optimizer = GradientEquilibrum(
    model.parameters(),
    lr=0.01,
)
print(optimizer)
out = optimizer.zero_grad()
print(out)
