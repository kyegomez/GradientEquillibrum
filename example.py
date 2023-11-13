import torch
import torch.nn as nn
from ge.main import GradientEquilibrum  # Import your optimizer class

# Define a sample model
class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Create a sample model and data
model = SampleModel()
data = torch.randn(64, 10)
target = torch.randn(64, 10)
loss_fn = nn.MSELoss()

# Initialize your GradientEquilibrum optimizer
optimizer = GradientEquilibrum(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(data)

    # Calculate the loss
    loss = loss_fn(output, target)

    # Backward pass
    loss.backward()

    # Update the model's parameters using the optimizer
    optimizer.step()

    # Print the loss for monitoring
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

# After training, you can use the trained model for inference
