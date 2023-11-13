[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Gradient Equilibrum
Gradient Equilibrium is a numerical optimization technique used to find the point at which a function reaches its global middle. This is different from traditional gradient descent methods, which seek to minimize or maximize a function. Instead, Gradient Equilibrium tries to find the point where the function value is at its average or equilibrium.


# Install
`pip install gradient-equilibrum`

# Usage
```python

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

```

## **Why Gradient Equilibrium?**

In many real-world scenarios, it's not always about finding the minimum or maximum. Sometimes, we might be interested in finding a balance or an average. This is where Gradient Equilibrium comes into play. For example, in load balancing problems or in scenarios where resources need to be evenly distributed, finding an equilibrium point is more relevant than finding extremes.

## **Algorithmic Pseudocode**

```
Function GradientEquilibrium(Function f, float learning_rate, int max_iterations):

    Initialize x = random value within the domain of f
    Initialize previous_x = x + 1  // Just to ensure we enter the loop

    For i = 1 to max_iterations and |previous_x - x| > small_value:
        previous_x = x
        
        // Compute gradient of f at x
        gradient = derivative(f, x)
        
        // Update x using gradient descent
        x = x - learning_rate * gradient

    End For

    Return x

End Function

Function derivative(Function f, float x):
    delta_x = small_value
    Return (f(x + delta_x) - f(x)) / delta_x
End Function
```


**How does the Algorithm Work?**

The Gradient Equilibrium algorithm starts by initializing a random value within the domain of the function. This value serves as our starting point. 

During each iteration, we calculate the gradient or derivative of the function at the current point. The gradient gives us the direction of steepest ascent. Since we are looking for the equilibrium, we move against the gradient by a factor of the learning rate. This step is similar to the gradient descent method but with a different goal in mind.

The algorithm stops iterating when the change between the current value and the previous value is less than a small threshold or when the maximum number of iterations is reached.

**Applications of Gradient Equilibrium**

1. **Load Balancing**: In distributed systems, ensuring that each server or node handles an approximately equal share of requests is crucial. Gradient Equilibrium can be used to find the optimal distribution.

2. **Resource Allocation**: Whether it's distributing funds, manpower, or any other resource, Gradient Equilibrium can help find the point where each division or department gets an average share.

3. **Economic Models**: In economics, equilibrium points where supply meets demand are of great significance. Gradient Equilibrium can be applied to find such points in complex economic models.

**Conclusion**

Gradient Equilibrium offers a fresh perspective on optimization problems. Instead of always seeking extremes, sometimes the middle ground or average is more relevant. With its straightforward approach and wide range of applications, Gradient Equilibrium is an essential tool for modern-day problem solvers.


# License 
MIT
