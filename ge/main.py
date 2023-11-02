import torch
from torch.autograd import grad


class GradientEquilibrium:
    def __init__(
        self,
        params,
        domain=(-10, 10),
        learning_rate=0.01,
        max_iterations=1000,
        tol=1e-7,
    ):
        """
        Initialize the GradientEquilibrium class.

        :param func: The function for which equilibrium point needs to be found
        :param domain: Tuple indicating the domain of the function
        :param learning_rate: Learning rate for gradient update
        :param max_iterations: Maximum number of iterations for the algorithm
        :param tol: Tolerance for convergence
        """
        self.params = params
        self.domain = domain
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol

    def _derivative(self, x):
        """
        Compute the derivative of the function using PyTorch's autograd.

        :param x: Input tensor
        :return: Gradient tensor
        """
        x = x.requires_grad_(True)
        y = self.func(x)
        (gradient,) = grad(y, x, create_graph=True)
        return gradient

    def find_equilibrium(self):
        """
        Find the equilibrium point for the function.

        :return: Equilibrium point tensor
        """
        x = torch.tensor(
            [torch.FloatTensor(1).uniform_(*self.domain)], requires_grad=True
        )

        for _ in range(self.max_iterations):
            prev_x = x.clone()
            gradient = self._derivative(x)
            with torch.no_grad():
                x -= self.learning_rate * gradient
                if torch.abs(prev_x - x) < self.tol:
                    break

        return x.item()

    def __repr__(self):
        return f"GradientEquilibrium(domain={self.domain}, learning_rate={self.learning_rate}, max_iterations={self.max_iterations}, tol={self.tol})"
