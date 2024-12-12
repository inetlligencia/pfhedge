import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch


import torch
from pfhedge.instruments import BrownianStock, EuropeanOption
from pfhedge.nn import Hedger


import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Module
from pfhedge.nn import BlackScholes, Clamp, MultiLayerPerceptron



def to_numpy(tensor: torch.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()


class NoTransactionBandNet(Module):
    def __init__(self, derivative):
        super().__init__()

        self.delta = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()

    def inputs(self):
        return self.delta.inputs() + ["prev_hedge"]

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.delta(input[..., :-1])
        width = self.mlp(input[..., :-1])

        min = delta - fn.leaky_relu(width[..., [0]])
        max = delta + fn.leaky_relu(width[..., [1]])

        return self.clamp(prev_hedge, min=min, max=max)


# Define the custom transaction cost function
class CustomTransactionCost:
    def __init__(self, cost_rate=0.01):
        """
        Initializes the transaction cost model.
        Args:
            cost_rate: Proportionality constant for transaction cost.
        """
        self.cost_rate = cost_rate

    def __call__(self, delta_change):
        """
        Computes transaction cost proportional to the 3/4 power of the delta change.
        Args:
            delta_change: The volume of stock traded (torch.Tensor).
        Returns:
            Transaction cost (torch.Tensor).
        """
        return self.cost_rate * torch.abs(delta_change) ** (3 / 4)
    

from pfhedge.nn import QuadraticCVaR
# import torch


class QuadraticCVaRWithCosts(QuadraticCVaR):
    """
    FROM GPT
    https://chatgpt.com/g/g-96E3TjGYg-pf-code-asst/c/674d2b52-698c-800a-ba8a-065aad39ff92
    copied link
    https://chatgpt.com/share/674db1b8-a0d4-800a-b1ae-be14fd3188b4
    

    A custom criterion that extends QuadraticCVaR to include transaction costs
    proportional to the 3/4 power of the traded volume.
    """
    def __init__(self, alpha=0.05, cost_rate=0.01):
        """
        Args:
            alpha: The significance level for CVaR.
            cost_rate: The proportionality constant for transaction costs.
        """
        super().__init__(alpha=alpha)
        self.cost_rate = cost_rate

    def compute_transaction_costs(self, delta_changes):
        """
        Computes transaction costs based on the 3/4 power of the traded volume.

        Args:
            delta_changes: Tensor of changes in hedge positions.

        Returns:
            A tensor of transaction costs.
        """
        return self.cost_rate * torch.abs(delta_changes) ** (3 / 4)

    def forward(self, hedge_pnl, costs, delta_changes):
        """
        Overrides the forward method to include transaction costs.

        Args:
            hedge_pnl: The profit and loss from hedging.
            costs: Additional costs not captured by delta changes (e.g., fees).
            delta_changes: Changes in hedge positions.

        Returns:
            The CVaR loss including transaction costs.
        """
        # Compute transaction costs
        transaction_costs = self.compute_transaction_costs(delta_changes)

        # Total costs include the computed transaction costs and any other costs
        total_costs = costs + transaction_costs

        # Adjust hedge PnL by subtracting total costs
        adjusted_hedge_pnl = hedge_pnl - total_costs

        # Compute QuadraticCVaR loss on adjusted PnL
        return super().forward(adjusted_hedge_pnl)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Default device:", DEVICE)


if __name__ == "__main__":
    # In each epoch, N_PATHS brownian motion time-series are generated.
    N_PATHS = 50_000
    # How many times a model is updated in the experiment.
    N_EPOCHS = 200
    # Define the underlying asset (Brownian motion stock process)
    stock = BrownianStock(cost=5e-4)
    # Define the derivative (European option) based on the stock
    option = EuropeanOption(stock, strike=100, maturity=30/252)
    # Define the transaction cost model
    transaction_cost_model = CustomTransactionCost(cost_rate=0.01)
    model = NoTransactionBandNet(derivative=option)
    # Initialize the hedger without a specific model for now
    hedger = Hedger(model=model, inputs=["log_moneyness", "time_to_maturity"])
    history = hedger.fit(derivative=option, n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)
    