from pfhedge.nn import Hedger
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer

# error: Skipping analyzing "tqdm": found module but no type hints or library stubs
from tqdm import tqdm  # type: ignore

from pfhedge._utils.hook import save_prev_output
from pfhedge._utils.lazy import has_lazy
from pfhedge._utils.operations import ensemble_mean
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.features import FeatureList
from pfhedge.features._base import Feature
from pfhedge.instruments.base import BaseInstrument
from pfhedge.instruments.derivative.base import BaseDerivative
from pfhedge.nn.functional import pl

class CostHedger(Hedger):
    def __init__(self, *args, **kwargs):
        # if 'cost' in kwargs:
        #             self.cost_function = kwargs['cost']
        #             del kwargs['cost']
        # else:
        #     self.cost_function = None

        super().__init__(*args, **kwargs)


    def calculate_transaction_cost(self, delta_hedge):
        """Calculates transaction cost proportional to notional^(3/4)."""
        notional_change = delta_hedge * self.spot  # Assuming self.spot exists
        cost = torch.abs(notional_change) ** (3/4)  # No cost_factor here
        return cost

    def compute_loss(
        self,
        derivative: BaseDerivative,
        hedge: Optional[List[BaseInstrument]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
        enable_grad: bool = True,
    ) -> Tensor:
        # ... existing compute_loss logic ...
        with torch.set_grad_enabled(enable_grad):

            def _get_loss() -> Tensor:
                derivative.simulate(n_paths=n_paths, init_state=init_state)
                portfolio = self.compute_portfolio(derivative, hedge=hedge)
                return self.criterion(portfolio, derivative.payoff())

            mean_loss = ensemble_mean(_get_loss, n_times=n_times)

        

        # Add transaction cost to the loss
        transaction_cost = self.calculate_transaction_cost(
            self.hedge(self.inputs, self.moneyness) - self.prev_hedge
        )

        # ... rest of compute_loss logic ...
        final = mean_loss + transaction_cost.mean()

        return final
    