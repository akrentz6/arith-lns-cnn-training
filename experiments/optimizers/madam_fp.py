import torch
from torch.optim import Optimizer

class Madam(Optimizer):

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta: float = 0.999,
        eps: float = 1e-6,
        grad_clamp: float = 10.0,
        weight_clamp: float = 3.0,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"Invalid beta: {beta} (expected in [0, 1))")
        if grad_clamp <= 0:
            raise ValueError(f"Invalid grad_clamp: {grad_clamp}")
        if weight_clamp is not None and weight_clamp <= 0:
            raise ValueError(f"Invalid weight_clamp: {weight_clamp}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(
            lr=lr,
            beta=beta,
            grad_clamp=grad_clamp,
            weight_clamp=weight_clamp,
            eps=eps,
            _group_initialized=False,  # internal
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                lr = group["lr"]
                beta = group["beta"]
                eps = group["eps"]
                grad_clamp = group["grad_clamp"]

                if len(state) == 0:
                    # First time we see this parameter
                    weight_clamp = group["weight_clamp"]
                    state["max"] = weight_clamp * torch.sqrt(torch.mean(p * p)).item()
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                state["exp_avg_sq"] = beta * state["exp_avg_sq"] + (1.0 - beta) * (grad ** 2)
                corr_exp_avg_sq = state["exp_avg_sq"] / (1.0 - beta ** state["step"]) + eps

                g_normed = grad / torch.sqrt(corr_exp_avg_sq)
                g_clipped = torch.clamp(g_normed, -grad_clamp, grad_clamp)

                p.data.mul_(torch.exp(-lr * g_clipped * torch.sign(p.data)))
                p.data.clamp_(-state["max"], state["max"])

        return loss