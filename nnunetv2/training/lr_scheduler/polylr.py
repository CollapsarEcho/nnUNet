from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class CustomCosineAnnealingLR(CosineAnnealingLR):
    """
    Inherits from CosineAnnealingLR, but only starts annealing in the last 30% steps.
    """

    def __init__(self, optimizer, max_steps, initial_lr, eta_min=0, last_epoch=-1):
        self.flat_steps = int(0.7 * max_steps)
        self.initial_lr = initial_lr
        self.total_steps = max_steps
        self._step_count = 0
        super().__init__(
            optimizer, T_max=max_steps - self.flat_steps, eta_min=eta_min, last_epoch=-1
        )
        for group in optimizer.param_groups:
            group["lr"] = initial_lr

    def step(self, epoch=None):
        if epoch is None or epoch == -1:
            epoch = self._step_count
            self._step_count += 1

        if epoch < self.flat_steps:
            for group in self.optimizer.param_groups:
                group["lr"] = self.initial_lr
        else:
            # Only start annealing after flat_steps
            # Let the base scheduler manage its own state
            super().step(epoch - self.flat_steps)

    def get_lr(self):
        if self._step_count <= self.flat_steps:
            return [self.initial_lr for _ in self.optimizer.param_groups]
        else:
            return super().get_lr()


# def cosine_anneal(epoch, max_epochs, initial_lr, exponent=0.9):
#     if epoch < (0.7 * max_epochs):
#         return initial_lr
#     else:
#         return (
#             initial_lr
#             * (
#                 1
#                 + math.cos(
#                     math.pi * (epoch - ((0.7 * max_epochs) - 1)) / (0.3 * max_epochs)
#                 )
#             )
#             / 2
#         )
