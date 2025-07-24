import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
import numpy as np
from nnunetv2.training.lr_scheduler.polylr import CustomCosineAnnealingLR


class nnUNetTrainerDiceCELoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, (
            "regions not supported by this trainer"
        )

        soft_dice_kwargs = {
            "batch_dice": self.configuration_manager.batch_dice,
            "do_bg": self.label_manager.has_regions,
            "smooth": 1e-5,
            "ddp": self.is_ddp,
        }

        ce_kwargs = {
            "weight": None,
            "ignore_index": self.label_manager.ignore_label
            if self.label_manager.has_ignore_label
            else -100,
        }

        loss = DC_and_CE_loss(
            soft_dice_kwargs,
            ce_kwargs,
            weight_ce=3,
            weight_dice=1,
            ignore_label=None,
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))]
            )
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainer_Mish_Ranger(nnUNetTrainerDiceCELoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 800

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = CustomCosineAnnealingLR(
            optimizer, initial_lr=self.initial_lr, T_max=self.num_epochs
        )
        return optimizer, lr_scheduler
    

        # def configure_optimizers(self):
        #     optimizer = Ranger(
        #         self.network.parameters(),
        #         lr=self.initial_lr,
        #         k=6,
        #         N_sma_threshhold=5,
        #         weight_decay=self.weight_decay,
        #     )

        #     # Total number of training epochs
        #     total_epochs = self.num_epochs

        #     # Define the flat (constant) phase as 70% of total epochs
        #     flat_epochs = int(total_epochs * 0.7)
        #     # The cosine annealing phase covers the remaining 30% of epochs
        #     cosine_epochs = total_epochs - flat_epochs

        #     # Scheduler for constant learning rate during the flat phase
        #     scheduler_flat = ConstantLR(optimizer, factor=1.0, total_iters=flat_epochs)

        #     # Cosine annealing scheduler: decays from base_lr (0.001) to eta_min (here set to 0)
        #     scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=0)

        #     # Chain the two schedulers: use flat phase first, then cosine annealing starting at flat_epochs
        #     lr_scheduler = SequentialLRNoEpoch(
        #         optimizer,
        #         schedulers=[scheduler_flat, scheduler_cosine],
        #         milestones=[flat_epochs],
        #     )

        #     # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        #     return optimizer, lr_scheduler
