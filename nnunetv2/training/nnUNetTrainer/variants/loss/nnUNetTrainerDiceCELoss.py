import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
import numpy as np


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
            weight_ce=1,
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


class nnUNetTrainerDiceCELoss_500epochs(nnUNetTrainerDiceCELoss):
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
        self.num_epochs = 500
