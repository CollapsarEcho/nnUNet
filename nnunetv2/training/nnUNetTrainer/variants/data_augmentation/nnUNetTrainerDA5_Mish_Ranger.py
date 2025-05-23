from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    BlankRectangleTransform,
    GaussianBlurTransform,
    GaussianNoiseTransform,
    MedianFilterTransform,
    SharpeningTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    Rot90Transform,
    SpatialTransform,
    TransposeAxesTransform,
)
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    OneOfTransform,
    RemoveLabelTransform,
    RenameTransform,
)
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from ranger import Ranger
from torch import autocast
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, SequentialLR

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.training.data_augmentation.compute_initial_patch_size import (
    get_patch_size,
)
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import (
    ApplyRandomBinaryOperatorTransform,
    MoveSegAsOneHotToData,
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import (
    DownsampleSegForDSTransform2,
)
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import (
    ConvertSegmentationToRegionsTransform,
)
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context


class SequentialLRNoEpoch(SequentialLR):
    def step(self, epoch=None):
        # Ignore the epoch argument and call the parent method without it
        super().step()


class nnUNetTrainerDA5(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger(self.network.parameters(), lr=self.initial_lr)
        # Calculate when to start cosine annealing (after 70% of epochs)
        cosine_start_epoch = int(0.7 * self.num_epochs)

        # Create a scheduler that keeps lr constant for 70% of epochs, then applies cosine annealing
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 1.0
            if epoch < cosine_start_epoch
            else 0.5
            * (
                1
                + np.cos(
                    np.pi
                    * (epoch - cosine_start_epoch)
                    / (self.num_epochs - cosine_start_epoch)
                )
            ),
        )
        return optimizer, lr_scheduler

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi)
            else:
                rotation_for_DA = (
                    -180.0 / 360 * 2.0 * np.pi,
                    180.0 / 360 * 2.0 * np.pi,
                )
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (
                    -180.0 / 360 * 2.0 * np.pi,
                    180.0 / 360 * 2.0 * np.pi,
                )
            else:
                rotation_for_DA = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(
            patch_size[-dim:],
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.7, 1.43),
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f"do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        matching_axes = np.array(
            [sum([i == j for j in patch_size]) for i in patch_size]
        )
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms = []
        tr_transforms.append(RenameTransform("target", "seg", True))

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=rotation_for_DA,
                angle_y=rotation_for_DA,
                angle_z=rotation_for_DA,
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=(0.7, 1.43),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.4,
                independent_scale_for_each_axis=True,
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3),
                    axes=valid_axes,
                    data_key="data",
                    label_key="seg",
                    p_per_sample=0.5,
                ),
            )

        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(
                    valid_axes, data_key="data", label_key="seg", p_per_sample=0.5
                )
            )

        tr_transforms.append(
            OneOfTransform(
                [
                    MedianFilterTransform(
                        (2, 8),
                        same_for_each_channel=False,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                    GaussianBlurTransform(
                        (0.3, 1.5),
                        different_sigma_per_channel=True,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                ]
            )
        )

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(
            BrightnessTransform(
                0, 0.5, per_channel=True, p_per_sample=0.1, p_per_channel=0.5
            )
        )

        tr_transforms.append(
            OneOfTransform(
                [
                    ContrastAugmentationTransform(
                        contrast_range=(0.5, 2),
                        preserve_range=True,
                        per_channel=True,
                        data_key="data",
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                    ContrastAugmentationTransform(
                        contrast_range=(0.5, 2),
                        preserve_range=False,
                        per_channel=True,
                        data_key="data",
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                ]
            )
        )

        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.25, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.15,
                ignore_axes=ignore_axes,
            )
        )

        tr_transforms.append(
            GammaTransform(
                (0.7, 1.5),
                invert_image=True,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.1,
            )
        )
        tr_transforms.append(
            GammaTransform(
                (0.7, 1.5),
                invert_image=True,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.1,
            )
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform(
                [[max(1, p // 10), p // 3] for p in patch_size],
                rectangle_value=np.mean,
                num_rectangles=(1, 5),
                force_square=False,
                p_per_sample=0.4,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                max_strength=_brightness_gradient_additive_max_strength,
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                _local_gamma_gamma,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(
                MaskTransform(
                    [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    mask_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            if ignore_label is not None:
                raise NotImplementedError("ignore label not yet supported in cascade")
            assert foreground_labels is not None, (
                "We need all_labels for cascade augmentations"
            )
            use_labels = [i for i in foreground_labels if i != 0]
            tr_transforms.append(MoveSegAsOneHotToData(1, use_labels, "seg", "data"))
            tr_transforms.append(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    p_per_sample=0.4,
                    key="data",
                    strel_size=(1, 8),
                    p_per_label=1,
                )
            )
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                )
            )

        tr_transforms.append(RenameTransform("seg", "target", True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(
                ConvertSegmentationToRegionsTransform(
                    list(regions) + [ignore_label]
                    if ignore_label is not None
                    else regions,
                    "target",
                    "target",
                )
            )

        if deep_supervision_scales is not None:
            tr_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales, 0, input_key="target", output_key="target"
                )
            )
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RenameTransform("target", "seg", True))
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(
                MoveSegAsOneHotToData(1, foreground_labels, "seg", "data")
            )

        val_transforms.append(RenameTransform("seg", "target", True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(
                ConvertSegmentationToRegionsTransform(
                    list(regions) + [ignore_label]
                    if ignore_label is not None
                    else regions,
                    "target",
                    "target",
                )
            )

        if deep_supervision_scales is not None:
            val_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales, 0, input_key="target", output_key="target"
                )
            )

        val_transforms.append(NumpyToTensor(["data", "target"], "float"))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None,
            ignore_label=self.label_manager.ignore_label,
        )

        # validation pipeline
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # we set transforms=None because this trainer still uses batchgenerators which expects transforms to be passed to
        dl_tr = nnUNetDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=None,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=None,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            del data
            lo = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = target != self.label_manager.ignore_label
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = (1 - target[:, -1:]).bool()
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1].bool()
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": lo.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }


class nnUNetTrainerDA5ord0(nnUNetTrainerDA5):
    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        matching_axes = np.array(
            [sum([i == j for j in patch_size]) for i in patch_size]
        )
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms = []
        tr_transforms.append(RenameTransform("target", "seg", True))

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=rotation_for_DA,
                angle_y=rotation_for_DA,
                angle_z=rotation_for_DA,
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=(0.7, 1.43),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=0,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=0,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.4,
                independent_scale_for_each_axis=True,
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3),
                    axes=valid_axes,
                    data_key="data",
                    label_key="seg",
                    p_per_sample=0.5,
                ),
            )

        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(
                    valid_axes, data_key="data", label_key="seg", p_per_sample=0.5
                )
            )

        tr_transforms.append(
            OneOfTransform(
                [
                    MedianFilterTransform(
                        (2, 8),
                        same_for_each_channel=False,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                    GaussianBlurTransform(
                        (0.3, 1.5),
                        different_sigma_per_channel=True,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                ]
            )
        )

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(
            BrightnessTransform(
                0, 0.5, per_channel=True, p_per_sample=0.1, p_per_channel=0.5
            )
        )

        tr_transforms.append(
            OneOfTransform(
                [
                    ContrastAugmentationTransform(
                        contrast_range=(0.5, 2),
                        preserve_range=True,
                        per_channel=True,
                        data_key="data",
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                    ContrastAugmentationTransform(
                        contrast_range=(0.5, 2),
                        preserve_range=False,
                        per_channel=True,
                        data_key="data",
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                ]
            )
        )

        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.25, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.15,
                ignore_axes=ignore_axes,
            )
        )

        tr_transforms.append(
            GammaTransform(
                (0.7, 1.5),
                invert_image=True,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.1,
            )
        )
        tr_transforms.append(
            GammaTransform(
                (0.7, 1.5),
                invert_image=True,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.1,
            )
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform(
                [[max(1, p // 10), p // 3] for p in patch_size],
                rectangle_value=np.mean,
                num_rectangles=(1, 5),
                force_square=False,
                p_per_sample=0.4,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                max_strength=_brightness_gradient_additive_max_strength,
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                _local_gamma_gamma,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(
                MaskTransform(
                    [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    mask_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            if ignore_label is not None:
                raise NotImplementedError("ignore label not yet supported in cascade")
            assert foreground_labels is not None, (
                "We need all_labels for cascade augmentations"
            )
            use_labels = [i for i in foreground_labels if i != 0]
            tr_transforms.append(MoveSegAsOneHotToData(1, use_labels, "seg", "data"))
            tr_transforms.append(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    p_per_sample=0.4,
                    key="data",
                    strel_size=(1, 8),
                    p_per_label=1,
                )
            )
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                )
            )

        tr_transforms.append(RenameTransform("seg", "target", True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(
                ConvertSegmentationToRegionsTransform(
                    list(regions) + [ignore_label]
                    if ignore_label is not None
                    else regions,
                    "target",
                    "target",
                )
            )

        if deep_supervision_scales is not None:
            tr_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales, 0, input_key="target", output_key="target"
                )
            )
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms


def _brightnessadditive_localgamma_transform_scale(x, y):
    return np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y])))


def _brightness_gradient_additive_max_strength(_x, _y):
    return (
        np.random.uniform(-5, -1)
        if np.random.uniform() < 0.5
        else np.random.uniform(1, 5)
    )


def _local_gamma_gamma():
    return (
        np.random.uniform(0.01, 0.8)
        if np.random.uniform() < 0.5
        else np.random.uniform(1.5, 4)
    )


class nnUNetTrainerDA5Segord0(nnUNetTrainerDA5):
    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        matching_axes = np.array(
            [sum([i == j for j in patch_size]) for i in patch_size]
        )
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms = []
        tr_transforms.append(RenameTransform("target", "seg", True))

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=rotation_for_DA,
                angle_y=rotation_for_DA,
                angle_z=rotation_for_DA,
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=(0.7, 1.43),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=0,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.4,
                independent_scale_for_each_axis=True,
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3),
                    axes=valid_axes,
                    data_key="data",
                    label_key="seg",
                    p_per_sample=0.5,
                ),
            )

        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(
                    valid_axes, data_key="data", label_key="seg", p_per_sample=0.5
                )
            )

        tr_transforms.append(
            OneOfTransform(
                [
                    MedianFilterTransform(
                        (2, 8),
                        same_for_each_channel=False,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                    GaussianBlurTransform(
                        (0.3, 1.5),
                        different_sigma_per_channel=True,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                ]
            )
        )

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(
            BrightnessTransform(
                0, 0.5, per_channel=True, p_per_sample=0.1, p_per_channel=0.5
            )
        )

        tr_transforms.append(
            OneOfTransform(
                [
                    ContrastAugmentationTransform(
                        contrast_range=(0.5, 2),
                        preserve_range=True,
                        per_channel=True,
                        data_key="data",
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                    ContrastAugmentationTransform(
                        contrast_range=(0.5, 2),
                        preserve_range=False,
                        per_channel=True,
                        data_key="data",
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                ]
            )
        )

        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.25, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.15,
                ignore_axes=ignore_axes,
            )
        )

        tr_transforms.append(
            GammaTransform(
                (0.7, 1.5),
                invert_image=True,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.1,
            )
        )
        tr_transforms.append(
            GammaTransform(
                (0.7, 1.5),
                invert_image=True,
                per_channel=True,
                retain_stats=True,
                p_per_sample=0.1,
            )
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform(
                [[max(1, p // 10), p // 3] for p in patch_size],
                rectangle_value=np.mean,
                num_rectangles=(1, 5),
                force_square=False,
                p_per_sample=0.4,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                max_strength=_brightness_gradient_additive_max_strength,
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                _local_gamma_gamma,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5,
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(
                MaskTransform(
                    [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    mask_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            if ignore_label is not None:
                raise NotImplementedError("ignore label not yet supported in cascade")
            assert foreground_labels is not None, (
                "We need all_labels for cascade augmentations"
            )
            use_labels = [i for i in foreground_labels if i != 0]
            tr_transforms.append(MoveSegAsOneHotToData(1, use_labels, "seg", "data"))
            tr_transforms.append(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    p_per_sample=0.4,
                    key="data",
                    strel_size=(1, 8),
                    p_per_label=1,
                )
            )
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                )
            )

        tr_transforms.append(RenameTransform("seg", "target", True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(
                ConvertSegmentationToRegionsTransform(
                    list(regions) + [ignore_label]
                    if ignore_label is not None
                    else regions,
                    "target",
                    "target",
                )
            )

        if deep_supervision_scales is not None:
            tr_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales, 0, input_key="target", output_key="target"
                )
            )
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms


class nnUNetTrainerDA5_10epochs(nnUNetTrainerDA5):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10


class nnUNetTrainerDA5_Ranger_CosAnneal(nnUNetTrainerDA5):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger(
            self.network.parameters(),
            lr=self.initial_lr,
            k=6,
            N_sma_threshhold=5,
            weight_decay=self.weight_decay,
        )

        # Total number of training epochs
        total_epochs = self.num_epochs

        # Define the flat (constant) phase as 70% of total epochs
        flat_epochs = int(total_epochs * 0.7)
        # The cosine annealing phase covers the remaining 30% of epochs
        cosine_epochs = total_epochs - flat_epochs

        # Scheduler for constant learning rate during the flat phase
        scheduler_flat = ConstantLR(optimizer, factor=1.0, total_iters=flat_epochs)

        # Cosine annealing scheduler: decays from base_lr (0.001) to eta_min (here set to 0)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=0)

        # Chain the two schedulers: use flat phase first, then cosine annealing starting at flat_epochs
        lr_scheduler = SequentialLRNoEpoch(
            optimizer,
            schedulers=[scheduler_flat, scheduler_cosine],
            milestones=[flat_epochs],
        )

        # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler


class nnUNetTrainerDA5_Ranger_CosAnneal_CEDice_800Epochs(
    nnUNetTrainerDA5_Ranger_CosAnneal
):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 800

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
