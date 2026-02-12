"""
nnUNet trainer with NO data augmentation.

For binary occupancy grid input (0/1 values), standard augmentation
(rotation, noise, blur, brightness, contrast, gamma, mirror) is not
meaningful. This trainer skips all augmentation and keeps only the
essential transforms: RemoveLabelTransform + DownsampleSegForDS.

Usage:
    nnUNetv2_train 501 3d_fullres 0 --npz -tr nnUNetTrainerNoDA
"""
from typing import Union, List, Tuple

import numpy as np
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerNoDA(nnUNetTrainer):
    """nnUNet trainer with all data augmentation disabled.

    Only essential transforms are kept:
    - RemoveLabelTransform: converts ignore label (-1) to 0
    - DownsampleSegForDS: prepares deep supervision targets

    Also sets initial_patch_size == patch_size (no rotation margin needed).
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # No rotation, no dummy 2D DA, no mirroring
        # initial_patch_size == patch_size since no spatial augmentation
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA = (-np.pi, np.pi)  # unused but required by interface
        do_dummy_2d_data_aug = False
        initial_patch_size = patch_size  # no margin needed
        mirror_axes = None  # disable mirroring
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        # No augmentation transforms. Only essential pipeline steps.
        transforms = []

        # Required: convert ignore label -1 to 0
        transforms.append(RemoveLabelTansform(-1, 0))

        # Required: deep supervision target downsampling
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
