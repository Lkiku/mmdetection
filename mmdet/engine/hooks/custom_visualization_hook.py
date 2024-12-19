import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from .visualization_hook import DetVisualizationHook


@HOOKS.register_module()
class CustomDetVisualizationHook(DetVisualizationHook):
    """Custom Detection Visualization Hook that shows augmented images during training."""

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: dict) -> None:
        """Run after every ``self.interval`` training iterations.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict): Data from dataloader.
            outputs (dict): Outputs from model.
        """
        if self.draw is False:
            return

        # Only show augmented images at intervals
        if (runner.iter + 1) % self.interval != 0:
            return

        # Get augmented image from data_batch
        aug_img = data_batch['inputs'][0].permute(1, 2, 0).cpu().numpy()
        # Convert to uint8 and ensure in [0, 255]
        aug_img = (aug_img * 255).astype(np.uint8)
        aug_img = mmcv.bgr2rgb(aug_img)

        # Add augmented image
        self._visualizer.add_image(
            'augmented_img',
            aug_img,
            step=runner.iter + 1)
