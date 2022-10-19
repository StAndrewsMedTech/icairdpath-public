from abc import ABCMeta, abstractmethod
from math import ceil
from repath.utils.geometry import Size
from typing import Dict, Tuple
from random import randint

from pandas.core.arrays import boolean

import numpy as np
import pandas as pd

from repath.utils.convert import to_frame_with_locations
from repath.utils.filters import pool2d
from repath.preprocess.patching.apply_transform import ApplyTransforms, SingleTransform

class PatchFinder(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, labels_image: np.array, slide_shape: Size) -> Tuple[pd.DataFrame, int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def labels_level(self):
        raise NotImplementedError


class GridPatchFinder(PatchFinder):
    def __init__(
        self,
        labels_level: int,
        patch_level: int,
        patch_size: int,
        stride: int,  # defined in terms of the labels image space
        border: int = 0,
        jitter: int = 0,
        remove_background: bool = True,
        apply_transforms: ApplyTransforms = SingleTransform(),
        pool_mode: str = "max"
    ) -> None:
        """ Note that the assumption is that the same settings will be used for a number of different patches.

        Args:
            labels (Dict[str, int]): Dict mapping string labels to indices in the labels image.
            labels_level (int): The magnification level of the labels image.
            patch_level (int): The magnification level at which to extract the pixels data for the patches.
            patch_size (int): The width and height of the patches in pixels at patches_level magnification.
            stride (int): The horizontal and vertical distance between each patch (the stide of the window).
            border (int, optional): [description]. Defaults to 0.
            jitter (int, optional): [description]. Defaults to None.
        """

        # assign values
        self.labels_level = labels_level
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.stride = stride
        self.border = border
        self.jitter = jitter
        self.remove_background = remove_background
        self.apply_transforms = apply_transforms
        self.pool_mode = pool_mode
        # some assumptions
        # 1. patch_size is some integer multiple of a pixel at labels_level
        # 2. patch_level is equal to or below labels_level
        # 3. stride is some integer multiple of a pixel at labels_level

    def __call__(self, labels_image: np.array, slide_shape: Size) -> Tuple[pd.DataFrame, int, int]:
        """Patch finders can be called with an array of rendered annotations and produce a patch index.

        Args:
            labels_image (np.array): An array containing a label index for each pixel at some magnification level.

        Returns:
            PatchIndex: A patch index containing data about how to retieve and label the patches for the slide.
        """
        scale_factor = 2 ** (self.labels_level - self.patch_level)
        kernel_size = int(self.patch_size / scale_factor)
        label_level_stride = int(self.stride / scale_factor)

        # TODO - Needs to select no the max label but the one with the most area? - needs thinking about this!
        # The pooling operation might be a parameter for the patch finder.
        patch_labels = pool2d(labels_image, kernel_size, label_level_stride, 0, pool_mode=self.pool_mode)

        # convert the 2d array of patch labels to a data frame
        df = to_frame_with_locations(patch_labels, "label")
        df.row *= self.stride
        df.column *= self.stride
        df = df.rename(columns={"row": "y", "column": "x"})
        df = df.reindex(columns=["x", "y", "label"])

        # for each patch, specify a transform
        df = self.apply_transforms(df)

        # calculate amount to subtract from top left for border and jitter
        subtract_top_left = ceil(self.border / 2) + self.jitter

        # for each row, add the border
        df['x'] = np.subtract(df['x'], subtract_top_left)
        df['y'] = np.subtract(df['y'], subtract_top_left)
        output_patch_size = self.patch_size + (self.border + self.jitter)

        # remove the background
        if self.remove_background:
            df = df[df.label != 0]  # TODO: put this in as a method that is optional on the slide patch index (or something)

        # clip the patch coordinates to the slide dimensions
        df['x'] = np.maximum(df['x'], 0)
        df['y'] = np.maximum(df['y'], 0)
        df['x'] = np.minimum(df['x'], slide_shape.width - output_patch_size)
        df['y'] = np.minimum(df['y'], slide_shape.height - output_patch_size)

        # return the index and the data required to extract the patches later
        return df, self.patch_level, output_patch_size

    def labels_level(self):
        raise self.labels_level


class RandomPatchFinder:
    pass
