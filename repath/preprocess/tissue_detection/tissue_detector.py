from abc import ABCMeta, abstractmethod
import math

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, block_reduce
from skimage.morphology import binary_closing


class PreFilter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError

class NullBlur(PreFilter):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image

class MedianBlur(PreFilter):
    def __init__(self, filter_size: int) -> None:
        # assign values
        self.filter_size = filter_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image_out = median_filter(image, size=self.filter_size)
        return image_out

class GaussianBlur(PreFilter):
    def __init__(self, sigma: int) -> None:
        # assign values
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image_out = gaussian_filter(image, sigma=self.sigma) 
        return image_out      


class MorphologyTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError

class NullTransform(MorphologyTransform):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Does not apply a transform """
        return image

class SimpleClosingTransform(MorphologyTransform):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Applies binary closing transform """
        mask_out = binary_closing(image)
        return mask_out

class SizedClosingTransform(MorphologyTransform):
    def __init__(self, level_in: int, expand_size: float = 50, level_zero_size: float = 0.25) -> None:
        # assign values
        self.level_in = level_in
        self.expand_size = expand_size
        self.level_zero_size = level_zero_size
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Applies binary closing transform with a neighbourhood off specified size"""
        area_of_pixel = 2 ** self.level_in * self.level_zero_size
        pixels_to_expand = math.ceil(self.expand_size / area_of_pixel)
        neighbourhood = np.ones((pixels_to_expand, pixels_to_expand))
        mask_out = binary_closing(image, neighbourhood)
        return mask_out

class FillHolesTransform(MorphologyTransform):
    def __init__(self, level_in: int, hole_size_to_fill: float = 250, level_zero_size: float = 0.25) -> None:
        # assign values
        self.level_in = level_in
        self.hole_size_to_fill = hole_size_to_fill
        self.level_zero_size = level_zero_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        area_of_pixel = 2 ** self.level_in * self.level_zero_size
        pixel_size_to_fill = self.hole_size_to_fill / area_of_pixel

        def fill_region_or_not(reg):
            size_to_fill = reg.area < pixel_size_to_fill
            colour_to_fill = reg.mean_intensity < 0.1
            fill_reg = size_to_fill & colour_to_fill
            return fill_reg

        # set background bigger than pixel value so for this nothing is counted as background
        label_image = label(image, background=256)
        regions = regionprops(label_image, image)
        regions_to_fill_mask = [fill_region_or_not(reg) for reg in regions]
        regions_to_fill = np.add(np.arange(len(regions)),1)[regions_to_fill_mask]

        mask_to_fill = np.isin(label_image, regions_to_fill)
        filled_holes = np.where(mask_to_fill, True, image)

        return filled_holes

class MaxPoolTransform(MorphologyTransform):
    def __init__(self, level_in: int, level_out: int) -> None:
        # assign values
        self.level_in = level_in
        self.level_out = level_out
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Applies max pool"""
        pool_size = 2 ** (self.level_out - self.level_in)
        image_out = block_reduce(image, (pool_size, pool_size), func=np.max)
        return image_out


class TissueDetector(metaclass=ABCMeta):
    def __init__(self, pre_filter = NullBlur(), morph_transform = NullTransform()) -> None:
        # assign values
        if not isinstance(pre_filter, list):
            self.pre_filter = [pre_filter]
        else:
            self.pre_filter = pre_filter
        if not isinstance(morph_transform, list):
            self.morph_transform = [morph_transform]
        else:
            self.morph_transform = morph_transform

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError


class TissueDetectorOTSU(TissueDetector):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """creates a dataframe of pixels locations labelled as tissue or not

        Based on the method proposed by wang et all
        1. Convert from RGB to HSV
        2. Perform automatic thresholding using Otsu's method on the H and S channels
        3. Combine the thresholded H and S channels

        Args:
            image: A scaled down WSI image. Must be r,g,b.

        Returns:
            An ndarray of booleans with the same dimensions as the input image
            True means foreground, False means background
        """
        # convert the image into the hsv colour space
        image_hsv = rgb2hsv(image)

        # apply filter
        for pf in self.pre_filter:
            image_hsv = pf(image_hsv)

        # use Otsu's method to find the thresholds for hue and saturation
        thresh_h = threshold_otsu(image_hsv[:, :, 0])
        thresh_s = threshold_otsu(image_hsv[:, :, 1])

        # mask the image to get determine which pixels with hue and saturation above their thresholds
        mask_h = image_hsv[:, :, 1] > thresh_h
        mask_s = image_hsv[:, :, 1] > thresh_s

        # combine the masks with an OR so any pixel above either threshold counts as foreground
        np_mask = np.logical_or(mask_h, mask_s)

        # apply morphological transforms
        for mt in self.morph_transform:
            np_mask = mt(np_mask)
        return np_mask


class TissueDetectorGreyScale(TissueDetector):
    def __init__(self, pre_filter = NullBlur(), morph_transform = NullTransform(), grey_level: float = 0.8) -> None:
        super().__init__(pre_filter, morph_transform)
        self.grey_level = grey_level

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """creates a dataframe of pixels locations labelled as tissue or not

        1. Convert PIL image to numpy array
        2. Convert from RGB to gray scale
        3. Get masks, any pixel that is less than a threshold (e.g. 0.8)

        Args:
        image: A scaled down WSI image. Must be r,g,b.

        Returns:
        An ndarray of booleans with the same dimensions as the input image
        True means foreground, False means background
        """
        # convert PIL image to numpy array
        image = np.asarray(image)

        # change pure black to pure white
        imager = image[:, :, 0] == 0
        imageg = image[:, :, 1] == 0
        imageb = image[:, :, 2] == 0
        image_mask = np.expand_dims(np.logical_and(np.logical_and(imager, imageg), imageb), axis=-1)
        image = np.where(image_mask, [255,255,255], image)
        image = np.array(image, dtype=np.uint8)

        # convert to gray-scale
        image_grey = rgb2gray(image)

        # apply filter
        for pf in self.pre_filter:
            image = pf(image)

        # get masks, any pixel that is less than 0.8
        np_mask = np.less_equal(image_grey, self.grey_level)

        # apply morphological transforms
        for mt in self.morph_transform:
            np_mask = mt(np_mask)

        return np_mask


class TissueDetectorAll(TissueDetector):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """creates a dataframe of all pixels in image labelled as foreground

        Args:
        image: A scaled down WSI image. Must be r,g,b.

        Returns:
        An ndarray of booleans with the same dimensions as the input image
        True means foreground
        """
        # convert PIL image to numpy array
        image = np.asarray(image)

        # apply filter
        for pf in self.pre_filter:
            image = pf(image)

        # get masks, all pixels
        np_mask = np.array(np.ones(image.shape[0:2]), dtype=bool)

        # apply morphological transforms
        for mt in self.morph_transform:
            np_mask = mt(np_mask)

        return np_mask

