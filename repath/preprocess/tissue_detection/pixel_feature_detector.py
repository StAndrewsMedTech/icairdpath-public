from abc import ABCMeta, abstractmethod
from typing import List

import itertools
from itertools import combinations_with_replacement
import numpy as np
from skimage import filters, feature
from skimage import img_as_float32
from concurrent.futures import ThreadPoolExecutor


class PixelFeature(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError

class EdgeFeature(PixelFeature):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return filters.sobel(image)

class TextureFeature(PixelFeature):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        H_elems = [
            np.gradient(np.gradient(image)[ax0], axis=ax1)
            for ax0, ax1 in combinations_with_replacement(range(image.ndim), 2)
        ]
        eigvals = feature.hessian_matrix_eigvals(H_elems)
        eigvals = np.dstack((np.expand_dims(eigvals[0, :, :], axis=-1), np.expand_dims(eigvals[1, :, :], axis=-1)))

        return eigvals



def calc_HE_color_deconv(im):
    """ calculates H&E channels for an image using color deconvolution

    Adapted from https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_deconvolution/color_deconvolution.py


    Args:
        im (ndarray): rgb value image on scale 0-255

    Returns:
        a numpy array of same size with two channels one for hematoxylin and one for eosin
    """
    # flatten image vector
    im_rgb = im.reshape((-1, im.shape[-1])).T

    # convert to optical density
    I_0 = 256
    im_rgb = im_rgb.astype(float) + 1
    im_sda = -np.log(im_rgb/(1.* I_0)) * 255/np.log(I_0)

    # work out stain vector
    w = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])
    stain0 = w[:, 0]
    stain1 = w[:, 1]
    stain2 = np.cross(stain0, stain1)
    wc = np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T
    wc = wc / np.sqrt((wc ** 2).sum(0))
    Q = np.linalg.inv(wc)

    # apply deconvolution
    sda_deconv = np.dot(Q, im_sda)

    # reshape and rescale for output
    stain_im = sda_deconv.T.reshape(im.shape[:-1] + (sda_deconv.shape[0],))
    stain_out = stain_im.clip(0, 255).astype(np.uint8)
    he_out = stain_out[:, :, 0:2]

    return he_out



class PixelFeatureDetector(object):
    def __init__(self, features_list: List, sigma_min: int = 1, sigma_max: int = 16, num_sigma: int = None, num_workers: int = None, h_e: bool = False, raw=False) -> None:
        if num_sigma is None:
            num_sigma = int(np.log2(sigma_max) - np.log2(sigma_min) + 1)
        sigmas = np.logspace(
            np.log2(sigma_min),
            np.log2(sigma_max),
            num=num_sigma,
            base=2,
            endpoint=True,
        )
        if raw:
            sigmas = np.hstack((0, sigmas))
        self.sigmas = sigmas
        self.num_workers = num_workers
        self.features_list = features_list
        self.h_e = h_e

    def per_channel(self, img):
        def singlescale_features_singlechannel(self, img, sigma):
            gaussian_filtered = filters.gaussian(img, sigma)
            results = gaussian_filtered
            for ft in self.features_list:
                res = ft(gaussian_filtered)
                results = np.dstack((results, res))

            return results

        img = np.ascontiguousarray(img_as_float32(img))

        for idx, sig in enumerate(self.sigmas):
            out_sigma = singlescale_features_singlechannel(self, img, sig)
            if idx == 0:
                out_sigmas = out_sigma
            else:
                out_sigmas = np.dstack((out_sigmas, out_sigma))

        return out_sigmas

    def __call__(self, image: np.ndarray) -> np.array:

        if self.h_e:
            he_channels = calc_HE_color_deconv(image)
            image = np.dstack((image, he_channels))

        for dim in range(image.shape[-1]):
            res = self.per_channel(image[..., dim])
            if dim == 0:
                all_results = res
            else:
                all_results = np.dstack((all_results, res))

        return all_results





