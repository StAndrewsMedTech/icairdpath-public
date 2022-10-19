import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from typing import List, Tuple, Callable

from repath.utils.convert import remove_item_from_dict

from repath.preprocess.patching.patch_index import CombinedIndex, CombinedPatchSet, SlidesIndex, SlidePatchSet


def simple_random(class_df: pd.DataFrame, sum_totals: int) -> pd.DataFrame:
    class_sample = class_df.sample(n=sum_totals, axis=0, replace=False)
    return class_sample


def simple_random_replacement(class_df: pd.DataFrame, sum_totals: int) -> pd.DataFrame:
    class_sample = class_df.sample(n=sum_totals, axis=0, replace=True)
    return class_sample


def weighted_random(class_df: pd.DataFrame, sum_totals: int) -> pd.DataFrame:
    class_df = class_df.assign(freq=class_df.groupby('slide_idx')['slide_idx'].transform('count').tolist())
    class_df = class_df.assign(weights= np.divide(1, class_df.freq))
    class_sample = class_df.sample(n=sum_totals, axis=0, replace=True, weights=class_df.weights)
    return class_sample


def balanced_sample(indexes: List[SlidesIndex], num_samples: int, floor_samples: int = 1000, 
                    sampling_policy: Callable[[pd.DataFrame, int], pd.DataFrame] = simple_random) -> CombinedPatchSet:

    index = CombinedIndex.for_slide_indexes(indexes)

    # work out how many of each type of patches you have in the index
    labels = np.unique(index.patches_df.label)
    sum_totals = [np.sum(index.patches_df.label == label) for label in labels]
    print(sum_totals) 
    # find the count for the class that has the lowest count, so we have balanced classes
    n_patches = min(sum_totals) 

    # limit the count for each class to the number of samples we want
    n_patches = min(n_patches, num_samples) 

    # make sure that have a minimun number of samples for each class if available
    # classes with smaller that floor with remain the same
    n_patches = max(n_patches, floor_samples) 

    sum_totals = np.minimum(sum_totals, n_patches)  # cap the number sample from each class to n_patches 

    # sample n patches
    sampled_patches = pd.DataFrame(columns=index.patches_df.columns)
    for idx, label in enumerate(labels):
        class_df = index.patches_df[index.patches_df.label == label]
        class_sample = sampling_policy(class_df, sum_totals[idx])
        sampled_patches = pd.concat((sampled_patches, class_sample), axis=0)

    index.patches_df = sampled_patches
    return index

def balanced_sample_exclude_class(indexes: List[SlidesIndex], num_samples: int, exclude_class: str, floor_samples: int = 1000, 
                    sampling_policy: Callable[[pd.DataFrame, int], pd.DataFrame] = simple_random) -> CombinedPatchSet:

    for si in indexes:
        sl_id = [ps.slide_idx for ps in si.patches]
        labs = [si.dataset.paths.label[sid] for sid in sl_id]
        for idx, lab in enumerate(labs):
            if lab == exclude_class:
                si.patches[idx].patches_df = pd.DataFrame(columns=["x", "y", "label", "transform"])

    index = CombinedIndex.for_slide_indexes(indexes)

    # work out how many of each type of patches you have in the index
    labels = np.unique(index.patches_df.label)
    sum_totals = [np.sum(index.patches_df.label == label) for label in labels]
    print(sum_totals)
    # find the count for the class that has the lowest count, so we have balanced classes
    n_patches = min(sum_totals)

    # limit the count for each class to the number of samples we want
    n_patches = min(n_patches, num_samples)

    # make sure that have a minimun number of samples for each class if available
    # classes with smaller that floor with remain the same
    n_patches = max(n_patches, floor_samples)
    sum_totals = np.minimum(sum_totals, n_patches)  # cap the number sample from each class to n_patches

    # sample n patches
    sampled_patches = pd.DataFrame(columns=index.patches_df.columns)
    for idx, label in enumerate(labels):
        class_df = index.patches_df[index.patches_df.label == label]
        class_sample = sampling_policy(class_df, sum_totals[idx])
        sampled_patches = pd.concat((sampled_patches, class_sample), axis=0)

    index.patches_df = sampled_patches
    return index


def per_slide_sample(indexes: List[SlidesIndex], num_samples: List[int]) -> CombinedPatchSet:

    index = CombinedIndex.for_slide_indexes(indexes)
    labels = np.unique(index.patches_df.label)

    # sample n patches per slide
    sampled_patches = pd.DataFrame(columns=index.patches_df.columns)
    for idx, label in enumerate(labels):
        nslide = num_samples[idx]
        class_df = index.patches_df[index.patches_df.label == label]
        for name, group in class_df.groupby('slide_idx'):
            ns = min(nslide, group.shape[0])
            slide_sample = group.sample(n=ns, axis=0, replace=False)
            sampled_patches = pd.concat((sampled_patches, slide_sample), axis=0)

    index.patches_df = sampled_patches
    return index
