from collections import namedtuple
from itertools import chain
from joblib import load
import os
from pathlib import Path
import threading
from typing import List, Sequence, Union
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
import torch
from torch import nn
from torchvision import transforms

from repath.data.datasets import Dataset
from repath.preprocess.patching.patch_finder import PatchFinder
from repath.preprocess.tissue_detection.tissue_detector import TissueDetector
from repath.preprocess.tissue_detection.pixel_feature_detector import TextureFeature, PixelFeatureDetector
from repath.preprocess.tissue_detection.blood_mucus_rework import apply_bloodm_detector, apply_bloodm_detector_3class
from repath.data.slides import Region
from repath.utils.convert import remove_item_from_dict
from repath.utils.geometry import Size
from repath.utils.paths import project_root


class PatchSet(Sequence):
    def __init__(
        self,
        dataset: Dataset,
        patch_size: int,
        level: int,
        patches_df: pd.DataFrame,
    ) -> None:
        self.dataset = dataset
        self.patch_size = patch_size
        self.level = level
        self.patches_df = patches_df

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        return self.patches_df.iloc[idx,]

    def summary(self) -> pd.DataFrame:
        """Gives a summary of number of patches for each class as a dataframe.

        Returns:
            pd.DataFrame: A summary dataframe defining number of patches for each class
        """
        by_label = self.patches_df.groupby("label").size()
        labels = {v: k for k, v in self.dataset.labels.items()}
        count_df = by_label.to_frame().T.rename(columns = labels)
        columns = list(labels.values())
        summary = pd.DataFrame(columns=columns)
        for l in labels.values():
            if l in count_df:
                summary[l] = count_df[l]
            else:
                summary[l] = 0
        summary = summary.replace(np.nan, 0)  # if there are no patches for some classes
        return summary


class CombinedPatchSet(PatchSet):
    def __init__(self, dataset: Dataset, patch_size: int, level: int, patches_df: pd.DataFrame) -> None:
        super().__init__(dataset, patch_size, level, patches_df)
        # columns of patches_df are x, y, label, slide_idx

    def save_patches(self, output_dir: Path, transforms: List[transforms.Compose] = None) -> None:
        """Saves patches after applying transforms to them 

        Args:
            output_dir (Path): The path to save the generated patches.
            transforms (List[transforms.Compose], optional): The transforms to be applied to each patch. Defaults to None.
        """
        for slide_idx, group in self.patches_df.groupby('slide_idx'):
            slide_path, _, _, _ = self.dataset[slide_idx]
            with self.dataset.slide_cls(slide_path) as slide:
                print(f"Writing patches for {self.dataset.to_rel_path(slide_path)}")
                for row in group.itertuples():
                    # read the patch image from the slide
                    region = Region.patch(row.x, row.y, self.patch_size, self.level)
                    image = slide.read_region(region)

                    # apply any transforms, as indexed in the 'transform' column
                    if transforms:
                        image = transforms[row.transform](image)

                    # get the patch label as a string
                    labels = {v: k for k, v in self.dataset.labels.items()}
                    label = labels[row.label]

                    # ensure the output directory exists
                    output_subdir = output_dir / label
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    files_in_subdir = output_subdir.glob('*.png')
                    for ff in files_in_subdir:
                        os.remove(ff)

                    # write out the slide
                    rel_slide_path = self.dataset.to_rel_path(slide_path)
                    _,tail = os.path.split(rel_slide_path)
                    slide_name_str = tail.split('.')[0]
                    patch_filename = slide_name_str + f"-{row.x}-{row.y}-{row.transform}.png"
                    image_path = output_dir / label / patch_filename
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(image_path), np.array(image))
                    # image.save(image_path)


class CombinedIndex(object):
    def __init__(self, cps: List[CombinedPatchSet]) -> None:
        self.datasets = [cp.dataset for cp in cps]
        self.patchsizes = [cp.patch_size for cp in cps]
        self.levels = [cp.level for cp in cps]
        patches_dfs = [cp.patches_df for cp in cps]
        patches_df = pd.concat(patches_dfs, axis=0)
        cps_index = [[idx] * len(cp) for idx, cp in enumerate(cps)]
        cps_index = [item for sublist in cps_index for item in sublist]
        patches_df['cps_idx'] = cps_index
        self.patches_df = patches_df

    def __len__(self):
        return len(self.patches_df)

    @classmethod
    def for_slide_indexes(cls, indexes: List['SlidesIndex']) -> 'CombinedIndex':
        """Combine a list of SlideIndices as one SlideIndex.

        Returns:
            SlideIndex: A single SLideIndex which is the combination of all given SlideIndices.
        """
        cps = [index.as_combined() for index in indexes]
        ci = cls(cps)
        return ci

    def save_patches(self, output_dir: Path, transforms: List[transforms.Compose] = None, affix: str = '', add_patches: bool = False) -> None:
        """Saves patches after applying transforms to them for more than one datastes.

        Args:
            output_dir (Path): The path to save the patches.
            transforms (List[transforms.Compose], optional): List of transforms to be applied to each patch. Defaults to None.
            affix (str, optional): A string added to the name of the patch before saving it. Defaults to ''.
        """
        if not add_patches:
            # check subdirectories exist and delete any existing patches
            for cps_idx, cps_group in self.patches_df.groupby('cps_idx'):
                # get the patch label as a string
                labels_dict = remove_item_from_dict(self.datasets[cps_idx].labels, 'background')
                labels = list(labels_dict.keys())
                for lab in labels:
                    # ensure the output directory exists
                    output_subdir = output_dir / lab
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    files_in_subdir = output_subdir.glob('*.png')
                    for ff in files_in_subdir:
                        os.remove(ff)

        for cps_idx, cps_group in self.patches_df.groupby('cps_idx'):
            for slide_idx, sl_group in cps_group.groupby('slide_idx'):
                slide_path, _, _, _ = self.datasets[cps_idx][slide_idx]
                stime = time.time()
                with self.datasets[cps_idx].slide_cls(slide_path) as slide:
                    print(f"Writing patches for {self.datasets[cps_idx].to_rel_path(slide_path)}")
                    for row in sl_group.itertuples():
                        # read the patch image from the slide
                        region = Region.patch(row.x, row.y, self.patchsizes[cps_idx], self.levels[cps_idx])
                        image = slide.read_region(region)

                        # apply any transforms, as indexed in the 'transform' column
                        if transforms:
                            image = transforms[row.transform-1](image)  # TODO: row.transforms should be 0 indexed by is 1 indexed

                        # get the patch label as a string
                        labels = {v: k for k, v in self.datasets[cps_idx].labels.items()}
                        label = labels[row.label]

                        # ensure the output directory exists
                        output_subdir = output_dir / label
                        output_subdir.mkdir(parents=True, exist_ok=True)

                        # write out the slide
                        rel_slide_path = self.datasets[cps_idx].to_rel_path(slide_path)
                        _,tail = os.path.split(rel_slide_path)
                        slide_name_str = tail.split('.')[0]
                        patch_filename = slide_name_str + f"-{row.x}-{row.y}{affix}.png"
                        image_path = output_dir / label / patch_filename
                        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                        cv2.imwrite(str(image_path), np.array(image))
                        # image.save(image_path)


    def save(self, output_dir: Path) -> None:
        """Saves a csv file for each slide.

        Args:
            output_dir (Path): The path to save the csv file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.patches_df.to_csv(output_dir / 'combined_index.csv', index=False)



class SlidePatchSet(PatchSet):
    def __init__(
        self, 
        slide_idx: int,
        dataset: Dataset,
        patch_size: int,
        level: int,
        patches_df: pd.DataFrame
    ) -> None:
        super().__init__(dataset, patch_size, level, patches_df)
        self.slide_idx = slide_idx
        abs_slide_path, self.annotation_path, self.label, tags = dataset[slide_idx]
        self.slide_path = dataset.to_rel_path(abs_slide_path)
        self.tags = [tg.strip() for tg in tags.split(';')]


    @classmethod
    def index_slide(cls, slide_idx: int, dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder, notblank: bool = True, bloodm: bool = False, bmmulti: bool = False):
        """[summary]

        Args:
            slide_idx (int): Index of the slide
            dataset (Dataset): An object that represents a set of slides and their annotations.
            tissue_detector (TissueDetector): A method for segmenting tissue from non-tissue in an slide.
            patch_finder (PatchFinder): 

        Returns:
            pathset: 
        """
        slide_path, annotation_path, rowlabel, _ = dataset[slide_idx]
        with dataset.slide_cls(slide_path) as slide:
            print(f"indexing {slide_path.name}")  # TODO: Add proper logging!
            if patch_finder.labels_level >= len(slide.dimensions):
                request_level = len(slide.dimensions) - 1
                lev_diff = patch_finder.labels_level - request_level
                max_level_dim = slide.dimensions[-1]
                requested_level_size = Size(max_level_dim.width // 2 ** lev_diff, max_level_dim.height // 2 ** lev_diff)
                labels_shape = requested_level_size.as_shape()
            else:
                labels_shape = slide.dimensions[patch_finder.labels_level].as_shape()
            if notblank:
                annotations = dataset.load_annotations(annotation_path, rowlabel)
                scale_factor = 2 ** patch_finder.labels_level
                labels_image = annotations.render(labels_shape, scale_factor)
            else:
                labels_image = np.ones(labels_shape)
            if bloodm:
                bloodm_detector = PixelFeatureDetector(features_list=[TextureFeature()], sigma_min = 1, sigma_max = 1, raw=True)
                bm_clf_path = project_root() / 'results' / 'blood_mucus_classifier' / 'bloodm_clf.joblib'
                assert bm_clf_path.is_file(),"Blood and mucus classifier doesn't exist."
                bloodm_clf = load(bm_clf_path)
                patch_size = 2**patch_finder.labels_level
                scale_factor = 2 ** (4 - patch_finder.patch_level)
                kernel_size = int(patch_size / scale_factor)
                if bmmulti:
                    output_labels = apply_bloodm_detector_3class(slide, 4, tissue_detector, bloodm_detector, bloodm_clf, kernel_size)
                    output_labels = output_labels[0:labels_image.shape[0], 0:labels_image.shape[1]]
                    tissue_mask = output_labels == 0
                    bm_mask = output_labels > 2
                    labels_image[tissue_mask] = 0
                    labels_image[bm_mask] = output_labels[bm_mask]
                else:
                    output_labels = apply_bloodm_detector(slide, 4, tissue_detector, bloodm_detector, bloodm_clf, kernel_size)
                    output_labels = output_labels[0:labels_image.shape[0], 0:labels_image.shape[1]]
                    tissue_mask = output_labels > 1
                    labels_image[~tissue_mask] = 0
            else:
                tissue_mask = tissue_detector(slide.get_thumbnail(patch_finder.labels_level))
                labels_image[~tissue_mask] = 0
            df, level, size = patch_finder(labels_image, slide.dimensions[patch_finder.patch_level])
            patchset = cls(slide_idx, dataset, size, level, df)
            return patchset

    @classmethod
    def index_and_save_slide(cls, slide_idx: int, dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder, 
        output_dir: Path, notblank: bool = True, bloodm: bool = False, bmmulti: bool = True):
        """[summary]

        Args:
            slide_idx (int): Index of the slide
            dataset (Dataset): An object that represents a set of slides and their annotations.
            tissue_detector (TissueDetector): A method for segmenting tissue from non-tissue in an slide.
            patch_finder (PatchFinder): 

        Returns:
            pathset: 
        """
        slide_path, annotation_path, rowlabel, _ = dataset[slide_idx]
        #print('rowlabel:',rowlabel)
        csv_path = dataset.to_rel_path(slide_path).with_suffix('.csv')
        csv_path = output_dir / csv_path
        csv_path.parents[0].mkdir(parents=True, exist_ok=True)
        if csv_path.is_file():
            patches_df = pd.read_csv(csv_path)
            output_size = patch_finder.patch_size + (patch_finder.border + patch_finder.jitter)
            patchset = SlidePatchSet(slide_idx, dataset, output_size, patch_finder.patch_level, patches_df)
        else:
            with dataset.slide_cls(slide_path) as slide:
                print(f"indexing {slide_path.name}")  # TODO: Add proper logging!
                if patch_finder.labels_level >= len(slide.dimensions):
                    request_level = len(slide.dimensions) - 1
                    lev_diff = patch_finder.labels_level - request_level
                    max_level_dim = slide.dimensions[-1]
                    requested_level_size = Size(max_level_dim.width // 2 ** lev_diff, max_level_dim.height // 2 ** lev_diff)
                    labels_shape = requested_level_size.as_shape()
                else:
                    labels_shape = slide.dimensions[patch_finder.labels_level].as_shape()
                if notblank:
                    annotations = dataset.load_annotations(annotation_path, rowlabel)
                    scale_factor = 2 ** patch_finder.labels_level
                    labels_image = annotations.render(labels_shape, scale_factor)
                else:
                    labels_image = np.ones(labels_shape)
                if bloodm:
                    bloodm_detector = PixelFeatureDetector(features_list=[TextureFeature()], sigma_min = 1, sigma_max = 1, raw=True)
                    bm_clf_path = project_root() / 'results' / 'blood_mucus_classifier' / 'bloodm_clf.joblib'
                    assert bm_clf_path.is_file(),"Blood and mucus classifier doesn't exist."
                    bloodm_clf = load(bm_clf_path)
                    patch_size = 2**patch_finder.labels_level
                    scale_factor = 2 ** (4 - patch_finder.patch_level)
                    kernel_size = int(patch_size / scale_factor)
                    output_labels = apply_bloodm_detector(slide, 4, tissue_detector, bloodm_detector, bloodm_clf, kernel_size)
                    output_labels = output_labels[0:labels_image.shape[0], 0:labels_image.shape[1]]
                    if bmmulti:
                        output_labels = apply_bloodm_detector_3class(slide, 4, tissue_detector, bloodm_detector, bloodm_clf, kernel_size)
                        output_labels = output_labels[0:labels_image.shape[0], 0:labels_image.shape[1]]
                        tissue_mask = output_labels == 0
                        bm_mask = output_labels > 2
                        labels_image[tissue_mask] = 0
                        labels_image[bm_mask] = output_labels[bm_mask]
                    else:
                        output_labels = apply_bloodm_detector(slide, 4, tissue_detector, bloodm_detector, bloodm_clf, kernel_size)
                        output_labels = output_labels[0:labels_image.shape[0], 0:labels_image.shape[1]]
                        tissue_mask = output_labels > 1
                        labels_image[~tissue_mask] = 0
                else:
                    tissue_mask = tissue_detector(slide.get_thumbnail(patch_finder.labels_level))
                    labels_image[~tissue_mask] = 0
                df, level, size = patch_finder(labels_image, slide.dimensions[patch_finder.patch_level])
                patchset = cls(slide_idx, dataset, size, level, df)
                #print(f"Saving {csv_path}")
                patchset.patches_df.to_csv(csv_path, index=False)
        return patchset


    @property
    def abs_slide_path(self):
        return self.dataset.to_abs_path(self.slide_path)

    def open_slide(self):
        return self.dataset.slide_cls(self.abs_slide_path)


class SlidesIndex(Sequence):
    def __init__(self, dataset: Dataset, patches: List[SlidePatchSet]) -> None:
        self.dataset = dataset
        self.patches = patches

    @classmethod
    def index_dataset(cls, dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder, notblank: bool = True, bloodm: bool = False, bmmulti: bool = False) -> 'SlidesIndex':
        """[summary]

        Returns:
            [type]: [description]
        """
        patchsets = [SlidePatchSet.index_slide(idx, dataset, tissue_detector, patch_finder, notblank, bloodm, bmmulti) for idx in range(len(dataset))]
        return cls(dataset, patchsets)

    @classmethod
    def index_and_save_dataset(cls, dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder, output_dir: Path, delete_existing: bool, notblank: bool = True, bloodm: bool = False, bmmulti: bool = False) -> 'SlidesIndex':
        """[summary]

        Returns:
            [type]: [description]
        """
        if delete_existing:
            files_in_dir = output_dir.glob('*.csv')
            for ff in files_in_dir:
                os.remove(ff)
        patchsets = [SlidePatchSet.index_and_save_slide(idx, dataset, tissue_detector, patch_finder, output_dir, notblank, bloodm, bmmulti) for idx in range(len(dataset))]
        return cls(dataset, patchsets)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    @property
    def patch_size(self):
        return self.patches[0].patch_size

    @property
    def level(self):
        return self.patches[0].level

    def summary(self) -> pd.DataFrame:
        """Creates a summary dataframe for patches.

        Returns:
            pd.DataFrame: [description]
        """
        summaries = [p.summary() for p in self.patches]
        slide_path = [p.slide_path for p in self.patches]
        slide_label = [p.label for p in self.patches]
        rtn = pd.concat(summaries)
        rtn['slide_path'] = slide_path
        rtn['slide_label'] = slide_label
        rtn = rtn.reset_index()
        rtn = rtn.drop('index', axis=1)
        rtn = rtn[['slide_path', 'slide_label'] + list(self.dataset.labels.keys())]
        return rtn

    def as_combined(self) -> CombinedPatchSet:
        """combine all patchsets into one.

        Returns:
            CombinedPatchSet: A combined patchset of all patchsets.
        """
        
        frames = [ps.patches_df for ps in self.patches]
        slide_indexes = [[ps.slide_idx]*len(ps) for ps in self.patches]
        slide_indexes = list(chain(*slide_indexes))
        patches_df =  pd.concat(frames, axis=0)
        patches_df['slide_idx'] = slide_indexes
        return CombinedPatchSet(self.dataset, self.patch_size, self.level, patches_df)

    def save(self, output_dir: Path) -> None:
        """Saves a csv file for each slide.

        Args:
            output_dir (Path): The path to save the csv file.
        """
        columns = ['slide_idx', 'csv_path', 'level', 'patch_size']
        index_df = pd.DataFrame(columns=columns)
        for ps in self.patches:
            # save out the csv file for this slide
            csv_path = ps.slide_path.with_suffix('.csv')
            csv_path = output_dir / csv_path
            csv_path.parents[0].mkdir(parents=True, exist_ok=True)
            print(f"Saving {csv_path}")
            ps.patches_df.to_csv(csv_path, index=False)

            # append information about slide to index
            info = np.array([ps.slide_idx, csv_path, ps.level, ps.patch_size])
            info = np.reshape(info, (1, 4))
            row = pd.DataFrame(info, columns=columns)
            index_df = index_df.append(row, ignore_index=True)

        # tidy up a bit and save the csv
        index_df = index_df.astype({"level": int, "patch_size": int})
        output_dir.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(output_dir / 'index.csv', index=False)

    @classmethod
    def load(cls, dataset: Dataset, input_dir: Path) -> 'SlidesIndex':
        def patchset_from_row(r) -> PatchSet:
            patches_df = pd.read_csv(input_dir / r.csv_path)
            return SlidePatchSet(int(r.slide_idx), dataset, int(r.patch_size), 
                                 int(r.level), patches_df)

        index = pd.read_csv(input_dir / 'index.csv')
        patches = [patchset_from_row(r) for r in index.itertuples()]
        rtn = cls(dataset, patches)
        return rtn
