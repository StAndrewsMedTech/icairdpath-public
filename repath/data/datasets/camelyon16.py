
import os
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import pandas as pd
from repath.data.annotations import AnnotationSet
from repath.data.annotations.asapxml import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.openslide import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root
from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, pre_re_curve



class Camelyon16(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)

    def load_annotations(self, file: Path, label: str) -> AnnotationSet:
            """ Load annotations form a given annotation file path.

            Args:
                file (Path): Path to the annotation file.  
            
            Returns:
                AnnotationSet which includes: annotations, labels, labels_order and fill_label
            """
            group_labels = {"Tumor": "tumor", "_0": "tumor", "_1": 'tumor', "_2": 'normal', 'Exclusion': 'normal', 'None': 'normal'}
            annotations = load_annotations(file, group_labels) if file else []
            labels_order = ["background", "tumor", "normal"]
            return AnnotationSet(annotations, self.labels, labels_order, "normal")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "normal": 1, "tumor": 2}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"normal": 0, "tumor": 1}


def training():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon16" / "raw" / "training"
    annotations_dir = root / "lesion_annotations"
    tumor_slide_dir = root / "tumor"
    normal_slide_dir = root / "normal"

    # all paths are relative to the dataset 'root'
    annotation_paths = sorted([p.relative_to(root) for p in annotations_dir.glob("*.xml")])
    tumor_slide_paths = sorted([p.relative_to(root) for p in tumor_slide_dir.glob("*.tif")])
    normal_slide_paths = sorted([p.relative_to(root) for p in normal_slide_dir.glob("*.tif")])

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = tumor_slide_paths + normal_slide_paths
    df["annotation"] =  annotation_paths + ["" for _ in range(len(normal_slide_paths))]
    df["label"] = ['tumor'] * len(tumor_slide_paths) + ['normal'] * len(normal_slide_paths)
    df["tags"] = ""

    return Camelyon16(root, df)


def training_small():
    # set up the paths to the slides and annotations
    cam16 = training()
    df = cam16.paths.sample(12, random_state=777)

    return Camelyon16(project_root() / cam16.root, df)


def testing():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for test dataset.

    Returns:
        DataFrame (pd.DataFrame): Test data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon16" / "raw" / "testing"
    annotations_dir = root / "lesion_annotations"
    slide_dir = root / "images"
    

    # all paths are relative to the dataset 'root'
    slide_paths = sorted([p.relative_to(root) for p in slide_dir.glob("*.tif")])
    annotation_paths = sorted([p.relative_to(root) for p in annotations_dir.glob("*.xml")])

    #get the slide name
    slide_names = []
    for path in slide_paths:
        head, tail = os.path.split(path)
        slide_names.append(tail.split('.')[0])

    #search for slides with annotations, add the annotation path if it exists else add empty string
    slides_annotations_paths = []
    for name in slide_names:
        a_path = ""
        for anno_path in annotation_paths:
            if  name in str(anno_path):
                a_path = anno_path
        slides_annotations_paths.append(a_path)
    
    #get the slide labels by reading the csv file
    csv_path = root / 'reference.csv'
    label_csv_file = pd.read_csv(csv_path, header = None)
    slide_labels = label_csv_file.iloc[:, 1]

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths 
    df["annotation"] = slides_annotations_paths
    df["label"] = slide_labels
    df["tags"] = ""

    return Camelyon16(root, df)




