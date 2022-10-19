from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from repath.data.annotations import AnnotationSet
from repath.data.annotations.geojson import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.isyntax import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root


class BloodMucus(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path, default_label: str) -> AnnotationSet:
        group_labels = {"background": "background", "ignore": "background", "Ignore*": "background", 
        "Tissue": "tissue", "tissue": "tissue", "blood": "blood", "mucus": "mucus", "blood/mucus": "blood_mucus"}
        labels = {"background": 0, "tissue": 1, "blood": 2, "mucus": 3, "blood_mucus": 4}
        annotations = load_annotations(file, group_labels, default_label) if file else []
        labels_order = ["background", "blood", "mucus", "blood_mucus", "tissue"]
        return AnnotationSet(annotations, labels, labels_order, "background")

    def load_annotated_area(self, file: Path) -> AnnotationSet:
        group_labels = {"background": "background", "ignore": "background", "Ignore*": "background", 
            "Tissue": "tissue", "tissue": "tissue", "blood": "blood", "mucus": "mucus", "blood/mucus": "blood_mucus", "annotated_area": "annotated_area"}
        labels = {"background": 0, "annotated_area": 5}
        # assumes the annotated area annotation is unlabelled
        default_label = "annotated_area"
        annotations = load_annotations(file, group_labels, default_label) if file else []
        # remove everything except annotated areas
        annotated_areas = []
        for ann in annotations:
            if ann.label not in ["tissue", "blood", "mucus", "blood_mucus"]:
                annotated_areas.append(ann)
        labels_order = ["background", "annotated_area"]
        return AnnotationSet(annotated_areas, labels, labels_order, "background")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "tissue": 1, "blood": 2, "mucus": 3, "blood_mucus": 4}


def training():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # first set up the paths to the slides without tissue annotations
    root_dir = project_root() / "data" / "icaird_blood"
    image_dir = root_dir   / "images"
    annot_dir = root_dir / "annotations"
    
    slide_paths = [p.relative_to(root_dir) for p in image_dir.glob("*.isyntax")]
    annot_paths = [p.relative_to(root_dir) for p in annot_dir.glob("*.txt")]

    slide_paths = sorted(slide_paths)
    annot_paths = sorted(annot_paths)

    # for debug
    #slide_paths = slide_paths[0:3]
    #annot_paths = annot_paths[0:3]
   
    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths
    df["annotation"] = annot_paths
    df["label"] = ""
    df["tags"] = ""

    return BloodMucus(root_dir, df)


def validation():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # first set up the paths to the slides without tissue annotations
    root_dir = project_root() / "data" / "icaird_blood"
    image_dir = root_dir   / "images_valid"
    annot_dir = root_dir / "annotations_valid"
    
    slide_paths = [p.relative_to(root_dir) for p in image_dir.glob("*.isyntax")]
    annot_paths = [p.relative_to(root_dir) for p in annot_dir.glob("*.txt")]

    slide_paths = sorted(slide_paths)
    annot_paths = sorted(annot_paths)
   
    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths
    df["annotation"] = annot_paths
    df["label"] = ""
    df["tags"] = ""

    return BloodMucus(root_dir, df)
