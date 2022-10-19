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


class TissueGame(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path, label: str) -> AnnotationSet:
        group_labels = {"background": "background", "Tissue": "tissue", "tissue": "tissue"}
        labels = {"background": 0, "tissue": 1}
        default_label = "tissue"
        annotations = load_annotations(file, group_labels, default_label) if file else []
        labels_order = ["background", "tissue"]
        return AnnotationSet(annotations, labels, labels_order, "background")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "tissue": 1}


def tissue_game():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # first set up the paths to the slides without tissue annotations
    root_dir = project_root() / "data" 
    image_dir_cv = project_root() / "data" / "icaird_tissue"  / "images_for_game"
    image_dir_en = project_root() / "data" / "icaird_blood"  / "images"
    
    slide_paths_cv = sorted([p.relative_to(root_dir) for p in image_dir_cv.glob("*.isyntax")])
    slide_paths_en = [p.relative_to(root_dir) for p in image_dir_en.glob("*.isyntax")]
    random.seed(123)
    slide_paths_en = random.sample(slide_paths_en, 45)
    slide_paths_en = sorted(slide_paths_en)
   
    # turn them into a data frame and pad with empty annotation paths
    df1 = pd.DataFrame()
    df1["slide"] = slide_paths_cv + slide_paths_en
    df1["annotation"] = ""
    df1["label"] = ""
    df1["tags"] = ""

    # next get paths to slides with tissue or not annotations
    annot_dir = project_root() / "data" / "icaird_tissue" / "annotations"
    image_dir_ann = project_root() / "data" / "icaird_tissue" / "images"
    slide_paths = sorted([p.relative_to(root_dir) for p in image_dir_ann.glob("*.isyntax")])
    annot_paths = sorted([p.relative_to(root_dir) for p in annot_dir.glob("*.txt")])
    df2 = pd.DataFrame()
    df2["slide"] = slide_paths
    df2["annotation"] = annot_paths
    df2["label"] = ""
    df2["tags"] = "tissue_ann"

    # combine dataframes
    df = pd.concat((df1, df2), axis=0, ignore_index=True)


    return TissueGame(root_dir, df)
