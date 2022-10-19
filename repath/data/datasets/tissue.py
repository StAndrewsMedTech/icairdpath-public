from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from repath.data.annotations import AnnotationSet
from repath.data.annotations.geojson import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.isyntax import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root


class TissueDetection(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path) -> AnnotationSet:
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


def tissue():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root_dir = project_root() / "data" / "icaird_tissue" 
    image_dir = root_dir / "images" 
    annot_dir = root_dir / "annotations"
    
    annot_paths = sorted([p.relative_to(root_dir) for p in annot_dir.glob("*.txt")])
    slide_paths = sorted([p.relative_to(root_dir) for p in image_dir.glob("*.isyntax")])
   
    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths
    df["annotation"] = annot_paths
    df["label"] = "unknown"  # ["cervical"] * 9 + ["endometrial"] * 7 - this labelling does not line up with the number of slides
    df["tags"] = ""

    return TissueDetection(root_dir, df)
