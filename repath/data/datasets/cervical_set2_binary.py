import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from repath.data.annotations import AnnotationSet
from repath.data.annotations.geojson import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.isyntax import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root


class Cervical(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path, label: str) -> AnnotationSet:
        group_labels = {"low_grade": "not_malignant", "Low Grade": "not_malignant", "Low grade": "not_malignant", "low grade" : "not_malignant",
            "high_grade": "not_malignant", "High Grade": "not_malignant", "High grade": "not_malignant", "high grade": "not_malignant",
            "malignant": "malignant", "Malignant":"malignant", 
            "Normal/inflammation": "not_malignant", "normal/inflammation": "not_malignant", "normal": "not_malignant"}
        annotations = load_annotations(file, group_labels, "not_malignant") if file else []
        labels_order = ["not_malignant",  "malignant"]

        return AnnotationSet(annotations, self.labels, labels_order, "not_malignant")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"background" : 0, "not_malignant": 1, "malignant": 2}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"not_malignant": 1, "malignant": 2}


class Cervical_subCategories(Cervical):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"normal": 1, "cin1": 2, "hpv":3 , "cin2": 4, "cin3": 5, "squamous_carcinoma": 6, "adenocarcinoma": 7, "cgin": 8, "other": 9}


def set2():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "icaird"
    image_dir = "iCAIRD"  
    annotations_dir = "iCAIRD_annotations"
    metadata_dir = root / "iCAIRD_metadata"
    
    csv_path = metadata_dir / "iCAIRD_Cervical_Data_algo2.csv"

    cervical_df = pd.read_csv(csv_path)

    def annotation_rel_path(df_row):
        batch_no = df_row.batch
        image_name = str(df_row.Filename)
        annot_name = image_name[:-7] + "tiff.txt"
        annot_name = '/batch' + str(int(batch_no)) + "/" + annot_name
        return annot_name

    cervical_df['annot_path'] = cervical_df.apply(annotation_rel_path, axis = 1)

    df = pd.DataFrame()
    df["slide"] =  str(image_dir)+ '/' + cervical_df['Filename']
    df["annotation"] = str(annotations_dir) + cervical_df['annot_path']
    df["label"] = cervical_df['Category']
    df["tags"] = cervical_df['subCategory'] + ";" + cervical_df['train_valid_v2']
   
    return Cervical(root, df)

