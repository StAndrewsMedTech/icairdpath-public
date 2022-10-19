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


class Endometrial(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path, label: str) -> AnnotationSet:
        # NOTE: Insufficient is only a slide label not a patch level label
        # Therefore all slides labelled as insufficient should return no patches

        group_labels = {"Malignant":"malignant", "malignant":"malignant",  
                        "Other/benign":"other_benign", "other_benign":"other_benign", "Other":"other_benign", "other":"other_benign", 
                        "Adenocarcinoma": "adenocarcinoma", "adenocarcinoma":"adenocarcinoma", 
                        "Sarcoma": "sarcoma", "sarcoma":"sarcoma", 
                        "hyperplasia_with_atypia": "hyperplasia_with_atypia", "hyperplasia":"hyperplasia_with_atypia", "Hyperplasia with atypia": "hyperplasia_with_atypia", 
                        "Hyperplasia with Atypia": "hyperplasia_with_atypia", "hyperplasia atypia": "hyperplasia_with_atypia", 
                        "CarcinoSarcoma" : "carcinosarcoma", "carcinosarcoma":"carcinosarcoma", "Carcinosarcoma": "carcinosarcoma", 
                        "Hormonal":"hormonal", "hormonal":"hormonal", 
                        "inactive_atrophic": "innactive_atrophic", "Inactive/atrophic":"innactive_atrophic","inactive":"innactive_atrophic", 
                        "Inactive/Atrophic":"innactive_atrophic", "Innactive/Atrophic":"innactive_atrophic",
                        "Menstrual":"menstrual", "menstrual":"menstrual",
                        "Proliferative":"proliferative", "proliferative":"proliferative", 
                        "Secretory":"secretory", "secretory":"secretory", 
                        "Cervical Tissue": "other_benign", "Cervical": "other_benign", "Cervix": "other_benign",
                        "Normal EndoCervix": "other_benign", 
                        "polyp": "other_benign", "endometrial polyp": "other_benign", 
                        "Hormonal Change": "hormonal", 
                        "EctoCervix": "other_benign", "Normal EctoCervix": "other_benign",
                        "benign": "other_benign",
                        "normal myometrium": "other_benign"}
        if label == "insufficient":
            # insufficient slides should be excluded in the sampler, so this is a placeholder
            label = "other_benign"

        annotations = load_annotations(file, group_labels, label) if file else []
        labels_order = ["background",  "other_benign", "hormonal", "innactive_atrophic", "menstrual", "proliferative", "secretory", "malignant", "adenocarcinoma", "carcinosarcoma", "hyperplasia_with_atypia", "sarcoma"]

        return AnnotationSet(annotations, self.labels, labels_order, label)

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"background" : 0, "insufficient":1, "other_benign":2, "hormonal":3, "innactive_atrophic":4, "menstrual":5, "proliferative":6, "secretory":7, "malignant":8, "adenocarcinoma":9, "carcinosarcoma":10, "hyperplasia_with_atypia":11, 
                "sarcoma":12}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"insufficient":1, "other_benign":2, "hormonal":3, "innactive_atrophic":4, "menstrual":5, "proliferative":6, "secretory":7, "malignant":8, "adenocarcinoma":9, "carcinosarcoma":10, "hyperplasia_with_atypia":11, 
                "sarcoma":12}


class Endometrial_subCategories(Endometrial):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"insufficient":1, "other_benign":2, "hormonal":3, "innactive_atrophic":4, "menstrual":5, "proliferative":6, "secretory":7, "malignant":8, "adenocarcinoma":9, "carcinosarcoma":10, "hyperplasia_with_atypia":11, 
                "sarcoma":12}


def subcat():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "icaird"
    image_dir = "iCAIRD"  
    annotations_dir = "iCAIRD_annotations"
    metadata_dir = root / "iCAIRD_metadata"
    
    csv_path = metadata_dir / "iCAIRD_Endometrial_Data_algo4.csv"

    endometrial_df = pd.read_csv(csv_path)

    def annotation_rel_path(df_row):
        batch_no = df_row.batch
        image_name = str(df_row.Filename)
        annot_name = image_name[:-7] + "tiff.txt"
        annot_name = '/batch' + str(int(batch_no)) + "/" + annot_name
        return annot_name

    endometrial_df['annot_path'] = endometrial_df.apply(annotation_rel_path, axis = 1)

    df = pd.DataFrame()
    df["slide"] =  str(image_dir)+ '/' + endometrial_df['Filename']
    df["annotation"] = str(annotations_dir) + endometrial_df['annot_path']
    df["label"] = endometrial_df['subCategory']
    df["tags"] = endometrial_df['Category'] + ";" + endometrial_df['train_valid_algo4']
   
    return Endometrial(root, df)

