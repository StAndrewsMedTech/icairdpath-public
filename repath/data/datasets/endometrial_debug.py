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
        #group_labels = {"Insufficient":"insufficient", "insufficient":"insufficient", "Malignant":"malignant", "malignant":"malignant",  "Other/benign":"other_benign", "other_benign":"other_benign",
        #                 "Adenocarcinoma": "adenocarcinoma", "adenocarcinoma":"adenocarcinoma", "Sarcoma": "sarcoma", "sarcoma":"sarcoma", "hyperplasia_with_atypia":"hyperplasia",
        #                 "hyperplasia":"hyperplasia", "Hyperplasia with atypia": "hyperplasia", "CarcinoSarcoma" : "carcinosarcoma", "carcinosarcoma":"carsinosarcoma",  "Other":"other", "other":"other", 
        #                 "Hormonal":"hormonal", "hormonal":"hormonal", "innactive_atrophic":"innactive", "Innactive/atrophic":"inacctive","innactive":"innactive", "Menstrual":"menstrual", 
        #                 "menstrual":"menstrual","Proliferative":"proliferative", "proliferative":"profilerative", "Secretory":"secretory", "secretory":"secretory"  }
        group_labels = {"Insufficient":"insufficient", "insufficient":"insufficient", 
                        "Malignant":"malignant", "malignant":"malignant",  
                        "Other/benign":"other_benign", "other_benign":"other_benign", "Other":"other_benign", "other":"other_benign", 
                        "Adenocarcinoma": "malignant", "adenocarcinoma":"malignant", 
                        "Sarcoma": "malignant", "sarcoma":"malignant", 
                        "hyperplasia_with_atypia":"malignant", "hyperplasia":"malignant", "Hyperplasia with atypia": "malignant", "Hyperplasia with Atypia": "malignant", 
                        "CarcinoSarcoma" : "malignant", "carcinosarcoma":"malignant",  
                        "Hormonal":"other_benign", "hormonal":"other_benign", 
                        "inactive_atrophic":"other_benign", "Inactive/atrophic":"other_benign","inactive":"other_benign", "Inactive/Atrophic":"other_benign", 
                        "Menstrual":"other_benign", "menstrual":"other_benign",
                        "Proliferative":"other_benign", "proliferative":"other_benign", 
                        "Secretory":"other_benign", "secretory":"other_benign", 
                        "Cervical Tissue": "other_benign", "Cervical": "other_benign",
                        "Normal EndoCervix": "other_benign", 
                        "polyp": "other_benign",  
                        "Hormonal Change": "other_benign", 
                        "EctoCervix": "other_benign", "Normal EctoCervix": "other_benign"}
        annotations = load_annotations(file, group_labels, label) if file else []
        labels_order = ["insufficient",  "other_benign",  "malignant"]
        return AnnotationSet(annotations, self.labels, labels_order, label)

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"background" : 0, "insufficient": 1, "other_benign": 2 , "malignant": 3 }

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"insufficient": 1, "other_benign": 2 ,  "malignant": 3}


class Endometrial_subCategories(Endometrial):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"insufficient": 1, "hormonal": 2, "innactive":3 , "menstrual": 4, "proliferative": 5, "secretory": 6, "adenocarcinoma": 7, "carcinosarcoma": 8, "hyperplasia": 9, 
                "other":10, "sarcoma":11}


def debug():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "icaird"
    image_dir = "iCAIRD"  
    annotations_dir = "iCAIRD_annotations"
    metadata_dir = root / "iCAIRD_metadata"
    
    csv_path = metadata_dir / "iCAIRD_Endometrial_Data_algo1.csv"

    endometrial_df = pd.read_csv(csv_path)

    def annotation_rel_path(df_row):
        batch_no = df_row.batch
        image_name = str(df_row.Filename)
        annot_name = image_name[:-7] + "tiff.txt"
        annot_name = '/batch' + str(batch_no) + "/" + annot_name
        return annot_name

    endometrial_df['annot_path'] = endometrial_df.apply(annotation_rel_path, axis = 1)

    df = pd.DataFrame()
    df["slide"] =  str(image_dir)+ '/' + endometrial_df['Filename']
    df["annotation"] = str(annotations_dir) + endometrial_df['annot_path']
    df["label"] = endometrial_df['Category']
    df["tags"] = endometrial_df['subCategory'] + ";" + endometrial_df['train_valid_algo1']
   
    return Endometrial(root, df)

