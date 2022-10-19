import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from repath.data.annotations import AnnotationSet
from repath.data.annotations.geojson import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.bioformats import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root


class Cervicalczi(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path, label: str) -> AnnotationSet:
        group_labels = {"low_grade": "low_grade", "Low Grade": "low_grade", "Low grade": "low_grade", "low grade" : "low_grade",
            "high_grade": "high_grade", "High Grade": "high_grade", "High grade": "high_grade", "high grade": "high_grade",
            "malignant": "malignant", "Malignant":"malignant", 
            "Normal/inflammation": "normal", "normal/inflammation": "normal", "normal": "normal"}
        annotations = load_annotations(file, group_labels, "normal") if file else []
        labels_order = ["low_grade",  "high_grade",  "malignant", "normal"]
        maxann = "normal_inflammation"
        for ann in annotations:
            annlab = ann.label
            if annlab == "low_grade":
                if maxann == "normal_inflammation":
                    maxann = annlab
            if annlab == "high_grade":
                if maxann == "normal_inflammation":
                    maxann = annlab
                if maxann =="low_grade":
                    maxann = annlab
            if annlab == "malignant":
                maxann = annlab

        if maxann != label:
            print(f'slide label does not match annotations, slide label is {label}, worst annotation is {maxann}')

        return AnnotationSet(annotations, self.labels, labels_order, "normal")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"background" : 0, "normal": 1, "low_grade": 2 , "high_grade": 3 ,  "malignant": 4}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"normal": 1, "low_grade": 2 , "high_grade": 3 ,  "malignant": 4}


class Cervical_subCategories(Cervicalczi):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"normal": 1, "cin1": 2, "hpv":3 , "cin2": 4, "cin3": 5, "squamous_carcinoma": 6, "adenocarcinoma": 7, "cgin": 8, "other": 9}


def train():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.
    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """

    # set up the paths to the slides and annotations
    root = project_root() / "data"/ "icaird"
    image_dir = "iCAIRD/iCAIRD_czi"  
    annotations_dir = "iCAIRD_annotations"
    metadata_dir = root / "iCAIRD_metadata/"

    csv_path =  metadata_dir / "iCAIRD_Cervical_Data_full_21_March_2022.csv"
    cervical_czi_df = pd.read_csv(csv_path)
    
    #cervical_czi_df = cervical_czi_df[cervical_czi_df['use_row'] == 1]
    cervical_czi_df = cervical_czi_df[cervical_czi_df['train/valid/test'] == 'train']
    cervical_czi_df = cervical_czi_df.reset_index(drop=True)

    def annotation_rel_path(df_row):
        #batch_no = df_row.batch
        image_name = str(df_row.Filename)
        annot_name = image_name[:-7] + ".txt"
        annot_name = '/Cervical/' + annot_name
        return annot_name

    cervical_czi_df['annot_path'] = cervical_czi_df.apply(annotation_rel_path, axis = 1)

    df = pd.DataFrame()
    df["slide"] =  str(image_dir)+ '/' + cervical_czi_df['Filename']
    df["annotation"] = str(annotations_dir) + cervical_czi_df['annot_path']
    df["label"] = cervical_czi_df['Category']
    df["tags"] = cervical_czi_df['subCategory'] + ";" + cervical_czi_df['train/valid/test']
    return Cervicalczi(root, df)

def valid():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """

    # set up the paths to the slides and annotations
    root = project_root() / "data"/ "icaird"
    image_dir = "iCAIRD"
    annotations_dir = "iCAIRD_annotations"
    metadata_dir = root / "iCAIRD_metadata/"

    csv_path =  metadata_dir / "iCAIRD_Cervical_Data_full_21_March_2022.csv"
    cervical_czi_df = pd.read_csv(csv_path)

    #cervical_czi_df = cervical_czi_df[cervical_czi_df['use_row'] == 1]
    cervical_czi_df = cervical_czi_df[cervical_czi_df['train/valid/test'] == 'valid']
    cervical_czi_df = cervical_czi_df.reset_index(drop=True)
    
    def annotation_rel_path(df_row):
        #batch_no = df_row.batch
        image_name = str(df_row.Filename)
        annot_name = image_name[:-7] + ".txt"
        annot_name = '/Cervical/' + annot_name
        return annot_name

    cervical_czi_df['annot_path'] = cervical_czi_df.apply(annotation_rel_path, axis = 1)

    df = pd.DataFrame()
    df["slide"] =  str(image_dir)+ '/' + cervical_czi_df['Filename']
    df["annotation"] = str(annotations_dir) + cervical_czi_df['annot_path']
    df["label"] = cervical_czi_df['Category']
    df["tags"] = cervical_czi_df['subCategory'] + ";" + cervical_czi_df['train/valid/test']
    return Cervicalczi(root, df)


def test():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """

    # set up the paths to the slides and annotations
    root = project_root() / "data"/ "icaird"
    image_dir = "iCAIRD"
    annotations_dir = "iCAIRD_annotations"
    metadata_dir = root / "iCAIRD_metadata/"

    csv_path =  metadata_dir / "iCAIRD_Cervical_Data_full_21_March_2022.csv"
    cervical_czi_df = pd.read_csv(csv_path)

    #cervical_czi_df = cervical_czi_df[cervical_czi_df['use_row'] == 1]
    cervical_czi_df = cervical_czi_df[cervical_czi_df['train/valid/test'] == 'test']
    cervical_czi_df = cervical_czi_df.reset_index(drop=True)
    
    def annotation_rel_path(df_row):
        #batch_no = df_row.batch
        image_name = str(df_row.Filename)
        annot_name = image_name[:-7] + ".txt"
        annot_name = '/Cervical/' + annot_name
        return annot_name

    cervical_czi_df['annot_path'] = cervical_czi_df.apply(annotation_rel_path, axis = 1)

    df = pd.DataFrame()
    df["slide"] =  str(image_dir)+ '/' + cervical_czi_df['Filename']
    df["annotation"] = str(annotations_dir) + cervical_czi_df['annot_path']
    df["label"] = cervical_czi_df['Category']
    df["tags"] = cervical_czi_df['subCategory'] + ";" + cervical_czi_df['train/valid/test']
    return Cervicalczi(root, df)

def set2():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data"/ "icaird"
    image_dir = "iCAIRD"  
    annotations_dir = "iCAIRD_annotations/Cervical/"
    metadata_dir = root / "iCAIRD_metadata/"
    csv_path =  metadata_dir / "iCAIRD_Cervical_Data_full_21_March_2022.csv"
    cervical_czi_df = pd.read_csv(csv_path)

    def annotation_rel_path(df_row):
        #batch_no = df_row.batch
        image_name = str(df_row.Filename)
        annot_name = image_name[:-7] + "txt"
        annot_name = annot_name
        return annot_name

    cervical_czi_df['annot_path'] = cervical_czi_df.apply(annotation_rel_path, axis = 1)

    df = pd.DataFrame()
    df["slide"] =  str(image_dir)+ '/' + cervical_czi_df['Filename']
    df["annotation"] = str(annotations_dir) + cervical_czi_df['annot_path']
    df["label"] = cervical_czi_df['Category']
    df["tags"] = cervical_czi_df['subCategory'] + ";" + cervical_czi_df['train/valid/test']
   
    return Cervicalczi(root, df)

