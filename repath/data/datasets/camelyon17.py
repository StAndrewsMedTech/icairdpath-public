import os
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import pandas as pd
from os import walk
from os.path import join
from repath.data.annotations import AnnotationSet
from repath.data.annotations.asapxml import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.openslide import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root
from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, pre_re_curve


class Camelyon17(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)

    def load_annotations(self, file: Path, label: str) -> AnnotationSet:
        group_labels = {"metastases": "tumor", "normal": "normal"}
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
        return {"negative": 0, "itc": 1,  "macro": 2, "micro": 3}

def training():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon17" / "raw" / "training"
    annotations_dir = root / "lesion_annotations"
    
    # all paths are relative to the dataset 'root'
    slide_paths = sorted([os.path.relpath(os.path.join(dirpath, file), root) for (dirpath, dirnames, filenames) in os.walk(root) for file in filenames if ".tif" in file])
    annotation_paths = sorted([os.path.relpath(os.path.join(dirpath, file), root) for (dirpath, dirnames, filenames) in os.walk(root) for file in filenames if ".xml" in file])

    # Extract names of slides exclusing their extensions
    slide_names = []
    for path in slide_paths:
        head, tail = os.path.split(path)
        slide_names.append(tail.split('.')[0])
    
    #annotation path to the slides which are annotated 
    slides_annotations_paths = []
    for name in slide_names:
        a_path = ""
        for anno_path in annotation_paths:
            if  name in anno_path:
                a_path = anno_path
        slides_annotations_paths.append(a_path)

    #Extract annotations names exclusing their extensions
    annotation_names = []
    for path in annotation_paths:
        head, tail =  os.path.split(path)
        annotation_names.append(tail.split('.')[0])

    #Slide level labels
    labels = pd.read_csv(root / 'stage_labels.csv')
    slides_labels_df =labels.loc[labels.stage.isin(["itc", "negative", "micro", "macro"])] 
    slide_level_labels = slides_labels_df.values.tolist()
    
    slide_labels = []
    for lst in slide_level_labels:
        if  lst[0].split('.')[0]  in slide_names:
            slide_labels.append(lst[1])
    
    #patient level labels
    patient_labels_df = labels.loc[labels.stage.isin(["pN0", "pN1", "pN2", "pN0(i+)", "pN1mi"])]
    patient_level_labels = patient_labels_df.values.tolist()
    

    patient_names =  patient_labels_df.patient.str.split('.')
    patient_names = [row[0] for row in patient_names ]
    
    def intersection(lst1, lst2): 
        return list(set(lst1) & set(lst2)) 
  
    slides_names_with_anno = intersection(slide_names, annotation_names)
    slides_names_with_no_anno = [item for item in slide_names if item not in slides_names_with_anno]
    
    #tags for each slide including the patient_name, patient-level_label and whethere slide is annotated or not
    tags = []
    for sname in slide_names:
        for row in patient_level_labels:
            name = row[0].split('.')[0]
            label = row[1]
            tag = ''
            if sname in slides_names_with_anno and name in sname:
                tag = name + ';' + label + ';' + 'annotated'
            elif sname in slides_names_with_no_anno and name in sname:
                tag = name + ';' + label 
            tags.append(tag)
    tags = [tag for tag in tags if tag != ""]
    
    # turn slides_path, slide-level label, annotations_path and tags  into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths 
    df["annotation"] = slides_annotations_paths 
    df["label"] = slide_labels
    df["tags"] = tags 

    return Camelyon17(root, df)


def training_small():
    # set up the paths to the slides and annotations
    cam17 = training()
    df = cam17.paths.sample(12, random_state=777)

    return Camelyon17(project_root() / cam17.root, df)



def testing():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for test set.

    Returns:
        DataFrame (pd.DataFrame): Test set data frame.
    """
    #path to the test slides
    root = project_root() / "data" / "camelyon17" / "raw" / "testing"
    test_slides_dir = root / "patients"
    slide_paths = sorted([p.relative_to(root) for p in test_slides_dir.glob("*.tif")])
    labels = pd.read_csv(root /'evaluation/submission_example.csv')

    slides_labels_df = labels.loc[labels.stage.isin(["itc", "negative", "micro", "macro"])] 
    slide_level_labels  = slides_labels_df.values.tolist()
    
    #slide-level labels for test data
    slide_names = []
    for path in slide_paths:
        head, tail = os.path.split(path)
        slide_names.append(tail)
    
    slide_labels = []
    for lst in slide_level_labels:
        if  lst[0]  in slide_names:
            slide_labels.append(lst[1])
    
    #patient-level labels for test data
    patient_labels_df = labels.loc[labels.stage.isin(["pN0", "pN1", "pN2", "pN0(i+)", "pN1mi"])]
    patient_level_labels = patient_labels_df.values.tolist()
    
    #tags including the patient name for each slide, and patient-level labels. 
    #Test slides do not have annotations so the annotation column is all blank for test slides.
    tags = []
    for lst in patient_level_labels:
        name = lst[0].split('.')[0]
        label = lst[1]
        for spath in slide_paths:
            if name in str(spath):
                tag = name + ';' + label
                tags.append(tag)
        
    
    # turn slides_paths, slide_level labels, and tags  into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths
    df["annotation"] = ""
    df["label"] = slide_labels
    df["tags"] = tags 

    return Camelyon17(root, df) 
