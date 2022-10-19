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

class Endometrial_subCategories(Endometrial):
     def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"adenocarcinoma": 0, "carcinosarcoma":1 , "sarcoma": 2, "hyperplasia": 3, "other": 4, "insufficient": 5, 
                "proliferative": 6, "secretory": 7 , "menstrual": 8, "innactive_atrophic": 9, "hormonal": 10}

class Endometrial_subCategories(Endometrial):
     def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"adenocarcinoma": 0, "carcinosarcoma":1 , "sarcoma": 2, "hyperplasia with atypia": 3, "other": 4, "insufficient": 5, 
                "proliferative": 6, "secretory": 7 , "menstrual": 8, "innactive_atrophic": 9, "hormonal": 10}


class Endometrial(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame, label: str) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path) -> AnnotationSet:
        group_labels = {"malignant": "malignant", "insufficient": "insufficient", "other_benign": "other_benign"}
        annotations = load_annotations(file, group_labels) if file else []
        labels_order = [ "malignant", "insufficient", "other_benign"]
        return AnnotationSet(annotations, self.labels, labels_order, "malignant")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"background" : 0, "malignant": 1 , "insufficient": 2 ,  "other_benign": 3}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"malignant": 1 , "insufficient": 2 ,  "other_benign": 3}

def training():
     """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "endometrial" / "raw" / "training" 
    annotations_dir = root / "annotations"
    
    #slide directories    
    adenocarcinoma_slide_dir = root / "malignant" / "adenocarcinoma"
    carcinosarcoma_slide_dir = root / "malignant" / "carcinosarcoma"
    sarcoma_slide_dir = root / "malignant" / "sarcoma"
    hyperplasia_slide_dir = root / "malignant" / "hyperplasia"
    other_slide_dir = root / "malignant" / "other"
    insufficient_slide_dir = root / "insufficient" 
    proliferative_slide_dir = root / "other_benign" / "proliferative"
    secretory_slide_dir = root / "other_benign" / "secretory"
    menstrual_slide_dir = root / "other_benign" / "menstrual"
    innactive_atrophic_atrophic_slide_dir = root / "other_benign" / "innactive_atrophic"
    Hormonal_slide_dir = root / "other_benign" / "hormonal"
    
    #annotation directories
    adenocarcinoma_anno_dir = annotations_dir / "malignant" / "adenocarcinoma"
    carcinosarcoma_anno_dir = annotations_dir / "malignant" / "carcinosarcoma"
    sarcoma_anno_dir = annotations_dir / "malignant" / "sarcoma"
    hyperplasia_anno_dir = annotations_dir / "malignant" / "hyperplasia"
    other_anno_dir = annotations_dir / "malignant" / "other"
    insufficient_anno_dir = annotations_dir / "insufficient" 
    proliferative_anno_dir = annotations_dir / "other_benign" / "proliferative"
    secretory_anno_dir = annotations_dir / "other_benign" / "secretory"
    menstrual_anno_dir = annotations_dir / "other_benign" / "menstrual"
    innactive_atrophic_atrophic_anno_dir = annotations_dir / "other_benign" / "innactive_atrophic"
    Hormonal_anno_dir = annotations_dir / "other_benign" / "hormonal"

    #slide paths
    adenocarcinoma_slide_paths = sorted([p.relative_to(root) for p in adenocarcinoma_slide_dir.glob("*.isyntax")])
    carcinosarcoma_slide_paths = sorted([p.relative_to(root) for p in carcinosarcoma_slide_dir.glob("*.isyntax")])
    sarcoma_slide_paths = sorted([p.relative_to(root) for p in sarcoma_slide_dir.glob("*.isyntax")])
    hyperplasia_slide_paths = sorted([p.relative_to(root) for p in hyperplasia_slide_dir.glob("*.isyntax")])
    other_slide_paths = sorted([p.relative_to(root) for p in other_slide_dir.glob("*.isyntax")])
    insufficient_slide_paths = sorted([p.relative_to(root) for p in insufficient_slide_dir.glob("*.isyntax")])
    proliferative_slide_paths = sorted([p.relative_to(root) for p in proliferative_slide_dir.glob("*.isyntax")])
    secretory_slide_paths = sorted([p.relative_to(root) for p in secretory_slide_dir.glob("*.isyntax")])
    menstrual_slide_paths = sorted([p.relative_to(root) for p in  menstrual_slide_dir.glob("*.isyntax")])
    innactive_atrophic_slide_paths = sorted([p.relative_to(root) for p in innactive_atrophic_slide_dir.glob("*.isyntax")])
    hormonal_slide_paths = sorted([p.relative_to(root) for p in hormonal_slide_dir.glob("*.isyntax")])

    #annotation paths
    adenocarcinoma_anno_paths = sorted([p.relative_to(root) for p in adenocarcinoma_anno_dir.glob("*.txt")])
    carcinosarcoma_anno_paths = sorted([p.relative_to(root) for p in carcinosarcoma_anno_dir.glob("*.txt")])
    sarcoma_anno_paths = sorted([p.relative_to(root) for p in sarcoma_anno_dir.glob("*.txt")])
    hyperplasia_anno_paths = sorted([p.relative_to(root) for p in hyperplasia_anno_dir.glob("*.txt")])
    other_anno_paths = sorted([p.relative_to(root) for p in other_anno_dir.glob("*.txt")])
    insufficient_anno_paths = sorted([p.relative_to(root) for p in insufficient_anno_dir.glob("*.txt")])
    proliferative_anno_paths = sorted([p.relative_to(root) for p in proliferative_anno_dir.glob("*.txt")])
    secretory_anno_paths = sorted([p.relative_to(root) for p in secretory_anno_dir.glob("*.txt")])
    menstrual_anno_paths = sorted([p.relative_to(root) for p in  menstrual_anno_dir.glob("*.txt")])
    innactive_atrophic_anno_paths = sorted([p.relative_to(root) for p in innactive_atrophic_anno_dir.glob("*.txt")])
    hormonal_anno_paths = sorted([p.relative_to(root) for p in Hormonal_anno_dir.glob("*.txt")])

    df = pd.DataFrame()
    df["slide"] = adenocarcinoma_slide_paths + carcinosarcoma_slide_paths + sarcoma_slide_paths + hyperplasia_slide_paths + other_slide_paths + insufficient_slide_paths + proliferative_slide_paths + \
                secretory_slide_paths +  menstrual_slide_paths + innactive_atrophic_slide_paths + hormonal_slide_paths
    
    df["annotations"] = adenocarcinoma_anno_paths + carcinosarcoma_anno_paths + sarcoma_anno_paths + hyperplasia_anno_paths + other_anno_paths + insufficient_anno_paths + proliferative_anno_paths + \
                secretory_anno_paths +  menstrual_anno_paths + innactive_atrophic_anno_paths + hormonal_anno_paths

    df["label"] = ['malignant'] * len(adenocarcinoma_slide_paths) + ['malignant'] * len(carcinosarcoma_slide_paths) + ['malignant'] * len(sarcoma_slide_paths) + ['malignant'] * len(hyperplasia_slide_paths) + ['malignant'] *  len(other_slide_paths) + ['insufficient'] * len(insufficient_slide_paths) + ['other_benign'] * len(proliferative_slide_paths) + ['other_benign'] * len(secretory_slide_paths) +   ['other_benign'] * len(menstrual_slide_paths) +  ['other_benign'] * len(innactive_atrophic_slide_paths) + ['other_benign'] * len(hormonal_slide_paths)



    df["tags"] = ['adenocarcinoma'] * len(adenocarcinoma_slide_paths) + ['carcinosarcoma'] * len(carcinosarcoma_slide_paths) + ['sarcoma'] * len(sarcoma_slide_paths) + ['hyperplasia'] * len(hyperplasia_slide_paths) + ['other'] *  len(other_slide_paths) + ['insufficient'] * len(insufficient_slide_paths) + ['proliferative'] * len(proliferative_slide_paths) + ['secretory'] * len(secretory_slide_paths) +   ['menstrual'] * len(menstrual_slide_paths) +  ['innactive_atrophic'] * len(innactive_atrophic_slide_paths) + ['hormonal'] * len(hormonal_slide_paths)


    
    return Cervical(root, df)



def testing():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for test dataset.

    Returns:
        DataFrame (pd.DataFrame): Test data frame
    """
    
     # set up the paths to the slides and annotations
    root = project_root() / "data" / "endometrial" / "raw" / "testing"
    annotations_dir = root / "annotations"

    #slide directories    
    adenocarcinoma_slide_dir = root / "malignant" / "adenocarcinoma"
    carcinosarcoma_slide_dir = root / "malignant" / "carcinosarcoma"
    sarcoma_slide_dir = root / "malignant" / "sarcoma"
    hyperplasia_slide_dir = root / "malignant" / "hyperplasia"
    other_slide_dir = root / "malignant" / "other"
    insufficient_slide_dir = root / "insufficient"
    proliferative_slide_dir = root / "other_benign" / "proliferative"
    secretory_slide_dir = root / "other_benign" / "secretory"
    menstrual_slide_dir = root / "other_benign" / "menstrual"
    innactive_atrophic_slide_dir = root / "other_benign" / "innactive_atrophic"
    Hormonal_slide_dir = root / "other_benign" / "hormonal"

    #annotation directories
    adenocarcinoma_anno_dir = annotations_dir / "malignant" / "adenocarcinoma"
    carcinosarcoma_anno_dir = annotations_dir / "malignant" / "carcinosarcoma"
    sarcoma_anno_dir = annotations_dir / "malignant" / "sarcoma"
    hyperplasia_anno_dir = annotations_dir / "malignant" / "hyperplasia"
    other_anno_dir = annotations_dir / "malignant" / "other"
    insufficient_anno_dir = annotations_dir / "insufficient"
    proliferative_anno_dir = annotations_dir / "other_benign" / "proliferative"
    secretory_anno_dir = annotations_dir / "other_benign" / "secretory"
    menstrual_anno_dir = annotations_dir / "other_benign" / "menstrual"
    innactive_atrophic_atrophic_anno_dir = annotations_dir / "other_benign" / "innactive_atrophic"
    Hormonal_anno_dir = annotations_dir / "other_benign" / "Hormonal"

    #slide paths
    adenocarcinoma_slide_paths = sorted([p.relative_to(root) for p in adenocarcinoma_slide_dir.glob("*.isyntax")])
    carcinosarcoma_slide_paths = sorted([p.relative_to(root) for p in carcinosarcoma_slide_dir.glob("*.isyntax")])
    sarcoma_slide_paths = sorted([p.relative_to(root) for p in sarcoma_slide_dir.glob("*.isyntax")])
    hyperplasia_slide_paths = sorted([p.relative_to(root) for p in hyperplasia_slide_dir.glob("*.isyntax")])
    other_slide_paths = sorted([p.relative_to(root) for p in other_slide_dir.glob("*.isyntax")])
    insufficient_slide_paths = sorted([p.relative_to(root) for p in insufficient_slide_dir.glob("*.isyntax")])
    proliferative_slide_paths = sorted([p.relative_to(root) for p in proliferative_slide_dir.glob("*.isyntax")])
    secretory_slide_paths = sorted([p.relative_to(root) for p in secretory_slide_dir.glob("*.isyntax")])
    menstrual_slide_paths = sorted([p.relative_to(root) for p in  menstrual_slide_dir.glob("*.isyntax")])
    innactive_atrophic_slide_paths = sorted([p.relative_to(root) for p in innactive_atrophic_slide_dir.glob("*.isyntax")])
    hormonal_slide_paths = sorted([p.relative_to(root) for p in hormonal_slide_dir.glob("*.isyntax")])


    #annotation paths
    adenocarcinoma_anno_paths = sorted([p.relative_to(root) for p in adenocarcinoma_anno_dir.glob("*.txt")])
    carcinosarcoma_anno_paths = sorted([p.relative_to(root) for p in carcinosarcoma_anno_dir.glob("*.txt")])
    sarcoma_anno_paths = sorted([p.relative_to(root) for p in sarcoma_anno_dir.glob("*.txt")])
    hyperplasia_anno_paths = sorted([p.relative_to(root) for p in hyperplasia_anno_dir.glob("*.txt")])
    other_anno_paths = sorted([p.relative_to(root) for p in other_anno_dir.glob("*.txt")])
    insufficient_anno_paths = sorted([p.relative_to(root) for p in insufficient_anno_dir.glob("*.txt")])
    proliferative_anno_paths = sorted([p.relative_to(root) for p in proliferative_anno_dir.glob("*.txt")])
    secretory_anno_paths = sorted([p.relative_to(root) for p in secretory_anno_dir.glob("*.txt")])
    menstrual_anno_paths = sorted([p.relative_to(root) for p in  menstrual_anno_dir.glob("*.txt")])
    innactive_atrophic_anno_paths = sorted([p.relative_to(root) for p in innactive_atrophic_anno_dir.glob("*.txt")])
    hormonal_anno_paths = sorted([p.relative_to(root) for p in Hormonal_anno_dir.glob("*.txt")])

    df = pd.DataFrame()
    df["slide"] = adenocarcinoma_slide_paths + carcinosarcoma_slide_paths + sarcoma_slide_paths + hyperplasia_slide_paths + other_slide_paths + insufficient_slide_paths + proliferative_slide_paths + \
                secretory_slide_paths +  menstrual_slide_paths + innactive_atrophic_slide_paths + hormonal_slide_paths

    df["annotations"] = adenocarcinoma_anno_paths + carcinosarcoma_anno_paths + sarcoma_anno_paths + hyperplasia_anno_paths + other_anno_paths + insufficient_anno_paths + proliferative_anno_paths + \
                secretory_anno_paths +  menstrual_anno_paths + innactive_atrophic_anno_paths + hormonal_anno_paths

    df["label"] = ['malignant'] * len(adenocarcinoma_slide_paths) + ['malignant'] * len(carcinosarcoma_slide_paths) + ['malignant'] * len(sarcoma_slide_paths) + ['malignant'] * len(hyperplasia_slide_paths) + ['malignant'] *  len(other_slide_paths) + ['insufficient'] * len(insufficient_slide_paths) + ['other_benign'] * len(proliferative_slide_paths) + ['other_benign'] * len(secretory_slide_paths) +   ['othe_-benign'] * len(menstrual_slide_paths) +  ['other_benign'] * len(innactive_atrophic_slide_paths) + ['other_benign'] * len(hormonal_slide_paths)



    df["tags"] = ['denocarcinoma'] * len(adenocarcinoma_slide_paths) + ['carcinosarcoma'] * len(carcinosarcoma_slide_paths) + ['sarcoma'] * len(sarcoma_slide_paths) + ['hyperplasia'] * len(hyperplasia_slide_paths) + ['other'] *  len(other_slide_paths) + ['insufficient'] * len(insufficient_slide_paths) + ['proliferative'] * len(proliferative_slide_paths) + ['secretory'] * len(secretory_slide_paths) +   ['menstrual'] * len(menstrual_slide_paths) +  ['innactive_atrophic'] * len(innactive_atrophic_slide_paths) + ['hormonal'] * len(hormonal_slide_paths)


    return Endometrial(root, df)




