import os
import pandas as pd
from pathlib import Path
from repath.utils.paths import project_root

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
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path, label: str) -> AnnotationSet:
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
    root = project_root() / "iCAIRD"  
    annotations_dir = root / "annotations"
    
    csv_path = root / "iCAIRD_endo_Data.csv"

    endo_df = pd.read_csv(csv_path)
    endo_train = endo_df[endo_df['train/valid/test']=='train']

    annotations = endo_train['Image Filename'].replace({'isyntax':'txt'}, regex=True)

    df = pd.DataFrame()
    df["slide"] =  str(os.path.relpath(root)) + '/' + endo_train['Image Filename']
    df["label"] = endo_train['Category']
    df["annotation"] = str(os.path.relpath(annotations_dir)) + '/' + annotations
    df["tags"] = endo_train['subCategory']
   
    return Endometrial(root, df)


def testing():
     """ Generated a data-frame of slide_path, annotation_path, label and tags for test dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "iCAIRD"  
    annotations_dir = root / "annotations"
    

    csv_path = root / "iCAIRD_Endometrial_Data.csv"

    endo_df = pd.read_csv(csv_path)
    endo_test= endo_df[endo_df['train/valid/test']=='test']

    annotations =  endo_test['Image Filename'].replace({'isyntax':'txt'}, regex=True)

    df = pd.DataFrame()
    df["slide"] =  str(os.path.relpath(root)) + '/' + endo_test['Image Filename']
    df["label"] = endo_test['Category']
    df["annotation"] = str(os.path.relpath(annotations_dir)) + '/' + annotations
    df["tags"] = endo_test['subCategory']
   
    return Endometrial(root, df)
