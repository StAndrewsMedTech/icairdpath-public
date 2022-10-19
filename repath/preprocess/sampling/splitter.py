from copy import deepcopy
import numpy as np
import pandas as pd
import random
from abc import ABCMeta, abstractmethod
from typing import Tuple

from repath.preprocess.patching import SlidesIndex


"""
Global stuff
"""

def get_subset_of_dataset(slides_index_to_match, whole_dataset):
    
    # get valid slides in valid slide index (the list of split between valid and train)
    slides_to_match = [pat.slide_path for pat in slides_index_to_match.patches]
    # create new dataset initially with both training and valid
    dataset_subset = deepcopy(whole_dataset)
    # chop new dataset down to be just slide in valid set
    mask = [str(sl) in str(slides_to_match) for sl in dataset_subset.paths.slide]
    dataset_subset.paths = dataset_subset.paths[mask]

    return dataset_subset


def get_subset_of_dataset_reindex(slides_index_to_match, whole_dataset):
    
    # get valid slides in valid slide index (the list of split between valid and train)
    slides_to_match = [pat.slide_path for pat in slides_index_to_match.patches]
    # create new dataset initially with both training and valid
    dataset_subset = deepcopy(whole_dataset)
    # chop new dataset down to be just slide in valid set
    mask = [str(sl) in str(slides_to_match) for sl in dataset_subset.paths.slide]
    dataset_subset.paths = dataset_subset.paths[mask]
    dataset_subset.paths = dataset_subset.paths.reset_index(drop=True)


    return dataset_subset


def split_cervical_tags(index: SlidesIndex):
    train_mask = [idx for idx, tag in enumerate(index.dataset.paths.tags) if tag.split(';')[1] == 'train']
    valid_mask = [idx for idx, tag in enumerate(index.dataset.paths.tags) if tag.split(';')[1] == 'valid'] 

    train_index = [index[idx] for idx in train_mask]
    valid_index = [index[idx] for idx in valid_mask]

    return SlidesIndex(index.dataset, train_index), SlidesIndex(index.dataset, valid_index)


def split_endometrial(index: SlidesIndex, train_percent: float) -> Tuple[SlidesIndex, SlidesIndex]:
    print("Splitting Endometrial")

    # get summaries of total patch classes for each slide
    summaries = index.summary()

    # count numbers of each slide label
    slide_label_dict = index.dataset.slide_labels
    slide_numbers_total = {}
    slide_numbers_train = {}
    slide_numbers_valid = {}
    for key in slide_label_dict.keys():
        n_slides = np.sum(summaries.slide_label == key)
        slide_numbers_total[key] = n_slides
        slide_numbers_train[key] = int(n_slides * train_percent)
        slide_numbers_valid[key] = n_slides - int(n_slides * train_percent)

        # divide each class of slide according to train percent
        slide_numbers = np.array(list(range(len(index.dataset))))

        # divide each class of slide according to train percent
        slide_numbers = np.array(list(range(len(index.dataset))))

        malignant_slides = slide_numbers[index.dataset.paths.label == 'malignant']
        #print("malignant_slides", malignant_slides)
        malignant_slides_train = np.random.choice(malignant_slides, slide_numbers_train['malignant'], replace=False)

        insufficient_slides = slide_numbers[index.dataset.paths.label == 'insufficient']
        #print("insufficient_slides", insufficient_slides)
        insufficient_slides_train = np.random.choice(insufficient_slides, slide_numbers_train['insufficient'], replace=False)

        other_benign_slides = slide_numbers[index.dataset.paths.label == 'other_benign']
        #print("other_benign_slides", other_benign_slides)
        other_benign_slides_train = np.random.choice(other_benign_slides, slide_numbers_train['other_benign'], replace=False)
        
        train_slide_numbers = np.hstack((malignant_slides_train, insufficient_slides_train, other_benign_slide_train))
        #print("train_slide_numbers", train_slide_numbers)

        valid_slide_numbers = [item for item in slide_numbers if item not in train_slide_numbers]
        #print("valid_slide_numbers", valid_slide_numbers)

        train_index = [index[idx] for idx in train_slide_numbers]
        valid_index = [index[idx] for idx in valid_slide_numbers]

        # sum total number of tumor patches in each set
        train_summaries = summaries.iloc[train_slide_numbers, :]
        valid_summaries = summaries.iloc[valid_slide_numbers, :]

        # sum total number of tumor patches in each set
        train_summaries = summaries.iloc[train_slide_numbers, :]
        valid_summaries = summaries.iloc[valid_slide_numbers, :]

        total_malignant_patches = np.sum(summaries.malignant)
        total_insufficient_patches = np.sum(summaries.insufficient)
        total_other_benign_patches = np.sum(summaries.other_benign)

        train_malignant_patches = np.sum(train_summaries.malignant)
        train_insufficient_patches = np.sum(train_summaries.insufficient)
        train_other_benign_patches = np.sum(train_summaries.other_benign)


        valid_malignant_patches = np.sum(valid_summaries.malignant)
        valid_insufficient_patches = np.sum(valid_summaries.insufficient)
        valid_other_benign_patches = np.sum(valid_summaries.other_benign)

         # some sort of check and adjustment if massively different to train_percent
        # training tumor patches should be train percent or close to it
        # close is defined as plus or minus 5% of train percent
        # so if train percent is 70, 10% of 70 is 7 so close is 70 plus or minus 3.5
        
        #check for malignant patches
        train_percent_5 = train_percent * 0.05
        train_low = (train_percent - train_percent_5) * total_malignant_patches
        train_high = (train_percent + train_percent_5) * total_malignant_patches
        if not (train_low <= train_malignant_patches <= train_high):
            print("Warning: splitting on slides has not got a balanced split of patches ")
        if train_malignant_patches == 0:
            train_slide_numbers = train_slide_numbers + [malignant_slides[0]]
            valid_slide_numbers = slide_numbers[~train_slide_numbers]
            train_index = index[train_slide_numbers]
            valid_index = index[valid_slide_numbers]

        #check for insufficient patches
        train_percent_5 = train_percent * 0.05
        train_low = (train_percent - train_percent_5) * total_insufficient_patches
        train_high = (train_percent + train_percent_5) * total_insufficient_patches
        if not (train_low <= train_insufficient_patches <= train_high):
            print("Warning: splitting on slides has not got a balanced split of patches ")
        if train_malignant_patches == 0:
            train_slide_numbers = train_slide_numbers + [insufficient_slides[0]]
            valid_slide_numbers = slide_numbers[~train_slide_numbers]
            train_index = index[train_slide_numbers]
            valid_index = index[valid_slide_numbers]

        #check for other_benign patches
        train_percent_5 = train_percent * 0.05
        train_low = (train_percent - train_percent_5) * total_other_benign_patches
        train_high = (train_percent + train_percent_5) * total_other_benign_patches
        if not (train_low <= train_other_benign_patches <= train_high):
            print("Warning: splitting on slides has not got a balanced split of patches ")
        if train_other_benign_patches == 0:
            train_slide_numbers = train_slide_numbers + [other_benign_slides[0]]
            valid_slide_numbers = slide_numbers[~train_slide_numbers]
            train_index = index[train_slide_numbers]
            valid_index = index[valid_slide_numbers]


        return SlidesIndex(index.dataset, train_index), SlidesIndex(index.dataset, valid_index)



def split_cervical(index: SlidesIndex, train_percent: float) -> Tuple[SlidesIndex, SlidesIndex]:
    print("Splitting Cervical")

    # get summaries of total patch classes for each slide
    summaries = index.summary()

    # count numbers of each slide label
    slide_label_dict = index.dataset.slide_labels
    slide_numbers_total = {}
    slide_numbers_train = {}
    slide_numbers_valid = {}
    for key in slide_label_dict.keys():
        n_slides = np.sum(summaries.slide_label == key)
        slide_numbers_total[key] = n_slides
        slide_numbers_train[key] = int(n_slides * train_percent)
        slide_numbers_valid[key] = n_slides - int(n_slides * train_percent)

        # divide each class of slide according to train percent
        slide_numbers = np.array(list(range(len(index.dataset))))
        
        low_grade_slides = slide_numbers[index.dataset.paths.label == 'low_grade']
        #print("low_grade_slides", low_grade_slides)
        low_grade_slides_train = np.random.choice(low_grade_slides, slide_numbers_train['low_grade'], replace=False)
        
        high_grade_slides = slide_numbers[index.dataset.paths.label == 'high_grade']
        #print("high_grade_slides", high_grade_slides)
        high_grade_slides_train = np.random.choice(high_grade_slides, slide_numbers_train['high_grade'], replace=False)

        malignant_slides = slide_numbers[index.dataset.paths.label == 'malignant']
        #print("malignant_slides", malignant_slides)
        malignant_slides_train = np.random.choice(malignant_slides, slide_numbers_train['malignant'], replace=False)

        normal_inflammation_slides = slide_numbers[index.dataset.paths.label == 'normal_inflammation']
        #print("normal_inflammation_slides", normal_inflammation_slides)
        normal_inflammation_slides_train = np.random.choice(normal_inflammation_slides, slide_numbers_train['normal_inflammation'], replace=False)
        
        train_slide_numbers = np.hstack((low_grade_slides_train, high_grade_slides_train, malignant_slide_train, normal_inflammation_slide_train))
        #print("train_slide_numbers", train_slide_numbers)

        valid_slide_numbers = [item for item in slide_numbers if item not in train_slide_numbers]
        #print("valid_slide_numbers", valid_slide_numbers)

        train_index = [index[idx] for idx in train_slide_numbers]
        valid_index = [index[idx] for idx in valid_slide_numbers]

        # sum total number of tumor patches in each set
        train_summaries = summaries.iloc[train_slide_numbers, :]
        valid_summaries = summaries.iloc[valid_slide_numbers, :]
        
        total_low_grade_patches = np.sum(summaries.low_grade)
        total_high_grade_patches = np.sum(summaries.high_grade)
        total_malignant_patches = np.sum(summaries.malignant)
        total_normal_inflammation_patches = np.sum(summaries.normal_inflammation)

        train_low_grade_patches = np.sum(train_summaries.low_grade)
        train_high_grade_patches = np.sum(train_summaries.high_grade)
        train_malignant_patches = np.sum(train_summaries.malignant)
        train_normal_inflammation_patches = np.sum(train_summaries.normal_inflammation)

        
        valid_low_grade_patches = np.sum(valid_summaries.low_grade)
        valid_high_grade_patches = np.sum(valid_summaries.high_grade)
        valid_malignant_patches = np.sum(valid_summaries.malignant)
        valid_normal_inflammation_patches = np.sum(valid_summaries.normal_inflammation)
        
        # some sort of check and adjustment if massively different to train_percent
        # training main category patches should be train percent or close to it
        # close is defined as plus or minus 5% of train percent
        # so if train percent is 70, 10% of 70 is 7 so close is 70 plus or minus 3.5
        
        #check for low_grade patches
        train_percent_5 = train_percent * 0.05
        train_low = (train_percent - train_percent_5) * total_low_grade_patches
        train_high = (train_percent + train_percent_5) * total_low_grade_patches
        if not (train_low <= train_low_grade_patches <= train_high):
            print("Warning: splitting on slides has not got a balanced split of patches ")
        if train_low_grade_patches == 0:
            train_slide_numbers = train_slide_numbers + [low_grade_slides[0]]
            valid_slide_numbers = slide_numbers[~train_slide_numbers]
            train_index = index[train_slide_numbers]
            valid_index = index[valid_slide_numbers]
        
        #check for high grade patches
        train_percent_5 = train_percent * 0.05
        train_low = (train_percent - train_percent_5) * total_high_grade_patches
        train_high = (train_percent + train_percent_5) * total_high_grade_patches

        if not (train_low <= train_high_grade_patches <= train_high):
            print("Warning: splitting on slides has not got a balanced split of patches ")
        if train_high_grade_patches == 0:
            train_slide_numbers = train_slide_numbers + [high_grade_slides[0]]
            valid_slide_numbers = slide_numbers[~train_slide_numbers]
            train_index = index[train_slide_numbers]
            valid_index = index[valid_slide_numbers]

        #check for malignant patches
        train_percent_5 = train_percent * 0.05
        train_low = (train_percent - train_percent_5) * total_malignant_patches
        train_high = (train_percent + train_percent_5) * total_malignant_patches

        if not (train_low <= train_malignant_patches <= train_high):
            print("Warning: splitting on slides has not got a balanced split of patches ")
        if train_malignant_patches == 0:
            train_slide_numbers = train_slide_numbers + [malignant_slides[0]]
            valid_slide_numbers = slide_numbers[~train_slide_numbers]
            train_index = index[train_slide_numbers]
            valid_index = index[valid_slide_numbers]

        #check for normal_inflammation patches
        train_percent_5 = train_percent * 0.05
        train_low = (train_percent - train_percent_5) * total_normal_inflammation_patches
        train_high = (train_percent + train_percent_5) * total_normal_inflammation_patches

        if not (train_low <= train_normal_inflammation_patches <= train_high):
            print("Warning: splitting on slides has not got a balanced split of patches ")
        if train_normal_inflammation_patches == 0:
            train_slide_numbers = train_slide_numbers + [normal_inflammation_slides[0]]
            valid_slide_numbers = slide_numbers[~train_slide_numbers]
            train_index = index[train_slide_numbers]
            valid_index = index[valid_slide_numbers]

        
        return SlidesIndex(index.dataset, train_index), SlidesIndex(index.dataset, valid_index)

        
def split_camelyon16(index: SlidesIndex, train_percent: float) -> Tuple[SlidesIndex, SlidesIndex]:
    print("Splitting Camelyon 16")

    # get summaries of total patch classes for each slide
    summaries = index.summary()

    # count numbers of each slide label
    slide_label_dict = index.dataset.slide_labels
    slide_numbers_total = {}
    slide_numbers_train = {}
    slide_numbers_valid = {}
    for key in slide_label_dict.keys():
        n_slides = np.sum(summaries.slide_label == key)
        slide_numbers_total[key] = n_slides
        slide_numbers_train[key] = int(n_slides * train_percent)
        slide_numbers_valid[key] = n_slides - int(n_slides * train_percent)

    #print("slide_numbers_train", slide_numbers_train)
    #print("slide_numbers_valid", slide_numbers_valid)

    # divide each class of slide according to train percent
    slide_numbers = np.array(list(range(len(index.dataset))))

    normal_slides = slide_numbers[index.dataset.paths.label == 'normal']
    #print("normal_slides", normal_slides)
    normal_slides_train = np.random.choice(normal_slides, slide_numbers_train['normal'], replace=False)
    #print("normal_slides_train", normal_slides_train)
    tumor_slides = slide_numbers[index.dataset.paths.label == 'tumor']
    #print("tumor_slides", tumor_slides)
    tumor_slides_train = np.random.choice(tumor_slides, slide_numbers_train['tumor'], replace=False)
    #print("tumor_slides_train", tumor_slides_train)
    train_slide_numbers = np.hstack((normal_slides_train, tumor_slides_train))
    #print("train_slide_numbers", train_slide_numbers)
    valid_slide_numbers = [item for item in slide_numbers if item not in train_slide_numbers]
    #print("valid_slide_numbers", valid_slide_numbers)
    train_index = [index[idx] for idx in train_slide_numbers]
    valid_index = [index[idx] for idx in valid_slide_numbers]

    # sum total number of tumor patches in each set
    train_summaries = summaries.iloc[train_slide_numbers, :]
    valid_summaries = summaries.iloc[valid_slide_numbers, :]
    total_tumor_patches = np.sum(summaries.tumor)
    train_tumor_patches = np.sum(train_summaries.tumor)
    valid_tumor_patches = np.sum(valid_summaries.tumor)

    # some sort of check and adjustment if massively different to train_percent
    # training tumor patches should be train percent or close to it
    # close is defined as plus or minus 5% of train percent
    # so if train percent is 70, 10% of 70 is 7 so close is 70 plus or minus 3.5
    train_percent_5 = train_percent * 0.05
    train_low = (train_percent - train_percent_5) * total_tumor_patches
    train_high = (train_percent + train_percent_5) * total_tumor_patches
    if not (train_low <= train_tumor_patches <= train_high):
        print("Warning: splitting on slides has not got a balanced split of patches ")
    if train_tumor_patches == 0:
        train_slide_numbers = train_slide_numbers + [tumor_slides[0]]
        valid_slide_numbers = slide_numbers[~train_slide_numbers]
        train_index = index[train_slide_numbers]
        valid_index = index[valid_slide_numbers]

    return SlidesIndex(index.dataset, train_index), SlidesIndex(index.dataset, valid_index)


def split_camelyon17(index: SlidesIndex, train_percent: float) -> Tuple[SlidesIndex, SlidesIndex]:

    # get summaries of total patch classes for each slide
    summaries = index.summary()

    # count numbers of each slide label
    slide_label_dict = index.dataset.slide_labels
    slide_numbers_total = {}
    slide_numbers_train = {}
    slide_numbers_valid = {}
    for key in slide_label_dict.keys():
        n_slides = np.sum(summaries.slide_label == key)
        slide_numbers_total[key] = n_slides
        slide_numbers_train[key] = int(n_slides * train_percent)
        slide_numbers_valid[key] = n_slides - int(n_slides * train_percent)

    # find annotated slides
    annotated_slides = [ps for ps in index if 'annotated' in ps.tags]
    unannotated_slides = [ps for ps in index if 'annotated' not in ps.tags]
    annotated_patients = [ps.tags[0] for ps in annotated_slides]
    all_patients = [ps.tags[0] for ps in index]

    # split patients into annotated and unannotated sets
    unique_all_patients = list(set(all_patients))
    unique_annotated_patients = list(set(annotated_patients))
    unique_unannotated_patients = [item for item in unique_all_patients if item not in unique_annotated_patients]

    # calculate numbers of annotated and unannoted patients for train set
    n_annotated_patients = len(unique_annotated_patients)
    n_train_annotated_patients = int(train_percent * n_annotated_patients)
    n_all_patients = len(unique_all_patients)
    n_train_all_patients = int(train_percent * n_all_patients)
    n_train_unannotated = n_train_all_patients - n_train_annotated_patients

    # sample annotated and unannotated training patients
    annotated_patients_train = np.random.choice(unique_annotated_patients, n_train_annotated_patients, replace=False)
    unannotated_patients_train = np.random.choice(unique_unannotated_patients, n_train_unannotated, replace=False)

    # create list of training and valid patients
    train_patients = annotated_patients_train.tolist() + unannotated_patients_train.tolist()
    valid_patients = [item for item in unique_all_patients if item not in train_patients]

    # split slides into train and valid
    train_index = SlidesIndex(index.dataset, [ps for ps in index if ps.tags[0] in train_patients])
    valid_index = SlidesIndex(index.dataset, [ps for ps in index if ps.tags[0] in valid_patients])

    # check split of patch classes
    train_summaries = train_index.summary()
    valid_summaries = valid_index.summary()
    total_tumor_patches = np.sum(summaries.tumor)
    train_tumor_patches = np.sum(train_summaries.tumor)
    valid_tumor_patches = np.sum(valid_summaries.tumor)

    # some sort of check and adjustment if massively different to train_percent
    # training tumor patches should be train percent or close to it
    # close is defined as plus or minus 5% of train percent
    # so if train percent is 70, 10% of 70 is 7 so close is 70 plus or minus 3.5
    train_percent_5 = train_percent * 0.05
    train_low = (train_percent - train_percent_5) * total_tumor_patches
    train_high = (train_percent + train_percent_5) * total_tumor_patches
    if not (train_low <= train_tumor_patches <= train_high):
        print("Warning: splitting on patients has not got a balanced split of patches")
        print(f'training tumor patches percent: {train_tumor_patches / total_tumor_patches}')

    # check split of slide classes
    slide_class_split = pd.DataFrame(index=['train', 'valid', 'total'], columns=['negative', 'itc', 'micro', 'macro'])
    for col in slide_class_split.columns:
        slide_class_split.loc['train'][col] = np.sum(train_summaries.slide_label == col)
        slide_class_split.loc['valid'][col] = np.sum(valid_summaries.slide_label == col)
        slide_class_split.loc['total'][col] = np.sum(summaries.slide_label == col)
    if (np.sum(np.less(slide_class_split.to_numpy(), 1)) > 0):
        print("Warning: splitting on patients has some slide classes in train or valid as zero")

    return train_index, valid_index

