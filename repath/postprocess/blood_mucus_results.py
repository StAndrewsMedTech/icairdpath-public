from pathlib import Path
from typing import List

import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd

from repath.data.datasets import Dataset
from repath.utils.metrics import conf_mat_plot_heatmap


def pool_blood_mucus(A, kernel_size, stride, padding):
    """
    2D Pooling

    Taken from https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy

    applies blood mucus patching function over kernel area

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
    """
    def blood_mucus_patching(patch):
        # assume background = 0, tissue = 1, everything else is either blood or mucus
        # if all values in patch are zero then returns zero
        # if any value in a patch is equal to one returns one
        # else returns 2 for blood or mucus
        if np.sum(patch) == 0:
            lab_out = 0
        elif np.sum(np.equal(patch, 1)) > 0:
            lab_out = 1
        else:
            lab_out = 2

        return lab_out

    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )

    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
    )

    A_w = A_w.reshape(-1, *kernel_size)

    A_flat = np.reshape(A_w, (A_w.shape[0], kernel_size[0]**2))
    pooled_vals = np.apply_along_axis(blood_mucus_patching, 1, A_flat)
    output_new = pooled_vals.reshape(output_shape)

    return output_new


def get_annot_areas(dset: Dataset, level: int) -> List[np.ndarray]:
    """ Creates a list of thumbnails that are binary image with just the area annotated.

    Validation annotations are very detailed in a small area. This function returns a 
    thumbnail at a given image that is black but with a white rectangle showing the
    area that has been annotated in detail by the pathologists. This can be used to 
    cut down results to just areas that are fully annotated.

    Args:
        dset (Dataset): the dataset containing slides and annotation paths
        level (int):  the level to create thumbnails at

    Returns:
        A list containing the annotated areas thumbnails as numpy arrays
    """
    annotz = []

    for dd in dset:
        slide_path, annot_path, _, _ = dd
        with dset.slide_cls(dset.root / slide_path) as slide:
            thumb = slide.get_thumbnail(level)
            annotations = dset.load_annotated_area(annot_path)
            if len(annotations.annotations) > 0:
                annot = annotations.render(thumb.shape[0:2], 2**level)
            else:
                annot = np.ones(thumb.shape, dtype=np.uint8) * 5
            annotz.append(annot)

    return annotz


def calc_confusion_mat_3class(true_img: np.ndarray, pred_img: np.ndarray) -> List[int]:
    """ Calculates a vector of confusion matrix values when input is 3 classes

    Calculates values for a confusion matrix for two numpy arrays of the same
    size where input values are equal to 0, 1, or 2.
    Input an array of annotations from pathologists with classes 0, 1 and 2 
    and an array of predicted labels of the same class, calculates values for
    the confusion matrices between the images.

    Args:
        true_img (numpy array): true labels image, annotations from pathologist
        pred_img (numpy array): predicted labels image from algorithm being tested

    Returns:
        A list of 9 values in order (true=0,pred=0), (true=0,pred=1), (true=0,pred=2),
        (true=1,pred=0), (true=1,pred=1), (true=1,pred=2), (true=2,pred=0), 
        (true=2,pred=1), (true=2,pred=2)
    """
    tru0_pred0 = np.sum(np.logical_and(true_img == 0, pred_img == 0))
    tru0_pred1 = np.sum(np.logical_and(true_img == 0, pred_img == 1))
    tru0_pred2 = np.sum(np.logical_and(true_img == 0, pred_img == 2))
    tru1_pred0 = np.sum(np.logical_and(true_img == 1, pred_img == 0))
    tru1_pred1 = np.sum(np.logical_and(true_img == 1, pred_img == 1))
    tru1_pred2 = np.sum(np.logical_and(true_img == 1, pred_img == 2))
    tru2_pred0 = np.sum(np.logical_and(true_img == 2, pred_img == 0))
    tru2_pred1 = np.sum(np.logical_and(true_img == 2, pred_img == 1))
    tru2_pred2 = np.sum(np.logical_and(true_img == 2, pred_img == 2))

    cm = [tru0_pred0, tru0_pred1, tru0_pred2, tru1_pred0, tru1_pred1, tru1_pred2, tru2_pred0, tru2_pred1, tru2_pred2]
    return cm


def calc_confusion_mat_2class(true_img, pred_img):
    """ Calculates a vector of confusion matrix values when input is 2 classes

    Calculates values for a confusion matrix for two numpy arrays of the same
    size where input values are equal to 0, 1.
    Input an array of annotations from pathologists with classes 0, 1 
    and an array of predicted labels of the same class, calculates values for
    the confusion matrices between the images.

    Args:
        true_img (numpy array): true labels image, annotations from pathologist
        pred_img (numpy array): predicted labels image from algorithm being tested

    Returns:
        A list of 4 values in order (true=0,pred=0), (true=0,pred=1), 
        (true=1,pred=0), (true=1,pred=1)
    """
    tru0_pred0 = np.sum(np.logical_and(true_img == 0, np.logical_not(pred_img == 1)))
    tru0_pred1 = np.sum(np.logical_and(true_img == 0, pred_img == 1))
    tru1_pred0 = np.sum(np.logical_and(true_img == 1, np.logical_not(pred_img == 1)))
    tru1_pred1 = np.sum(np.logical_and(true_img == 1, pred_img == 1))
    cm2 = [tru0_pred0, tru0_pred1, tru1_pred0, tru1_pred1]
    return cm2


def save_confusion_mat(cm: np.ndarray, save_dir: Path, exp_name: str):
    """ Output results of confusion matrices as csv and image

    For a vector of confusion matrix values of either 3 class or 2 class
    saves to a directory a csv of the vector and create a confusion matrix image
    which is output as a png

    Args:
        cm (numpy array): confusion matrix values
        save_dir (pathlib path): output directory
        exp_name (string): a string used in title of confusion matrix image

    """
    cm_pd = np.array(cm, dtype=np.uint)
    cm_pd = np.expand_dims(cm_pd, axis=0)
    title_cm = 'confusion matrix blood mucus ' + exp_name
    if len(cm) == 4:
        cm_pd = pd.DataFrame(cm_pd, columns=['tn', 'fp', 'fn', 'tp'])
        cm_pd.to_csv(save_dir / 'confusion_matrix_2class.csv')
        display_labels = ['not tissue', 'tissue']
        cm_np = np.reshape(cm, (2,2))
        cm_np = np.array(cm_np,dtype=np.uint)
        cm_out = conf_mat_plot_heatmap(cm_np, display_labels, title_cm)
        cm_out.get_figure().savefig(save_dir / 'confusion_matrix_2class.png')
    else:
        cm_pd = pd.DataFrame(cm_pd, columns=['true0pred0', 'true0pred1', 'true0pred2', 'true1pred0', 'true1pred1', 'true1pred2', 'true2pred0', 'true2pred1', 'true2pred2'])
        cm_pd.to_csv(save_dir / 'confusion_matrix_3class.csv')
        display_labels = ['background', 'tissue', 'blood or mucus']
        cm_np = np.reshape(cm, (3,3))
        cm_np = np.array(cm_np,dtype=np.uint)
        cm_out = conf_mat_plot_heatmap(cm_np, display_labels, title_cm)
        cm_out.get_figure().savefig(save_dir / 'confusion_matrix_3class.png')
