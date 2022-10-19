import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import measure


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    mask_labels = np.unique(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275 / (resolution * pow(2, level))
    print(len(properties), len(mask_labels))
    for i in range(0, len(properties)):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(mask_labels[i + 1])
    return Isolated_Tumor_Cells


def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image

    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made

    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections

        TP_probs:   A list containing the probabilities of the True positive detections

        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)

        detection_summary:   A python dictionary object with keys that are the labels
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate].
        Lesions that are missed by the algorithm have an empty value.

        FP_summary:   A python dictionary object with keys that represent the
        false positive finding number and values that contain detection
        details [confidence score, X-coordinate, Y-coordinate].
    """

    max_label = np.amax(evaluation_mask)
    print("max_label", max_label)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1, max_label + 1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []

    FP_counter = 0
    if (is_tumor):
        for i in range(0, len(Xcorr)):
            HittedLabel = evaluation_mask[int(Ycorr[i] / pow(2, level)), int(Xcorr[i] / pow(2, level))]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter += 1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i] > TP_probs[HittedLabel - 1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel - 1] = Probs[i]
    else:
        for i in range(0, len(Xcorr)):
            FP_probs.append(Probs[i])
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
            FP_counter += 1

    num_of_tumors = max_label - len(Isolated_Tumor_Cells);
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary


def computeFROC_new(FP_list, TP_list, total_tumor, n_imgs):
    """Generates the data required for plotting the FROC curve

    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    total_FPs, total_TPs = [], []
    all_probs = sorted(set(FP_list + TP_list))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(FP_list) >= Thresh).sum())
        total_TPs.append((np.asarray(TP_list) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs) / float(n_imgs)
    total_sensitivity = np.asarray(total_TPs) / float(total_tumor)
    return total_FPs, total_sensitivity


def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve
    
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
         
    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))
    return  total_FPs, total_sensitivity



def calc_average_froc(total_FPs, total_sensitivity):
    output = np.column_stack((total_FPs, total_sensitivity))
    FPPI_vals = [8, 4, 2, 1, 0.5, 0.25]
    total_sens = 0
    for fppi in FPPI_vals:
        fppi_mask = total_FPs <= fppi
        fppi_mask_list = fppi_mask.tolist()
        index_out = fppi_mask_list.index(True)
        sens_out = output[index_out, 1]
        total_sens += sens_out

    average_sens = total_sens / 6
    return(output, average_sens)


def evaluate_froc(mask_dir: Path, lesions: pd.DataFrame,
                  evaluation_mask_level: int, level_zero_resolution: float) -> pd.DataFrame:
    result_file_list = lesions.filename.unique()
    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    FP_list = []
    TP_list = []
    total_tumor = 0
    n_imgs = 0
    n_tumor_in = 0
    caseNum = 0
    for case in result_file_list:
        lesions_slide = lesions[lesions.filename == case]
        lesions_slide = lesions_slide[lesions_slide.pixels > 0]
        n_tumor_in += lesions_slide.shape[0]
        Probs = lesions_slide.prob_score.to_list()
        Xcorr = lesions_slide.centre_x.astype(int).to_list()
        Ycorr = lesions_slide.centre_y.astype(int).to_list()
        # test files are named test convert name to tumor so names match
        #if case[0:4] == 'test':
        #    case = 'tumor' + case[4:]
        file_exists = Path(os.path.join(mask_dir, case) + '.png').is_file()
        print(Path(os.path.join(mask_dir, case) + '.png'), file_exists)
        if (file_exists):
            evaluation_mask = cv2.imread(os.path.join(mask_dir, case) + '.png', 0)
            ITC_labels = computeITCList(evaluation_mask, level_zero_resolution, evaluation_mask_level)
        else:
            evaluation_mask = 0
            ITC_labels = []

        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        results = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, file_exists, evaluation_mask, ITC_labels, evaluation_mask_level)
        FP_list = FP_list + list(results[0])
        TP_list = TP_list + list(results[1])
        print("results",results[2])
        total_tumor += results[2]
        n_imgs += 1
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], \
        FP_summary[1][caseNum] = results

        caseNum += 1

    # Compute FROC curve
    total_FPs, total_sensitivity = computeFROC_new(FP_list, TP_list, total_tumor, n_imgs)
    # combine into array and calcuate average
    output, average_sens = calc_average_froc(total_FPs, total_sensitivity)
    output = pd.DataFrame(output, columns=['total_FPs', 'total_sensitivity'])

    return output, average_sens