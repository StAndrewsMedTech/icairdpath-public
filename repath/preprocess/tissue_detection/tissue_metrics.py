import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import repath.data.datasets.tissue as tissue
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.tissue_detection import TissueDetector, TissueDetectorAll
from repath.utils.export import convert_mask_to_json
from repath.utils.geometry import Size
from repath.utils.metrics import conf_mat_plot_heatmap
from repath.utils.paths import project_root


def calc_tissue_conf_mat(tissue_detector_test: TissueDetector, label: str, level_in: int, base_dir: Path) -> None:
    """
    For a tissue detector this calculates the confusion matrix compared to ground truths in tissue detection dataset.
    Writes out patch indexes for truth and tissue detection to base_dir / label 
    Writes out a png image of the confidence matrix to base_dir / label called label_confidence_matrix.png
    
    Args:
            tissue_detector_test (TissueDetector): A class of tissue detector to test
            label (str): A label to add to filenames for naming output of this experiment
            level_in (int): The level at which to carry out the tissue detection
            best_dir (path): directory to write out data

    """
    tissue_dataset = tissue.tissue()
    psize = 2 ** level_in
    patch_finder = GridPatchFinder(labels_level=level_in, patch_level=0, patch_size=psize, stride=psize, remove_background=False)
    # The tissue detector is applied over the top of the patch finder
    # tissue detector all classes everything as foreground so will not change the foreground background labels from the annotations
    tissue_detector_all = TissueDetectorAll()
    tissue_patchsets_labelled = SlidesIndex.index_dataset(tissue_dataset, tissue_detector_all, patch_finder)
    tissue_patches_labelled = CombinedIndex.for_slide_indexes([tissue_patchsets_labelled])
    
    # create blank slides with just tissue detector labels
    tissue_patchsets_detected = SlidesIndex.index_dataset(tissue_dataset, tissue_detector_test, patch_finder, notblank=False)
    tissue_patches_detected = CombinedIndex.for_slide_indexes([tissue_patchsets_detected])

    # calculate confusion matrix
    TP = np.sum(np.logical_and(tissue_patches_detected.patches_df.label, tissue_patches_labelled.patches_df.label))
    FP = np.sum(np.logical_and(tissue_patches_detected.patches_df.label, np.logical_not(tissue_patches_labelled.patches_df.label)))
    FN = np.sum(np.logical_and(np.logical_not(tissue_patches_detected.patches_df.label), tissue_patches_labelled.patches_df.label))
    TN = np.sum(np.logical_and(np.logical_not(tissue_patches_detected.patches_df.label), np.logical_not(tissue_patches_labelled.patches_df.label)))
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP) 
    confmat = np.array([[TN, FP], [FN, TP]], dtype=np.int)

    # output results
    print(f'Accuracy: {round(accuracy, 5)}, Recall: {round(recall, 5)}, Precision: {round(precision, 5)}')
    print(f'TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}')

    heatmap_title = "Tissue Detector " + label + " - Accuracy :" + str(round(accuracy, 5))
    cm_out = conf_mat_plot_heatmap(confmat, ['background', 'foreground'], heatmap_title)
    output_name = label + '_confidence_matrix.png'
    outdir = base_dir / label
    outdir.mkdir(parents=True, exist_ok=True)
    cm_out.get_figure().savefig(outdir / output_name)


def get_output_images(tissue_detector: TissueDetector, label: str, level_in: int,  base_dir: Path):
    datset = tissue.tissue()
    tissue_detector_all = TissueDetectorAll()
    output_dir_colour = base_dir / label / 'colour_output_images'
    output_dir_colour.mkdir(parents=True, exist_ok=True)
    output_dir_outline = base_dir / label / 'outline_output_images'
    output_dir_outline.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(datset.paths.slide):
        print(path)
        annot_path = datset.paths.annotation.iloc[idx]
        test_path = project_root() / datset.root / path
        truth_path = project_root() / datset.root / annot_path
        with datset.slide_cls(test_path) as slide:
            thumb = slide.get_thumbnail(level_in)
            tissue_mask_detected = tissue_detector(thumb)
            if level_in >= len(slide.dimensions):
                request_level = len(slide.dimensions) - 1
                lev_diff = level_in - request_level
                max_level_dim = slide.dimensions[-1]
                requested_level_size = Size(max_level_dim.width // 2 ** lev_diff, max_level_dim.height // 2 ** lev_diff)
                labels_shape = requested_level_size.as_shape()
            else:
                labels_shape = slide.dimensions[level_in].as_shape()
            tissue_annotations = datset.load_annotations(truth_path)
            scale_factor = 2 ** level_in
            tissue_mask_annotated = tissue_annotations.render(labels_shape, scale_factor)
            tn_arr, fp_arr, fn_arr, tp_arr = get_slide_output_masks(tissue_mask_annotated, tissue_mask_detected)
            colour_thumb = create_colour_output(tn_arr, fp_arr, fn_arr, tp_arr)
            outpath_colour = output_dir_colour / str(Path(test_path).stem + '.jpg')
            colour_thumb.save(outpath_colour)
            outline_thumb = create_overlay_output(thumb, tissue_mask_detected)
            outpath_outline = output_dir_outline / str(Path(test_path).stem + '.jpg')
            outline_thumb.save(outpath_outline)


def get_slide_output_masks(true_mask, dets_mask):
   
    # convert to binary masks
    true_mask = true_mask > 0
    dets_mask = dets_mask > 0
    # work out tp, fp etc arrays
    tp_arr = np.expand_dims(np.logical_and(true_mask, dets_mask), axis=-1)
    fp_arr = np.expand_dims(np.logical_and(np.logical_not(true_mask), dets_mask), axis=-1)
    fn_arr = np.expand_dims(np.logical_and(np.logical_not(dets_mask), true_mask), axis=-1)
    tn_arr = np.expand_dims(np.logical_and(np.logical_not(true_mask), np.logical_not(dets_mask)), axis=-1)

    return tn_arr, fp_arr, fn_arr, tp_arr


def create_colour_output(tn_arr, fp_arr, fn_arr, tp_arr):
    
    # create a coloured ouput array
    color_result = np.zeros((tp_arr.shape[0], tp_arr.shape[1], 3))
    color_result = np.where(tp_arr, [255, 255, 255], color_result)
    color_result = np.where(fp_arr, [0, 0, 255], color_result)
    color_result = np.where(fn_arr, [255, 0, 0], color_result)
    color_result = np.where(tn_arr, [0, 0, 0], color_result)
    color_im = Image.fromarray(np.array(color_result, dtype=np.uint8))
    
    return color_im


def create_overlay_output(thumb, tissue_mask_detected):
    dets_mask = np.array(tissue_mask_detected*255, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(dets_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(thumb, contours, -1, (0,139,139))
    img_pil = Image.fromarray(np.array(img, dtype=np.uint8))

    return img_pil


def write_contours_to_file(tissue_detector: TissueDetector, label: str,  level_in: int, base_dir: Path, level_out: int=5):
    datset = tissue.tissue()
    output_dir = base_dir / label / 'json_files'
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in datset.paths.slide:
        print(path)
        test_path = project_root() / datset.root / path
        with datset.slide_cls(test_path) as slide:
            thumb = slide.get_thumbnail(level_in)
            tissue_mask = tissue_detector(thumb)
            path_label_name = 'tissue_' + label.lower()
            tissue_json = convert_mask_to_json(tissue_mask, path_label_name, level_in, level_out=level_out)
            output_path = output_dir / str(path.stem + '.json')
            with open(output_path, 'w') as outfile:
                json.dump(tissue_json, outfile)
