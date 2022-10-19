from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from repath.utils.convert import to_frame_with_locations
from repath.utils.paths import project_root


def convert_mask_to_contours_json_no_grandkids(im_resize, label):
    # get contours of binary mask
    contours, hierarchy  = cv2.findContours(np.array(im_resize, dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    json_points = []

    for ct in range(hierarchy.shape[1]):
        got_parent = hierarchy[0, ct, 3] >= 0
        if not got_parent:
            child_mask = hierarchy[0, :, 3] == ct
            got_kids = np.sum(child_mask) > 0
            if got_kids:
                polygon1 = contours[ct]
                pts = []
                for pt in range(polygon1.shape[0]):
                    pnt = polygon1[pt][0].tolist()
                    pts.append(pnt)
                json_polygon = [pts]
                kids = np.array(contours, dtype=object)[np.arange(len(contours))[child_mask]]
                for kid in kids:
                    pts = []
                    for pt in range(kid.shape[0]):
                        pnt = kid[pt][0].tolist()
                        pts.append(pnt)
                    json_polygon.append(pts)
                json_points.append(json_polygon)
            else:
                pts = []
                for pt in range(contours[ct].shape[0]):
                    pnt = contours[ct][pt][0].tolist()
                    pts.append(pnt)
                json_points.append([pts])
                
    json_dict = {label: json_points}
    return json_dict



def convert_mask_to_json(binary_mask: np.ndarray, label: str, level_in: int, level_out: int = 5):
    # convert to PIL image
    im_out = Image.fromarray(np.array(binary_mask*255, dtype=np.uint8))

    # calculate new image size
    level_diff = level_in - level_out
    size_diff = 2 ** level_diff
    new_size = (im_out.size[0] * size_diff, im_out.size[1] * size_diff)

    # resize image
    im_resize = im_out.resize(new_size, Image.BOX)
    
    # get json dictionary of contours
    json_dict = convert_mask_to_contours_json(im_resize, label)

    return json_dict


def convert_mask_to_contours_json(im_resize, label):
    # get contours of binary mask
    contours, hierarchy  = cv2.findContours(np.array(im_resize, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    json_points = []

    has_parent = [ct > -1 for ct in hierarchy[0, :, 3]]
    has_child = [ct > -1 for ct in hierarchy[0, :, 2]]
    no_parent_no_child = np.logical_and(np.logical_not(has_parent), np.logical_not(has_child))
    parent_no_child = np.logical_and(has_parent, np.logical_not(has_child))
    child_no_parent = np.logical_and(np.logical_not(has_parent), has_child)
    parent_and_child = np.logical_and(has_parent, has_child)
    parent_and_child_list = np.arange(len(contours))[parent_and_child]
    grandkids_mask = np.isin(hierarchy[0, :, 3], parent_and_child_list)
    no_parent_no_child_list = np.arange(len(contours))[no_parent_no_child]
    grandkids_list = np.arange(len(contours))[grandkids_mask]
    parents_list = np.arange(len(contours))[child_no_parent]
    top_level_list = np.hstack((no_parent_no_child_list, grandkids_list, parents_list))
    for ct in top_level_list:
        child_mask = hierarchy[0, :, 3] == ct
        got_kids = np.sum(child_mask) > 0
        if got_kids:
            polygon1 = contours[ct]
            pts = []
            for pt in range(polygon1.shape[0]):
                pnt = polygon1[pt][0].tolist()
                pts.append(pnt)
            json_polygon = [pts]
            kids = np.array(contours, dtype=object)[np.arange(len(contours))[child_mask]]
            for kid in kids:
                pts = []
                for pt in range(kid.shape[0]):
                    pnt = kid[pt][0].tolist()
                    pts.append(pnt)
                json_polygon.append(pts)
            json_points.append(json_polygon)
        else:
            pts = []
            for pt in range(contours[ct].shape[0]):
                pnt = contours[ct][pt][0].tolist()
                pts.append(pnt)
            json_points.append([pts])

    json_dict = {label: json_points}

    return json_dict


def patches_for_game(tissue_detector_test: 'TissueDetector', label: str, level_in: int, base_dir: Path) -> None:
    """
    For a tissue detector this gets tissue no tissue patches for gamification app
    
    Args:
            tissue_detector_test (TissueDetector): A class of tissue detector to test
            label (str): A label to add to filenames for naming output of this experiment
            level_in (int): The level at which to carry out the tissue detection
            best_dir (path): directory to write out data

    """
    tissue_dataset = tissue.tissue()
    psize = 2 ** level_in
    patch_finder = GridPatchFinder(labels_level=level_in, patch_level=0, patch_size=psize, stride=psize, remove_background=False)

    # create blank slides with just tissue detector labels
    tissue_patchsets_detected = SlidesIndex.index_dataset(tissue_dataset, tissue_detector_test, patch_finder, notblank=False)

    # find patches close to edges
    tissue_patchsets_edges = find_patches_close_to_edge(tissue_dataset, tissue_patchsets_detected, tissue_detector_test, level_in)

    # combine into one
    tissue_patches_edges = CombinedIndex.for_slide_indexes([tissue_patchsets_edges])

    # filter to get only edge patches
    tissue_patches_edges.patches_df = tissue_patches_edges.patches_df[tissue_patches_edges.patches_df.game_patch == True]

    # get sample of tissue_patches
    tissue_patches_sample = tissue_patches_edges.patches_df[tissue_patches_edges.patches_df.label == 1]
    tissue_patches_sample = tissue_patches_sample.sample(n=1000, axis=0)

    # get sample of non tissue_patches
    non_tissue_patches_sample = tissue_patches_edges.patches_df[tissue_patches_edges.patches_df.label == 0]
    non_tissue_patches_sample = non_tissue_patches_sample.sample(n=1000, axis=0)

    # combine into one set
    combined_sample = pd.concat((tissue_patches_sample, non_tissue_patches_sample), axis=0)
    tissue_patches_edges.patches_df = combined_sample

    # save patches
    tissue_patches_edges.save_patches(output_dir=Path(project_root(),"experiments","tissue","patches_for_game"))


def find_patches_close_to_edge(datset, slides_index, tissue_detector, level_in, how_close: int = 6):

    for sps in slides_index:
        path = datset.paths.slide[sps.slide_idx]
        test_path = test_path = project_root() / datset.root / path
        with datset.slide_cls(test_path) as slide:
            thumb = slide.get_thumbnail(level_in)
        tissue_mask_detected = tissue_detector(thumb)
        # find patches close to the contours of the image
        blank_im = np.zeros(tissue_mask_detected.shape, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(np.array(tissue_mask_detected, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_im = cv2.drawContours(blank_im, contours, -1, [255,255,255], how_close)
        contour_im = contour_im > 0
        df = to_frame_with_locations(contour_im, "game_patch")
        col_nams = sps.patches_df.columns.tolist()
        col_nams.append("game_patch")
        patches_df = pd.concat((sps.patches_df, df.iloc[:, 2:3]), axis=1, ignore_index=True)
        patches_df.columns = col_nams
        sps.patches_df = patches_df
        
    return slides_index