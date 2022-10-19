from joblib import load
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose, ToTensor

#from repath.data.slides.isyntax import Slide
#from repath.data.slides.bioformats import Slide
from repath.final_algo.single_slide_dataset import SingleSlideDataset
from repath.postprocess.slide_level_metrics import calculate_slide_features_single_endoupdate, calculate_slide_features_single_cervicalupdate
from repath.preprocess.patching import GridPatchFinder
#from repath.preprocess.tissue_detection.blood_mucus_rework import apply_bloodm_detector
from repath.preprocess.tissue_detection.pixel_feature_detector import TextureFeature, PixelFeatureDetector
from repath.utils.filters import pool2d
from repath.utils.seeds import set_seed


def to_heatmap(patches_df: pd.DataFrame, class_name: str, patch_size: int, stride: int, slide_dims) -> np.array:
    patches_df.columns = [colname.lower() for colname in patches_df.columns]
    class_name = class_name.lower()

    # find core patch size
    if stride is not None:
        pool_size = int(patch_size / stride)
        base_patch_size = stride
    else:
        base_patch_size = patch_size

    # remove border and convert to column, row
    patches_df['column'] = np.divide(patches_df['x'], base_patch_size)
    patches_df['row'] = np.divide(patches_df['y'], base_patch_size)

    max_rows = int(math.ceil(slide_dims.height / base_patch_size)) 
    max_cols = int(math.ceil(slide_dims.width / base_patch_size)) 

    # create a blank thumbnail
    thumbnail_out = np.zeros((max_rows, max_cols))

    # for each row in dataframe set the value of the pixel specified by row and column to the probability in clazz
    for rw in range(patches_df.shape[0]):
        df_row = patches_df.iloc[rw]
        thumbnail_out[int(df_row.row), int(df_row.column)] = df_row[class_name]

    if stride is not None:
        thumbnail_out = pool2d(thumbnail_out, pool_size, pool_size, 0, "avg")

    return thumbnail_out


def single_slide_prediction(input_slide: Path, tissue_type: 'str', device_idx = 0, annots = None):

    if tissue_type == 'endo':
        from repath.data.slides.isyntax import Slide
        from repath.preprocess.tissue_detection.blood_mucus_rework import apply_bloodm_detector
        import repath.final_algo.endometrial_settings as settings
        import repath.data.datasets.endometrial_6 as endom
        dset = endom.train()
        load_annots = dset.load_annotations
    if tissue_type == 'cerv':
        from repath.data.slides.isyntax import Slide
        import repath.final_algo.cervical_settings as settings
        import repath.data.datasets.cervical_set2 as cervi
        dset = cervi.train()
        load_annots = dset.load_annotations
    if tissue_type == 'cziCerv':
        from repath.data.slides.bioformats2 import Slide
        import repath.final_algo.cervical_czi_settings as settings
        from repath.data.slides.bioformats2 import setup, silence_javabridge, shutdown
        setup()
        silence_javabridge()
    if tissue_type == 'endoczi':
        from repath.data.slides.bioformats2 import Slide
        from repath.preprocess.tissue_detection.blood_mucus_rework import apply_bloodm_detector
        import repath.final_algo.endometrial_settings as settings
        from repath.data.slides.bioformats2 import setup, silence_javabridge, shutdown
        setup()
        silence_javabridge()
    # open slide get patch locations
    with Slide(input_slide) as slide:
        slide_dims = slide.dimensions[settings.patch_level]
        if annots is not None:
            annot_path, rowlabel = annots
            annotations = load_annots(annot_path, rowlabel)
            scale_factor = 2 ** settings.feature_level
            labels_shape = slide.dimensions[settings.feature_level].as_shape()
            labels_image = annotations.render(labels_shape, scale_factor)
        if tissue_type == 'cerv':
            thumbnail = slide.get_thumbnail(settings.feature_level)
            tissue_mask = settings.tissue_detector(thumbnail)
            labels_image = tissue_mask
        if tissue_type == 'cziCerv':
            thumbnail = slide.get_thumbnail(settings.feature_level)
            tissue_mask = settings.tissue_detector(thumbnail)
            labels_image = tissue_mask
        if tissue_type == 'endo':
            bloodm_detector = PixelFeatureDetector(features_list=[TextureFeature()], sigma_min = 1, sigma_max = 1, raw=True)
            bloodm_clf = load(settings.bm_clf_path)
            bm_patch_size = 2 ** settings.feature_level
            scale_factor = 2 ** (4 - settings.patch_level)
            kernel_size = int(bm_patch_size / scale_factor)
            output_labels = apply_bloodm_detector(slide, 4, settings.tissue_detector, bloodm_detector, bloodm_clf, kernel_size)
            output_labels = output_labels[0:labels_image.shape[0], 0:labels_image.shape[1]]
            tissue_mask = output_labels > 1
            labels_image[~tissue_mask] = 0
        if tissue_type == 'endoczi':
            bloodm_detector = PixelFeatureDetector(features_list=[TextureFeature()], sigma_min = 1, sigma_max = 1, raw=True)
            bloodm_clf = load(settings.bm_clf_path)
            bm_patch_size = 2 ** settings.feature_level
            scale_factor = 2 ** (4 - settings.patch_level)
            kernel_size = int(bm_patch_size / scale_factor)
            output_labels = apply_bloodm_detector(slide, 4, settings.tissue_detector, bloodm_detector, bloodm_clf, kernel_size)
            tissue_mask = output_labels > 1
            labels_image = tissue_mask

        patch_finder = GridPatchFinder(settings.feature_level, settings.patch_level, settings.patch_size, settings.stride, pool_mode=settings.pool_mode)
        labels_image = np.array(labels_image, dtype=np.int)
        df, _, _ = patch_finder(labels_image, slide_dims)
        df = df.reset_index(drop=True)

    # create pytorch loader
    transform = Compose([ToTensor()])
    dataset = SingleSlideDataset(df, Slide, input_slide, settings.patch_size, settings.patch_level, transform)
    dataset.open_slide()
    test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=settings.batch_size)
    
    # Run patch level model
    set_seed(123)
    model = settings.PatchClassifier.load_from_checkpoint(checkpoint_path=settings.cp_path)

    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    num_samples = len(test_loader) * settings.batch_size

    prob_out = np.zeros((num_samples, settings.num_classes))

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            data, target = batch
            data = data.to(device)
            output = model(data)
            sm = torch.nn.Softmax(1)
            output_sm = sm(output)
            pred_prob = output_sm.cpu().numpy()  # rows: batch_size, cols: num_classes

            start = idx * settings.batch_size
            end = start + pred_prob.shape[0]
            prob_out[start:end, :] = pred_prob

            if len(test_loader) < 11:
                nn = 1
            elif len(test_loader) < 101:
                nn = 10
            elif len(test_loader) < 251:
                nn = 25
            else:
                nn = 50

            if idx % nn == 0:
                print('Batch {} of {}'.format(idx, len(test_loader)-1))

    dataset.close_slide()

    #if tissue_type == 'cziCerv' or tissue_type == 'endoczi':
    #    shutdown()
    
    # create patch results dataframe            
    prob_out = prob_out[0:df.shape[0], :]
    prob_out = pd.DataFrame(prob_out, columns=settings.class_names)
    patch_results = pd.concat((df, prob_out), axis=1)

    heatmaps = []
    for class_name in settings.class_names:
        heatmap = to_heatmap(patch_results, class_name, settings.patch_size, settings.stride, slide_dims)
        heatmaps.append(heatmap)


    if tissue_type == 'endo' or tissue_type == 'endoczi':
        malig_hm = heatmaps[0]
        other_hm = heatmaps[1]
        tissue_area = np.sum((malig_hm + other_hm) > 0)

        features_ml = calculate_slide_features_single_endoupdate(malig_hm, tissue_area, "_ML")
        features_ob = calculate_slide_features_single_endoupdate(other_hm, tissue_area, "_OB")

        features_df = pd.concat((features_ml, features_ob), axis=1)
        features_df['tissue_area'] = tissue_area
    if tissue_type == 'cerv' or tissue_type == 'cziCerv':
        highg_hm = heatmaps[0]
        lowg_hm = heatmaps[1]
        malig_hm = heatmaps[2]
        normal_hm = heatmaps[3]
        
        tissue_area = np.sum((malig_hm + highg_hm + lowg_hm + normal_hm) > 0)

        features_ml = calculate_slide_features_single_cervicalupdate(malig_hm, tissue_area, "_ML")
        features_hg = calculate_slide_features_single_cervicalupdate(highg_hm, tissue_area, "_HG")
        features_lg = calculate_slide_features_single_cervicalupdate(lowg_hm, tissue_area, "_LG")
        features_ni = calculate_slide_features_single_cervicalupdate(normal_hm, tissue_area, "_NI")

        features_df = pd.concat((features_ml, features_hg, features_lg, features_ni), axis=1)

    slide_model = load(settings.trained_model)
    predictions = slide_model.predict(features_df)
    probabilities = slide_model.predict_proba(features_df)
    probabilities = pd.DataFrame(probabilities, columns=settings.slide_class_names)
    print("predicted slide class:", predictions[0])
    print(probabilities)

    return predictions, probabilities, patch_results, heatmaps
