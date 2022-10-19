import numpy as np
import pandas as pd
import time
from repath.utils.paths import project_root
from repath.final_algo.single_slide import single_slide_prediction


start = time.time()
def loop_over_set_of_slides(tissue_type, datset, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if tissue_type == 'endo':
        input_path = project_root() / 'data' / 'icaird' / 'iCAIRD_metadata' / 'iCAIRD_Endometrial_7.csv'
        input_df = pd.read_csv(input_path)
        input_df = input_df[input_df['use_row'] == 1]
        dset_df = input_df[input_df['train_valid_test'] == datset]
        from endometrial_settings import slide_class_names
        image_dir = project_root() / 'data' / 'icaird' / 'iCAIRD'
        annot_dir = project_root() / 'data' / 'icaird' / 'iCAIRD_annotations' / 'Endometrial'
    if tissue_type == 'cerv':
        input_path = project_root() / 'data' / 'icaird' / 'iCAIRD_metadata' / 'iCAIRD_Cervical_Data_Full.csv'
        input_df = pd.read_csv(input_path)
        dset_df = input_df[input_df['train/valid/test'] == datset]
        from cervical_settings import slide_class_names
        image_dir = project_root() / 'data' / 'icaird' / 'iCAIRD'
        annot_dir = project_root() / 'data' / 'icaird' / 'iCAIRD_annotations' / 'Cervical'
        dset_df.rename(columns={'Cat': 'Category'})
    if tissue_type == 'cziCerv':
        input_path = project_root() / 'data' / 'icaird' / 'iCAIRD_metadata' / 'iCAIRD_Cervical_czi.csv'
        input_df = pd.read_csv(input_path)
        dset_df = input_df[input_df['train/valid/test'] == datset]
        from cervical_czi_settings import slide_class_names
        image_dir = project_root() / 'data' / 'icaird' / 'iCAIRD_czi'/ 'cervical_czi'
        annot_dir = None
    if tissue_type == 'endoczi':
        input_path = project_root() / 'data' / 'icaird' / 'iCAIRD_metadata' / 'iCAIRD_Endometrial_czi.csv'
        input_df = pd.read_csv(input_path)
        dset_df = input_df[input_df['use_row'] == 1]
        from endometrial_settings import slide_class_names
        image_dir = project_root() / 'data' / 'icaird' / 'iCAIRD_czi'/ 'endometrial_czi'
        annot_dir = None
    if output_path.is_file():
        dset_df = pd.read_csv(output_path)
    else:
        dset_df = dset_df.reset_index(drop=True)
        dset_df['prediction'] = ""
        dset_df['time'] = 0
        for scn in slide_class_names:
            dset_df[scn] = 0

    predz = []
    probz = []
    patch_results =[]
    
    for idx, rw in dset_df.iterrows():
        start = time.time()
        print("Slide", idx + 1, "of", dset_df.shape[0])
        filepath = image_dir / rw['Filename']
        patch_filename = output_path.parent / (filepath.stem + '.csv')
        if annot_dir is not None:
            annot_path = annot_dir / (filepath.stem + '.txt')
            labelname = rw['Category']
            annot_info = [annot_path, labelname]
        else:
            annot_info = None
        if patch_filename.is_file():
            continue
        else:
            pred, prob, patch_result ,heatmap = single_slide_prediction(filepath, tissue_type, 1, annots=annot_info)
            predz.append(pred)
            probz.append(prob)
            patch_results.append(patch_result)
            dset_df['prediction'].iloc[idx]=pred[0]
            dset_df['time'].iloc[idx] = time.time()-start
            for scn in slide_class_names:
                dset_df[scn].iloc[idx]=prob[scn].iloc[0]
            dset_df.to_csv(output_path, index=False)
            patch_result.to_csv(patch_filename)
    patch_results_out = pd.concat(patch_results, axis = 0)
    patch_results_out = patch_results_out.reset_index(drop = True)

    
    patch_results_out.to_csv(output_path.parent / 'patch_results.csv', index=False)



#loop_over_set_of_slides('cerv', 'test', project_root() / 'results' / 'cervical_final_May2022' / 'slide_results' / 'test_update' / 'test_slide_results.csv')
#loop_over_set_of_slides('cziCerv', 'test', project_root() / 'results' / 'cervical_final_May2022' / 'slide_results' / 'test_update' / 'czi_test_slide_results.csv')
#loop_over_set_of_slides('endoczi', 'test', project_root() / 'results' / 'endo_czi' / 'czi_slide_results.csv')
loop_over_set_of_slides('endo', 'test', project_root() / 'results' / 'endo_final_patch_level_test' / 'czi_slide_results.csv')