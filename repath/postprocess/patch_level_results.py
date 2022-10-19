from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from multiprocess import Pool
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, confusion_matrix, cohen_kappa_score
import torch
from torch.utils.data import DataLoader

from repath.preprocess.patching import CombinedIndex
from repath.postprocess.results import SlidesIndexResults
from repath.postprocess.slide_dataset import FolderClassDataset
from repath.utils.convert import remove_item_from_dict
from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, pre_re_curve, save_conf_mat_plot, save_conf_mat_plot_ci, binary_curves, conf_mat_plot_heatmap


def eval_on_device(args):
    batch_size = 128
    device_idx, classifier, valid_set, number_classes = args
    
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=8, worker_init_fn=np.random.seed(123))
    classifier.eval()
    classifier.to(device)

    num_samples = len(valid_loader) * valid_loader.batch_size

    prob_out = np.zeros((num_samples, number_classes))
    targ_out = np.zeros((num_samples))

    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            data, target = batch
            data = data.to(device)
            output = classifier(data)
            sm = torch.nn.Softmax(1)
            output_sm = sm(output)
            pred_prob = output_sm.cpu().numpy()  # rows: batch_size, cols: num_classes
            #print("pred_prob:", pred_prob)
            start = idx * valid_loader.batch_size
            end = start + pred_prob.shape[0]
            #print("pred_prob:", pred_prob.shape)
            prob_out[start:end, :] = pred_prob
            targ_out[start:end] = target

            if idx % 100 == 0:
                print('Batch {} of {} on GPU {}'.format(idx, len(valid_loader), device_idx))

    targ_out = np.expand_dims(targ_out, axis=-1)
    res_out = np.hstack((prob_out, targ_out))

    return res_out


def calc_patch_level_metrics(patches_df: pd.DataFrame, poslabel: int = 2, posname: str = 'tumor', optimal_threshold: float = 0.5) -> pd.DataFrame:
    # for posname column create a boolean mask of values greater than threshold (true means patch is detected as tumor)
    posname_mask = np.greater_equal(patches_df[posname], optimal_threshold)
    # get prediction for each patch which are either poslabel or 0
    predictions = np.where(posname_mask, posname, 'other')
    truenames = np.where(patches_df.label == poslabel, posname, 'other')
    # calculate accuracy for poslabel
    patch_accuracy = np.sum(truenames == predictions) / patches_df.shape[0]
    # calculate number of true postives etc - not using scikit learn function as it is slow
    tn, fp, fn, tp = conf_mat_raw(truenames, predictions,
                                  labels=['other', posname]).ravel()
    # calculate patch recall for poslabel
    patch_recall = tp / (tp + fn)
    # calculate patch specificity for poslabel
    patch_specificity = tn / (tn + fp)
    # calculate patch precision for poslabel
    patch_precision = tp / (tp + fp)
    # write results to list
    patch_results_out = [patch_accuracy, tn, fp, fn, tp, patch_recall, patch_specificity, patch_precision]

    return patch_results_out


def patch_level_metrics(slide_results: List[SlidesIndexResults], save_dir: Path, data_title: str,
                        posname: str = 'tumor', optimal_threshold: float = 0.5, 
                        ci: bool = False, nreps: int = 1000) -> pd.DataFrame:
    # check save directory exists if not make it
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # combine into one set of patches
    all_patches = CombinedIndex.for_slide_indexes(slide_results) 
    
    # remove background from dict and get number that corresponds to poslabel
    # assumes all label dicts are same for multiple datasets so just uses the first one in list
    class_labels = remove_item_from_dict(all_patches.datasets[0].labels, 'background')
    print(class_labels)
    poslabel = class_labels[posname]

    # get one number summaries
    patch_results_out = calc_patch_level_metrics(all_patches.patches_df, poslabel, posname, optimal_threshold)

    # use precision recall from scikit - calculates every threshold
    # patch_precisions, patch_recalls, patch_thresholds = precision_recall_curve(all_patches.patches_df.label,
    #                                                                            all_patches.patches_df[posname],
    #                                                                            pos_label=poslabel)
    patch_precisions, patch_recalls, _, patch_thresholds = binary_curves(all_patches.patches_df.label.to_numpy(), all_patches.patches_df[posname].to_numpy(), poslabel)

    # calculate pr auc
    patch_pr_auc = auc(patch_recalls, patch_precisions)
    # add to list of results
    patch_results_out.append(patch_pr_auc)
    
    # write out precision recall curve - without CI csv and png
    patch_curve = pd.DataFrame(list(zip(patch_precisions, patch_recalls, patch_thresholds)), 
                               columns=['patch_precisions', 'patch_recalls', 'patch_thresholds'])
    patch_curve.to_csv(save_dir / 'patch_pr_curve.csv', index=False)
    title_pr = "Patch Classification Precision-Recall Curve for \n" + data_title
    pr_curve_plt = plotROC(patch_recalls[1:], patch_precisions[1:], patch_pr_auc, title_pr, 'Recall', 'Precision', y_axis_lim = [0, 1])
    pr_curve_plt.savefig(save_dir/ "patch_pr_curve.png")

    # convert list to dataframe with row name - results
    col_names = ['accuracy', 'tn', 'fp', 'fn', 'tp', 'recall', 'specificity', 'precision', 'auc']
    patch_results_out = pd.DataFrame(np.reshape(patch_results_out, (1, len(patch_results_out))), columns=col_names)
    patch_results_out.index = ['results']
    
    # create confidence matrix plot and write out
    title_cm = "Patch Classification Confusion Matrix for \n" + data_title + "\n accuracy = " + str(round(patch_results_out.loc['results', 'accuracy'], 4))
    save_conf_mat_plot(patch_results_out[['tn', 'fp', 'fn', 'tp']], ['normal', 'tumor'], title_cm, save_dir)

    if ci:
        # create empty list to store results this will be end up as a list of list
        # more efficient to convert list of lists to pandas dataframe once than append row by row
        patch_ci = []
        # will calculate precision at a specified set of recall levels, this will be the same length for each sample
        # if used precision_recall_curve recalls and length would vary due to different numbers of thresholds
        nrecall_levs = 1001
        # create empty numpy array for storing precisions
        precisions1000 = np.empty((nreps, nrecall_levs))
        # set recall levels
        recall_levels = np.linspace(0.0, 1.0, nrecall_levs)
        for rep in range(nreps):
            print(rep)
            # create bootstrap sample
            sample_patches = all_patches.patches_df.sample(frac=1.0, replace=True)
            # get one number summaries
            ci_results = calc_patch_level_metrics(sample_patches, poslabel, posname, optimal_threshold)
            # get precisions and store
            pre1000 = pre_re_curve(sample_patches.label.to_numpy(), sample_patches[posname].to_numpy(), poslabel, recall_levels)
            precisions1000[rep, :] = pre1000
            # get pr auc
            ci_pr_auc = auc(recall_levels[1:], pre1000[1:])
            # append to this set of results
            ci_results.append(ci_pr_auc)
            # append ci_results to create list of lists of ci_results
            patch_ci.append(ci_results)
 
        # convert list of lists to a dataframe
        patch_ci_df = pd.DataFrame(patch_ci, columns=col_names)

        # name to rows to give sample numbers
        patch_ci_df.index = ['sample_' + str(x) for x in range(nreps)]
        # create confidence intervals for each 
        patch_results_out_ci = patch_ci_df.quantile([0.025, 0.975])
        # rename rows of dataframe
        patch_results_out_ci.index = ['ci_lower_bound', 'ci_upper_bound']

        # concatenate results to give results, confidence interval then all samples
        patch_results_out = pd.concat((patch_results_out, patch_results_out_ci, patch_ci_df), axis=0)

        # create confidence interval for precision recall curve
        precisions_ci = np.quantile(precisions1000, [0.025, 0.975], axis=0)
        # create dataframe with precision recall curve confidence interval
        patch_curve_ci = pd.DataFrame(np.hstack((precisions_ci.T, np.reshape(recall_levels, (nrecall_levs, 1)))), 
                                                 columns=['patch_precisions_lower', 'patch_precisions_upper', 'patch_recalls'])
        # write out precision recall curve confidence interval
        patch_curve_ci.to_csv(save_dir / 'patch_pr_curve_ci.csv', index=False)

        # create pr curve with confidence interval
        title_pr = "Patch Classification Precision-Recall Curve for \n" + data_title
        pr_curve_plt = plotROCCI(patch_recalls, patch_precisions, recall_levels, precisions_ci, patch_pr_auc,
                                 patch_results_out_ci.auc.to_list(), title_pr, 'Recall', 'Precision')
        pr_curve_plt.savefig(save_dir / "patch_pr_curve_ci.png")

        # create confidence matrix plot with confidence interval and write out
        acc_res = round(patch_results_out.loc['results', 'accuracy'], 4)
        acc_low = round(patch_results_out.loc['ci_lower_bound', 'accuracy'], 4)
        acc_high = round(patch_results_out.loc['ci_upper_bound', 'accuracy'], 4)
        summary_value_string = "\n accuracy = " + str(acc_res) + "(" + str(acc_low) + ", " + str(acc_high) + ")"
        title_cm = "Patch Classification Confusion Matrix for \n" + data_title +  summary_value_string
        save_conf_mat_plot_ci(patch_results_out[['tn', 'fp', 'fn', 'tp']], ['normal', 'tumor'], title_cm, save_dir)

    # write out patch summary result dataframe
    patch_results_out.to_csv(save_dir / 'patch_results.csv')


def patch_level_metrics_multi(slide_results: List[SlidesIndexResults], save_dir: Path, data_title: str) -> pd.DataFrame:
    # check save directory exists if not make it
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # combine into one set of patches
    all_patches = CombinedIndex.for_slide_indexes(slide_results) 
    
    # remove background from dict and get number that corresponds to poslabel
    # assumes all label dicts are same for multiple datasets so just uses the first one in list
    class_labels = remove_item_from_dict(all_patches.datasets[0].labels, 'background')

    predicted_probabilities = all_patches.patches_df.loc[:, list(class_labels.keys())]
    predicted_class = predicted_probabilities.idxmax(axis=1)
    predicted_label = predicted_class.replace(class_labels)

    cm = conf_mat_raw(all_patches.patches_df.label.to_numpy(), predicted_label.to_numpy(), labels=class_labels.values())
    cm_out = cm.ravel()

    col_heads = []
    accuracy = 0
    for idt, tt in enumerate(class_labels.keys()):
        for idp, pp in enumerate(class_labels.keys()):
            gen_label = 'true_' + tt + '_pred_' + pp
            col_heads.append(gen_label)
            if pp == tt:
                accuracy = accuracy + cm[idt, idp]
    accuracy = accuracy / np.sum(cm_out)

    patch_results_out = pd.DataFrame(np.reshape(cm_out, (1, len(cm_out))), columns=col_heads, index=['results'])
    patch_results_out['accuracy'] = [accuracy]

    cm_img = conf_mat_plot_heatmap(cm, class_labels.keys(), data_title)
    out_path = 'confusion_matrix.png'
    cm_img.get_figure().savefig(save_dir / out_path)

    cm_pc = cm/cm.sum(axis=1, keepdims=True)
    cm_pc = np.multiply(cm_pc, 100)
    cm_pc = np.around(cm_pc, 2)
    data_title = data_title + '\n Accuracy= ' + str(round(accuracy, 4)) 
    cm_img_pc = conf_mat_plot_heatmap(cm_pc, class_labels.keys(), data_title)
    out_path_pc = 'confusion_matrix_percent.png'
    cm_img_pc.get_figure().savefig(save_dir / out_path_pc)


    # write out patch summary result dataframe
    patch_results_out.to_csv(save_dir / 'patch_results.csv')


def patch_level_metrics_multi_balanced(classifier, transform, patch_folder, save_dir: Path, data_title: str) -> pd.DataFrame:
   
    # check save directory exists if not make it
    save_dir.mkdir(parents=True, exist_ok=True)

    valid_set_norm = FolderClassDataset(patch_folder / "normal", 3, transform=transform)
    valid_set_lowg = FolderClassDataset(patch_folder / "low_grade", 1, transform=transform)
    valid_set_high = FolderClassDataset(patch_folder / "high_grade", 0, transform=transform)
    valid_set_malg = FolderClassDataset(patch_folder / "malignant", 2, transform=transform)

    vset_list = [valid_set_norm, valid_set_lowg, valid_set_high, valid_set_malg]
    inputs = zip(range(4), [classifier]*4, vset_list, [4]*4)

    pool = Pool()
    results = pool.map(eval_on_device, inputs)
    pool.close()
    pool.join()

    for rr in range(4):
        rez = results[rr]
        vset_len = len(vset_list[rr])
        rez = rez[0:vset_len, :]
        if rr == 0:
            results_all = rez
        else:
            results_all = np.vstack((results_all, rez))

    probs_all = results_all[:, 0:4]
    labels_all = results_all[:, 4:5]
    labels_all = np.array(labels_all, dtype=int)
    preds_all = np.argmax(probs_all, axis=1)
    preds_all = np.expand_dims(preds_all, axis=-1)

    cm_patch = conf_mat_raw(labels_all, preds_all, [0, 1, 2, 3])
    cm_patch = cm_patch[[3,1,0,2],:]
    cm_patch = cm_patch[:,[3,1,0,2]]
    cm_img = conf_mat_plot_heatmap(cm_patch, ['normal', 'low grade', 'high grade', 'malignant'], data_title)
    out_path = 'confusion_matrix_balanced.png'
    cm_img.get_figure().savefig(save_dir / out_path)

    cm_pc = cm_patch/cm_patch.sum(axis=1, keepdims=True)
    cm_pc = np.multiply(cm_pc, 100)
    cm_pc = np.around(cm_pc, 2)
    cm_img_pc = conf_mat_plot_heatmap(cm_pc, ['normal', 'low grade', 'high grade', 'malignant'], data_title)
    out_path_pc = 'confusion_matrix_percent_balanced.png'
    cm_img_pc.get_figure().savefig(save_dir / out_path_pc)


def patch_level_metrics_multi_balanced_endo(classifier, transform, patch_folder, save_dir: Path, data_title: str) -> pd.DataFrame:
   
    # check save directory exists if not make it
    save_dir.mkdir(parents=True, exist_ok=True)

    valid_set_norm = FolderClassDataset(patch_folder / "other_benign", 1, transform=transform)
    valid_set_malg = FolderClassDataset(patch_folder / "malignant", 0, transform=transform)

    vset_list = [valid_set_norm, valid_set_malg]
    inputs = zip(range(2,4), [classifier]*2, vset_list, [2]*2)

    pool = Pool()
    results = pool.map(eval_on_device, inputs)
    pool.close()
    pool.join()

    for rr in range(2):
        rez = results[rr]
        vset_len = len(vset_list[rr])
        rez = rez[0:vset_len, :]
        if rr == 0:
            results_all = rez
        else:
            results_all = np.vstack((results_all, rez))

    probs_all = results_all[:, 0:2]
    labels_all = results_all[:, 2:3]
    labels_all = np.array(labels_all, dtype=int)
    preds_all = np.argmax(probs_all, axis=1)
    preds_all = np.expand_dims(preds_all, axis=-1)

    cm_patch = conf_mat_raw(labels_all, preds_all, [1, 0])
    cm_patch = cm_patch[[0,1],:]
    cm_patch = cm_patch[:,[0,1]]
    cm_img = conf_mat_plot_heatmap(cm_patch, ['other_benign', 'malignant'], data_title)
    out_path = 'confusion_matrix_balanced.png'
    cm_img.get_figure().savefig(save_dir / out_path)

    cm_pc = cm_patch/cm_patch.sum(axis=1, keepdims=True)
    cm_pc = np.multiply(cm_pc, 100)
    cm_pc = np.around(cm_pc, 2)
    cm_img_pc = conf_mat_plot_heatmap(cm_pc, ['other_benign', 'malignant'], data_title)
    out_path_pc = 'confusion_matrix_percent_balanced.png'
    cm_img_pc.get_figure().savefig(save_dir / out_path_pc)
    

def patch_level_metrics_multi_balanced_endo_bm(classifier, transform, patch_folder, save_dir: Path, data_title: str) -> pd.DataFrame:
   
    # check save directory exists if not make it
    save_dir.mkdir(parents=True, exist_ok=True)

    valid_set_norm = FolderClassDataset(patch_folder / "other_benign", 4, transform=transform)
    valid_set_malg = FolderClassDataset(patch_folder / "malignant", 2, transform=transform)
    valid_set_blod = FolderClassDataset(patch_folder / "blood", 0, transform=transform)
    valid_set_mucs = FolderClassDataset(patch_folder / "bloodmucus", 1, transform=transform)
    valid_set_blmu = FolderClassDataset(patch_folder / "mucus", 3, transform=transform)

    vset_list = [valid_set_norm, valid_set_malg, valid_set_blod, valid_set_mucs, valid_set_blmu]
    inputs = zip(range(1,6), [classifier]*5, vset_list, [5]*5)

    pool = Pool()
    results = pool.map(eval_on_device, inputs)
    pool.close()
    pool.join()

    for rr in range(5):
        rez = results[rr]
        vset_len = len(vset_list[rr])
        rez = rez[0:vset_len, :]
        if rr == 0:
            results_all = rez
        else:
            results_all = np.vstack((results_all, rez))

    probs_all = results_all[:, 0:5]
    labels_all = results_all[:, 5:6]
    labels_all = np.array(labels_all, dtype=int)
    preds_all = np.argmax(probs_all, axis=1)
    preds_all = np.expand_dims(preds_all, axis=-1)

    cm_patch = conf_mat_raw(labels_all, preds_all, [0, 1, 2, 3, 4])
    cm_patch = cm_patch[[0,1,2,3,4],:]
    cm_patch = cm_patch[:,[0,1,2,3,4]]
    cm_img = conf_mat_plot_heatmap(cm_patch, ['other_benign', 'malignant', 'blood', 'bloodmucus', 'mucus'], data_title)
    out_path = 'confusion_matrix_balanced.png'
    cm_img.get_figure().savefig(save_dir / out_path)

    cm_pc = cm_patch/cm_patch.sum(axis=1, keepdims=True)
    cm_pc = np.multiply(cm_pc, 100)
    cm_pc = np.around(cm_pc, 2)
    cm_img_pc = conf_mat_plot_heatmap(cm_pc, ['other_benign', 'malignant', 'blood', 'bloodmucus', 'mucus'], data_title)
    out_path_pc = 'confusion_matrix_percent_balanced.png'
    cm_img_pc.get_figure().savefig(save_dir / out_path_pc)    
