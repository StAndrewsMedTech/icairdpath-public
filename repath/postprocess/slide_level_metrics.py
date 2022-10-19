from abc import ABC, abstractmethod
from joblib import dump, load
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from skimage.measure import label, regionprops
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from xgboost import XGBClassifier

from repath.postprocess.instance_segmentors import ConnectedComponents, DBScan
from repath.postprocess.results import SlidesIndexResults, SlidePatchSetResults
from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, fpr_tpr_curve, save_conf_mat_plot, save_conf_mat_plot_ci, binary_curves


class SlideClassifier(ABC):
    def __init__(self, slide_labels: Dict) -> None:
        self.slide_labels = slide_labels

    @abstractmethod
    def calculate_slide_features(self, result) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict_slide_level(self) -> None:
        pass

    def calc_features(self, results: SlidesIndexResults, output_dir: Path) -> pd.DataFrame:
        
        features_all_slides = []

        for result in results:
            features_out = self.calculate_slide_features(result,root_dir=results.output_dir)
            features_all_slides.append(features_out)

        outcols = features_all_slides[0].columns.values

        features_all_slides = pd.concat(features_all_slides)

        # write out lesions
        output_dir.mkdir(parents=True, exist_ok=True)
        features_all_slides.to_csv(output_dir / 'features.csv', index=False)
  

    def calc_slide_metrics_binary(self, title, output_dir, ci=True, nreps=1000, posname='tumor'):
        output_dir.mkdir(parents=True, exist_ok=True)
        slide_results = pd.read_csv(output_dir / 'slide_results.csv')

        # Accuracy - number of matching labels / total number of slides
        slide_accuracy = np.sum(slide_results.true_label == slide_results.predictions) / slide_results.shape[0]
        print(f'accuracy: {slide_accuracy}')

        # confusion matrix
        conf_mat = conf_mat_raw(slide_results.true_label.to_numpy(),
                                    slide_results.predictions.to_numpy(),
                                    labels=self.slide_labels.keys())
        conf_mat = conf_mat.ravel().tolist()

        # ROC curve for 
        pos_probs = [float(prob) for prob in slide_results[posname].tolist()]
        precision, tpr, fpr, roc_thresholds = binary_curves(slide_results.true_label.to_numpy(),
                                                            np.array(pos_probs), pos_label=posname)
        auc_out = auc(fpr, tpr)

        # write out precision recall curve - without CI csv and png
        slide_curve = pd.DataFrame(list(zip(fpr, tpr, roc_thresholds)), 
                                   columns=['fpr', 'tpr', 'thresholds'])
        slide_curve.to_csv(output_dir / 'slide_pr_curve.csv', index=False)
        title_pr = "Slide Classification ROC Curve for \n" + title
        pr_curve_plt = plotROC(fpr, tpr, auc_out, title_pr, "False Positive Rate", "True Positive Rate")
        pr_curve_plt.savefig(output_dir / 'slide_pr_curve.png')

        # combine single figure results into dataframe
        slide_metrics_out = [slide_accuracy] + conf_mat + [auc_out]
        slide_metrics_out = np.reshape(slide_metrics_out, (1, 6))
        slide_metrics_out = pd.DataFrame(slide_metrics_out, columns=['accuracy', 'tn', 'fp', 'fn', 'tp', 'auc'])
        slide_metrics_out.index = ['results']

        # create confidence matrix plot and write out
        title_cm = "Slide Classification Confusion Matrix for \n" + title +"\n accuracy = " + str(round(slide_accuracy, 4))
        save_conf_mat_plot(slide_metrics_out[['tn', 'fp', 'fn', 'tp']], self.slide_labels.keys(), title_cm, output_dir)

        if ci:
            slide_accuracy1000 = np.empty((nreps, 1))
            conf_mat1000 = np.empty((nreps, 4))
            auc_out1000 = np.empty((nreps, 1))
            nrecall_levs = 101
            fpr_levels = np.linspace(0.0, 1.0, nrecall_levs)
            tpr1000 = np.empty((nreps, nrecall_levs))
            for rep in range(nreps):
                sample_slide_results = slide_results.sample(frac=1.0, replace=True)
                slide_accuracy = np.sum(sample_slide_results.true_label == sample_slide_results.predictions) / \
                                sample_slide_results.shape[0]
                slide_accuracy1000[rep, 0] = slide_accuracy
                conf_mat = conf_mat_raw(sample_slide_results.true_label.to_numpy(),
                                            sample_slide_results.predictions.to_numpy(), labels=self.slide_labels.keys())
                conf_mat = conf_mat.ravel().tolist()
                conf_mat1000[rep, :] = conf_mat

                #pos_probs = [float(prob) for prob in sample_slide_results[posname].tolist()]
                tpr_lev = fpr_tpr_curve(sample_slide_results.true_label.to_numpy(), sample_slide_results[posname].to_numpy(),
                                                        pos_label=posname, recall_levels=fpr_levels)
                auc_samp = auc(fpr_levels[1:], tpr_lev[1:])
                auc_out1000[rep, 0] = auc_samp
                #tpr_lev = np.interp(fpr_levels, fpr, tpr)
                tpr1000[rep, :] = tpr_lev
            
            # combine single figure metrics to dataframe
            samples_df = pd.DataFrame(np.hstack((slide_accuracy1000, conf_mat1000, auc_out1000)), 
                                      columns=['accuracy', 'tn', 'fp', 'fn', 'tp', 'auc'])
            samples_df.index = ['sample_' + str(x) for x in range(nreps)]
            slide_metrics_ci = samples_df.quantile([0.025, 0.975])
            slide_metrics_ci.index = ['ci_lower_bound', 'ci_upper_bound']
            slide_metrics_out = pd.concat((slide_metrics_out, slide_metrics_ci, samples_df), axis=0)
            title_cm = title_cm + " (" + str(round(slide_metrics_ci.loc["ci_lower_bound", "accuracy"], 4)) 
            title_cm = title_cm + ", " + str(round(slide_metrics_ci.loc["ci_upper_bound", "accuracy"], 4)) + ")"
            # create confidence matrix plot with confidence interval and write out
            save_conf_mat_plot_ci(slide_metrics_out[['tn', 'fp', 'fn', 'tp']], self.slide_labels.keys(), title_cm, output_dir)

            # write out the curve information
            tprCI = np.quantile(tpr1000, [0.025, 0.975], axis=0)
            # fpr, tpr, roc_thresholds - slide level roc curve for c16
            slide_curve = pd.DataFrame(np.hstack((tprCI.T, np.reshape(fpr_levels, (nrecall_levs, 1)))),
                                       columns=['tpr_lower', 'tpr_upper', 'fpr'])
            slide_curve.to_csv(output_dir / 'slide_pr_curve_ci.csv', index=False)
            slide_curve_plt = plotROCCI(fpr, tpr, fpr_levels, tprCI, auc_out, slide_metrics_ci.auc.tolist(),
                                        title_pr, "False Positive Rate", "True Positive Rate")
            slide_curve_plt.savefig(output_dir / "slide_pr_curve_ci.png")

        slide_metrics_out.to_csv(output_dir / 'slide_metrics.csv')

    def calc_slide_metrics_multi(self, title, output_dir, ci=True, nreps=1000, labelorder=None, newlabels=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        slide_results = pd.read_csv(output_dir / 'slide_results.csv')

        if labelorder == None:
            labelorder = self.slide_labels.keys()

        # Accuracy - number of matching labels / total number of slides
        slide_accuracy = np.sum(slide_results.true_label == slide_results.predictions) / slide_results.shape[0]
        print(f'accuracy: {slide_accuracy}')
        
        # confusion matrix for multi class
        conf_mat = conf_mat_raw(slide_results.true_label.to_numpy(),
                                    slide_results.predictions.to_numpy(),
                                    labels=labelorder)
        conf_mat = conf_mat.ravel().tolist()
        pred_tiled_labels = list(self.slide_labels.keys()) * len(self.slide_labels.keys())
        true_tiled_labels = [item for item in self.slide_labels.keys() for i in range(len(self.slide_labels.keys()))]
        confmat_labels = [f'true_{vals[0]}_pred_{vals[1]}' for vals in list(zip(true_tiled_labels, pred_tiled_labels))]
        column_labels = ['accuracy'] + confmat_labels

        output_list = [slide_accuracy] + conf_mat
        output_arr = np.reshape(np.array(output_list), (1, len(output_list)))
        slide_metrics_out = pd.DataFrame(output_arr)
        slide_metrics_out.columns = column_labels
        slide_metrics_out.index = ['results']

        # create confidence matrix plot and write out
        if newlabels == None:
            newlabels = labelorder
        title_cm = "Slide Classification Confusion Matrix for \n" + title + "\n accuracy = " + str(round(slide_accuracy,4))
        save_conf_mat_plot(slide_metrics_out.iloc[:, 1:], newlabels, title_cm, output_dir)

        if ci:
            slide_accuracy1000 = np.empty((nreps, 1))
            conf_mat1000 = np.empty((nreps, len(self.slide_labels.keys())**2))
            for rep in range(nreps):
                sample_slide_results = slide_results.sample(frac=1.0, replace=True)
                slide_accuracy = np.sum(sample_slide_results.true_label == sample_slide_results.predictions) / \
                                sample_slide_results.shape[0]
                slide_accuracy1000[rep, 0] = slide_accuracy
                conf_mat = conf_mat_raw(sample_slide_results.true_label.to_numpy(),
                                            sample_slide_results.predictions.to_numpy(), labels=labelorder)
                conf_mat = conf_mat.ravel().tolist()
                conf_mat1000[rep, :] = conf_mat

            # combine single figure metrics to dataframe
            samples_df = pd.DataFrame(np.hstack((slide_accuracy1000, conf_mat1000)), 
                                      columns=column_labels)
            samples_df.index = ['sample_' + str(x) for x in range(nreps)]
            slide_metrics_ci = samples_df.quantile([0.025, 0.975])
            slide_metrics_ci.index = ['ci_lower_bound', 'ci_upper_bound']
            slide_metrics_out = pd.concat((slide_metrics_out, slide_metrics_ci, samples_df), axis=0)
            title_cm = title_cm + " (" + str(round(slide_metrics_ci.loc["ci_lower_bound", "accuracy"], 4)) 
            title_cm = title_cm + ", " + str(round(slide_metrics_ci.loc["ci_upper_bound", "accuracy"], 4)) + ")"

            # create confidence matrix plot with confidence interval and write out
            save_conf_mat_plot_ci(slide_metrics_out.iloc[:, 1:], labelorder, title_cm, output_dir)

        slide_metrics_out.to_csv(output_dir / 'slide_metrics.csv')

    def calc_slide_metrics(self, title, output_dir, ci=True, nreps=1000, posname='tumor', labelorder=None, newlabels=None):
        if len(self.slide_labels) == 2:
            self.calc_slide_metrics_binary(title, output_dir, ci=ci, nreps=1000, posname=posname)
        else:
            self.calc_slide_metrics_multi(title, output_dir, ci=ci, nreps=1000, labelorder=labelorder, newlabels=newlabels)


class SlideClassifierWang(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str, posname: str = 'tumor') -> pd.DataFrame:
        def get_global_features(img_grey: np.array, labelled_image: np.ndarray, tissue_area: int) -> Tuple[float, float]:
            """ Create features based on whole slide properties of a given trheshold

            Args:
                img_grey: a greyscale heatmap where each pixel represents a patch. pixel values from 0, 255
                    with 255 representing a probability of one
                thresh: the threshold probability to use to bbinarize the image
                tiss_area: the area of the image that is tissue in pixels

            Returns:
                A tuple containing:
                    area_ratio - the ratio of number of pixels over the given probability to the tissue area
                    prob_area - the sum of probability of all the pixels over the threshold divided by the tissue area
            """

            # measure connected components
            reg_props_t = regionprops(labelled_image)
            # get area for each region
            img_areas = [reg.area for reg in reg_props_t]
            # get total area of tumor regions
            metastatic_area = np.sum(img_areas)
            # get area ratio
            area_ratio = metastatic_area / tissue_area

            # get list of regions
            labels_t = np.unique(labelled_image)
            # create empty list of same size
            lab_list_t = np.zeros((len(labels_t), 1))
            # for each region
            for lab in range(1, len(labels_t)):
                # get a mask of just that region
                mask = labelled_image == lab
                # sum the probability over the region in the mask
                tot_prob = np.sum(np.divide(img_grey[mask], 255))
                # add to empty list
                lab_list_t[lab, 0] = tot_prob
            # sum over whole list
            tot_prob_t = np.sum(lab_list_t)
            # diveide by tissue area
            prob_area = tot_prob_t / tissue_area

            return area_ratio, prob_area

        def get_region_features(reg) -> list:
            """ Get list of properties of a ragion

            Args:
                reg: a region from regionprops function

            Returns:
                A list of 11 region properties

            """
            # get area of region
            reg_area = reg.area
            # eccentricity - for an ellipse with same second moments as region
            # divide distance between focal points by length of major axis
            reg_eccent = reg.eccentricity
            # extent ratio of pixels in region to pixels in bounding box
            reg_extent = reg.extent
            # area of bounding box of region
            reg_bbox_area = reg.bbox_area
            # major axis length of ellipse with same second moment of area
            reg_maj_ax_len = reg.major_axis_length
            # highest probabaility in the region
            reg_max_int = reg.max_intensity
            # mean probability voer he region
            reg_mean_int = reg.mean_intensity
            # lowest probability in the region
            reg_min_int = reg.min_intensity
            # Rrtio of pixels in the region to pixels of the convex hull image.
            reg_solid = reg.solidity
            # cacluate aspect ration of region bounding box
            reg_bbox = reg.bbox
            reg_aratio = (reg_bbox[2] - reg_bbox[0]) / (reg_bbox[3] - reg_bbox[1])

            output_list = [reg_area, reg_eccent, reg_extent, reg_bbox_area, reg_maj_ax_len, reg_max_int,
                        reg_mean_int, reg_min_int, reg_aratio, reg_solid]
            return output_list

    
        print(f'calculating features for {result.slide_path.stem}')

        heatmap = result.to_heatmap(posname)
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        # area of tissue is the number of rows in results dataframe
        tissue_area = result.patches_df.shape[0]

        # set thresholds for global features
        threshz = [0.5, 0.6, 0.7, 0.8, 0.9]

        # create storage for global features as an 1xn array. There are two features for each threshold.
        glob_list = np.zeros((1, 2 * len(threshz)))
        # for each threshold calculate two features and store in array
        for idx, th in enumerate(threshz):
            segmentor = ConnectedComponents(th)
            labelled_image = segmentor.segment(heatmap)
            outvals = get_global_features(heatmap, labelled_image, tissue_area)
            glob_list[0, (idx * 2)] = outvals[0]
            glob_list[0, (idx * 2 + 1)] = outvals[1]

        # get two largest areas at 0.5 thresh
        segmentor = ConnectedComponents(0.5)
        labelled_image = segmentor.segment(heatmap)

        # measure connected components
        reg_props_5 = regionprops(labelled_image, intensity_image=heatmap)

        # get area for each region
        img_areas_5 = [reg.area for reg in reg_props_5]

        # get labels for each region
        img_label_5 = [reg.label for reg in reg_props_5]

        # sort in descending order
        toplabels = [x for _, x in sorted(zip(img_areas_5, img_label_5), reverse=True)][0:2]

        # create empty 1x20 array to store ten feature values each for top 2 lesions
        loc_list = np.zeros((1, 20))

        # per lesion add to store - labels start from 1 need to subtract 1 for zero indexing
        for rg in range(2):
            if len(img_areas_5) > rg:
                reg = reg_props_5[toplabels[rg] - 1]
                outvals = get_region_features(reg)
            else:
                outvals = [0] * 10
            loc_list[0, (rg * 10):((rg + 1) * 10)] = outvals

        # combine global features and lesion features into one array
        features_list = np.hstack((glob_list, loc_list))

        # create column names
        out_cols = ["area_ratio_5", "prob_score_5", "area_ratio_6", "prob_score_6", "area_ratio_7", "prob_score_7",
                    "area_ratio_8", "prob_score_8", "area_ratio_9", "prob_score_9",
                    "area_1", "eccentricity_1", "extent_1", "bbox_area_1", "major_axis_1", "max_intensity_1",
                    "mean_intensity_1", "min_intensity_1", "aspect_ratio_1", "solidity_1",
                    "area_2", "eccentricity_2", "extent_2", "bbox_area_2", "major_axis_2", "max_intensity_2",
                    "mean_intensity_2", "min_intensity_2", "aspect_ratio_2", "solidity_2"]

        # convert to dataframe with column names
        features_df = pd.DataFrame(features_list, columns=out_cols)
        features_df['slidename'] = result.slide_path.stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags

        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']

        # fit or load (NB not all experiments will fit a sepearate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            slide_model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)


class SlideClassifierLee(SlideClassifier):

    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str, posname: str = 'tumor') -> pd.DataFrame:
        def get_region_features(reg) -> list:
            """ Get list of properties of a ragion

            Args:
                reg: a region from regionprops function

            Returns:
                A list of 8 region properties

            """
            # get area of region
            reg_area = reg.area
            # major_axis_length of a regoin
            reg_major_axis = reg.major_axis_length
            # minor_axis_length of a region
            reg_minor_axis = reg.minor_axis_length
            # density of a region
            reg_density = 1 / reg_area
            # mean, max , min  probability of a region
            reg_mean_intensity = reg.mean_intensity
            reg_max_intensity = reg.max_intensity
            reg_min_intensity = reg.min_intensity

            output_list = [reg_area, reg_major_axis, reg_minor_axis, reg_density, reg_mean_intensity, reg_max_intensity,
                        reg_min_intensity]

            return output_list

        print(f'calculating features for {result.slide_path.stem}')

        heatmap = result.to_heatmap(posname)
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        # get two largest areas at 0.5 thresh
        segmentor = DBScan(0.58, eps=3, min_samples=20)
        labelled_image = segmentor.segment(heatmap)
        labelled_image = np.array(labelled_image, dtype='int')

        # measure connected components
        reg_props = regionprops(labelled_image, intensity_image=heatmap)

        # get area for each region
        img_areas = [reg.area for reg in reg_props]

        # get labels for each region
        img_label = [reg.label for reg in reg_props]

        # sort in descending order
        toplabels = [x for _, x in sorted(zip(img_areas, img_label), reverse=True)][0:3]

        # create empty 1x8 array to store 7 feature values each for top 3 lesions
        feature_list = np.zeros((1, 21))

        # labels in image are nto zero indexed reg props are so need to adjust for non zero indexing
        for rg in range(3):
            if len(img_areas) > rg:
                toplab = toplabels[rg]
                topindex = img_label.index(toplab)
                reg = reg_props[topindex]
                outvals = get_region_features(reg)
            else:
                outvals = [0] * 7
            feature_list[0, (rg * 7):((rg + 1) * 7)] = outvals

        out_cols = ["major_axis_1", "minor_axis_1", "area_1", "density_1", "mean_probability_1", "max_probability_1",
                    "min_probability_1", "major_axis_2", "minor_axis_2", "area_2", "density_2", "mean_probability_2", 
                    "max_probability_2", "min_probability_2", "major_axis_3", "minor_axis_3", "area_3", "density_3", 
                    "mean_probability_3", "max_probability_3", "min_probability_3"]
        # convert to dataframe with column names
        features_df = pd.DataFrame(feature_list, columns=out_cols)
        features_df['slidename'] = result.slide_path.stem
        features_df['slide_label'] = result.label
        features_df['tags'] = result.tags

        return features_df


    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        labels = [lab.lower() for lab in labels]
        names = features['slidename']
        tags = features['tags']

        # fit or load (NB not all experiments will fit a separate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            slide_model = XGBClassifier()
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ["true_label"]
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)


class SlideClassifierLiu(SlideClassifier):

    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str, posname: str = 'tumor') -> pd.DataFrame:
        print(f'calculating features for {result.slide_path.stem}')

        output_list = [np.max(result.patches_df[posname]), result.slide_path.stem, result.label, result.tags]
        output_arr = np.reshape(np.array(output_list), (1, 4))

        out_cols = ["max_probability", 'slidename', 'slide_label', 'tags']
        # convert to dataframe with column names
        feature_df = pd.DataFrame(output_arr, columns=out_cols)

        return feature_df

    def predict_slide_level(self, features_dir: Path):
        features_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']

        just_features = just_features.astype(np.float)
        predictions = np.array(just_features.max_probability > 0.998)
        predictions = np.where(predictions, "tumor", "normal")
        # Probabilities for each class
        probabilities = np.array(just_features.max_probability)
        probabilities = probabilities.reshape((probabilities.shape[0], 1))
        probabilities = np.hstack((np.subtract(1, probabilities), probabilities))
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ["true_label"]
        slides_labels_df["true_label"] = [lab.lower() for lab in slides_labels_df["true_label"]]
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)


class SlideClassifierWangMultiCervical(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str):
        def calculate_slide_features_single(heatmap, tissue_area, suffix) -> pd.DataFrame:
            def get_global_features(img_grey: np.array, labelled_image: np.ndarray, tissue_area: int) -> Tuple[float, float]:
                """ Create features based on whole slide properties of a given trheshold
                Args:
                    img_grey: a greyscale heatmap where each pixel represents a patch. pixel values from 0, 255
                        with 255 representing a probability of one
                    thresh: the threshold probability to use to bbinarize the image
                    tiss_area: the area of the image that is tissue in pixels
                Returns:
                    A tuple containing:
                        area_ratio - the ratio of number of pixels over the given probability to the tissue area
                        prob_area - the sum of probability of all the pixels over the threshold divided by the tissue area
                """

                # measure connected components
                reg_props_t = regionprops(labelled_image)
                # get area for each region
                img_areas = [reg.area for reg in reg_props_t]
                # get total area of tumor regions
                metastatic_area = np.sum(img_areas)
                
                # get list of regions
                labels_t = np.unique(labelled_image)
                # create empty list of same size
                lab_list_t = np.zeros((len(labels_t), 1))
                # for each region
                for lab in range(1, len(labels_t)):
                    # get a mask of just that region
                    mask = labelled_image == lab
                    # sum the probability over the region in the mask
                    tot_prob = np.sum(np.divide(img_grey[mask], 255))
                    # add to empty list
                    lab_list_t[lab, 0] = tot_prob
                # sum over whole list
                tot_prob_t = np.sum(lab_list_t)
                if tissue_area > 0:
                    # get area ratio
                    area_ratio = metastatic_area / tissue_area
                    # diveide by tissue area
                    prob_area = tot_prob_t / tissue_area
                else:
                    area_ratio = 0
                    prob_area = 0

                return area_ratio, prob_area

            def get_region_features(reg) -> list:
                """ Get list of properties of a ragion
                Args:
                    reg: a region from regionprops function
                Returns:
                    A list of 11 region properties
                """
                # get area of region
                reg_area = reg.area
                # eccentricity - for an ellipse with same second moments as region
                # divide distance between focal points by length of major axis
                reg_eccent = reg.eccentricity
                # extent ratio of pixels in region to pixels in bounding box
                reg_extent = reg.extent
                # area of bounding box of region
                reg_bbox_area = reg.bbox_area
                # major axis length of ellipse with same second moment of area
                reg_maj_ax_len = reg.major_axis_length
                # highest probabaility in the region
                reg_max_int = reg.max_intensity
                # mean probability voer he region
                reg_mean_int = reg.mean_intensity
                # lowest probability in the region
                reg_min_int = reg.min_intensity
                # Rrtio of pixels in the region to pixels of the convex hull image.
                reg_solid = reg.solidity
                # cacluate aspect ration of region bounding box
                reg_bbox = reg.bbox
                # cacluate area of convex hull surrounding area
                reg_convex = reg.convex_area
                # cacluate area of region with all holes filled
                reg_filled = reg.filled_area
                # minor axis length of ellipse with same second moment of area
                reg_min_ax_len = reg.minor_axis_length
                # diamter of circle with same area as region
                reg_equiv_dia = reg.equivalent_diameter
                # eulers characteristic - no of connected components minus holes 
                reg_euler = reg.euler_number
                # perimeter of region
                reg_perim = reg.perimeter
                reg_aratio = (reg_bbox[2] - reg_bbox[0]) / (reg_bbox[3] - reg_bbox[1])
                
                output_list = [reg_area, reg_eccent, reg_extent, reg_bbox_area, reg_maj_ax_len, reg_max_int,
                               reg_mean_int, reg_min_int, reg_aratio, reg_solid, reg_convex, reg_filled, 
                               reg_min_ax_len, reg_equiv_dia, reg_euler, reg_perim]
                return output_list

            assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

            # area of tissue is the number of rows in results dataframe
            #tissue_area = result.patches_df.shape[0]

            # set thresholds for global features
            threshz = [0.5, 0.6, 0.7, 0.8, 0.9]

            # create storage for global features as an 1xn array. There are two features for each threshold.
            glob_list = np.zeros((1, 2 * len(threshz)))
            # for each threshold calculate two features and store in array
            for idx, th in enumerate(threshz):
                segmentor = ConnectedComponents(th)
                labelled_image = segmentor.segment(heatmap)
                outvals = get_global_features(heatmap, labelled_image, tissue_area)
                glob_list[0, (idx * 2)] = outvals[0]
                glob_list[0, (idx * 2 + 1)] = outvals[1]

            # get two largest areas at 0.5 thresh
            segmentor = ConnectedComponents(0.5)
            labelled_image = segmentor.segment(heatmap)

            # measure connected components
            reg_props_5 = regionprops(labelled_image, intensity_image=heatmap)

            # get area for each region
            img_areas_5 = [reg.area for reg in reg_props_5]

            # get labels for each region
            img_label_5 = [reg.label for reg in reg_props_5]
            
            nregs = 10

            # sort in descending order
            toplabels = [x for _, x in sorted(zip(img_areas_5, img_label_5), reverse=True)][0:nregs]

            # create empty 1x20 array to store ten feature values each for top 2 lesions
            loc_list = np.zeros((1, 16*nregs+1))

            # per lesion add to store - labels start from 1 need to subtract 1 for zero indexing
            for rg in range(nregs):
                if len(img_areas_5) > rg:
                    reg = reg_props_5[toplabels[rg] - 1]
                    outvals = get_region_features(reg)
                else:
                    outvals = [0] * 16
                loc_list[0, (rg * 16):((rg + 1) * 16)] = outvals

            loc_list[0, -1] = len(img_areas_5)

            # combine global features and lesion features into one array
            features_list = np.hstack((glob_list, loc_list))

            # create column names
            out_cols = ["area_ratio_5", "prob_score_5", "area_ratio_6", "prob_score_6", "area_ratio_7", "prob_score_7",
                        "area_ratio_8", "prob_score_8", "area_ratio_9", "prob_score_9"]
            reg_cols = ["area_", "eccentricity_", "extent_", "bbox_area_", "major_axis_", "max_intensity_",
                        "mean_intensity_", "min_intensity_", "aspect_ratio_", "solidity_", "convex_hull_", "filled_area_",
                        "minor_axis_length_", "equivalent_diameter_", "euler_", "perimeter_"]
            for rg in range(nregs):
                rgcols = [cl + str(rg + 1) for cl in reg_cols]
                out_cols = out_cols + rgcols
                
            out_cols = out_cols + ['no_regions']
                
            out_cols = [cl + suffix for cl in out_cols]
            # convert to dataframe with column names
            features_df = pd.DataFrame(features_list, columns=out_cols)
            reg_area_name = 'all_region_areas' + suffix
            features_df[reg_area_name] = ''
            if len(img_areas_5) > 0:
                area_str = str(img_areas_5[0])
                for ar in range(1, len(img_areas_5)):
                        area_str = area_str + ";" + str(img_areas_5[ar])
            else:
                area_str = ''
            features_df[reg_area_name][0] = area_str 
            
            return features_df

        csvpath = result.slide_path.with_suffix('.csv')
       
        malig_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
        malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
        highg_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_high_grade.png')
        highg_hm = np.asarray(Image.open(highg_hm_path)) / 255
        lowg_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_low_grade.png')
        lowg_hm = np.asarray(Image.open(lowg_hm_path)) / 255
        normal_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_normal.png')
        normal_hm = np.asarray(Image.open(normal_hm_path)) / 255
        tissue_area = np.sum((malig_hm + highg_hm + lowg_hm + normal_hm) > 0)
        
        features_ml = calculate_slide_features_single(malig_hm, tissue_area, "_ML")
        features_hg = calculate_slide_features_single(highg_hm, tissue_area, "_HG")
        features_lg = calculate_slide_features_single(lowg_hm, tissue_area, "_LG")
        features_ni = calculate_slide_features_single(normal_hm, tissue_area, "_NI")
 
        features_df = pd.concat((features_ml, features_hg, features_lg, features_ni), axis=1)
        features_df['slidename'] = Path(csvpath).stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags
        features_df['tissue_area'] = tissue_area
        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')
        print('features shape:',features.shape)
        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']
        labels2 = labels
        print("labels", np.unique(labels2))
        # fit or load (NB not all experiments will fit a sepearate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            #slide_model = RandomForestClassifier(n_estimators=700, bootstrap = True,criterion= 'gini',  max_features = 'sqrt')
            slide_model = RandomForestClassifier(n_estimators=220, min_samples_split=2, min_samples_leaf=2,
                    max_features='auto', max_depth=21, criterion='entropy', class_weight='balanced_subsample', bootstrap=True)
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        
        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)

class SlideClassifierWangMulti(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str):
        def calculate_slide_features_single(heatmap, tissue_area, suffix) -> pd.DataFrame:
            def get_global_features(img_grey: np.array, labelled_image: np.ndarray, tissue_area: int) -> Tuple[float, float]:
                """ Create features based on whole slide properties of a given trheshold

                Args:
                    img_grey: a greyscale heatmap where each pixel represents a patch. pixel values from 0, 255
                        with 255 representing a probability of one
                    thresh: the threshold probability to use to bbinarize the image
                    tiss_area: the area of the image that is tissue in pixels

                Returns:
                    A tuple containing:
                        area_ratio - the ratio of number of pixels over the given probability to the tissue area
                        prob_area - the sum of probability of all the pixels over the threshold divided by the tissue area
                """

                # measure connected components
                reg_props_t = regionprops(labelled_image)
                # get area for each region
                img_areas = [reg.area for reg in reg_props_t]
                # get total area of tumor regions
                metastatic_area = np.sum(img_areas)
                # get area ratio
                area_ratio = metastatic_area / tissue_area

                # get list of regions
                labels_t = np.unique(labelled_image)
                # create empty list of same size
                lab_list_t = np.zeros((len(labels_t), 1))
                # for each region
                for lab in range(1, len(labels_t)):
                    # get a mask of just that region
                    mask = labelled_image == lab
                    # sum the probability over the region in the mask
                    tot_prob = np.sum(np.divide(img_grey[mask], 255))
                    # add to empty list
                    lab_list_t[lab, 0] = tot_prob
                # sum over whole list
                tot_prob_t = np.sum(lab_list_t)
                if tissue_area > 0:
                    # get area ratio
                    area_ratio = metastatic_area / tissue_area
                    # diveide by tissue area
                    prob_area = tot_prob_t / tissue_area
                else:
                    area_ratio = 0
                    prob_area = 0

                return area_ratio, prob_area

            def get_region_features(reg) -> list:
                """ Get list of properties of a ragion

                Args:
                    reg: a region from regionprops function

                Returns:
                    A list of 11 region properties

                """
                # get area of region
                reg_area = reg.area
                # eccentricity - for an ellipse with same second moments as region
                # divide distance between focal points by length of major axis
                reg_eccent = reg.eccentricity
                # extent ratio of pixels in region to pixels in bounding box
                reg_extent = reg.extent
                # area of bounding box of region
                reg_bbox_area = reg.bbox_area
                # major axis length of ellipse with same second moment of area
                reg_maj_ax_len = reg.major_axis_length
                # highest probabaility in the region
                reg_max_int = reg.max_intensity
                # mean probability voer he region
                reg_mean_int = reg.mean_intensity
                # lowest probability in the region
                reg_min_int = reg.min_intensity
                # Rrtio of pixels in the region to pixels of the convex hull image.
                reg_solid = reg.solidity
                # cacluate aspect ration of region bounding box
                reg_bbox = reg.bbox
                reg_aratio = (reg_bbox[2] - reg_bbox[0]) / (reg_bbox[3] - reg_bbox[1])

                output_list = [reg_area, reg_eccent, reg_extent, reg_bbox_area, reg_maj_ax_len, reg_max_int,
                            reg_mean_int, reg_min_int, reg_aratio, reg_solid]
                return output_list

            assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

            # area of tissue is the number of rows in results dataframe
            #tissue_area = result.patches_df.shape[0]

            # set thresholds for global features
            threshz = [0.5, 0.6, 0.7, 0.8, 0.9]

            # create storage for global features as an 1xn array. There are two features for each threshold.
            glob_list = np.zeros((1, 2 * len(threshz)))
            # for each threshold calculate two features and store in array
            for idx, th in enumerate(threshz):
                segmentor = ConnectedComponents(th)
                labelled_image = segmentor.segment(heatmap)
                outvals = get_global_features(heatmap, labelled_image, tissue_area)
                glob_list[0, (idx * 2)] = outvals[0]
                glob_list[0, (idx * 2 + 1)] = outvals[1]

            # get two largest areas at 0.5 thresh
            segmentor = ConnectedComponents(0.5)
            labelled_image = segmentor.segment(heatmap)

            # measure connected components
            reg_props_5 = regionprops(labelled_image, intensity_image=heatmap)

            # get area for each region
            img_areas_5 = [reg.area for reg in reg_props_5]

            # get labels for each region
            img_label_5 = [reg.label for reg in reg_props_5]

            # sort in descending order
            toplabels = [x for _, x in sorted(zip(img_areas_5, img_label_5), reverse=True)][0:2]

            # create empty 1x20 array to store ten feature values each for top 2 lesions
            loc_list = np.zeros((1, 20))

            # per lesion add to store - labels start from 1 need to subtract 1 for zero indexing
            for rg in range(2):
                if len(img_areas_5) > rg:
                    reg = reg_props_5[toplabels[rg] - 1]
                    outvals = get_region_features(reg)
                else:
                    outvals = [0] * 10
                loc_list[0, (rg * 10):((rg + 1) * 10)] = outvals

            # combine global features and lesion features into one array
            features_list = np.hstack((glob_list, loc_list))

            # create column names
            out_cols = ["area_ratio_5", "prob_score_5", "area_ratio_6", "prob_score_6", "area_ratio_7", "prob_score_7",
                        "area_ratio_8", "prob_score_8", "area_ratio_9", "prob_score_9",
                        "area_1", "eccentricity_1", "extent_1", "bbox_area_1", "major_axis_1", "max_intensity_1",
                        "mean_intensity_1", "min_intensity_1", "aspect_ratio_1", "solidity_1",
                        "area_2", "eccentricity_2", "extent_2", "bbox_area_2", "major_axis_2", "max_intensity_2",
                        "mean_intensity_2", "min_intensity_2", "aspect_ratio_2", "solidity_2"]
            out_cols = [cl + suffix for cl in out_cols]

            # convert to dataframe with column names
            features_df = pd.DataFrame(features_list, columns=out_cols)

            return features_df

        csvpath = result.slide_path.with_suffix('.csv')
        #results = pd.read_csv(root_dir / 'results' / csvpath)
        #tissue_area = results.shape[0]
        malig_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
        malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
        highg_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_high_grade.png')
        highg_hm = np.asarray(Image.open(highg_hm_path)) / 255
        lowg_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_low_grade.png')
        lowg_hm = np.asarray(Image.open(lowg_hm_path)) / 255
        normal_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_normal.png')
        normal_hm = np.asarray(Image.open(normal_hm_path)) / 255
        tissue_area = np.sum((malig_hm + highg_hm + lowg_hm + normal_hm) > 0)
        
        features_ml = calculate_slide_features_single(malig_hm, tissue_area, "_ML")
        features_hg = calculate_slide_features_single(highg_hm, tissue_area, "_HG")
        features_lg = calculate_slide_features_single(lowg_hm, tissue_area, "_LG")
        
        features_df = pd.concat((features_ml, features_hg, features_lg), axis=1)
        features_df['slidename'] = Path(csvpath).stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags

        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']

        print("labels", np.unique(labels))

        # fit or load (NB not all experiments will fit a sepearate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1500, num = 10)]
            criterion = ['gini', 'entropy']
            max_features = ['auto', 'sqrt', 'log2']
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1,2,4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            #class_weight = ['balanced', 'balanced_subsample'] 
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'criterion': criterion ,
                           'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'bootstrap': bootstrap}
                            #'class_weight': class_weight}
            rf = RandomForestClassifier()
            slide_model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 300, cv = 5, verbose=0, random_state=123, n_jobs = -1)


            #slide_model = RandomForestClassifier(n_estimators=400,random_state = 123, criterion = 'entropy', min_samples_split = 4 , max_depth = 10 , max_features = 'sqrt', bootstrap = True)
            #slide_model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')
        print('\n \nBest model parameters:\n',slide_model.best_params_)
        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)


class SlideClassifierWangMultiEndo(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str):
        def calculate_slide_features_single(heatmap, tissue_area, suffix) -> pd.DataFrame:
            def get_global_features(img_grey: np.array, labelled_image: np.ndarray, tissue_area: int) -> Tuple[float, float]:
                """ Create features based on whole slide properties of a given trheshold

                Args:
                    img_grey: a greyscale heatmap where each pixel represents a patch. pixel values from 0, 255
                        with 255 representing a probability of one
                    thresh: the threshold probability to use to bbinarize the image
                    tiss_area: the area of the image that is tissue in pixels

                Returns:
                    A tuple containing:
                        area_ratio - the ratio of number of pixels over the given probability to the tissue area
                        prob_area - the sum of probability of all the pixels over the threshold divided by the tissue area
                """

                # measure connected components
                reg_props_t = regionprops(labelled_image)
                # get area for each region
                img_areas = [reg.area for reg in reg_props_t]
                # get total area of tumor regions
                metastatic_area = np.sum(img_areas)
                

                # get list of regions
                labels_t = np.unique(labelled_image)
                # create empty list of same size
                lab_list_t = np.zeros((len(labels_t), 1))
                # for each region
                for lab in range(1, len(labels_t)):
                    # get a mask of just that region
                    mask = labelled_image == lab
                    # sum the probability over the region in the mask
                    tot_prob = np.sum(np.divide(img_grey[mask], 255))
                    # add to empty list
                    lab_list_t[lab, 0] = tot_prob
                # sum over whole list
                tot_prob_t = np.sum(lab_list_t)
                
                if tissue_area > 0:
                    # get area ratio
                    area_ratio = metastatic_area / tissue_area
                    # diveide by tissue area
                    prob_area = tot_prob_t / tissue_area
                else:
                    area_ratio = 0
                    prob_area = 0

                return area_ratio, prob_area

            def get_region_features(reg) -> list:
                """ Get list of properties of a ragion

                Args:
                    reg: a region from regionprops function

                Returns:
                    A list of 11 region properties

                """
                # get area of region
                reg_area = reg.area
                # eccentricity - for an ellipse with same second moments as region
                # divide distance between focal points by length of major axis
                reg_eccent = reg.eccentricity
                # extent ratio of pixels in region to pixels in bounding box
                reg_extent = reg.extent
                # area of bounding box of region
                reg_bbox_area = reg.bbox_area
                # major axis length of ellipse with same second moment of area
                reg_maj_ax_len = reg.major_axis_length
                # highest probabaility in the region
                reg_max_int = reg.max_intensity
                # mean probability voer he region
                reg_mean_int = reg.mean_intensity
                # lowest probability in the region
                reg_min_int = reg.min_intensity
                # Rrtio of pixels in the region to pixels of the convex hull image.
                reg_solid = reg.solidity
                # cacluate aspect ration of region bounding box
                reg_bbox = reg.bbox
                reg_aratio = (reg_bbox[2] - reg_bbox[0]) / (reg_bbox[3] - reg_bbox[1])

                output_list = [reg_area, reg_eccent, reg_extent, reg_bbox_area, reg_maj_ax_len, reg_max_int,
                            reg_mean_int, reg_min_int, reg_aratio, reg_solid]
                return output_list

            assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

            # area of tissue is the number of rows in results dataframe
            #tissue_area = result.patches_df.shape[0]

            # set thresholds for global features
            threshz = [0.5, 0.6, 0.7, 0.8, 0.9]

            # create storage for global features as an 1xn array. There are two features for each threshold.
            glob_list = np.zeros((1, 2 * len(threshz)))
            # for each threshold calculate two features and store in array
            for idx, th in enumerate(threshz):
                segmentor = ConnectedComponents(th)
                labelled_image = segmentor.segment(heatmap)
                outvals = get_global_features(heatmap, labelled_image, tissue_area)
                glob_list[0, (idx * 2)] = outvals[0]
                glob_list[0, (idx * 2 + 1)] = outvals[1]

            # get two largest areas at 0.5 thresh
            segmentor = ConnectedComponents(0.5)
            labelled_image = segmentor.segment(heatmap)

            # measure connected components
            reg_props_5 = regionprops(labelled_image, intensity_image=heatmap)

            # get area for each region
            img_areas_5 = [reg.area for reg in reg_props_5]

            # get labels for each region
            img_label_5 = [reg.label for reg in reg_props_5]

            # sort in descending order
            toplabels = [x for _, x in sorted(zip(img_areas_5, img_label_5), reverse=True)][0:2]

            # create empty 1x20 array to store ten feature values each for top 2 lesions
            loc_list = np.zeros((1, 20))

            # per lesion add to store - labels start from 1 need to subtract 1 for zero indexing
            for rg in range(2):
                if len(img_areas_5) > rg:
                    reg = reg_props_5[toplabels[rg] - 1]
                    outvals = get_region_features(reg)
                else:
                    outvals = [0] * 10
                loc_list[0, (rg * 10):((rg + 1) * 10)] = outvals

            # combine global features and lesion features into one array
            features_list = np.hstack((glob_list, loc_list))

            # create column names
            out_cols = ["area_ratio_5", "prob_score_5", "area_ratio_6", "prob_score_6", "area_ratio_7", "prob_score_7",
                        "area_ratio_8", "prob_score_8", "area_ratio_9", "prob_score_9",
                        "area_1", "eccentricity_1", "extent_1", "bbox_area_1", "major_axis_1", "max_intensity_1",
                        "mean_intensity_1", "min_intensity_1", "aspect_ratio_1", "solidity_1",
                        "area_2", "eccentricity_2", "extent_2", "bbox_area_2", "major_axis_2", "max_intensity_2",
                        "mean_intensity_2", "min_intensity_2", "aspect_ratio_2", "solidity_2"]
            out_cols = [cl + suffix for cl in out_cols]

            # convert to dataframe with column names
            features_df = pd.DataFrame(features_list, columns=out_cols)

            return features_df

        csvpath = result.slide_path.with_suffix('.csv')
        #results = pd.read_csv(root_dir / 'results' / csvpath)
        #tissue_area = results.shape[0]
        malig_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
        malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
        other_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_other_benign.png')
        other_hm = np.asarray(Image.open(other_hm_path)) / 255
        tissue_area = np.sum((malig_hm + other_hm) > 0)

        features_ml = calculate_slide_features_single(malig_hm, tissue_area, "_ML")
        features_ob = calculate_slide_features_single(other_hm, tissue_area, "_OB")
        
        features_df = pd.concat((features_ml, features_ob), axis=1)
        features_df['slidename'] = Path(csvpath).stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags

        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']

        print("labels", np.unique(labels))

        # fit or load (NB not all experiments will fit a sepearate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            slide_model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)



class SlideClassifierWangBinary(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str):
        def calculate_slide_features_single(heatmap, tissue_area, suffix) -> pd.DataFrame:
            def get_global_features(img_grey: np.array, labelled_image: np.ndarray, tissue_area: int) -> Tuple[float, float]:
                """ Create features based on whole slide properties of a given trheshold

                Args:
                    img_grey: a greyscale heatmap where each pixel represents a patch. pixel values from 0, 255
                        with 255 representing a probability of one
                    thresh: the threshold probability to use to bbinarize the image
                    tiss_area: the area of the image that is tissue in pixels

                Returns:
                    A tuple containing:
                        area_ratio - the ratio of number of pixels over the given probability to the tissue area
                        prob_area - the sum of probability of all the pixels over the threshold divided by the tissue area
                """

                # measure connected components
                reg_props_t = regionprops(labelled_image)
                # get area for each region
                img_areas = [reg.area for reg in reg_props_t]
                # get total area of tumor regions
                metastatic_area = np.sum(img_areas)
                # get area ratio
                area_ratio = metastatic_area / tissue_area

                # get list of regions
                labels_t = np.unique(labelled_image)
                # create empty list of same size
                lab_list_t = np.zeros((len(labels_t), 1))
                # for each region
                for lab in range(1, len(labels_t)):
                    # get a mask of just that region
                    mask = labelled_image == lab
                    # sum the probability over the region in the mask
                    tot_prob = np.sum(np.divide(img_grey[mask], 255))
                    # add to empty list
                    lab_list_t[lab, 0] = tot_prob
                # sum over whole list
                tot_prob_t = np.sum(lab_list_t)
                # diveide by tissue area
                prob_area = tot_prob_t / tissue_area

                return area_ratio, prob_area

            def get_region_features(reg) -> list:
                """ Get list of properties of a ragion

                Args:
                    reg: a region from regionprops function

                Returns:
                    A list of 11 region properties

                """
                # get area of region
                reg_area = reg.area
                # eccentricity - for an ellipse with same second moments as region
                # divide distance between focal points by length of major axis
                reg_eccent = reg.eccentricity
                # extent ratio of pixels in region to pixels in bounding box
                reg_extent = reg.extent
                # area of bounding box of region
                reg_bbox_area = reg.bbox_area
                # major axis length of ellipse with same second moment of area
                reg_maj_ax_len = reg.major_axis_length
                # highest probabaility in the region
                reg_max_int = reg.max_intensity
                # mean probability voer he region
                reg_mean_int = reg.mean_intensity
                # lowest probability in the region
                reg_min_int = reg.min_intensity
                # Rrtio of pixels in the region to pixels of the convex hull image.
                reg_solid = reg.solidity
                # cacluate aspect ration of region bounding box
                reg_bbox = reg.bbox
                reg_aratio = (reg_bbox[2] - reg_bbox[0]) / (reg_bbox[3] - reg_bbox[1])

                output_list = [reg_area, reg_eccent, reg_extent, reg_bbox_area, reg_maj_ax_len, reg_max_int,
                            reg_mean_int, reg_min_int, reg_aratio, reg_solid]
                return output_list

            assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

            # area of tissue is the number of rows in results dataframe
            #tissue_area = result.patches_df.shape[0]

            # set thresholds for global features
            threshz = [0.5, 0.6, 0.7, 0.8, 0.9]

            # create storage for global features as an 1xn array. There are two features for each threshold.
            glob_list = np.zeros((1, 2 * len(threshz)))
            # for each threshold calculate two features and store in array
            for idx, th in enumerate(threshz):
                segmentor = ConnectedComponents(th)
                labelled_image = segmentor.segment(heatmap)
                outvals = get_global_features(heatmap, labelled_image, tissue_area)
                glob_list[0, (idx * 2)] = outvals[0]
                glob_list[0, (idx * 2 + 1)] = outvals[1]

            # get two largest areas at 0.5 thresh
            segmentor = ConnectedComponents(0.5)
            labelled_image = segmentor.segment(heatmap)

            # measure connected components
            reg_props_5 = regionprops(labelled_image, intensity_image=heatmap)

            # get area for each region
            img_areas_5 = [reg.area for reg in reg_props_5]

            # get labels for each region
            img_label_5 = [reg.label for reg in reg_props_5]

            # sort in descending order
            toplabels = [x for _, x in sorted(zip(img_areas_5, img_label_5), reverse=True)][0:2]

            # create empty 1x20 array to store ten feature values each for top 2 lesions
            loc_list = np.zeros((1, 20))

            # per lesion add to store - labels start from 1 need to subtract 1 for zero indexing
            for rg in range(2):
                if len(img_areas_5) > rg:
                    reg = reg_props_5[toplabels[rg] - 1]
                    outvals = get_region_features(reg)
                else:
                    outvals = [0] * 10
                loc_list[0, (rg * 10):((rg + 1) * 10)] = outvals

            # combine global features and lesion features into one array
            features_list = np.hstack((glob_list, loc_list))

            # create column names
            out_cols = ["area_ratio_5", "prob_score_5", "area_ratio_6", "prob_score_6", "area_ratio_7", "prob_score_7",
                        "area_ratio_8", "prob_score_8", "area_ratio_9", "prob_score_9",
                        "area_1", "eccentricity_1", "extent_1", "bbox_area_1", "major_axis_1", "max_intensity_1",
                        "mean_intensity_1", "min_intensity_1", "aspect_ratio_1", "solidity_1",
                        "area_2", "eccentricity_2", "extent_2", "bbox_area_2", "major_axis_2", "max_intensity_2",
                        "mean_intensity_2", "min_intensity_2", "aspect_ratio_2", "solidity_2"]
            out_cols = [cl + suffix for cl in out_cols]

            # convert to dataframe with column names
            features_df = pd.DataFrame(features_list, columns=out_cols)

            return features_df

        csvpath = result.slide_path.with_suffix('.csv')
        #results = pd.read_csv(root_dir / 'results' / csvpath)
        #tissue_area = results.shape[0]
        malig_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
        malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
        normal_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_not_malignant.png')
        normal_hm = np.asarray(Image.open(normal_hm_path)) / 255
        tissue_area = np.sum((malig_hm + normal_hm) > 0)

        features_ml = calculate_slide_features_single(malig_hm, tissue_area, "_ML")
        
        #features_df = pd.concat((features_ml), axis=1)
        features_df = features_ml
        features_df['slidename'] = Path(csvpath).stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags

        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        labels = np.where(labels=='malignant', 'malignant', 'not_malignant')
        names = features['slidename']
        tags = features['tags']

        print("labels", np.unique(labels))

        # fit or load (NB not all experiments will fit a sepearate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            slide_model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)
            

class SlideClassifierLeeMulti(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str) -> pd.DataFrame:
        def calculate_slide_features_single(heatmap,tissue_area, posname):
            def get_region_features(reg) -> list:
                """ Get list of properties of a ragion

                Args:
                    reg: a region from regionprops function

                Returns:
                    A list of 8 region properties

                """
                # get area of region
                reg_area = reg.area
                # major_axis_length of a regoin
                reg_major_axis = reg.major_axis_length
                # minor_axis_length of a region
                reg_minor_axis = reg.minor_axis_length
                # density of a region
                reg_density = 1 / reg_area
                # mean, max , min  probability of a region
                reg_mean_intensity = reg.mean_intensity
                reg_max_intensity = reg.max_intensity
                reg_min_intensity = reg.min_intensity

                output_list = [reg_area, reg_major_axis, reg_minor_axis, reg_density, reg_mean_intensity, reg_max_intensity,
                            reg_min_intensity]

                return output_list

            print(f'calculating features for {result.slide_path.stem}')
            #heatmap = result.to_heatmap(posname)
            assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"
            
            # get two largest areas at 0.5 thresh
            
            segmentor = DBScan(0.58, eps=3, min_samples=20)
            labelled_image = segmentor.segment(heatmap)
            labelled_image = np.array(labelled_image, dtype='int')

            # measure connected components
            reg_props = regionprops(labelled_image, intensity_image=heatmap)

            # get area for each region
            img_areas = [reg.area for reg in reg_props]

            # get labels for each region
            img_label = [reg.label for reg in reg_props]

            # sort in descending order
            toplabels = [x for _, x in sorted(zip(img_areas, img_label), reverse=True)][0:3]

            # create empty 1x8 array to store 7 feature values each for top 3 lesions
            feature_list = np.zeros((1, 21))

            # labels in image are nto zero indexed reg props are so need to adjust for non zero indexing
            for rg in range(3):
                if len(img_areas) > rg:
                    toplab = toplabels[rg]
                    topindex = img_label.index(toplab)
                    reg = reg_props[topindex]
                    outvals = get_region_features(reg)
                else:
                    outvals = [0] * 7

            feature_list[0, (rg * 7):((rg + 1) * 7)] = outvals

            out_cols = ["major_axis_1", "minor_axis_1", "area_1", "density_1", "mean_probability_1", "max_probability_1",
                            "min_probability_1", "major_axis_2", "minor_axis_2", "area_2", "density_2", "mean_probability_2", 
                            "max_probability_2", "min_probability_2", "major_axis_3", "minor_axis_3", "area_3", "density_3", 
                            "mean_probability_3", "max_probability_3", "min_probability_3"]
            # convert to dataframe with column names
            out_cols = [cl + posname for cl in out_cols]
            features_df = pd.DataFrame(feature_list, columns=out_cols)
            # features_df['slidename'] = result.slide_path.stem
            # features_df['slide_label'] = result.label
            # features_df['tags'] = result.tags

            return features_df    
        csvpath = result.slide_path.with_suffix('.csv')
        malig_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
        malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
        highg_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_high_grade.png')
        highg_hm = np.asarray(Image.open(highg_hm_path)) / 255
        lowg_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_low_grade.png')
        lowg_hm = np.asarray(Image.open(lowg_hm_path)) / 255
        normal_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_normal.png')
        normal_hm = np.asarray(Image.open(normal_hm_path)) / 255
        tissue_area = np.sum((malig_hm + highg_hm + lowg_hm + normal_hm) > 0)

        features_ml = calculate_slide_features_single(malig_hm, tissue_area, "_ML")
        features_hg = calculate_slide_features_single(highg_hm, tissue_area, "_HG")
        features_lg = calculate_slide_features_single(lowg_hm, tissue_area, "_LG")
        
        features_df = pd.concat((features_ml, features_hg, features_lg), axis=1)
        features_df['slidename'] = Path(csvpath).stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags

        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        labels = [lab.lower() for lab in labels]
        names = features['slidename']
        tags = features['tags']

        # fit or load (NB not all experiments will fit a separate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            slide_model = XGBClassifier(booster = 'gbtree', objective='multi:softprob', eval_metric = 'merror', eta = 0.5, gamma = 0.5 , max_depth = 10, base_score = 0.3)
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')
        #print('\nBest model parameters:\n', slide_model.best_params_)
        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ["true_label"]
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)


def calculate_slide_features_single_endoupdate(heatmap, tissue_area, suffix) -> pd.DataFrame:
    def get_global_features(img_grey: np.array, labelled_image: np.ndarray, tissue_area: int) -> Tuple[float, float]:
        """ Create features based on whole slide properties of a given trheshold

        Args:
            img_grey: a greyscale heatmap where each pixel represents a patch. pixel values from 0, 255
                with 255 representing a probability of one
            thresh: the threshold probability to use to bbinarize the image
            tiss_area: the area of the image that is tissue in pixels

        Returns:
            A tuple containing:
                area_ratio - the ratio of number of pixels over the given probability to the tissue area
                prob_area - the sum of probability of all the pixels over the threshold divided by the tissue area
        """

        # measure connected components
        reg_props_t = regionprops(labelled_image)
        # get area for each region
        img_areas = [reg.area for reg in reg_props_t]
        # get total area of tumor regions
        metastatic_area = np.sum(img_areas)
        

        # get list of regions
        labels_t = np.unique(labelled_image)
        # create empty list of same size
        lab_list_t = np.zeros((len(labels_t), 1))
        # for each region
        for lab in range(1, len(labels_t)):
            # get a mask of just that region
            mask = labelled_image == lab
            # sum the probability over the region in the mask
            tot_prob = np.sum(np.divide(img_grey[mask], 255))
            # add to empty list
            lab_list_t[lab, 0] = tot_prob
        # sum over whole list
        tot_prob_t = np.sum(lab_list_t)
        
        if tissue_area > 0:
            # get area ratio
            area_ratio = metastatic_area / tissue_area
            # diveide by tissue area
            prob_area = tot_prob_t / tissue_area
        else:
            area_ratio = 0
            prob_area = 0

        return area_ratio

    def get_region_features(reg) -> list:
        """ Get list of properties of a ragion

        Args:
            reg: a region from regionprops function

        Returns:
            A list of 11 region properties

        """
        # get area of region
        reg_area = reg.area
        # eccentricity - for an ellipse with same second moments as region
        # divide distance between focal points by length of major axis
        reg_eccent = reg.eccentricity
        # extent ratio of pixels in region to pixels in bounding box
        reg_extent = reg.extent
        # area of bounding box of region
        reg_bbox_area = reg.bbox_area
        # major axis length of ellipse with same second moment of area
        reg_maj_ax_len = reg.major_axis_length
        # highest probabaility in the region
        reg_max_int = reg.max_intensity
        # mean probability voer he region
        reg_mean_int = reg.mean_intensity
        # Rrtio of pixels in the region to pixels of the convex hull image.
        reg_solid = reg.solidity
        # cacluate aspect ration of region bounding box
        reg_bbox = reg.bbox
        # minor axis length of ellipse with same second moment of area
        reg_min_ax_len = reg.minor_axis_length
        # eulers characteristic - no of connected components minus holes 
        reg_euler = reg.euler_number
        # perimeter of region
        reg_perim = reg.perimeter
        #reg_aratio = (reg_bbox[2] - reg_bbox[0]) / (reg_bbox[3] - reg_bbox[1])

        output_list = [reg_area, reg_eccent, reg_extent, reg_bbox_area, reg_maj_ax_len, reg_max_int,
                        reg_mean_int, reg_solid, reg_min_ax_len]
        return output_list

    assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

    # area of tissue is the number of rows in results dataframe
    #tissue_area = result.patches_df.shape[0]

    # set thresholds for global features
    threshz = [0.6, 0.9]

    # create storage for global features as an 1xn array. There are two features for each threshold.
    glob_list = np.zeros((1, len(threshz)))
    # for each threshold calculate two features and store in array
    for idx, th in enumerate(threshz):
        segmentor = ConnectedComponents(th)
        labelled_image = segmentor.segment(heatmap)
        outvals = get_global_features(heatmap, labelled_image, tissue_area)
        glob_list[0, (idx)] = outvals

    # get two largest areas at 0.5 thresh
    segmentor = ConnectedComponents(0.5)
    labelled_image = segmentor.segment(heatmap)

    # measure connected components
    reg_props_5 = regionprops(labelled_image, intensity_image=heatmap)

    # get area for each region
    img_areas_5 = [reg.area for reg in reg_props_5]

    # get labels for each region
    img_label_5 = [reg.label for reg in reg_props_5]
    
    # create column names
    out_cols = ["area_ratio_6", "area_ratio_9"]
    reg_cols = ["area_", "eccentricity_", "extent_", "bbox_area_", "major_axis_", "max_intensity_",
                "mean_intensity_", "solidity_", "minor_axis_length_"]
    
    if suffix == '_ML':  
        nregs = 3
        mask = [True] * len(reg_cols)
    else:
        nregs = 2
        mask = [True, False, False, True, True, False, True, False, True]
        
    # sort in descending order
    toplabels = [x for _, x in sorted(zip(img_areas_5, img_label_5), reverse=True)][0:nregs]

    # create empty 1x20 array to store ten feature values each for top 2 lesions
    loc_list = np.zeros((1, len(reg_cols)*nregs))

    # per lesion add to store - labels start from 1 need to subtract 1 for zero indexing
    for rg in range(nregs):
        if len(img_areas_5) > rg:
            reg = reg_props_5[toplabels[rg] - 1]
            outvals = get_region_features(reg)
        else:
            outvals = [0] * len(reg_cols)
        loc_list[0, (rg * len(reg_cols)):((rg + 1) * len(reg_cols))] = outvals

    loc_list[0, -1] = len(img_areas_5)

    
    # combine global features and lesion features into one array
    features_list = np.hstack((glob_list, loc_list))


    for rg in range(nregs):
        rgcols = [cl + str(rg + 1) for cl in reg_cols]
        out_cols = out_cols + rgcols
        
    #out_cols = out_cols + ['no_regions']
    #mask =  [True] * 2 + mask * nregs + [True]
    mask =  [True] * 2 + mask * nregs
    
    out_cols = [cl + suffix for cl in out_cols]

    # convert to dataframe with column names
    features_df = pd.DataFrame(features_list, columns=out_cols)
    features_df = features_df.loc[:, mask]
    reg_area_name = 'n_regions_gt10' + suffix
    nreggt10 = 0
    if len(img_areas_5) > 0:
        for ar in range(len(img_areas_5)):
            if img_areas_5[ar] > 10:
                nreggt10 += 1

    features_df[reg_area_name] = nreggt10

    return features_df

class SlideClassifierEndoUpdate(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str):
        csvpath = result.slide_path.with_suffix('.csv')
        #results = pd.read_csv(root_dir / 'results' / csvpath)
        #tissue_area = results.shape[0]
        malig_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
        malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
        other_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_other_benign.png')
        other_hm = np.asarray(Image.open(other_hm_path)) / 255
        tissue_area = np.sum((malig_hm + other_hm) > 0)

        features_ml = calculate_slide_features_single_endoupdate(malig_hm, tissue_area, "_ML")
        features_ob = calculate_slide_features_single_endoupdate(other_hm, tissue_area, "_OB")
        
        features_df = pd.concat((features_ml, features_ob), axis=1)
        features_df['tissue_area'] = tissue_area
        features_df['slidename'] = Path(csvpath).stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags
        

        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, valid_features_dir: Path, retrain=False):
        def custom_eval_func(predt, dtrain):
            true_labs = np.array(dtrain.get_label(), dtype=np.int64)
            pred_labs = np.argmax(predt, axis=1)
            acc = np.sum(true_labs != pred_labs)
            sens = np.sum(np.logical_and(true_labs == 1, np.logical_not(pred_labs == 1)))
            tot = acc + sens
            return "custommetric", sens
        
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']
        #subcat = tags.str.split(';', expand=True).iloc[:, 0]
        #labels2 = labels.where(subcat!='hyperplasia_with_atypia', other=subcat)
        labels2 = labels
        weights = np.ones(len(labels2))
        weights = np.where(labels2 == "other_benign", 0.05, weights)
        weights = np.where(labels2 == "insufficient", 0.1, weights)

        featuresv = pd.read_csv(valid_features_dir / 'features.csv')

        just_featuresv = featuresv.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labelsv = featuresv['slide_label']
        namesv = featuresv['slidename']
        tagsv = featuresv['tags']
        
        print("labels", np.unique(labels2))

        # fit or load (NB not all experiments will fit a sepearate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            #slide_model = RandomForestClassifier(n_estimators=120, bootstrap = True, max_features = None, criterion='gini', max_depth=6)
            slide_model = XGBClassifier(n_estimators=200, max_depth=int(2), verbosity=0)
            slide_model.fit(just_features, labels2, sample_weight=weights, eval_metric=custom_eval_func, eval_set=[(just_featuresv, labelsv)], early_stopping_rounds=20, verbose=False)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        #preds_probs_df.columns = ["predictions"] + ['hyperplasia_with_atypia', 'insufficient', 'malignant', 'other_benign'] #list(self.slide_labels.keys())
        preds_probs_df.columns = ["predictions"] + ['insufficient', 'malignant', 'other_benign'] #list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels2)
        slides_labels_df.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)
        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)

        just_featuresv = just_featuresv.astype(np.float)
        predictionsv = slide_model.predict(just_featuresv)
        # Probabilities for each class
        probabilitiesv = slide_model.predict_proba(just_featuresv)
        # combine predictions and probailities into a dataframe
        reshaped_predictionsv = predictionsv.reshape((predictionsv.shape[0], 1))
        preds_probs_dfv = pd.DataFrame(np.hstack((reshaped_predictionsv, probabilitiesv)))
        #preds_probs_df.columns = ["predictions"] + ['hyperplasia_with_atypia', 'insufficient', 'malignant', 'other_benign'] #list(self.slide_labels.keys())
        preds_probs_dfv.columns = ["predictions"] + ['insufficient', 'malignant', 'other_benign'] #list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_dfv = pd.DataFrame(labelsv)
        slides_labels_dfv.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_dfv = pd.DataFrame(namesv)
        slides_names_dfv.columns = ["slide_names"]
        tags_dfv = pd.DataFrame(tagsv)
        tags_dfv.columns = ["tags"]
        slide_resultsv = pd.concat((slides_names_dfv, slides_labels_dfv, tags_dfv, preds_probs_dfv), axis=1)

        slide_resultsv.to_csv(valid_features_dir / 'slide_results.csv', index=False)


def calculate_slide_features_single_cervicalupdate(heatmap, tissue_area, suffix) -> pd.DataFrame:
    def get_global_features(img_grey: np.array, labelled_image: np.ndarray, tissue_area: int) -> Tuple[float, float]:
        """ Create features based on whole slide properties of a given trheshold
        Args:
            img_grey: a greyscale heatmap where each pixel represents a patch. pixel values from 0, 255
                with 255 representing a probability of one
            thresh: the threshold probability to use to bbinarize the image
            tiss_area: the area of the image that is tissue in pixels
        Returns:
            A tuple containing:
                area_ratio - the ratio of number of pixels over the given probability to the tissue area
                prob_area - the sum of probability of all the pixels over the threshold divided by the tissue area
        """

        # measure connected components
        reg_props_t = regionprops(labelled_image)
        # get area for each region
        img_areas = [reg.area for reg in reg_props_t]
        # get total area of tumor regions
        metastatic_area = np.sum(img_areas)
        
        
        # get list of regions
        labels_t = np.unique(labelled_image)
        # create empty list of same size
        lab_list_t = np.zeros((len(labels_t), 1))
        # for each region
        for lab in range(1, len(labels_t)):
            # get a mask of just that region
            mask = labelled_image == lab
            # sum the probability over the region in the mask
            tot_prob = np.sum(np.divide(img_grey[mask], 255))
            # add to empty list
            lab_list_t[lab, 0] = tot_prob
        # sum over whole list
        tot_prob_t = np.sum(lab_list_t)
        
        if tissue_area > 0:
            # get area ratio
            area_ratio = metastatic_area / tissue_area
            # diveide by tissue area
            prob_area = tot_prob_t / tissue_area
        else:
            area_ratio = 0
            prob_area = 0

        return area_ratio
        
    def get_region_features(reg) -> list:
        """ Get list of properties of a ragion
        Args:
            reg: a region from regionprops function
        Returns:
            A list of 12 region properties
        """
        # get area of region
        reg_area = reg.area
        # area of bounding box of region
        reg_bbox_area = reg.bbox_area
        # major axis length of ellipse with same second moment of area
        reg_maj_ax_len = reg.major_axis_length
        # highest probabaility in the region
        reg_max_int = reg.max_intensity
        # mean probability voer he region
        reg_mean_int = reg.mean_intensity
        # lowest probability in the region
        reg_min_int = reg.min_intensity
        # cacluate area of convex hull surrounding area
        reg_convex = reg.convex_area
        # cacluate area of region with all holes filled
        reg_filled = reg.filled_area
        # minor axis length of ellipse with same second moment of area
        reg_min_ax_len = reg.minor_axis_length
        # diamter of circle with same area as region
        reg_equiv_dia = reg.equivalent_diameter
            # eulers characteristic - no of connected components minus holes 
        reg_euler = reg.euler_number
        # perimeter of region
        reg_perim = reg.perimeter

        output_list = [reg_area, reg_bbox_area, reg_maj_ax_len, reg_max_int, reg_mean_int,reg_min_int, reg_convex, reg_filled, reg_min_ax_len, reg_equiv_dia, reg_euler, reg_perim]
        return output_list
    
    assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"
    
    
    # set thresholds for global features
    threshz = [0.5, 0.7, 0.9]
    
    # create storage for global features as an 1xn array. There are two features for each threshold.
    glob_list = np.zeros((1, len(threshz)))
    # for each threshold calculate two features and store in array
    for idx, th in enumerate(threshz):
        segmentor = ConnectedComponents(th)
        labelled_image = segmentor.segment(heatmap)
        outvals = get_global_features(heatmap, labelled_image, tissue_area)
        glob_list[0, (idx)] = outvals

    # get two largest areas at 0.5 thresh
    segmentor = ConnectedComponents(0.5)
    labelled_image = segmentor.segment(heatmap)

    # measure connected components
    reg_props_5 = regionprops(labelled_image, intensity_image=heatmap)

    # get area for each region
    img_areas_5 = [reg.area for reg in reg_props_5]

    # get labels for each region
    img_label_5 = [reg.label for reg in reg_props_5]

    # create column names
    out_cols = ["area_ratio_5", "area_ratio_7", "area_ratio_9"]
    reg_cols = ["area_", "bbox_area_", "major_axis_", "max_intensity_", "mean_intensity_", "min_intensity_",
                    "convex_hull_", "filled_area_", "minor_axis_length_", "equivalent_diameter_", "euler_","perimeter_"]

    if suffix == '_ML':
        nregs = 7 
        mask = [True] * len(reg_cols)
    elif suffix == "_HG":
        nregs = 5 
        mask = [True,True,True,True,True,False,True,True,True,True,True,True]
    elif suffix =="_LG":
        nregs = 3
        mask = [False,True,True,True,True,True,True,True,True,False,True,False]
    else:
        nregs = 2 
        mask = [False,True,True,False,True,False,False,True,True,False,False,False]

    # sort in descending order
    toplabels = [x for _, x in sorted(zip(img_areas_5, img_label_5), reverse=True)][0:nregs]

    # create empty 1x20 array to store ten feature values each for top 2 lesions
    loc_list = np.zeros((1, len(reg_cols)*nregs))

    # per lesion add to store - labels start from 1 need to subtract 1 for zero indexing
    for rg in range(nregs):
        if len(img_areas_5) > rg:
            reg = reg_props_5[toplabels[rg] - 1]
            outvals = get_region_features(reg)
        else:
            outvals = [0] * len(reg_cols)
        loc_list[0, (rg * len(reg_cols)):((rg + 1) * len(reg_cols))] = outvals

    loc_list[0, -1] = len(img_areas_5)

    # combine global features and lesion features into one array
    features_list = np.hstack((glob_list, loc_list))
    #print(len(features_list))

    for rg in range(nregs):
        rgcols = [cl + str(rg + 1) for cl in reg_cols]
        out_cols = out_cols + rgcols

    mask =  [True] * 3 + mask * nregs
    
    out_cols = [cl + suffix for cl in out_cols]
    
    # convert to dataframe with column names
    features_df = pd.DataFrame(features_list, columns=out_cols)
    features_df = features_df.loc[:, mask]
    
    reg_area_name = 'n_regions_gt10' + suffix
    nreggt10 = 0
    if len(img_areas_5) > 0:
        for ar in range(len(img_areas_5)):
            if img_areas_5[ar] > 10:
                nreggt10 += 1

    features_df[reg_area_name] = nreggt10

    return features_df


class SlideClassifierCervicalUpdate(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, root_dir: str):
        csvpath = result.slide_path.with_suffix('.csv')
        malig_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
        malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
        highg_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_high_grade.png')
        highg_hm = np.asarray(Image.open(highg_hm_path)) / 255
        lowg_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_low_grade.png')
        lowg_hm = np.asarray(Image.open(lowg_hm_path)) / 255
        normal_hm_path = root_dir / 'heatmaps' / (Path(csvpath).stem + '_normal.png')
        normal_hm = np.asarray(Image.open(normal_hm_path)) / 255
        tissue_area = np.sum((malig_hm + highg_hm + lowg_hm + normal_hm) > 0)

        features_ml = calculate_slide_features_single_cervicalupdate(malig_hm, tissue_area, "_ML")
        features_hg = calculate_slide_features_single_cervicalupdate(highg_hm, tissue_area, "_HG")
        features_lg = calculate_slide_features_single_cervicalupdate(lowg_hm, tissue_area, "_LG")
        features_ni = calculate_slide_features_single_cervicalupdate(normal_hm, tissue_area, "_NI")

        features_df = pd.concat((features_ml, features_hg, features_lg, features_ni), axis=1)
        features_df['slidename'] = Path(csvpath).stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags
        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, valid_features_dir: Path, retrain=False):
        def custom_eval_func(predt, dtrain):
            true_labs = np.array(dtrain.get_label(), dtype=np.int64)
            pred_labs = np.argmax(predt, axis=1)
            acc = np.sum(true_labs != pred_labs)
            sens = np.sum(np.logical_and(true_labs == 1, np.logical_not(pred_labs == 1)))
            tot = acc + sens
            return "custommetric", sens
        
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']
        labels2 = labels
        
        weights = np.ones(len(labels2))
        weights = np.where(labels2 == "low_grade", 0.1, weights)
        weights = np.where(labels2 == "high_grade", 0.1, weights)
        weights = np.where(labels2 == "normal", 0.1, weights)


        featuresv = pd.read_csv(valid_features_dir / 'features.csv')

        just_featuresv = featuresv.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labelsv = featuresv['slide_label']
        namesv = featuresv['slidename']
        tagsv = featuresv['tags']

        print("labels", np.unique(labels2))

        # fit or load (NB not all experiments will fit a sepearate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            slide_model = XGBClassifier(n_estimators=60, max_depth=int(3), verbosity=0)
            slide_model.fit(just_features, labels2, sample_weight=weights, 
                            eval_metric=custom_eval_func, eval_set=[(just_featuresv, labelsv)],
                            early_stopping_rounds=20, verbose=False)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + ['high_grade', 'lowg_grade', 'malignant', 'normal']
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels2)
        slides_labels_df.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)
        
        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)

        just_featuresv = just_featuresv.astype(np.float)
        predictionsv = slide_model.predict(just_featuresv)
        # Probabilities for each class
        probabilitiesv = slide_model.predict_proba(just_featuresv)
        # combine predictions and probailities into a dataframe
        reshaped_predictionsv = predictionsv.reshape((predictionsv.shape[0], 1))
        preds_probs_dfv = pd.DataFrame(np.hstack((reshaped_predictionsv, probabilitiesv)))
        preds_probs_dfv.columns = ["predictions"] +['high_grade', 'lowg_grade', 'malignant', 'normal']
        # create slide label dataframe
        slides_labels_dfv = pd.DataFrame(labelsv)
        slides_labels_dfv.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_dfv = pd.DataFrame(namesv)
        slides_names_dfv.columns = ["slide_names"]
        tags_dfv = pd.DataFrame(tagsv)
        tags_dfv.columns = ["tags"]
        slide_resultsv = pd.concat((slides_names_dfv, slides_labels_dfv, tags_dfv, preds_probs_dfv), axis=1)

        slide_resultsv.to_csv(valid_features_dir / 'slide_results.csv', index=False)