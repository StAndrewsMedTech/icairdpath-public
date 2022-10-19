from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops

from abc import ABC, abstractmethod

from repath.postprocess.instance_segmentors import ConnectedComponents, DBScan
from repath.postprocess.results import SlidePatchSetResults, SlidesIndexResults
from repath.postprocess.lesion_metrics import evaluate_froc
from repath.utils.metrics import plotROC, plotROCCI


class LesionFinder(ABC):
    def __init__(
        self,
        mask_dir: Path,
        output_dir: Path
    ) -> None:
        self.mask_dir = mask_dir
        self.output_dir = output_dir

    @abstractmethod
    def find_slide_lesions(self, result) -> pd.DataFrame:
        pass

    @abstractmethod
    def calc_lesions(self, result):
        pass

    def calculate_ci_lesions(self, lesions_all_slides: pd.DataFrame, nreps: int = 1000):
        froc1000 = []
        tot_fps = np.linspace(0,8,801)
        tot_fps = tot_fps[1:]
        sens1000 = np.empty((nreps, 800))

        for i in range(nreps):
            sample_lesions = lesions_all_slides.sample(frac=1.0, replace=True)
            froc_curve, froc = evaluate_froc(self.mask_dir, sample_lesions, 5, 0.243) 
            froc1000.append(froc)
            total_FPs_inc = np.flip(froc_curve.total_FPs)
            total_sens_inc = np.flip(froc_curve.total_sensitivity)
            sens_lev = np.interp(tot_fps, total_FPs_inc, total_sens_inc)
            sens1000[i, :] = sens_lev

        froc_df = pd.DataFrame(froc1000, columns=['froc'])
        froc_df.index = ['sample_' + str(x) for x in range(nreps)]
        # create confidence interval
        froc_ci = froc_df.quantile([0.025, 0.975])
        # rename rows of dataframe
        froc_ci.index = ['ci_lower_bound', 'ci_upper_bound']
        froc_df = pd.concat([froc_ci, froc_df], axis=0)

        # crete confidence interval for sensitivity
        froc_curve_ci = np.quantile(sens1000, [0.025, 0.975], axis=0)
        tot_fps = np.reshape(tot_fps, (len(tot_fps), 1))
        print('tot fps: ', tot_fps.shape, ' froc_curve_ci: ', froc_curve_ci.T.shape)
        froc_curve_ci = np.hstack((tot_fps, froc_curve_ci.T))
        froc_curve_ci = pd.DataFrame(froc_curve_ci, columns = ['total_fps', 'ci_lower_bound', 'ci_upper_bound'])

        return froc_df, froc_curve_ci

    def calc_lesion_results(self, title: str, ci: bool = False, nreps: int = 1000):
        # check save directory exists if not make it

        lesions_all_slides = pd.read_csv(self.output_dir / 'lesions.csv')

        froc_curve, froc = evaluate_froc(self.mask_dir, lesions_all_slides, 5, 0.243)
        froc_curve.to_csv(self.output_dir / 'froc_curve.csv', index=False)

        froc_plot_title = "Free Receiver Operating Characteristic Curve for Lesion Detection \n" + title
        froc_plot = plotROC(froc_curve.total_FPs, froc_curve.total_sensitivity, froc, froc_plot_title,
                            "Average False Positives", "Metastatis Detection Sensitivity", [0, 8], [0, 1])
        froc_plot.savefig(self.output_dir / 'froc_curve.png')

        froc_df = pd.DataFrame([froc], columns=['froc'])
        froc_df.index = ['results']

        if ci:
            froc_ci_df, froc_curve_ci = self.calculate_ci_lesions(lesions_all_slides, nreps)

            froc_curve_ci.to_csv(self.output_dir / 'froc_curve_ci.csv', index=False)

            froc_curve_just_ci = np.array(froc_curve_ci[['ci_lower_bound', 'ci_upper_bound']].T)
            print(froc_curve_just_ci.shape)
            # create froc curve with confidence interval
            froc_curve_plt = plotROCCI(froc_curve.total_FPs, froc_curve.total_sensitivity, 
                                       froc_curve_ci.total_fps, froc_curve_just_ci, 
                                       froc, froc_ci_df.froc.loc[['ci_lower_bound', 'ci_upper_bound']].to_list(), 
                                       froc_plot_title, "Average False Positives", "Metastatis Detection Sensitivity", [0, 8])
            froc_curve_plt.savefig(self.output_dir / "froc_curve_ci.png")

            froc_df = pd.concat([froc_df, froc_ci_df], axis=0)

        froc_df.to_csv(self.output_dir / 'froc.csv')



class LesionFinderLee(LesionFinder):
    def find_slide_lesions(self, result: SlidePatchSetResults, posname: str = 'tumor') -> pd.DataFrame:
        print(result.slide_path.stem)
        # create heatmap from results 
        heatmap = result.to_heatmap(posname)
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        # create labelled image
        labelled_image = DBScan(0.5).segment(heatmap)

        # get region_props
        labelled_image = np.array(labelled_image, dtype='int')
        reg_props = regionprops(labelled_image, intensity_image=heatmap)

        if len(reg_props) > 0:
            # get centre for each region
            img_cents = [reg.centroid for reg in reg_props]

            # pixels per patch 
            pixels_per_patch = result.patch_size - result.border

            # convert img_cents to level zero scale
            img_cents = np.multiply(img_cents, pixels_per_patch)

            # convert img_cents into x y rather than row column
            img_cents = np.hstack((img_cents, img_cents))
            img_cents = img_cents[:, 1:3]

            # probability score for each region
            prob_score = np.reshape(np.array([np.sum(reg.intensity_image) for reg in reg_props]), (len(reg_props), 1))

            # save number of pixels per region
            img_area = np.reshape(np.array([reg.area for reg in reg_props]), (len(reg_props), 1))

            # combine to give output
            output_df = pd.DataFrame(
                np.hstack((prob_score, img_cents, img_area)),
                columns=["prob_score", "centre_x", "centre_y", "pixels"])
            output_df["filename"] = result.slide_path.stem
        else:
            output_df = pd.DataFrame(columns=["prob_score", "centre_x", "centre_y", "pixels", "filename"]) 

        return output_df 

    def calc_lesions(self, results: SlidesIndexResults) -> pd.DataFrame:
        
        lesions_all_slides = pd.DataFrame(columns=["prob_score", "centre_x", "centre_y", "pixels", "filename"])

        for result in results:
            lesions_out = self.find_slide_lesions(result)
            lesions_all_slides = pd.concat((lesions_all_slides, lesions_out), axis=0, ignore_index=True)

        # write out lesions
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lesions_all_slides.to_csv(self.output_dir / 'lesions.csv', index=False)


class LesionFinderWang(LesionFinder):
    def find_slide_lesions(self, result: SlidePatchSetResults, result_post: SlidePatchSetResults, 
                     posname: str = 'tumor') -> pd.DataFrame:
        print(result_post.slide_path.stem)
        # create heatmaps from results 
        heatmap = result.to_heatmap(posname)
        heatmap_hnm = result_post.to_heatmap(posname)
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap pre in wrong format"
        assert heatmap_hnm.dtype == 'float' and np.max(heatmap_hnm) <= 1.0 and np.min(heatmap_hnm) >= 0.0, "Heatmap post in wrong format"

        # create labelled image
        labelled_image = ConnectedComponents(0.9).segment(heatmap)

        # get region_props
        reg_props = regionprops(labelled_image, intensity_image=heatmap)
        reg_props_hnm = regionprops(labelled_image, intensity_image=heatmap_hnm)

        if len(reg_props) > 0:
            # get centre for each region
            img_cents = [reg.centroid for reg in reg_props]

            # pixels per patch 
            pixels_per_patch = result.patch_size - result.border

            # convert img_cents to level zero scale
            img_cents = np.multiply(img_cents, pixels_per_patch)

            # convert img_cents into x y rather than row column
            img_cents = np.hstack((img_cents, img_cents))
            img_cents = img_cents[:, 1:3]

            # probability score for each region
            prob_score_orig = [np.sum(reg.intensity_image) for reg in reg_props]
            prob_score_hnm = [np.sum(reg.intensity_image) for reg in reg_props_hnm]
            prob_score = np.divide(np.add(prob_score_orig, prob_score_hnm), 2)
            prob_score = np.reshape(np.array(prob_score), (len(prob_score), 1))

            # save number of pixels per region
            img_area = np.reshape(np.array([reg.area for reg in reg_props]), (len(reg_props), 1))

            # combine to give output
            output_df = pd.DataFrame(
                np.hstack((prob_score, img_cents, img_area)),
                columns=["prob_score", "centre_x", "centre_y", "pixels"])
            output_df["filename"] = result_post.slide_path.stem
        else:
            output_df = pd.DataFrame(columns=["prob_score", "centre_x", "centre_y", "pixels"])

        return output_df

    def calc_lesions(self, results_pre: SlidesIndexResults, results_post: SlidesIndexResults) -> pd.DataFrame:
        
        # get lesions
        lesions_all_slides = pd.DataFrame(columns=["prob_score", "centre_x", "centre_y", "pixels", "filename"])

        for result_pre, result_post in zip(results_pre, results_post):
            lesions_out = self.find_slide_lesions(result_pre, result_post)
            lesions_all_slides = pd.concat((lesions_all_slides, lesions_out), axis=0, ignore_index=True)
        # write out lesions
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lesions_all_slides.to_csv(self.output_dir / 'lesions.csv', index=False)


class LesionFinderLiu(LesionFinder):
    def find_slide_lesions(self, result: SlidePatchSetResults, posname: str = 'tumor', prob_cutoff: float = 0.5, 
                     patch_radius: int = 6) -> pd.DataFrame:
        print(result.slide_path.stem)
        # set probabilities below the cutoff value to zero
        result.patches_df[posname] = np.where(result.patches_df[posname] < prob_cutoff, 0, result.patches_df[posname])
        print("nperslide: ", np.sum(result.patches_df['tumor'] > 0))
        # find radius size in pixels rather than patches
        ### TODO: check if patch size at this point includes the border or not, need it without the border
        ### HACK patch size here includes border we need it without border so have hard coded for now :(
        radius_size = patch_radius * 128
        print(radius_size, patch_radius, result.patch_size)

        # list to store lesions quicker to convert to dataframe from list once than append each row
        list_lesions = []

        # loop over find highest probability as a tumor centre, then set all pixels in a 6 pixel radius to zero
        while np.max(result.patches_df[posname]) > 0:
            # find row with highest probability
            max_rw = result.patches_df.iloc[np.argmax(result.patches_df[posname])]
            # create output for that tumor centre
            row_out = pd.DataFrame([max_rw[posname], max_rw.x, max_rw.y]).T
            row_out.columns = ['prob_score', 'centre_x', 'centre_y']

            # for each point calcualte distance to this point
            result.patches_df['distance'] = np.sqrt(np.add(np.square(np.subtract(result.patches_df.x, max_rw.x)),
                                                           np.square(np.subtract(result.patches_df.y, max_rw.y))))
            # find rows that are closer than radius size
            radius_mask = result.patches_df.distance < radius_size
            # find number of pixels are within radius greater than prob cutoff equivalent of size 
            row_out['pixels'] = np.sum(radius_mask)
            # set probability for all rows that are within radius to zero
            result.patches_df[posname] = np.where(radius_mask, 0, result.patches_df[posname])

            # add to output
            list_lesions.append(row_out)

        output_df = pd.DataFrame(columns=["prob_score", "centre_x", "centre_y", "pixels"])
        for ll in range(len(list_lesions)):
            output_df = pd.concat((output_df, list_lesions[ll]))

        output_df["filename"] = result.slide_path.stem

        return output_df 

    def calc_lesions(self, results: SlidesIndexResults) -> pd.DataFrame:
        
        lesions_all_slides = pd.DataFrame(columns=["prob_score", "centre_x", "centre_y", "pixels", "filename"])

        for result in results:
            lesions_out = self.find_slide_lesions(result)
            lesions_all_slides = pd.concat((lesions_all_slides, lesions_out), axis=0, ignore_index=True)

        # write out lesions
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lesions_all_slides.to_csv(self.output_dir / 'lesions.csv', index=False)

