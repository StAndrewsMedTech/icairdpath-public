from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from skimage.measure import label
from sklearn.cluster import DBSCAN

from repath.utils.convert import to_frame_with_locations


class InstanceSegmentor(ABC):
    @abstractmethod
    def segment(self, heatmap: np.ndarray) -> np.ndarray:
        pass


class ConnectedComponents(InstanceSegmentor):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def segment(self, heatmap: np.ndarray) -> np.ndarray:
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        # heatmap is array r * c * float (range 0 - 1)
        # threshold heatmap into binary image
        binary_img = np.greater_equal(heatmap, self.threshold)

        # find connected components, label background as 0
        labelled_img = label(binary_img, background=0)
        return labelled_img


class DBScan(InstanceSegmentor):
    def __init__(self, threshold: float, eps: float = 0.5, min_samples: int = 5):

        self.threshold = threshold
        self.eps = eps
        self.min_samples = min_samples

    def segment(self, heatmap: np.ndarray) -> np.ndarray:
        # heatmap is array r * c * float (range 0 - 1)
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        frame = to_frame_with_locations(heatmap)

        frame['labels'] = 0

        # Threshold frame over value=0.5
        th_frame = frame.loc[frame['value'] > self.threshold]
        # feed the row and column of dataframe as input array to DBSCAN
        X = th_frame.loc[:, ["row", "column"]]
        X = np.array(X)

        # catch empty heatmaps
        if X.shape[0] == 0:
            return np.zeros(heatmap.shape)

        # Step2: feed array of points to DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        # get max label
        max_label = np.max(db.labels_)
        outliers_mask = db.labels_ != -1
        new_labels_start = max_label + 1
        new_labels_stop = new_labels_start + len(db.labels_)
        new_labels = range(new_labels_start, new_labels_stop)
        db_labels = db.labels_
        # db_labels = db_labels[outliers_mask]
        db_labels = np.where(outliers_mask, db_labels, new_labels)

        # Step3: Convert the new dataframe back to image
        df1 = frame[["row", "column", "labels"]]
        df2 = th_frame[["row", "column"]].copy()
        # df2 = df2[outliers_mask]
        df2.loc[:, "labels"] = db_labels.T
        # th_frame.loc[:, "labels"] = db_labels.T
        # df2 = th_frame.loc[:, ["row", "column", "labels"]]
        df = df2.combine_first(df1)

        # convert the dataframe back to image
        labels = df['labels'].to_numpy()
        img_arr = labels.reshape(heatmap.shape[0], heatmap.shape[1])

        labelled_img = np.array(img_arr, dtype=int)
        return labelled_img
