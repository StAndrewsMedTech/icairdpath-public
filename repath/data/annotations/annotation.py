from typing import List, Dict

import cv2
import numpy as np

from repath.utils.geometry import PointF, Shape

annotation_types = ["Dot", "Polygon", "Spline", "Rectangle"]


class Annotation:
    def __init__(
        self, name: str, annotation_type: str, label: str, vertices: List[PointF]
    ):
        assert annotation_type in annotation_types
        self.name = name
        self.type = annotation_type
        self.label = label
        self.coordinates = vertices

    def draw(self, image: np.array, labels: Dict[str, int], factor: float):
        """Renders the annotation into the image.

        Args:
            image (np.array): Array to write the annotations into, must have dtype float.
            labels (Dict[str, int]): The value to write into the image for each type of label.
            factor (float): How much to scale (by divison) each vertex by.
        """
        fill_colour = labels[self.label]
        vertices = np.array(self.coordinates) / factor
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(image, [vertices], (fill_colour))


class AnnotationSet:
    def __init__(
        self,
        annotations: List[Annotation],
        labels: Dict[str, int],
        labels_order: List[str],
        fill_label: str,
    ) -> None:
        self.annotations = annotations
        self.labels = labels
        self.labels_order = labels_order
        self.fill_label = fill_label

    def render(self, shape: Shape, factor: float) -> np.array:
        """Renders associated annotation data into the image.

        Args:
            shape (Shape): A NamedTuple of (num_rows, num_cols)
            factor (float): Scaling factor

        Returns:
            image (np.array): A scaled image with associated annotations
        """
        annotations = sorted(
            self.annotations, key=lambda a: self.labels_order.index(a.label)
        )
        image = np.full(shape, self.labels[self.fill_label], dtype=float)
        for a in annotations:
            a.draw(image, self.labels, factor)
        return image.astype("int")
