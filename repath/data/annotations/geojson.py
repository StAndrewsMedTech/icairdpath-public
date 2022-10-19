from pathlib import Path
import json
from typing import List, Dict

from repath.data.annotations.annotation import Annotation


def json_load(filepath, **kwargs):
    with open(filepath, "r") as f:
        data = json.load(f, **kwargs)
    return data


def base_shape(coord_list):
    vertices = [(float(c[0]), float(c[1])) for c in coord_list]
    return(vertices)


def gjson_polygon(polygon: List, label: str, default_label: str):
    polygon_vertices = [base_shape(bs) for bs in polygon]    
    outer_polygon = Annotation(label, 'Polygon', label, polygon_vertices[0])
    annotations_list = [outer_polygon]
    if len(polygon) > 1:
        for poly in polygon_vertices[1:]:
            inner_polygon = Annotation(label, 'Polygon', default_label, poly)
            annotations_list.append(inner_polygon)

    return annotations_list


def annotation_from_feature(feature: Dict, group_labels: Dict[str, str],
                            default_label: str) -> Annotation:
    """ Gets annotation tags from Json features

    Args:
        feature : Geojson data
        group_labels(Dict[str, str]): A dictionary of strings defining labels.

    Returns:
        Annotations tags such as name, type, label and vertices

    """
    geometry = feature['geometry']
    geometry_type = geometry['type']
    coordinates = geometry['coordinates']
    properties = feature['properties']

    if 'classification' in properties.keys():
        classification = properties['classification']
        label = classification['name']
    else:
        print("unlabelled annotation")
        label = default_label

    assert label in group_labels.keys(), f'Unknown annotation group {label}'
    label = group_labels[label]
    if label == 'malignant' and (default_label =='cgin' or default_label == 'adenocarcinoma'):
        label = default_label

    if geometry_type == 'Polygon':
        annotations_list = gjson_polygon(coordinates, label, default_label)
    elif geometry_type == 'MultiPolygon':
        annotations_list = [gjson_polygon(crd, label, default_label) for crd in coordinates]
        annotations_list = [item for sublist in annotations_list for item in sublist]
    else:
        ### HACK to return anything else
        annotations_list = []
        print(geometry_type, properties)
    
    return annotations_list


def load_annotations(json_path: Path, group_labels: Dict[str, str], default_label: str) -> List[Annotation]:
    file_in = json_load(json_path)
    features = file_in['features']
    annotations_list = []
    for feat in features:
        annots = annotation_from_feature(feat, group_labels, default_label)
        annotations_list.append(annots)
    
    annotations_list = [item for sublist in annotations_list for item in sublist]

    return annotations_list
