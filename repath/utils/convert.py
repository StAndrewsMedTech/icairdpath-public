import numpy as np
from PIL import Image

import pandas as pd


def pil_to_np(image: Image) -> np.ndarray:
    """ Convert a PIL image into a Numpy array

    Args:
        image: the PIL image

    Returns:
        a Numpy array

    """
    arr = np.asarray(image)
    return arr


def np_to_pil(arr: np.ndarray) -> Image:
    """ Convert a Numpy array into a PIL image

    Args:
        arr: a Numpy array

    Returns:
        the PIL image

    """
    if arr.dtype == "bool":
        arr = arr.astype("uint8") * 255
    elif arr.dtype == "float64" or arr.dtype == "float32":
        arr = (arr * 255).astype("uint8")
    return Image.fromarray(arr)


def to_frame_with_locations(
    array: np.ndarray, value_name: str = "value"
) -> pd.DataFrame:
    """ Create a data frame with row and column locations for every value in the 2D array
    Args:
        array: a Numpy array
        value_name: a string with the column name for the array values to be output in
    Returns:
        a pandas data frame of row, column, value where each value is the value of np array at row, column
    """
    series = pd.DataFrame(array).stack()
    frame = pd.DataFrame(series)
    frame.reset_index(inplace=True)
    samples = frame.rename(
        columns={"level_0": "row", "level_1": "column", 0: value_name}
    )
    samples["row"] = samples["row"].astype(int)
    samples["column"] = samples["column"].astype(int)
    return samples


def remove_item_from_dict(dict_in: dict, key_to_remove: str) -> dict:
    """ remove one key value pair from a dictionary by specifying the key to remove
    Args:
        dict_in: dictionary to remove an item from
        key_to_remove: the key of the key value pair to be removed
    Returns:
        the dictionary without the specified item
    """
    dict_out = dict(dict_in)
    del dict_out[key_to_remove]
    return dict_out


def average_patches(slides_index, naugs: int, new_dir):
    """ averages results from patch sets with multiple augments per patch
    """
    for ps in slides_index:
        df = ps.patches_df
        ps.patches_df = df.groupby(['x', 'y'], as_index=False).mean()

    slides_index.output_dir = new_dir
    
    return slides_index


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst 
