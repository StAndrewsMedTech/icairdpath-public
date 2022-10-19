import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from repath.data.slides.isyntax import Slide
from repath.data.slides.slide import Region


class SingleSlideDataset(Dataset):
    def __init__(self, df: pd.DataFrame, slide: Slide, slide_path: Path, patch_size, patch_level, transform = None, augments = None) -> None:
        super().__init__()
        self.df = df
        self.slide = Slide(slide_path)
        self.transform = transform
        self.patch_size = patch_size
        self.patch_level = patch_level

    def open_slide(self):
        self.slide.open()

    def close_slide(self):
        self.slide.close()

    def to_patch(self, p: tuple) -> Image:
        region = Region.patch(p.x, p.y, self.patch_size, self.patch_level)
        image = self.slide.read_region(region)
        image = image.convert('RGB')
        return image

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        patch_info = self.df.iloc[idx]
        image = self.to_patch(patch_info)
        label = patch_info.label
        if self.transform is not None:
            image = self.transform(image)

        return image, label