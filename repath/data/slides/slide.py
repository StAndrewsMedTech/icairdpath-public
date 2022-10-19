from abc import ABCMeta, abstractmethod
import copy
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from PIL import Image
from repath.utils.geometry import Point, Size


class Region(NamedTuple):
    level: int
    location: Point
    size: Size

    @classmethod
    def patch(cls, x, y, size, level):
        location = Point(x, y)
        size = Size(size, size)
        return Region(level, location, size)

    @classmethod
    def make(cls, x, y, width, height, level):
        location = Point(x, y)
        size = Size(width, height)
        return Region(level, location, size)


class SlideBase(metaclass=ABCMeta):
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    @abstractmethod
    def path(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> List[Size]:
        raise NotImplementedError

    @abstractmethod
    def read_region(self, region: Region) -> Image:
        raise NotImplementedError

    @abstractmethod
    def read_regions(self, regions: List[Region]) -> Image:
        raise NotImplementedError

    def get_thumbnail(self, level: int) -> np.array:
        """ Generated the thumbnail of the slide at a given level.

        Args:
            level (int): The level for generating  thumbnail.

        Returns:
            The thumbnail of the slide at the given level as a numpy array

        """
        if level >= len(self.dimensions):
            request_level = len(self.dimensions) - 1
        else:
            request_level = level

        size = self.dimensions[request_level]
        region = Region(level=request_level, location=Point(0, 0), size=size)
        im = self.read_region(region)
        im = im.convert("RGB")
        
        if level != request_level:
            w, h = im.size
            lev_diff = level - request_level
            new_w = int(w / 2 ** lev_diff)
            new_h = int(h / 2 ** lev_diff)
            new_im = copy.deepcopy(im)
            new_im = new_im.resize((new_w, new_h))
        else:
            new_im = copy.deepcopy(im)

        new_im = np.asarray(new_im)

        return new_im
