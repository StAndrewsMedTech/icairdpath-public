from pathlib import Path
from typing import List

from PIL import Image
from openslide import open_slide

from repath.data.slides.slide import SlideBase, Region
from repath.utils.geometry import Size


class Slide(SlideBase):
    
    def __init__(self, path: Path) -> None:
        self._path = path
        self._osr = None

    def open(self) -> None:
        self._osr = open_slide(str(self._path))

    def close(self) -> None:
        self._osr.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dimensions(self) -> List[Size]:
        # TODO: how should these be clipped? so they are power of 2 scale factor compatable
        return [Size(*dim) for dim in self._osr.level_dimensions]

    def read_region(self, region: Region) -> Image:
        return self._osr.read_region(region.location, region.level, region.size)

    def read_regions(self, regions: List[Region]) -> Image:
        # TODO: this call could be parallelised
        # though pytorch loaders will do this for us
        regions = [self.read_region(region) for region in regions]
        return regions
