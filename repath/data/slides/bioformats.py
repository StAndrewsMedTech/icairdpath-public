from pathlib import Path
from typing import List

from PIL import Image

import javabridge
import bioformats

from repath.data.slides.slide import SlideBase, Region
from repath.utils.geometry import Size


def setup():
    javabridge.start_vm(class_path=bioformats.JARS)


def shutdown():
    javabridge.kill_vm()


class Slide(SlideBase):
    
    def __init__(self, path: Path) -> None:
        self._path = path
        self._key = hashlib.sha1(str(path).encode()).hexdigest()
        self._dims = []

def open(self) -> None:
    self._reader()

def close(self) -> None:
    bioformats.release_image_reader(self._key)

@property
def path(self) -> Path:
    return self._path

@property
def dimensions(self) -> List[Size]:
    if len(self._dims) == 0:
        num_levels = self._reader.rdr.getImageCount() # TODO: this need to be taken from the slide
        self._dims = [self._get_dims(level) for level in range(num_levels)]
    return self._dims

def read_region(self, region: Region) -> Image:
    reader = self._reader()
    reader.rdr.setSeries(region.level)
    region = reader.read(XYWH=(region.location.x, region.location.y, region.size.width, region.size.height))
    return region

def read_regions(self, regions: List[Region]) -> Image:
    regions = [self.read_region(region) for region in regions]
    return regions

def _reader(self):
    return bioformats.get_image_reader(self._key, self._path)

def _get_dims(level: int) -> Size:
    reader = self._reader()
    reader.rdr.setSeries(level)
    width = reader.rdr.getSizeX()
    height = reader.rdr.getSizeY()
    return Size(width, height)
