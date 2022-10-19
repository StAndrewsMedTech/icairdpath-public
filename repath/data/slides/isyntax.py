from pathlib import Path
from types import TracebackType
from typing import List, Tuple

from concurrent import futures
from multiprocessing import cpu_count
import numpy as np
from PIL import Image

from repath.data.slides.slide import SlideBase, Region
from repath.utils.geometry import Size, Point

import repath.data.slides.libraries.pixelengine as pixelengine
import repath.data.slides.libraries.softwarerendercontext as softwarerendercontext
import repath.data.slides.libraries.softwarerenderbackend as softwarerenderbackend

render_context = softwarerendercontext.SoftwareRenderContext()
render_backend = softwarerenderbackend.SoftwareRenderBackend()
pe = pixelengine.PixelEngine(render_backend, render_context)

# custom types for iSyntax
DataEnvelopeRect = Tuple[int, int, int, int]  # [start_x, end_x, start_y, end_y]


class Slide(SlideBase):
    # remember there is one of these for each subimage!
    def __init__(self, path: Path, image_index: int = 0) -> None:
        super().__init__()
        self._path = path
        self.image_index = image_index

    def open(self) -> None:
        self.input = pe["in"]
        self.input.open(str(self.path))
        self.max_level = self.input.numLevels()
        self.source_view = self.input.SourceView()
        self.samples_per_pixel = self.source_view.samplesPerPixel()

        # in isyntax each level has a different origin and dimensions
        # that are specifier in the level 0 coordinate system
        # self.origins = [self.get_origin(lvl) for lvl in range(self.max_level)]
        self.dims = [self.get_dimension(lvl) for lvl in range(self.max_level)]
        # self.dims = [Size(3000, 1000)] * 8

    def close(self) -> None:
        self.input.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dimensions(self) -> List[Size]:
        return self.dims

    def read_region(self, region: Region) -> Image:
        return self.read_regions([region])[0]

    def read_regions(self, regions: List[Region]) -> List:

        output_images = [None] * len(regions)

        # all regions must have the same level
        assert (
            len(np.unique([r.level for r in regions])) == 1
        ), "All regions must have same level."

        def region_to_range(region: Region) -> List:
            level, loc, size = region

            # scale to base level coords system
            scale = 2 ** level
            x = loc.x * scale
            y = loc.y * scale
            w = size.width * scale
            h = size.height * scale

            #origin = self.origins[level]
            origin = Point(0, 0)
            range = [
                x + origin.x,
                x + origin.x + w,
                y + origin.y,
                y + origin.y + h,
                level,
            ]
            return range

        def get_region_size(reg):
            x_start, x_end, y_start, y_end, lvl = reg.range
            ranges = self.source_view.dimensionRanges(lvl)
            # this calculation ensures that the width and height are on
            # the pixel bounds for this level
            w = int(1 + (x_end - x_start) / ranges[0][1])
            h = int(1 + (y_end - y_start) / ranges[1][1])
            return w, h

        def process_image(pix, reg_idx, w, h):
            # print(f"Processing region {reg_idx}")
            try:
                image = Image.frombuffer(
                    "RGB", (int(w), int(h)), pix, "raw", "RGB", 0, 1
                )
                # hack! - the iSyntax get region code is inclusive so we will crop
                # of the last pixel to get the image we want
                image = image.crop((0, 0, int(w - 1), int(h - 1)))
                output_images[reg_idx] = image
            except Exception as e:
                TracebackType.print_exc()

        def get_pixels(reg, buff_size):
            buff = np.empty(int(buff_size), dtype=np.uint8)
            reg.get(buff)
            return buff

        envelopes = self.source_view.dataEnvelopes(regions[0].level)
        ranges = [region_to_range(r) for r in regions]
        pe_regions = self.source_view.requestRegions(
            ranges, envelopes, True, [0, 0, 0], pe.BufferType.RGB
        )

        jobs = []
        with futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            while len(pe_regions) > 0:
                regions_ready = pe.waitAny(pe_regions)
                for region in regions_ready:
                    region_idx = ranges.index(
                        region.range
                    )  # where the region is in the list
                    width, height = get_region_size(region)
                    buffer_size = width * height * self.samples_per_pixel
                    pixels = get_pixels(region, buffer_size)
                    pe_regions.remove(region)
                    jobs.append(
                        executor.submit(
                            process_image, pixels, region_idx, width, height
                        )
                    )

        futures.wait(jobs, return_when=futures.ALL_COMPLETED)

        return output_images

    def get_envelope_rect(self, level: int) -> DataEnvelopeRect:
        envelopes = self.source_view.dataEnvelopes(level)
        rect = envelopes.asRectangles()
        return rect

    def get_origin(self, level: int) -> Point:
        start_x, _, start_y, _ = self.get_envelope_rect(level)
        origin = Point(start_x, start_y)
        return origin

    def get_dimension(self, level: int) -> Size:
        start_x = 0 
        start_y = 0
        end_x = 0
        end_y = 0
        rects = self.get_envelope_rect(level)
        for rect in rects:
            _, rect_end_x, _, rect_end_y = rect
            end_x = max(end_x, rect_end_x)
            end_y = max(end_y, rect_end_y)
        width = end_x - start_x
        height = end_y - start_y
        return Size(width // 2 ** level, height // 2 ** level)
