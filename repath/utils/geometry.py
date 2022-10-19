from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int


class PointF(NamedTuple):
    x: float
    y: float


class Address(NamedTuple):
    row: int
    col: int


class Size(NamedTuple):
    width: int
    height: int

    def as_shape(self):
        return Shape(self.height, self.width)


class Shape(NamedTuple):
    num_rows: int
    num_cols: int

    def as_size(self):
        return Size(self.num_cols, self.num_rows)
