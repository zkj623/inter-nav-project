"""Optional body footprint for generic route and collision queries."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class BodyFootprint:
    x_min: float
    x_max: float
    half_width: float

    def __post_init__(self):
        if (not np.isfinite([self.x_min, self.x_max, self.half_width]).all()
                or not self.x_min < 0 < self.x_max or self.half_width <= 0):
            raise ValueError('Body footprint requires finite bounds around the base origin')

    def samples(self, resolution, margin=0.):
        first, last, width = self.x_min-margin, self.x_max+margin, self.half_width+margin
        return np.meshgrid(np.linspace(first, last, int(math.ceil((last-first)/resolution))+1),
                           np.linspace(-width, width, int(math.ceil(2*width/resolution))+1))
