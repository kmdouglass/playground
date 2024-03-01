from dataclasses import dataclass
from math import inf

from ezray.core.general_ray_tracing import SurfaceType
from ezray.core.general_ray_tracing import (
    Conic as CoreConic,
    Image as CoreImage,
    Object as CoreObject,
    Stop as CoreStop,
    Toric as CoreToric,
)


@dataclass(frozen=True)
class Conic:
    semi_diameter: float
    conic_constant: float = 0
    radius_of_curvature: float = inf
    surface_type: SurfaceType = SurfaceType.REFRACTING

    def into_surface(self) -> CoreConic:
        return CoreConic(
            semi_diameter=self.semi_diameter,
            conic_constant=self.conic_constant,
            radius_of_curvature=self.radius_of_curvature,
            surface_type=self.surface_type,
        )


@dataclass(frozen=True)
class Image:
    def into_surface(self) -> CoreImage:
        return CoreImage()


@dataclass(frozen=True)
class Object:
    def into_surface(self) -> CoreObject:
        return CoreObject()


@dataclass(frozen=True)
class Stop:
    semi_diameter: float

    def into_surface(self) -> CoreStop:
        return CoreStop(semi_diameter=self.semi_diameter)


@dataclass(frozen=True)
class Toric:
    semi_diameter: float
    conic_constant: float = 0
    radius_of_curvature: float = inf
    radius_of_revolution: float = inf
    surface_type: SurfaceType = SurfaceType.REFRACTING

    def into_surface(self) -> CoreToric:
        return CoreToric(
            semi_diameter=self.semi_diameter,
            conic_constant=self.conic_constant,
            radius_of_curvature=self.radius_of_curvature,
            radius_of_revolution=self.radius_of_revolution,
            surface_type=self.surface_type,
        )


type SurfaceSpec = Conic | Image | Object | Stop | Toric
