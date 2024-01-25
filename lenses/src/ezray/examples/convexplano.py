from math import inf

from ezray import Gap, Surface, SurfaceType, System

model = System(
    [
        Surface(diameter=25, surface_type=SurfaceType.OBJECT),
        Gap(refractive_index=1.0, thickness=inf),
        Surface(
            diameter=25,
            radius_of_curvature=25.8,
            surface_type=SurfaceType.REFRACTING,
        ),
        Gap(refractive_index=1.515, thickness=5.3),
        Surface(diameter=25, surface_type=SurfaceType.REFRACTING),
        Gap(refractive_index=1.0, thickness=46.6),
        Surface(diameter=25, surface_type=SurfaceType.IMAGE),
    ]
)
