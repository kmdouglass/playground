from math import inf

from ezray import Gap, Surface, SurfaceType, System

model = System(
    [
        Surface(semi_diameter=12.5, surface_type=SurfaceType.OBJECT),
        Gap(refractive_index=1.0, thickness=inf),
        Surface(
            semi_diameter=12.5,
            radius_of_curvature=25.8,
            surface_type=SurfaceType.REFRACTING,
        ),
        Gap(refractive_index=1.515, thickness=5.3),
        Surface(semi_diameter=12.5, surface_type=SurfaceType.REFRACTING),
        Gap(refractive_index=1.0, thickness=46.6),
        Surface(semi_diameter=12.5, surface_type=SurfaceType.IMAGE),
    ]
)
