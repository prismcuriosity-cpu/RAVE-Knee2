"""Surface extraction, thickness, curvature and bulge."""

from confcarti.thickness.curvature import (
    CurvatureResult,
    boundary_influence_zone,
    compute_curvature,
    congruence_index,
    curvature_summary,
    curvedness,
    geodesic_boundary_distance,
    shape_index,
)

__all__ = [
    "CurvatureResult",
    "boundary_influence_zone",
    "compute_curvature",
    "congruence_index",
    "curvature_summary",
    "curvedness",
    "geodesic_boundary_distance",
    "shape_index",
]
