"""The curvature API that ``confcarti.thickness.bulge`` borrows.

Why this file exists
--------------------
``bulge.py`` uses four names from ``curvature.py`` and defines none of them:

    _radius_neighbourhoods(mesh, radius_mm)      -> local_form_residual
    _tangent_frames(normals)                     -> local_form_residual
    boundary_influence_zone(mesh, margin)        -> compute_bulge
    smooth_scalar_field(mesh, field, iterations) -> thickness_residual_bulge

In the notebook every module shares one global namespace, so those calls resolve
to whatever the curvature cell last defined. They are an interface, not private
helpers, and nothing pinned them — so the curvature rewrite changed three of the
four, and ``compute_bulge`` died at the first one:

    TypeError: _radius_neighbourhoods() missing 2 required positional
               arguments: 'targets' and 'radius_mm'

The other two changed quietly, which is worse: ``boundary_influence_zone`` gained
a ``metric`` argument defaulting to geodesic, and ``smooth_scalar_field`` gained
``preserve_nan`` defaulting to True. Both are the right behaviour for bulge as
well, and both silently move its numbers.

These tests pin the call signatures. They are deliberately written the way
``bulge.py`` calls them — positionally, from a mesh — so that a future change to
the subset/chunked internals cannot break the borrowing module without a red test.
"""

from __future__ import annotations

import inspect

import numpy as np
import trimesh

from confcarti.thickness.curvature import (
    _ball_neighbourhoods,
    _radius_neighbourhoods,
    _tangent_frames,
    boundary_influence_zone,
    compute_curvature,
    smooth_scalar_field,
)


def _open_patch(radius_mm: float = 20.0, cap_mm: float = 18.0) -> trimesh.Trimesh:
    """An open spherical cap — the shape bulge is actually run on."""
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=radius_mm)
    keep = sphere.vertices[:, 2] > radius_mm - cap_mm
    cap = trimesh.Trimesh(
        vertices=sphere.vertices,
        faces=sphere.faces[keep[sphere.faces].all(axis=1)],
        process=False,
    )
    cap.remove_unreferenced_vertices()
    return cap


# --------------------------------------------------------------------------- #
# The signature that broke
# --------------------------------------------------------------------------- #


def test_radius_neighbourhoods_takes_a_mesh_and_a_radius():
    """``_radius_neighbourhoods(mesh, radius_mm)`` — exactly how bulge calls it."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)

    index, mask = _radius_neighbourhoods(mesh, 3.0)

    assert index.shape == mask.shape
    assert index.shape[0] == len(mesh.vertices)
    assert mask.any(axis=1).all(), "every vertex must have at least one neighbour"
    # Padding repeats the centre vertex, so masked-out entries are never garbage.
    rows, cols = np.where(~mask)
    assert np.array_equal(index[rows, cols], rows)


def test_radius_neighbourhoods_signature_is_positional_mesh_first():
    """Pin the parameter names and order, not just that a two-arg call works."""
    parameters = list(inspect.signature(_radius_neighbourhoods).parameters)
    assert parameters[:2] == ["mesh", "radius_mm"]


def test_radius_neighbourhoods_matches_the_subset_helper():
    """The mesh-level entry point is the all-vertices case of the subset one."""
    from scipy.spatial import cKDTree

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    whole_index, whole_mask = _radius_neighbourhoods(mesh, 3.0)
    subset_index, subset_mask = _ball_neighbourhoods(
        vertices, cKDTree(vertices), np.arange(len(vertices)), 3.0
    )

    assert np.array_equal(whole_index, subset_index)
    assert np.array_equal(whole_mask, subset_mask)


def test_radius_neighbourhoods_grows_with_radius():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    small = _radius_neighbourhoods(mesh, 1.0)[1].sum()
    large = _radius_neighbourhoods(mesh, 3.0)[1].sum()
    assert large > small


# --------------------------------------------------------------------------- #
# The three other borrowed names
# --------------------------------------------------------------------------- #


def test_tangent_frames_takes_normals_and_returns_an_orthonormal_basis():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    u, v = _tangent_frames(normals)

    assert u.shape == v.shape == normals.shape
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.abs(np.einsum("ij,ij->i", u, normals)).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", v, normals)).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", u, v)).max() < 1e-9


def test_boundary_influence_zone_still_accepts_a_positional_margin():
    """Bulge calls ``boundary_influence_zone(mesh, margin)`` with no metric."""
    mesh = _open_patch()
    parameters = list(inspect.signature(boundary_influence_zone).parameters)
    assert parameters[:2] == ["mesh", "margin_mm"]

    zone = boundary_influence_zone(mesh, 6.0)
    assert zone.dtype == bool
    assert zone.shape == (len(mesh.vertices),)
    assert zone.any() and not zone.all(), "an 18 mm cap at a 6 mm margin keeps some"


def test_smooth_scalar_field_still_accepts_three_positional_arguments():
    """Bulge calls ``smooth_scalar_field(bci, thickness, iterations)``."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    field = np.asarray(mesh.vertices[:, 2], dtype=np.float64)

    smoothed = smooth_scalar_field(mesh, field, 3)

    assert smoothed.shape == field.shape
    assert np.isfinite(smoothed).all()
    assert smoothed.std() < field.std(), "smoothing should reduce variation"


def test_smoothing_does_not_conduct_across_a_nan_hole():
    """``preserve_nan=True`` is the default, and it blocks diffusion through a hole.

    This is the behaviour change bulge inherited. A denuded patch is a hole in the
    thickness field; letting values fill in and then diffuse onward carries
    thickness *across* full-thickness cartilage loss over the 60 iterations the
    pipeline uses.
    """
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    field = np.zeros(len(mesh.vertices), dtype=np.float64)
    hole = mesh.vertices[:, 2] > 5.0
    field[hole] = np.nan
    field[mesh.vertices[:, 2] < -8.0] = 1.0

    preserved = smooth_scalar_field(mesh, field, 20)
    filled = smooth_scalar_field(mesh, field, 20, preserve_nan=False)

    assert np.isnan(preserved[hole]).all()
    assert np.isfinite(filled[hole]).any()


# --------------------------------------------------------------------------- #
# End to end: the cell that actually failed
# --------------------------------------------------------------------------- #


def _sphere_control_residual(mesh, radius_mm):
    """Bulge's local form fit, driven through the borrowed neighbourhood API."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    index, mask = _radius_neighbourhoods(mesh, radius_mm)
    u, v = _tangent_frames(normals)
    offsets = vertices[index] - vertices[:, None, :]
    x = np.einsum("nkj,nj->nk", offsets, u)
    y = np.einsum("nkj,nj->nk", offsets, v)
    z = np.einsum("nkj,nj->nk", offsets, normals)

    design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=2)
    weights = mask.astype(np.float64)[:, :, None]
    gram = np.einsum("nkp,nkq->npq", design * weights, design)
    rhs = np.einsum("nkp,nk->np", design * weights, z * mask)
    residual = -np.linalg.solve(gram + np.eye(6)[None] * 1e-9, rhs[:, :, None])[:, 0, 0]

    interior = ~boundary_influence_zone(mesh, 0.5 * radius_mm, metric="geodesic")
    assert interior.any(), f"nothing interior at radius {radius_mm}"
    return float(np.abs(residual[interior]).max())


def test_local_form_fit_runs_on_an_open_patch():
    """Reproduces bulge's use of the borrowed neighbourhoods, on bulge's shape.

    The failing notebook cell was a form-radius sweep over a bumped sphere. This
    exercises the same path -- radius neighbourhoods plus a tangent-plane quadric
    over an open patch -- on the sweep's control case, a clean sphere.

    The bound is not zero, and should not be. A quadric cannot represent a
    spherical cap exactly, so the fit leaves a fourth-order residual that grows
    with ``form_radius / radius_of_curvature`` -- the false-positive floor the
    bulge module documents. On a 20 mm sphere it is ~0.004 mm at a 6 mm form
    radius and ~0.06 mm at 12 mm, and the original notebook run measured
    0.0644 mm RMS for the same 12 mm control. Asserting "no bulge on a sphere"
    at a tight tolerance would be asserting something false.
    """
    mesh = _open_patch()
    assert _sphere_control_residual(mesh, 6.0) < 0.01
    assert _sphere_control_residual(mesh, 12.0) < 0.10


def test_false_positive_floor_grows_with_form_radius():
    """Pin the documented property, not an arbitrary tolerance.

    The floor is a real limitation of local form removal on a curved surface, and
    the bulge module warns about it in prose. Here it is as a test, so that a
    change which quietly flattened or inflated it would be caught.
    """
    mesh = _open_patch()
    floors = [_sphere_control_residual(mesh, r) for r in (6.0, 8.0, 12.0)]
    assert floors == sorted(floors), f"floor is not monotone in radius: {floors}"
    assert floors[-1] > 4 * floors[0], "floor should grow markedly with radius"


def test_curvature_still_works_after_the_split():
    """The curvature path that motivated the signature change must still hold."""
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=10.0)
    result = compute_curvature(sphere, radius_mm=2.0, smoothing_iterations=0)

    assert result.estimable_fraction > 0.99
    assert abs(np.nanmedian(result.mean_curvature) - 0.1) < 0.005
