"""Curvature on *open* surfaces -- the case the 6 mm boundary margin destroyed.

Closed-form validation on a sphere and a plane always passed, because both are
closed or unbounded and neither exercises the boundary rule. A cartilage plate is
neither: it is an open sheet 20-25 mm across whose every vertex is within 12 mm
of a rim. These tests pin the behaviour on shapes of that description, where the
answer is still known in closed form.
"""

from __future__ import annotations

import numpy as np
import trimesh

from confcarti.thickness.curvature import (
    boundary_influence_zone,
    compute_curvature,
    geodesic_boundary_distance,
)


# --------------------------------------------------------------------------- #
# Fixtures: open patches whose curvature is known exactly
# --------------------------------------------------------------------------- #


def cylinder_strip(
    radius_mm: float = 22.0,
    width_mm: float = 22.0,
    arc_mm: float = 24.0,
    edge_mm: float = 0.5,
) -> trimesh.Trimesh:
    """An open strip cut from a cylinder: ``H = 1/(2R)``, ``K = 0`` exactly.

    Defaults approximate a tibial plate: ~22 x 24 mm on a 22 mm radius of
    curvature, triangulated at the 0.5 mm edge length of a marching-cubes
    surface extracted at OAI-ZIB resolution.
    """
    n_axial = int(round(width_mm / edge_mm)) + 1
    n_circ = int(round(arc_mm / edge_mm)) + 1
    axial = np.linspace(-width_mm / 2.0, width_mm / 2.0, n_axial)
    theta = np.linspace(-arc_mm / (2.0 * radius_mm), arc_mm / (2.0 * radius_mm), n_circ)

    tt, aa = np.meshgrid(theta, axial, indexing="ij")
    vertices = np.stack(
        [radius_mm * np.cos(tt), radius_mm * np.sin(tt), aa], axis=-1
    ).reshape(-1, 3)

    faces = []
    for i in range(n_circ - 1):
        for j in range(n_axial - 1):
            a = i * n_axial + j
            b = a + 1
            c = a + n_axial
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    # Outward normals: the strip bulges away from the cylinder axis.
    if np.dot(mesh.vertex_normals[0], mesh.vertices[0] * [1, 1, 0]) < 0:
        mesh.invert()
    return mesh


def folded_sheet(
    long_arm_mm: float = 25.0,
    short_arm_mm: float = 6.0,
    gap_mm: float = 1.5,
    width_mm: float = 30.0,
    edge_mm: float = 0.5,
) -> trimesh.Trimesh:
    """A sheet folded back on itself, with one arm shorter than the other.

    This is the trochlear groove / intercondylar notch in miniature, and the
    reason a Euclidean distance-to-rim is the wrong measurement. The free edge
    of the short arm sits ``gap_mm`` from the middle of the long arm through
    space, and ``long_arm - short_arm + bend`` millimetres from it along the
    surface. Both arms are flat, so ``H = K = 0`` everywhere except in the bend.
    """
    bend_radius = gap_mm / 2.0
    n_bend = max(int(round(np.pi * bend_radius / edge_mm)), 4)
    angles = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_bend)

    long_x = np.arange(-long_arm_mm, 0.0, edge_mm)
    short_x = np.arange(0.0, -short_arm_mm, -edge_mm)[1:]

    profile = np.concatenate(
        [
            np.stack([long_x, np.full_like(long_x, bend_radius)], axis=1),
            np.stack([bend_radius * np.cos(angles), bend_radius * np.sin(angles)], axis=1),
            np.stack([short_x, np.full_like(short_x, -bend_radius)], axis=1),
        ]
    )

    z = np.arange(-width_mm / 2.0, width_mm / 2.0 + edge_mm / 2.0, edge_mm)
    n_p, n_z = len(profile), len(z)
    vertices = np.stack(
        [
            np.repeat(profile[:, 0], n_z),
            np.repeat(profile[:, 1], n_z),
            np.tile(z, n_p),
        ],
        axis=1,
    )

    faces = []
    for i in range(n_p - 1):
        for j in range(n_z - 1):
            a = i * n_z + j
            faces.append([a, a + n_z, a + 1])
            faces.append([a + 1, a + n_z, a + n_z + 1])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def sphere_cap(radius_mm: float = 25.0, cap_mm: float = 22.0, edge_mm: float = 0.5):
    """An open cap cut from a sphere: ``H = 1/R``, ``K = 1/R^2`` exactly."""
    sphere = trimesh.creation.icosphere(subdivisions=6, radius=radius_mm)
    keep = sphere.vertices[:, 2] > radius_mm - cap_mm
    faces = sphere.faces[keep[sphere.faces].all(axis=1)]
    cap = trimesh.Trimesh(vertices=sphere.vertices, faces=faces, process=False)
    cap.remove_unreferenced_vertices()
    del edge_mm
    return cap


# --------------------------------------------------------------------------- #
# The regression the fix is for
# --------------------------------------------------------------------------- #


def test_open_strip_is_mostly_estimable():
    """A tibial-plate-sized strip must yield curvature over most of its area.

    With the old rule (``mask_boundary=True`` at ``2 * radius_mm`` measured in
    Euclidean space) this patch has *no* estimable vertex at all: every point of
    a 22 mm strip is within 6 mm of an edge somewhere. That was the 76-87 %
    loss seen on real knees, in its purest form.
    """
    mesh = cylinder_strip()
    res = compute_curvature(mesh, radius_mm=3.0, smoothing_iterations=3)

    res_old = compute_curvature(
        mesh, radius_mm=3.0, smoothing_iterations=3,
        mask_boundary=True, boundary_margin_mm=6.0, boundary_metric="euclidean",
    )

    assert res.estimable_fraction > 0.80, (
        f"only {100 * res.estimable_fraction:.0f}% estimable; the support test is "
        "still discarding most of a plate-sized patch"
    )
    # The old rule keeps ~20% of a 22 x 24 mm plate, which is where the
    # "curvature is not estimable on 76-87% of this surface" warnings in the
    # original run came from -- the fixture reproduces them to within a few
    # points on a shape whose curvature is known exactly.
    assert res_old.estimable_fraction < 0.25, (
        "fixture no longer reproduces the old failure "
        f"({100 * res_old.estimable_fraction:.0f}% kept)"
    )


def test_open_strip_curvature_is_accurate():
    """Recovered ``H`` and ``K`` match the closed form on the open strip."""
    radius = 22.0
    mesh = cylinder_strip(radius_mm=radius)
    res = compute_curvature(mesh, radius_mm=3.0, smoothing_iterations=3)

    h_true = 1.0 / (2.0 * radius)
    h_hat = np.nanmedian(res.mean_curvature)
    k_hat = np.nanmedian(res.gaussian_curvature)

    assert abs(h_hat - h_true) / h_true < 0.10, f"H {h_hat:.5f} vs {h_true:.5f}"
    assert abs(k_hat) < 2e-4, f"K {k_hat:.6f} should be ~0 on a cylinder"


def test_open_cap_curvature_is_accurate():
    """Recovered ``H`` and ``K`` match the closed form on an open spherical cap."""
    radius = 25.0
    mesh = sphere_cap(radius_mm=radius)
    res = compute_curvature(mesh, radius_mm=3.0, smoothing_iterations=3)

    assert res.estimable_fraction > 0.80
    assert abs(np.nanmedian(res.mean_curvature) - 1.0 / radius) / (1.0 / radius) < 0.10
    assert abs(np.nanmedian(res.gaussian_curvature) - 1.0 / radius**2) / (
        1.0 / radius**2
    ) < 0.20


def test_rim_bias_is_bounded():
    """The rim strip the old margin deleted is now measured, and measured well.

    The margin was not arbitrary -- a one-sided quadric really is biased -- so
    the fix is only a fix if the vertices it restores carry an accurate value,
    not merely a value. Checked on the outer 3 mm of the strip.
    """
    radius = 22.0
    mesh = cylinder_strip(radius_mm=radius)
    res = compute_curvature(mesh, radius_mm=3.0, smoothing_iterations=3)

    distance = geodesic_boundary_distance(mesh)
    rim = (distance <= 3.0) & np.isfinite(res.mean_curvature)
    assert rim.sum() > 100, "no restored rim vertices to check"

    h_true = 1.0 / (2.0 * radius)
    rim_bias = (np.median(res.mean_curvature[rim]) - h_true) / h_true
    assert abs(rim_bias) < 0.15, f"rim bias {100 * rim_bias:.0f}%"


# --------------------------------------------------------------------------- #
# The pieces the fix rests on
# --------------------------------------------------------------------------- #


def test_geodesic_boundary_distance_does_not_cut_through_a_fold():
    """Euclidean distance-to-rim reaches around a fold; geodesic does not.

    On the folded sheet, points in the middle of the long arm are 1.5 mm from
    the short arm's free edge through space and ~20 mm from any edge along the
    surface. A Euclidean margin deletes them; the geodesic one keeps them.
    """
    mesh = folded_sheet()

    geodesic = geodesic_boundary_distance(mesh)
    euclidean_zone = boundary_influence_zone(mesh, 3.0, metric="euclidean")
    geodesic_zone = boundary_influence_zone(mesh, 3.0, metric="geodesic")

    assert np.isfinite(geodesic).all()
    assert geodesic_zone.sum() < euclidean_zone.sum(), (
        "the geodesic margin should keep vertices that are only near a rim "
        "through space"
    )
    # Specifically: vertices deep in the long arm, opposite the short arm's rim.
    deep = geodesic > 5.0
    assert (euclidean_zone & deep).sum() > 50, "fixture does not exercise the fold"
    assert not (geodesic_zone & deep).any()


def test_fold_does_not_contaminate_the_fit():
    """A KD-tree ball spans the fold; the fitted curvature must not.

    Both arms of the folded sheet are flat, so ``H = 0`` there exactly. If
    neighbours from the facing arm -- 1.5 mm away through space, 20 mm away
    along the surface -- were allowed into the quadric, the fit would see a
    surface doubling back and report large spurious curvature. This pins the
    normal-agreement and out-of-plane filters that keep them out.
    """
    mesh = folded_sheet()
    res = compute_curvature(mesh, radius_mm=3.0, smoothing_iterations=0)

    facing = (
        (np.abs(mesh.vertices[:, 1] - 0.75) < 1e-6)          # long arm
        & (mesh.vertices[:, 0] > -5.0)                        # opposite the short arm
        & (np.abs(mesh.vertices[:, 2]) < 10.0)                # away from the side rims
    )
    values = res.mean_curvature[facing]
    finite = values[np.isfinite(values)]

    assert finite.size > 50, "no vertices facing the short arm were estimated"
    assert np.abs(np.median(finite)) < 0.01, (
        f"median H {np.median(finite):.4f} mm^-1 on a flat arm: the fit is "
        "reaching across the fold"
    )


def test_closed_sphere_still_matches_closed_form():
    """The closed-surface behaviour that already worked must not regress."""
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=10.0)
    res = compute_curvature(sphere, radius_mm=2.0, smoothing_iterations=0)

    assert res.estimable_fraction > 0.99
    assert abs(np.nanmedian(res.mean_curvature) - 0.1) < 0.005
    assert abs(np.nanmedian(res.gaussian_curvature) - 0.01) < 0.002


def test_plane_is_flat():
    """A plane has zero curvature everywhere, including near its own edges."""
    n = 61
    grid = np.linspace(-15.0, 15.0, n)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    vertices = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.append([a, a + n, a + 1])
            faces.append([a + 1, a + n, a + n + 1])
    plane = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)

    res = compute_curvature(plane, radius_mm=3.0, smoothing_iterations=0)
    assert res.estimable_fraction > 0.80
    assert abs(np.nanmedian(res.mean_curvature)) < 1e-6
    assert abs(np.nanmedian(res.gaussian_curvature)) < 1e-6


def test_smoothing_does_not_resurrect_rejected_vertices():
    """NaN must survive the diffusion passes.

    The original ordering smoothed first and masked second, so a biased rim fit
    diffused three rings into the interior before being deleted. The order is now
    reversed and the smoother preserves NaN; this pins both.
    """
    mesh = cylinder_strip()
    res = compute_curvature(
        mesh, radius_mm=3.0, smoothing_iterations=5,
        mask_boundary=True, boundary_margin_mm=4.0,
    )
    zone = boundary_influence_zone(mesh, 4.0, metric="geodesic")
    assert not np.isfinite(res.mean_curvature[zone]).any()


def test_adaptive_radius_can_be_disabled():
    """``adaptive_radius=False`` gives a strict single-scale field."""
    mesh = cylinder_strip()
    res = compute_curvature(
        mesh, radius_mm=3.0, adaptive_radius=False, smoothing_iterations=0
    )
    radii = res.fit_radius_mm[np.isfinite(res.fit_radius_mm)]
    assert np.allclose(radii, 3.0)


def test_summary_reports_its_own_coverage():
    """A curvature mean without an estimable fraction is not interpretable."""
    from confcarti.thickness.curvature import curvature_summary

    mesh = cylinder_strip()
    summary = curvature_summary(compute_curvature(mesh, radius_mm=3.0))
    assert 0.0 <= summary["estimable_fraction"] <= 1.0
    assert summary["estimable_fraction"] > 0.80
    assert np.isfinite(summary["fit_radius_median_mm"])
