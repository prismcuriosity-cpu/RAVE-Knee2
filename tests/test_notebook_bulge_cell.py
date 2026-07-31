"""The bulge module, exercised as the notebook actually runs it.

Nothing in the repo imports ``bulge.py`` -- it exists only as a notebook cell --
so the curvature rewrite changed three names underneath it with no test in sight.
This file closes that hole by executing the curvature and bulge cells out of the
generated notebook, in order, exactly as the kernel does, and then running the
section-4 validation sweep that raised

    TypeError: _radius_neighbourhoods() missing 2 required positional
               arguments: 'targets' and 'radius_mm'

It is a slow test by unit-test standards and worth every second: it is the only
thing here that sees the two modules in the same namespace.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

NOTEBOOK = Path(__file__).resolve().parent.parent / "ConfCarti_Full_Pipeline_fixed.ipynb"


@pytest.fixture(scope="module")
def notebook_namespace():
    """Curvature then bulge, in one namespace — the notebook's execution model."""
    if not NOTEBOOK.is_file():
        pytest.skip(f"{NOTEBOOK.name} not generated; run notebook_patch/apply_fixes.py")

    cells = json.loads(NOTEBOOK.read_text())["cells"]
    sources = [
        "".join(c["source"]) if c["cell_type"] == "code" else None for c in cells
    ]

    namespace: dict = {"__name__": "__main__"}
    for anchor in ("def compute_curvature(", "def compute_bulge("):
        matches = [s for s in sources if s and anchor in s]
        assert len(matches) == 1, f"{anchor!r} matched {len(matches)} cells"
        exec(compile(matches[0], f"<{anchor}>", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def bumped_sphere():
    """The section-4 fixture: a 1.5 mm Gaussian bump on a 20 mm sphere."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
    v = np.asarray(sphere.vertices)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    w = np.exp(-((u - np.array([0, 0, 1.0])) ** 2).sum(1) / (2 * 0.05))
    bumped = sphere.copy()
    bumped.vertices = v + u * (1.5 * w)[:, None]
    return sphere, bumped


def test_every_code_cell_parses():
    if not NOTEBOOK.is_file():
        pytest.skip(f"{NOTEBOOK.name} not generated")
    for i, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))       # raises with the cell index


def test_bulge_sweep_runs_and_detects(notebook_namespace, bumped_sphere):
    """The cell that raised. It must run, and it must find the bump."""
    _sphere, bumped = bumped_sphere
    compute_bulge = notebook_namespace["compute_bulge"]
    bulge_summary = notebook_namespace["bulge_summary"]

    recovered = {}
    for radius in (8.0, 12.0, 15.0, 20.0):
        summary = bulge_summary(compute_bulge(bumped, form_radius_mm=radius))
        recovered[radius] = summary["bulge_max_height_mm"] / 1.5

    # Amplitudes are under-estimated by design -- a local form fit absorbs part
    # of the feature it isolates -- so this pins the band the original run
    # reported (0.05 to 0.54), not a high-recovery claim.
    assert 0.0 < recovered[8.0] < 0.15
    assert 0.25 < recovered[12.0] < 0.60
    assert 0.35 < recovered[15.0] < 0.65
    assert 0.25 < recovered[20.0] < 0.55


def test_clean_sphere_control_has_no_bulge_area(notebook_namespace, bumped_sphere):
    """The sweep's control: a sphere has form but no bulge."""
    sphere, _bumped = bumped_sphere
    summary = notebook_namespace["bulge_summary"](
        notebook_namespace["compute_bulge"](sphere, form_radius_mm=12.0)
    )
    assert summary["bulge_area_mm2"] == pytest.approx(0.0)
    assert summary["bulge_rms_mm"] < 0.10        # the documented false-positive floor


def test_bulge_states_its_own_neighbour_cap(notebook_namespace):
    """Bulge must not inherit the curvature module's cap.

    Curvature raised the shared ``max_neighbours`` default from 96 to 256 because
    a 3 mm ball at 0.5 mm resolution holds ~113 vertices. Bulge inherited it and
    its recovered amplitudes moved by a factor of several hundred at some radii
    (see the breakdown test below). Both modules now state the cap they were
    validated at.
    """
    import inspect

    for name in ("local_form_residual", "compute_bulge"):
        parameters = inspect.signature(notebook_namespace[name]).parameters
        assert "max_neighbours" in parameters, f"{name} does not state its cap"
        assert parameters["max_neighbours"].default == 96


def test_robust_fit_breaks_down_on_a_wide_feature(notebook_namespace, bumped_sphere):
    """A documented limitation, pinned so it stays visible.

    The Tukey reweighting in ``local_form_residual`` assumes the feature is a
    minority of the form ball. This bump is 145 vertices wide and a 12 mm ball
    holds ~230, so with the *full* neighbourhood the reweighting flips: it treats
    the sphere as the outlier population, fits the bump, and the residual
    collapses to zero. The 96-point cap keeps proportionally more far-field
    points and holds the fit on the sphere.

    That makes the recovered amplitude sensitive to a sampling parameter, which
    is a real caveat on the bulge validation numbers -- not a bug introduced
    here, but not something to leave undocumented either. Raising
    ``robust_iterations`` to 0 removes the sensitivity at the cost of a
    non-robust fit.
    """
    _sphere, bumped = bumped_sphere
    local_form_residual = notebook_namespace["local_form_residual"]

    capped = np.nanmax(local_form_residual(bumped, radius_mm=12.0, max_neighbours=96))
    full = np.nanmax(local_form_residual(bumped, radius_mm=12.0, max_neighbours=1024))
    assert capped > 0.3, "the capped fit should still see the bump"
    assert full < 0.05, "the full-neighbourhood robust fit is expected to break down"

    # With the reweighting off, the two agree: the sensitivity is the robust
    # step, not the neighbourhood itself.
    no_robust_capped = np.nanmax(
        local_form_residual(bumped, radius_mm=12.0, robust_iterations=0, max_neighbours=96)
    )
    no_robust_full = np.nanmax(
        local_form_residual(bumped, radius_mm=12.0, robust_iterations=0, max_neighbours=1024)
    )
    assert abs(no_robust_capped - no_robust_full) < 0.05
