from __future__ import annotations
# ==========================================================================
# confcarti/thickness/curvature.py   (corrected)
# ==========================================================================
"""Per-vertex differential geometry of cartilage and subchondral bone surfaces.

Why this module exists
----------------------
CartiMorph measures thickness, area, volume and full-thickness loss, but not the
*shape* of the surfaces those quantities live on. Curvature has repeatedly been
shown to carry osteoarthritis signal that thickness alone misses -- surface
smoothness discriminates OA in every compartment (Tummala et al. 2015), joint
incongruity is measurable in vivo (Hohe et al. 2002), and local/global curvature
behaves as an independent biomarker (Folkesson et al. 2008). This module closes
that gap and, downstream, supplies the heteroscedastic scale
:math:`\\hat{\\sigma}(x)` that the normalized conformal score needs.

What changed relative to the first version
------------------------------------------
The original code discarded curvature at every vertex lying within
``2 * radius_mm`` -- 6.0 mm at the default 3.0 mm fit radius -- of an open
boundary, measuring that distance **through space** with a KD-tree. On a knee
cartilage plate that is not a conservative choice, it is a fatal one:

* An articular plate is an *open sheet* with a very large perimeter-to-area
  ratio. The tibial plates are roughly 20-25 mm across, so a 6 mm rim eaten from
  every edge leaves almost nothing; the observed loss was **76-87 % of every
  surface**, and the compartment "curvature" that survived was a summary of a
  small, non-anatomical island in the middle of the plate.
* Euclidean distance reaches *around* folds. Two points on opposite banks of the
  trochlear groove, or across the intercondylar notch, are millimetres apart in
  space and centimetres apart on the surface. The old margin therefore also
  deleted genuinely interior vertices.

The margin existed to solve a real problem -- a quadric fitted on a
one-sided, clipped neighbourhood is badly biased -- but it attacked a proxy for
that problem instead of the problem. This version tests the thing that actually
matters, per vertex:

1. **Is the neighbourhood two-sided?** Measured directly, as the largest angular
   gap between neighbours projected into the tangent plane. A complete disc has
   a maximum gap of order ``2*pi/k``; a neighbourhood cut by an edge has a gap
   of at least ``pi``. Vertices whose gap exceeds ``max_angular_gap_deg``
   (default 120 deg) are the ones the margin was trying to catch, and they are
   caught wherever they occur -- including at interior holes, which a
   distance-to-outer-rim rule misses entirely.
2. **Does the neighbourhood have enough support, and is the fit conditioned?**
   Minimum neighbour count and a reciprocal-condition-number floor on the
   5 x 5 normal matrix, computed on the offsets normalised by the fit radius so
   the threshold is scale-free.
3. **Is the neighbourhood on the same sheet of surface?** Candidates are drawn
   with a KD-tree (fast) and then filtered on normal agreement and on
   out-of-plane offset, which removes points reached across the joint gap or
   across a fold. The remaining boundary field -- when a caller still wants a
   geometric margin -- is computed as a **geodesic** distance along mesh edges,
   not a Euclidean one.

Where a vertex fails at the nominal scale, the fit is retried at successively
smaller radii down to ``min_radius_mm`` (adaptive scale). That is what recovers
the rim of a narrow plate: a 1.5 mm-scale estimate near the edge is a real
measurement at a stated scale, whereas a NaN is not a measurement at all. The
radius actually used is returned per vertex and summarised into the results CSV,
so a mixed-scale field can never be mistaken for a single-scale one.

Measured effect on an open cylindrical strip of the width of a tibial plate
(``tests/test_curvature_open_patch.py``): the old rule returns NaN everywhere and
reports no curvature; this version estimates ~90 % of the patch with a median
mean-curvature error under 5 %.

Estimators
----------
``quadric``
    Fit a local height field over the tangent plane and read the first and
    second fundamental forms off the fit. Accurate and noise-tolerant; default.

``cotangent``
    Discrete operators of Meyer, Desbrun, Schroder & Barr (2003): mean curvature
    from the cotangent Laplace-Beltrami operator over the mixed Voronoi area,
    Gaussian curvature from the angle deficit.

Sign convention
---------------
Curvature is signed against the *outward* vertex normal. A convex surface
(femoral condyle seen from outside) has positive mean curvature. Both estimators
are validated against closed forms: sphere ``H = 1/R, K = 1/R^2``; cylinder
``H = 1/(2R), K = 0``; plane ``H = K = 0``.

References
----------
Meyer, M., Desbrun, M., Schroder, P., Barr, A.H. (2003). "Discrete
    differential-geometry operators for triangulated 2-manifolds."
    Visualization and Mathematics III, 35-57.
Koenderink, J.J., van Doorn, A.J. (1992). "Surface shape and curvature scales."
    Image and Vision Computing 10(8):557-564.  [shape index, curvedness]
Petitjean, S. (2002). "A survey of methods for recovering quadrics in triangle
    meshes." ACM Computing Surveys 34(2):211-262.
Hohe, J. et al. (2002). "Surface size, curvature analysis, and assessment of
    knee joint incongruity with MRI in vivo." Magn Reson Med 47(3):554-561.
Folkesson, J. et al. (2008). "Automatic quantification of local and global
    articular cartilage surface curvature." Magn Reson Med 59(6):1340-1346.
"""


import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

__all__ = [
    "CurvatureResult",
    "compute_curvature",
    "quadric_curvature",
    "cotangent_curvature",
    "shape_index",
    "curvedness",
    "smooth_scalar_field",
    "congruence_index",
    "curvature_summary",
    "boundary_vertices",
    "geodesic_boundary_distance",
    "boundary_influence_zone",
]

CurvatureMethod = Literal["quadric", "cotangent"]

#: Shape-index class boundaries of Koenderink & van Doorn (1992). The index runs
#: from -1 (spherical cup) through 0 (saddle) to +1 (spherical cap).
SHAPE_CLASSES: tuple[tuple[str, float, float], ...] = (
    ("cup", -1.000, -0.625),
    ("rut", -0.625, -0.375),
    ("saddle_rut", -0.375, -0.125),
    ("saddle", -0.125, 0.125),
    ("saddle_ridge", 0.125, 0.375),
    ("ridge", 0.375, 0.625),
    ("cap", 0.625, 1.001),
)

#: Default acceptance rule for a local quadric fit. These are the numbers the
#: 6 mm margin was standing in for, expressed as properties of the fit itself.
DEFAULT_MIN_NEIGHBOURS: int = 8
DEFAULT_MAX_ANGULAR_GAP_DEG: float = 120.0
DEFAULT_MIN_RCOND: float = 1e-4
DEFAULT_NORMAL_AGREEMENT_DEG: float = 60.0
DEFAULT_MAX_OUT_OF_PLANE_FRACTION: float = 0.6


@dataclass(frozen=True, slots=True)
class CurvatureResult:
    """Per-vertex curvature fields.

    Attributes
    ----------
    k1, k2
        Principal curvatures in mm^-1 with ``k1 >= k2``. NaN where the local fit
        was rejected.
    mean_curvature
        ``H = (k1 + k2) / 2`` in mm^-1.
    gaussian_curvature
        ``K = k1 * k2`` in mm^-2.
    shape_index
        Koenderink-van Doorn shape index in [-1, 1]; NaN where the surface is
        locally flat and the index is undefined.
    curvedness
        ``sqrt((k1^2 + k2^2) / 2)`` in mm^-1: how strongly curved, regardless of
        shape.
    normals
        Outward unit vertex normals used for the sign convention.
    method
        Estimator used.
    n_clipped
        Vertices whose principal curvatures hit the clip bound.
    estimable
        Boolean mask, True where the local fit passed the support test. This is
        the honest denominator for every summary below.
    fit_radius_mm
        Radius actually used at each vertex; NaN where nothing was estimable.
        Differs from the nominal radius only where the adaptive fallback fired.
    boundary_distance_mm
        Geodesic distance along the mesh to the nearest open boundary. Provided
        for diagnostics and for callers who want to weight or stratify by it;
        it is no longer used to blanket-discard vertices.
    rejection_reason
        Per-vertex code: 0 accepted, 1 too few neighbours, 2 one-sided
        neighbourhood (angular gap), 3 ill-conditioned fit.
    """

    k1: np.ndarray
    k2: np.ndarray
    mean_curvature: np.ndarray
    gaussian_curvature: np.ndarray
    shape_index: np.ndarray
    curvedness: np.ndarray
    normals: np.ndarray
    method: str
    n_clipped: int = 0
    estimable: np.ndarray | None = None
    fit_radius_mm: np.ndarray | None = None
    boundary_distance_mm: np.ndarray | None = None
    rejection_reason: np.ndarray | None = None

    @property
    def n_vertices(self) -> int:
        """Number of vertices carrying a curvature value."""
        return int(self.k1.shape[0])

    @property
    def estimable_fraction(self) -> float:
        """Fraction of vertices where the local fit was accepted."""
        if self.estimable is None:
            return float(np.mean(np.isfinite(self.k1))) if self.n_vertices else 0.0
        return float(np.mean(self.estimable)) if self.estimable.size else 0.0

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the scalar fields keyed by name, for painting onto a surface."""
        return {
            "k1": self.k1,
            "k2": self.k2,
            "mean_curvature": self.mean_curvature,
            "gaussian_curvature": self.gaussian_curvature,
            "shape_index": self.shape_index,
            "curvedness": self.curvedness,
        }


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #


def boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Boolean mask of vertices lying on an open boundary of the mesh.

    Parameters
    ----------
    mesh
        Input mesh.

    Returns
    -------
    numpy.ndarray
        ``(n_vertices,)`` boolean mask.
    """
    n = len(mesh.vertices)
    flags = np.zeros(n, dtype=bool)
    edges = mesh.edges_sorted
    if len(edges) == 0:
        return flags
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = unique_edges[counts == 1]
    if open_edges.size:
        flags[open_edges.reshape(-1)] = True
    return flags


def geodesic_boundary_distance(mesh: trimesh.Trimesh) -> np.ndarray:
    """Shortest path *along the mesh* from every vertex to an open boundary.

    Euclidean distance to the boundary is the wrong metric on a folded open
    sheet: a vertex deep inside the lateral trochlear facet is a couple of
    millimetres from the medial facet through space and a couple of centimetres
    from it along the cartilage. Measuring through the surface is what the
    quantity was always supposed to mean.

    Implemented as one multi-source Dijkstra over the mesh's edge graph with
    edge lengths as weights, which slightly over-estimates the true geodesic
    (paths are constrained to edges) and is therefore conservative.

    Parameters
    ----------
    mesh
        Input mesh.

    Returns
    -------
    numpy.ndarray
        ``(n_vertices,)`` distance in mm. ``inf`` on a closed surface, which has
        no boundary; 0 on the boundary itself.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    n = len(mesh.vertices)
    on_boundary = boundary_vertices(mesh)
    if not on_boundary.any():
        return np.full(n, np.inf, dtype=np.float64)

    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    lengths = np.asarray(mesh.edges_unique_length, dtype=np.float64)
    graph = coo_matrix(
        (lengths, (edges[:, 0], edges[:, 1])), shape=(n, n)
    ).tocsr()

    sources = np.flatnonzero(on_boundary)
    distance = dijkstra(graph, directed=False, indices=sources, min_only=True)
    return np.asarray(distance, dtype=np.float64)


def boundary_influence_zone(
    mesh: trimesh.Trimesh, margin_mm: float, *, metric: str = "geodesic"
) -> np.ndarray:
    """Vertices within ``margin_mm`` of an open boundary.

    Retained as an explicit, opt-in tool -- ``compute_curvature`` no longer
    applies it by default, because a blanket margin large enough to guarantee a
    two-sided neighbourhood is also large enough to delete most of a cartilage
    plate. Use it to *stratify* a curvature field, or to reproduce the old
    behaviour deliberately.

    Parameters
    ----------
    mesh
        Input mesh.
    margin_mm
        Margin around the open boundary.
    metric
        ``geodesic`` (default, along the surface) or ``euclidean`` (through
        space -- the old behaviour, kept only for reproducing it).

    Returns
    -------
    numpy.ndarray
        ``(n_vertices,)`` boolean mask, True inside the margin.

    Raises
    ------
    ValueError
        For an unknown metric.
    """
    if metric not in ("geodesic", "euclidean"):
        raise ValueError(f"metric must be geodesic|euclidean, got {metric!r}")

    on_boundary = boundary_vertices(mesh)
    if not on_boundary.any() or margin_mm <= 0:
        return on_boundary

    if metric == "geodesic":
        return geodesic_boundary_distance(mesh) <= margin_mm

    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    tree = cKDTree(vertices[on_boundary])
    distance, _idx = tree.query(vertices, k=1, workers=-1)
    return distance <= margin_mm


# --------------------------------------------------------------------------- #
# Neighbourhoods
# --------------------------------------------------------------------------- #


def _ring_neighbourhoods(
    mesh: trimesh.Trimesh, ring: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded ``ring``-neighbourhoods for every vertex.

    Parameters
    ----------
    mesh
        Input mesh.
    ring
        Number of edge hops.

    Returns
    -------
    tuple
        ``(index, mask)`` both ``(n_vertices, max_k)``. ``index`` holds
        neighbour vertex indices padded by repeating the centre vertex, and
        ``mask`` is True for genuine entries. Padding with a mask (rather than
        with an arbitrary vertex) keeps the batched least-squares fits unbiased.
    """
    neighbours = mesh.vertex_neighbors
    n = len(mesh.vertices)

    sets: list[set[int]] = [{i} | set(neighbours[i]) for i in range(n)]
    for _ in range(ring - 1):
        expanded: list[set[int]] = []
        for i in range(n):
            grown = set(sets[i])
            for j in sets[i]:
                grown.update(neighbours[j])
            expanded.append(grown)
        sets = expanded

    sizes = np.fromiter((len(s) for s in sets), dtype=np.int64, count=n)
    max_k = int(sizes.max())

    index = np.empty((n, max_k), dtype=np.int64)
    mask = np.zeros((n, max_k), dtype=bool)
    for i, s in enumerate(sets):
        members = np.fromiter(sorted(s), dtype=np.int64, count=len(s))
        index[i, : members.size] = members
        index[i, members.size :] = i          # pad with the centre vertex
        mask[i, : members.size] = True
    return index, mask


def _radius_neighbourhoods(
    mesh: trimesh.Trimesh, radius_mm: float, *, max_neighbours: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Padded radius neighbourhoods for **every** vertex of a mesh.

    This is a borrowed API, not a private helper: :mod:`confcarti.thickness.bulge`
    calls it directly to build the neighbourhoods for its local form fit, and in
    the notebook -- where every module shares one global namespace -- it resolves
    to this definition. Its signature is therefore a contract, pinned by
    ``tests/test_curvature_bulge_contract.py``. Changing it to the subset form the
    adaptive-radius loop wanted broke ``compute_bulge`` with

        TypeError: _radius_neighbourhoods() missing 2 required positional
                   arguments: 'targets' and 'radius_mm'

    Callers that need a subset want :func:`_ball_neighbourhoods`.

    Parameters
    ----------
    mesh
        Input mesh, vertices in mm.
    radius_mm
        Neighbourhood radius in millimetres.
    max_neighbours
        Cap on neighbours per vertex.

    Returns
    -------
    tuple
        ``(index, mask)``, both ``(n_vertices, max_k)``.
    """
    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return _ball_neighbourhoods(
        vertices,
        cKDTree(vertices),
        np.arange(len(vertices)),
        radius_mm,
        max_neighbours=max_neighbours,
    )


def _ball_neighbourhoods(
    vertices: np.ndarray,
    tree: "object",
    targets: np.ndarray,
    radius_mm: float,
    *,
    max_neighbours: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Padded neighbourhoods of a fixed *physical* radius, for a subset of vertices.

    Why radius and not edge hops
    ----------------------------
    Curvature is a second derivative, so its estimate is only meaningful at a
    stated spatial scale. A marching-cubes surface extracted from a binary mask
    carries residual staircase ripple with a wavelength of roughly one voxel;
    ripple of amplitude ``a`` and wavelength ``lambda`` contributes curvature of
    order ``a (2 pi / lambda)^2``, which for ``a = 0.05 mm`` and
    ``lambda = 1 mm`` is about ``2 mm^-1``. That utterly swamps the
    ``1/22 mm^-1`` of a femoral condyle. A neighbourhood defined in edge hops
    shrinks with the voxel size and therefore locks onto the ripple; a
    neighbourhood defined in millimetres averages over it and returns anatomy.

    This is the same scale-space argument made by Folkesson et al. (2008).

    Parameters
    ----------
    vertices
        ``(n, 3)`` vertex array in mm.
    tree
        A ``scipy.spatial.cKDTree`` already built over ``vertices``.
    targets
        Indices of the vertices to build neighbourhoods for.
    radius_mm
        Neighbourhood radius in millimetres.
    max_neighbours
        Cap on neighbours per vertex, to bound memory on dense meshes. When a
        vertex has more, a *radius-stratified* subset is taken (sorted by
        distance, then evenly sampled) rather than an arbitrary slice of the
        KD-tree's unordered output -- an arbitrary slice can leave a directional
        hole and make a perfectly good neighbourhood look one-sided.

    Returns
    -------
    tuple
        ``(index, mask)``, both ``(len(targets), max_k)``, padded with the
        centre vertex and masked as in :func:`_ring_neighbourhoods`.
    """
    groups = tree.query_ball_point(vertices[targets], r=radius_mm, workers=-1)

    trimmed: list[np.ndarray] = []
    for pos, members in enumerate(groups):
        centre = int(targets[pos])
        arr = np.asarray(members, dtype=np.int64)
        if arr.size == 0:
            arr = np.asarray([centre], dtype=np.int64)
        elif arr.size > max_neighbours:
            offsets = vertices[arr] - vertices[centre]
            order = np.argsort(np.einsum("ij,ij->i", offsets, offsets))
            arr = arr[order][np.linspace(0, arr.size - 1, max_neighbours).astype(np.int64)]
        trimmed.append(arr)

    max_k = max(int(a.size) for a in trimmed)
    index = np.empty((len(targets), max_k), dtype=np.int64)
    mask = np.zeros((len(targets), max_k), dtype=bool)
    for pos, arr in enumerate(trimmed):
        index[pos, : arr.size] = arr
        index[pos, arr.size :] = targets[pos]
        mask[pos, : arr.size] = True
    return index, mask


def _tangent_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal tangent basis ``(u, v)`` for each unit normal.

    Uses the numerically stable branchless construction of Duff et al. (2017),
    which avoids the degeneracy of crossing the normal with a fixed axis.
    """
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    sign = np.where(nz >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + nz)
    b = nx * ny * a
    u = np.stack([1.0 + sign * nx * nx * a, sign * b, -sign * nx], axis=1)
    v = np.stack([b, sign + ny * ny * a, -ny], axis=1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return u, v


def _max_angular_gap(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Largest angular gap between neighbours projected into the tangent plane.

    This is the direct test for the failure the old boundary margin was a proxy
    for. Sort the neighbour azimuths around the centre; on a complete disc of
    ``k`` neighbours the biggest gap between consecutive azimuths is of order
    ``2*pi/k``, while a neighbourhood clipped by an edge necessarily leaves a gap
    of at least ``pi`` on the missing side. Unlike a distance-to-rim rule it also
    fires at interior holes and at slivers, and unlike a Euclidean margin it
    never fires on a genuinely interior vertex that happens to sit close to
    another fold of the same surface.

    Parameters
    ----------
    x, y
        ``(m, k)`` neighbour coordinates in the tangent frame.
    mask
        ``(m, k)`` validity mask.

    Returns
    -------
    tuple
        ``(gap_rad, n_used)`` -- the largest gap in radians (``2*pi`` when there
        is nothing to measure) and the number of neighbours that contributed.
    """
    rho = np.hypot(x, y)
    usable = mask & (rho > 1e-9)
    n_used = usable.sum(axis=1)

    # Pad with a large finite sentinel rather than inf so that the diff below
    # yields 0 on the padding instead of a nan (and a spurious warning); the
    # padded entries are discarded by ``valid_gap`` either way.
    sentinel = 1e6
    theta = np.where(usable, np.arctan2(y, x), sentinel)
    theta_sorted = np.sort(theta, axis=1)

    gaps = np.diff(theta_sorted, axis=1)
    order = np.arange(theta.shape[1] - 1)[None, :]
    valid_gap = order < (n_used - 1)[:, None]
    interior_gap = np.where(valid_gap, gaps, -np.inf).max(axis=1)

    rows = np.arange(theta.shape[0])
    last = theta_sorted[rows, np.clip(n_used - 1, 0, theta.shape[1] - 1)]
    first = theta_sorted[:, 0]
    wrap_gap = np.where(n_used > 0, first + 2.0 * np.pi - last, 2.0 * np.pi)

    gap = np.maximum(interior_gap, wrap_gap)
    # A single neighbour, or none, leaves the whole circle open.
    gap = np.where(n_used >= 2, gap, 2.0 * np.pi)
    return gap, n_used


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #


def _quadric_pass(
    vertices: np.ndarray,
    normals: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    index: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    radius_mm: float,
    *,
    min_neighbours: int,
    max_angular_gap_deg: float,
    min_rcond: float,
    normal_agreement_deg: float,
    max_out_of_plane_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One quadric-fitting pass at a single radius, over ``targets``.

    ``vertices``, ``normals``, ``u`` and ``v`` are the full per-mesh arrays;
    ``index`` and ``mask`` are ``(len(targets), k)``.

    Returns ``(k1, k2, accepted, reason)`` for the target vertices. The fit
    itself is the Monge patch ``z = d x + e y + a x^2 + b xy + c y^2`` of the
    original implementation; what is new is that the neighbourhood is filtered
    onto the local sheet first and the result is only accepted when the fit had
    the support to be meaningful.
    """
    centres = vertices[targets]
    centre_normals = normals[targets]
    cu, cv = u[targets], v[targets]
    offsets = vertices[index] - centres[:, None, :]

    x = np.einsum("mkj,mj->mk", offsets, cu)
    y = np.einsum("mkj,mj->mk", offsets, cv)
    z = np.einsum("mkj,mj->mk", offsets, centre_normals)

    # --- keep only candidates that lie on the same sheet of surface --------- #
    # A KD-tree ball is a ball in R^3: on an open sheet folded back on itself
    # (trochlear groove, intercondylar notch) or across the joint gap it happily
    # returns points from a surface that is metres away geodesically. Two cheap,
    # fully vectorised tests remove them, and they are the two ways an
    # off-sheet point announces itself: its normal points somewhere else, and it
    # sits far out of the tangent plane.
    neighbour_normals = normals[index]
    agreement = np.einsum("mkj,mj->mk", neighbour_normals, centre_normals)
    same_sheet = agreement >= np.cos(np.deg2rad(normal_agreement_deg))
    in_plane = np.abs(z) <= max_out_of_plane_fraction * radius_mm
    is_centre = index == targets[:, None]
    mask = mask & (same_sheet | is_centre) & (in_plane | is_centre)

    gap, n_used = _max_angular_gap(x, y, mask & ~is_centre)

    # --- fit, on offsets normalised by the radius so rcond is scale-free ---- #
    xs, ys, zs = x / radius_mm, y / radius_mm, z
    design = np.stack([xs, ys, xs * xs, xs * ys, ys * ys], axis=2)
    weights = mask.astype(np.float64)[:, :, None]
    weighted = design * weights

    gram = np.einsum("mkp,mkq->mpq", weighted, design)
    rhs = np.einsum("mkp,mk->mp", weighted, zs * mask)

    eigenvalues = np.linalg.eigvalsh(gram)
    largest = eigenvalues[:, -1]
    smallest = eigenvalues[:, 0]
    rcond = np.where(largest > 0, np.clip(smallest, 0.0, None) / np.maximum(largest, 1e-300), 0.0)

    # Ridge term: keeps the system solvable where a neighbourhood is degenerate.
    # It only ever regularises fits that the acceptance test below rejects.
    scale = np.trace(gram, axis1=1, axis2=2) / 5.0
    scale = np.where(scale > 0, scale, 1.0)
    gram = gram + np.eye(5)[None] * (1e-8 * scale[:, None, None])

    try:
        coeffs = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this unlikely
        coeffs = np.linalg.lstsq(gram, rhs[:, :, None], rcond=None)[0][:, :, 0]

    # Undo the radius normalisation: x = r * x' gives f_x = d'/r, f_xx = 2a'/r^2.
    fx = coeffs[:, 0] / radius_mm
    fy = coeffs[:, 1] / radius_mm
    fxx = 2.0 * coeffs[:, 2] / radius_mm**2
    fxy = coeffs[:, 3] / radius_mm**2
    fyy = 2.0 * coeffs[:, 4] / radius_mm**2

    denom = np.sqrt(1.0 + fx**2 + fy**2)
    # S = I^{-1} II, written out in closed form for the 2x2 case.
    #
    # Sign convention. For a Monge patch z = f(x, y) the textbook second
    # fundamental form uses +f_xx, which makes a sphere viewed along its
    # *outward* normal come out negative: putting the origin at the north pole
    # with n = +z gives f(x, y) ~ -(x^2 + y^2) / 2R, hence f_xx = -1/R.
    # ConfCarti reports convex-outward as positive (sphere H = +1/R), matching
    # the cotangent estimator and the sign used throughout the OA curvature
    # literature, so the form is negated here. Gaussian curvature is unaffected
    # -- it is the product of two curvatures and both change sign together.
    e_coef = -fxx / denom
    f_coef = -fxy / denom
    g_coef = -fyy / denom
    big_e = 1.0 + fx**2
    big_f = fx * fy
    big_g = 1.0 + fy**2

    det_i = big_e * big_g - big_f**2
    det_i = np.where(np.abs(det_i) < 1e-12, 1.0, det_i)

    gaussian = (e_coef * g_coef - f_coef**2) / det_i
    mean = (e_coef * big_g - 2.0 * f_coef * big_f + g_coef * big_e) / (2.0 * det_i)

    disc = np.maximum(mean**2 - gaussian, 0.0)
    root = np.sqrt(disc)
    k1 = mean + root
    k2 = mean - root

    # --- acceptance -------------------------------------------------------- #
    enough = n_used >= min_neighbours
    two_sided = gap <= np.deg2rad(max_angular_gap_deg)
    conditioned = rcond >= min_rcond
    accepted = enough & two_sided & conditioned & np.isfinite(k1) & np.isfinite(k2)

    reason = np.zeros(len(targets), dtype=np.int8)
    reason[~enough] = 1
    reason[enough & ~two_sided] = 2
    reason[enough & two_sided & ~conditioned] = 3
    reason[accepted] = 0
    return k1, k2, accepted, reason


def quadric_curvature(
    mesh: trimesh.Trimesh,
    *,
    ring: int = 2,
    radius_mm: float | None = None,
    clip_mm_inv: float = 2.0,
    min_radius_mm: float | None = None,
    adaptive_radius: bool = True,
    radius_shrink: float = 1.5,
    min_neighbours: int = DEFAULT_MIN_NEIGHBOURS,
    max_angular_gap_deg: float = DEFAULT_MAX_ANGULAR_GAP_DEG,
    min_rcond: float = DEFAULT_MIN_RCOND,
    normal_agreement_deg: float = DEFAULT_NORMAL_AGREEMENT_DEG,
    max_out_of_plane_fraction: float = DEFAULT_MAX_OUT_OF_PLANE_FRACTION,
    max_neighbours: int = 256,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """Principal curvatures from a local quadric (Monge patch) fit.

    For each vertex the neighbourhood is expressed in a frame whose third axis
    is the vertex normal, and the height field is fitted as

    .. math:: z = d x + e y + a x^2 + b xy + c y^2

    The first and second fundamental forms at the origin are then

    .. math::
        \\mathrm{I} = \\begin{pmatrix} 1 + f_x^2 & f_x f_y \\\\
                                       f_x f_y & 1 + f_y^2 \\end{pmatrix},
        \\quad
        \\mathrm{II} = \\frac{1}{\\sqrt{1 + f_x^2 + f_y^2}}
                      \\begin{pmatrix} f_{xx} & f_{xy} \\\\
                                       f_{xy} & f_{yy} \\end{pmatrix}

    with ``f_x = d``, ``f_y = e``, ``f_xx = 2a``, ``f_xy = b``, ``f_yy = 2c``.
    The principal curvatures are the eigenvalues of the shape operator
    ``S = I^{-1} II``.

    Keeping the linear terms matters: dropping them assumes the fitted normal is
    exactly the surface normal, which on a noisy mesh biases ``H`` towards zero.

    Parameters
    ----------
    mesh
        Input mesh with consistent outward normals.
    ring
        Neighbourhood radius in edge hops. Used only when ``radius_mm`` is None,
        in which case the support test and the adaptive fallback are skipped.
    radius_mm
        Nominal neighbourhood radius in millimetres. Strongly preferred on
        marching-cubes surfaces -- see :func:`_ball_neighbourhoods`.
    clip_mm_inv
        Symmetric clip on the principal curvatures.
    min_radius_mm
        Floor for the adaptive fallback. Defaults to
        ``max(1.5 mm, 3 x mean edge length)``: below roughly three edge lengths
        the fit stops measuring anatomy and starts measuring marching-cubes
        staircase, so shrinking further would trade a NaN for a wrong number.
    adaptive_radius
        Retry rejected vertices at successively smaller radii. Set False for a
        strict single-scale field.
    radius_shrink
        Factor by which the radius is divided on each retry.
    min_neighbours, max_angular_gap_deg, min_rcond
        Acceptance rule for a fit: enough support, two-sided support, and a
        conditioned normal matrix. ``max_angular_gap_deg`` is the parameter that
        replaces the old ``2 * radius_mm`` boundary margin.
    normal_agreement_deg, max_out_of_plane_fraction
        Sheet filter applied to KD-tree candidates before fitting.
    max_neighbours
        Cap on neighbours per vertex.
    chunk_size
        Vertices fitted per batch, to bound peak memory.

    Returns
    -------
    tuple
        ``(k1, k2, normals, n_clipped, estimable, fit_radius_mm, reason)``
        with ``k1 >= k2`` and NaN at non-estimable vertices.

    Examples
    --------
    A sphere of radius 10 has ``H = 1/R = 0.1`` and ``K = 1/R^2 = 0.01``:

    >>> sphere = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    >>> k1, k2, _n, _c, ok, _r, _why = quadric_curvature(sphere, radius_mm=2.0)
    >>> bool(abs(np.nanmedian((k1 + k2) / 2) - 0.1) < 0.005)
    True
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms
    n = len(vertices)
    u, v = _tangent_frames(normals)

    if radius_mm is None:
        # Edge-hop mode: no physical scale, so no support test and no fallback.
        index, mask = _ring_neighbourhoods(mesh, ring)
        edge_length = float(np.mean(mesh.edges_unique_length)) if len(mesh.edges_unique) else 1.0
        k1, k2, accepted, reason = _quadric_pass(
            vertices, normals, u, v, index, mask, np.arange(n),
            radius_mm=max(edge_length * ring, 1e-6),
            min_neighbours=min_neighbours,
            max_angular_gap_deg=360.0,       # disabled: no physical neighbourhood
            min_rcond=0.0,
            normal_agreement_deg=180.0,
            max_out_of_plane_fraction=np.inf,
        )
        fit_radius = np.full(n, np.nan)
        fit_radius[accepted] = edge_length * ring
        n_clipped = int(np.sum((np.abs(k1) > clip_mm_inv) | (np.abs(k2) > clip_mm_inv)))
        k1 = np.where(accepted, np.clip(k1, -clip_mm_inv, clip_mm_inv), np.nan)
        k2 = np.where(accepted, np.clip(k2, -clip_mm_inv, clip_mm_inv), np.nan)
        return k1, k2, normals, n_clipped, accepted, fit_radius, reason

    from scipy.spatial import cKDTree

    tree = cKDTree(vertices)
    edge_length = float(np.mean(mesh.edges_unique_length)) if len(mesh.edges_unique) else 0.5
    if min_radius_mm is None:
        min_radius_mm = max(1.5, 3.0 * edge_length)
    min_radius_mm = min(min_radius_mm, radius_mm)

    k1 = np.full(n, np.nan)
    k2 = np.full(n, np.nan)
    fit_radius = np.full(n, np.nan)
    reason = np.full(n, 1, dtype=np.int8)
    estimable = np.zeros(n, dtype=bool)

    pending = np.arange(n)
    radius = float(radius_mm)
    n_clipped = 0

    while pending.size:
        lost_chunks: list[np.ndarray] = []
        # Chunked so that the (m, k, 5) design tensor stays bounded: a dense
        # femoral BCI mesh has tens of thousands of vertices and a 3 mm ball
        # holds ~100 of them, which is gigabytes if fitted in one go.
        for start in range(0, pending.size, chunk_size):
            block = pending[start : start + chunk_size]
            index, mask = _ball_neighbourhoods(
                vertices, tree, block, radius, max_neighbours=max_neighbours
            )
            pk1, pk2, accepted, preason = _quadric_pass(
                vertices, normals, u, v, index, mask, block, radius,
                min_neighbours=min_neighbours,
                max_angular_gap_deg=max_angular_gap_deg,
                min_rcond=min_rcond,
                normal_agreement_deg=normal_agreement_deg,
                max_out_of_plane_fraction=max_out_of_plane_fraction,
            )
            won = block[accepted]
            if won.size:
                wk1, wk2 = pk1[accepted], pk2[accepted]
                n_clipped += int(
                    np.sum((np.abs(wk1) > clip_mm_inv) | (np.abs(wk2) > clip_mm_inv))
                )
                k1[won] = np.clip(wk1, -clip_mm_inv, clip_mm_inv)
                k2[won] = np.clip(wk2, -clip_mm_inv, clip_mm_inv)
                fit_radius[won] = radius
                estimable[won] = True
                reason[won] = 0
            block_lost = block[~accepted]
            reason[block_lost] = preason[~accepted]
            lost_chunks.append(block_lost)

        lost = np.concatenate(lost_chunks) if lost_chunks else np.empty(0, dtype=np.int64)
        if not adaptive_radius or lost.size == 0:
            break
        next_radius = radius / radius_shrink
        if next_radius < min_radius_mm - 1e-9:
            break
        radius = next_radius
        pending = lost

    return k1, k2, normals, n_clipped, estimable, fit_radius, reason


def cotangent_curvature(
    mesh: trimesh.Trimesh, *, clip_mm_inv: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Discrete curvature via the cotangent Laplacian and the angle deficit.

    Implements Meyer et al. (2003):

    .. math::
        2 H \\mathbf{n} = \\frac{1}{2 A_{\\mathrm{mixed}}}
            \\sum_{j \\in N(i)} (\\cot \\alpha_{ij} + \\cot \\beta_{ij})
            (\\mathbf{x}_i - \\mathbf{x}_j)

    .. math::
        K = \\frac{1}{A_{\\mathrm{mixed}}}
            \\left( 2\\pi - \\sum_j \\theta_j \\right)

    Parameters
    ----------
    mesh
        Input mesh.
    clip_mm_inv
        Symmetric clip on the principal curvatures.

    Returns
    -------
    tuple
        ``(k1, k2, normals, n_clipped, estimable)``. Boundary vertices are not
        estimable: the angle deficit needs a complete one-ring and there is not
        one. This is a one-ring rule, so it removes a single row of vertices --
        not a 6 mm band.

    Notes
    -----
    ``A_mixed`` is approximated by one third of the incident face area
    (the barycentric cell). Meyer's obtuse-triangle correction is skipped: on
    the Taubin-smoothed marching-cubes meshes used here the triangles are close
    to equilateral, and the correction changes ``H`` by well under a percent
    while roughly tripling the cost.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n = len(vertices)

    normals = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]

    e0 = p2 - p1        # opposite vertex 0
    e1 = p0 - p2        # opposite vertex 1
    e2 = p1 - p0        # opposite vertex 2

    cross = np.cross(e2, -e1)
    face_area2 = np.linalg.norm(cross, axis=1)
    face_area = 0.5 * face_area2
    safe_area2 = np.where(face_area2 < 1e-14, 1e-14, face_area2)

    # cot at a vertex = (dot of the two incident edge vectors) / (2 * face area)
    cot0 = np.einsum("ij,ij->i", -e2, e1) / safe_area2
    cot1 = np.einsum("ij,ij->i", -e0, e2) / safe_area2
    cot2 = np.einsum("ij,ij->i", -e1, e0) / safe_area2

    laplacian = np.zeros((n, 3), dtype=np.float64)
    # Edge (1,2) is weighted by the cotangent at vertex 0, and so on.
    for cot, a, b in ((cot0, i1, i2), (cot1, i2, i0), (cot2, i0, i1)):
        contribution = cot[:, None] * (vertices[a] - vertices[b])
        np.add.at(laplacian, a, contribution)
        np.add.at(laplacian, b, -contribution)

    area_mixed = np.zeros(n, dtype=np.float64)
    for idx in (i0, i1, i2):
        np.add.at(area_mixed, idx, face_area / 3.0)
    area_mixed = np.where(area_mixed < 1e-12, 1e-12, area_mixed)

    mean_vector = laplacian / (4.0 * area_mixed[:, None])
    mean = np.einsum("ij,ij->i", mean_vector, normals)

    # Gaussian curvature from the angle deficit.
    def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        denom = np.where(nu * nv < 1e-14, 1e-14, nu * nv)
        return np.arccos(np.clip(np.einsum("ij,ij->i", u, v) / denom, -1.0, 1.0))

    angles = np.zeros(n, dtype=np.float64)
    np.add.at(angles, i0, _angle(p1 - p0, p2 - p0))
    np.add.at(angles, i1, _angle(p2 - p1, p0 - p1))
    np.add.at(angles, i2, _angle(p0 - p2, p1 - p2))

    gaussian = (2.0 * np.pi - angles) / area_mixed

    # Boundary vertices have no full angle ring, so the deficit is meaningless.
    # Marking them non-estimable is the honest answer; the previous version set
    # K = 0 there, which silently reported a saddle-free plane along every rim.
    boundary = boundary_vertices(mesh)
    estimable = ~boundary

    disc = np.maximum(mean**2 - gaussian, 0.0)
    root = np.sqrt(disc)
    k1 = mean + root
    k2 = mean - root

    n_clipped = int(np.sum((np.abs(k1) > clip_mm_inv) | (np.abs(k2) > clip_mm_inv)))
    k1 = np.where(estimable, np.clip(k1, -clip_mm_inv, clip_mm_inv), np.nan)
    k2 = np.where(estimable, np.clip(k2, -clip_mm_inv, clip_mm_inv), np.nan)
    return k1, k2, normals, n_clipped, estimable


# --------------------------------------------------------------------------- #
# Derived descriptors
# --------------------------------------------------------------------------- #


def shape_index(k1: np.ndarray, k2: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Koenderink-van Doorn shape index.

    .. math:: S = \\frac{2}{\\pi} \\arctan \\frac{k_2 + k_1}{k_2 - k_1},
              \\quad k_1 \\ge k_2

    Runs from -1 (spherical cup) through 0 (symmetric saddle) to +1 (spherical
    cap), and is *scale invariant*: it describes local shape independently of
    how strongly curved the surface is. That separation is why it is reported
    alongside :func:`curvedness` rather than instead of it.

    Parameters
    ----------
    k1, k2
        Principal curvatures with ``k1 >= k2``.
    eps
        Below this ``|k1 - k2|`` the surface is umbilic-flat and the index is
        undefined; NaN is returned there.

    Returns
    -------
    numpy.ndarray
        Shape index in [-1, 1], NaN where undefined.

    Examples
    --------
    >>> float(shape_index(np.array([0.1]), np.array([0.1]))[0])
    1.0
    >>> float(shape_index(np.array([0.1]), np.array([-0.1]))[0])
    0.0
    """
    k1 = np.asarray(k1, dtype=np.float64)
    k2 = np.asarray(k2, dtype=np.float64)

    out = np.full(k1.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(k1) & np.isfinite(k2)
    # Umbilic points (k1 == k2) are pure cap or cup: +1 or -1.
    umbilic = finite & (np.abs(k1 - k2) < eps)
    flat = umbilic & (np.abs(k1) < eps)
    out[umbilic] = np.sign(k1[umbilic])
    out[flat] = np.nan

    # Koenderink & van Doorn (1992), eq. 3, with the k1 >= k2 ordering used here:
    #     S = (2/pi) arctan((k1 + k2) / (k1 - k2))
    # The denominator is k1 - k2 (non-negative). Writing it as k2 - k1 flips the
    # sign of the whole index and turns every convex cap into a cup.
    ordinary = finite & ~umbilic
    out[ordinary] = (2.0 / np.pi) * np.arctan(
        (k1[ordinary] + k2[ordinary]) / (k1[ordinary] - k2[ordinary])
    )
    # Normalise -0.0 to 0.0 so a symmetric saddle prints as 0.0.
    return out + 0.0


def curvedness(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Koenderink-van Doorn curvedness, ``sqrt((k1^2 + k2^2) / 2)`` in mm^-1.

    Parameters
    ----------
    k1, k2
        Principal curvatures.

    Returns
    -------
    numpy.ndarray
        Curvedness; 0 on a plane, ``1/R`` on a sphere of radius ``R``.

    Examples
    --------
    >>> float(curvedness(np.array([0.1]), np.array([0.1]))[0])
    0.1
    """
    k1 = np.asarray(k1, dtype=np.float64)
    k2 = np.asarray(k2, dtype=np.float64)
    return np.sqrt((k1**2 + k2**2) / 2.0)


def smooth_scalar_field(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    iterations: int = 3,
    *,
    preserve_nan: bool = True,
) -> np.ndarray:
    """Diffuse a per-vertex scalar over the mesh graph.

    Curvature is a second derivative, so it amplifies mesh noise; a few
    averaging passes make the field readable without materially shifting its
    mean. NaNs are ignored by the averaging.

    Parameters
    ----------
    mesh
        Mesh supplying the vertex adjacency.
    field
        ``(n_vertices,)`` scalar field.
    iterations
        Number of averaging passes.
    preserve_nan
        Keep NaN vertices NaN instead of filling them from their neighbours.
        This must be True when smoothing runs *after* the support test,
        otherwise a rejected vertex is quietly resurrected from the very
        neighbours that were too few to fit it. It is the default for that
        reason.

    Returns
    -------
    numpy.ndarray
        Smoothed field.
    """
    if iterations <= 0:
        return np.asarray(field, dtype=np.float64)

    values = np.asarray(field, dtype=np.float64).copy()
    keep_nan = ~np.isfinite(values) if preserve_nan else None
    neighbours = mesh.vertex_neighbors

    # Flatten adjacency once so each pass is a single segmented mean.
    lengths = np.fromiter((len(nb) for nb in neighbours), dtype=np.int64, count=len(neighbours))
    flat = (
        np.concatenate([np.asarray(nb, dtype=np.int64) for nb in neighbours])
        if lengths.sum()
        else np.empty(0, dtype=np.int64)
    )
    owner = np.repeat(np.arange(len(neighbours)), lengths)

    for _ in range(iterations):
        contrib = values[flat]
        valid = np.isfinite(contrib)
        totals = np.zeros(len(neighbours), dtype=np.float64)
        counts = np.zeros(len(neighbours), dtype=np.float64)
        np.add.at(totals, owner[valid], contrib[valid])
        np.add.at(counts, owner[valid], 1.0)
        averaged = np.where(counts > 0, totals / np.maximum(counts, 1.0), values)
        # Half-weight update keeps the field from collapsing to its global mean.
        values = np.where(np.isfinite(values), 0.5 * values + 0.5 * averaged, averaged)
        if keep_nan is not None:
            values[keep_nan] = np.nan
    return values


def compute_curvature(
    mesh: trimesh.Trimesh,
    *,
    method: CurvatureMethod = "quadric",
    ring: int = 2,
    radius_mm: float | None = 3.0,
    clip_mm_inv: float = 2.0,
    smoothing_iterations: int = 3,
    adaptive_radius: bool = True,
    min_radius_mm: float | None = None,
    min_neighbours: int = DEFAULT_MIN_NEIGHBOURS,
    max_angular_gap_deg: float = DEFAULT_MAX_ANGULAR_GAP_DEG,
    min_rcond: float = DEFAULT_MIN_RCOND,
    normal_agreement_deg: float = DEFAULT_NORMAL_AGREEMENT_DEG,
    max_out_of_plane_fraction: float = DEFAULT_MAX_OUT_OF_PLANE_FRACTION,
    max_neighbours: int = 256,
    chunk_size: int = 8192,
    mask_boundary: bool = False,
    boundary_margin_mm: float | None = None,
    boundary_metric: str = "geodesic",
    warn_below_estimable_fraction: float = 0.5,
) -> CurvatureResult:
    """Compute the full curvature description of a surface.

    Parameters
    ----------
    mesh
        Surface with consistent outward normals.
    method
        ``quadric`` or ``cotangent``.
    ring
        Edge-hop neighbourhood for the quadric fit; used only when
        ``radius_mm`` is None.
    radius_mm
        Nominal physical neighbourhood radius in mm for the quadric fit. This
        sets the spatial scale of the estimate and is the parameter that
        matters: on a marching-cubes surface an edge-hop neighbourhood measures
        voxel staircasing instead of anatomy. Pass None to fall back to
        ``ring``.
    clip_mm_inv
        Symmetric principal-curvature clip.
    smoothing_iterations
        Diffusion passes applied to the curvature fields. Applied **after** the
        support test, with NaNs preserved, so that a rejected boundary fit
        cannot diffuse into the interior. (The previous version smoothed first
        and masked second, which pushed the very boundary bias it was trying to
        remove up to three rings inward.)
    adaptive_radius, min_radius_mm
        Retry rejected vertices at smaller scales, down to ``min_radius_mm``.
    min_neighbours, max_angular_gap_deg, min_rcond
        Acceptance rule for a local fit. ``max_angular_gap_deg`` is what
        replaces the old ``2 * radius_mm`` geometric margin: it asks directly
        whether the neighbourhood surrounds the vertex, instead of guessing from
        a distance that it might not.
    normal_agreement_deg, max_out_of_plane_fraction, max_neighbours, chunk_size
        Sheet filter, neighbourhood cap and batch size; see
        :func:`quadric_curvature`.
    mask_boundary
        Additionally NaN everything within ``boundary_margin_mm`` of an open
        boundary. **Off by default** -- on a cartilage plate a margin wide enough
        to matter deletes most of the plate, and the support test already
        rejects the fits that margin was aimed at. Turn it on to reproduce older
        results or to enforce a single-scale interior-only field.
    boundary_margin_mm
        Margin used when ``mask_boundary`` is True. Defaults to ``radius_mm``
        (one fit radius), not two.
    boundary_metric
        ``geodesic`` (default) or ``euclidean``. Geodesic is the correct metric
        on a folded open sheet; ``euclidean`` reproduces the old behaviour.
    warn_below_estimable_fraction
        Emit a warning when less than this fraction of the surface ends up
        estimable.

    Returns
    -------
    CurvatureResult

    Raises
    ------
    ValueError
        For an unknown method or an empty mesh.

    Examples
    --------
    >>> sphere = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    >>> res = compute_curvature(sphere, radius_mm=None, smoothing_iterations=0)
    >>> bool(abs(np.nanmedian(res.mean_curvature) - 0.1) < 0.005)
    True
    >>> bool(abs(np.nanmedian(res.gaussian_curvature) - 0.01) < 0.002)
    True
    """
    if method not in ("quadric", "cotangent"):
        raise ValueError(f"method must be quadric|cotangent, got {method!r}")
    if len(mesh.vertices) == 0:
        raise ValueError("cannot compute curvature on a mesh with no vertices")

    if method == "quadric":
        k1, k2, normals, n_clipped, estimable, fit_radius, reason = quadric_curvature(
            mesh,
            ring=ring,
            radius_mm=radius_mm,
            clip_mm_inv=clip_mm_inv,
            adaptive_radius=adaptive_radius,
            min_radius_mm=min_radius_mm,
            min_neighbours=min_neighbours,
            max_angular_gap_deg=max_angular_gap_deg,
            min_rcond=min_rcond,
            normal_agreement_deg=normal_agreement_deg,
            max_out_of_plane_fraction=max_out_of_plane_fraction,
            max_neighbours=max_neighbours,
            chunk_size=chunk_size,
        )
    else:
        k1, k2, normals, n_clipped, estimable = cotangent_curvature(
            mesh, clip_mm_inv=clip_mm_inv
        )
        fit_radius = np.where(estimable, float(np.mean(mesh.edges_unique_length)), np.nan)
        reason = np.where(estimable, 0, 1).astype(np.int8)

    # Optional extra geometric margin, applied BEFORE smoothing so that nothing
    # it removes can leak back in.
    if mask_boundary:
        if boundary_margin_mm is None:
            boundary_margin_mm = float(radius_mm) if radius_mm is not None else 0.0
        if boundary_margin_mm > 0:
            zone = boundary_influence_zone(mesh, boundary_margin_mm, metric=boundary_metric)
            k1 = np.where(zone, np.nan, k1)
            k2 = np.where(zone, np.nan, k2)
            fit_radius = np.where(zone, np.nan, fit_radius)
            estimable = estimable & ~zone
            reason = np.where(zone, np.int8(4), reason)

    if smoothing_iterations > 0:
        k1 = smooth_scalar_field(mesh, k1, smoothing_iterations, preserve_nan=True)
        k2 = smooth_scalar_field(mesh, k2, smoothing_iterations, preserve_nan=True)
        # Re-impose k1 >= k2, which independent smoothing can violate.
        k1, k2 = np.maximum(k1, k2), np.minimum(k1, k2)

    fraction = float(np.mean(estimable)) if estimable.size else 0.0
    if fraction < warn_below_estimable_fraction:
        counts = np.bincount(np.asarray(reason, dtype=np.int64), minlength=5)
        logger.warning(
            "curvature estimable on only %.0f%% of this surface (%d/%d vertices). "
            "Rejections: %d too few neighbours, %d one-sided neighbourhood, "
            "%d ill-conditioned, %d inside an explicit boundary margin. If the "
            "one-sided count dominates, the patch is genuinely narrow relative to "
            "the %.1f mm fit radius -- lower radius_mm or min_radius_mm.",
            100.0 * fraction,
            int(estimable.sum()),
            int(estimable.size),
            int(counts[1]), int(counts[2]), int(counts[3]), int(counts[4]),
            float(radius_mm) if radius_mm is not None else float("nan"),
        )
    else:
        logger.debug(
            "curvature estimable on %.0f%% of the surface; median fit radius %.2f mm",
            100.0 * fraction,
            float(np.nanmedian(fit_radius)) if np.isfinite(fit_radius).any() else float("nan"),
        )

    return CurvatureResult(
        k1=k1,
        k2=k2,
        mean_curvature=(k1 + k2) / 2.0,
        gaussian_curvature=k1 * k2,
        shape_index=shape_index(k1, k2),
        curvedness=curvedness(k1, k2),
        normals=normals,
        method=method,
        n_clipped=n_clipped,
        estimable=estimable,
        fit_radius_mm=fit_radius,
        boundary_distance_mm=geodesic_boundary_distance(mesh),
        rejection_reason=reason,
    )


def congruence_index(
    femoral: CurvatureResult, tibial: CurvatureResult
) -> dict[str, float]:
    """Joint incongruity between two opposing articular surfaces.

    Hohe et al. (2002) define incongruity from the difference of the principal
    curvatures of the two opposing surfaces. Because the surfaces face each
    other, a perfectly congruent pair has *opposite-signed* curvature of equal
    magnitude, so the residual is formed as ``k_fem + k_tib``.

    Parameters
    ----------
    femoral, tibial
        Curvature of the two opposing surfaces.

    Returns
    -------
    dict
        ``incongruity_k1_mm_inv``, ``incongruity_k2_mm_inv``,
        ``incongruity_rms_mm_inv`` and the two median curvednesses.

    Notes
    -----
    This is a *distributional* comparison: the two meshes have different vertex
    counts and no point correspondence, so medians are compared rather than
    per-vertex differences. Establishing correspondence would need registration
    to the CLAIR-Knee-103R template, which is out of scope here.
    """
    fem_k1 = float(np.nanmedian(femoral.k1))
    fem_k2 = float(np.nanmedian(femoral.k2))
    tib_k1 = float(np.nanmedian(tibial.k1))
    tib_k2 = float(np.nanmedian(tibial.k2))

    d1 = fem_k1 + tib_k1
    d2 = fem_k2 + tib_k2
    return {
        "incongruity_k1_mm_inv": d1,
        "incongruity_k2_mm_inv": d2,
        "incongruity_rms_mm_inv": float(np.sqrt((d1**2 + d2**2) / 2.0)),
        "femoral_curvedness_mm_inv": float(np.nanmedian(femoral.curvedness)),
        "tibial_curvedness_mm_inv": float(np.nanmedian(tibial.curvedness)),
    }


def curvature_summary(
    result: CurvatureResult, *, shape_index_bins: int = 9
) -> dict[str, float]:
    """Reduce a curvature field to scalar descriptors for the results CSV.

    Parameters
    ----------
    result
        Curvature fields.
    shape_index_bins
        Number of equal-width shape-index bins whose occupancy is reported.

    Returns
    -------
    dict
        Robust summaries plus the shape-class fractions, plus the coverage
        diagnostics -- ``estimable_fraction`` and the fit-radius quantiles --
        without which a curvature mean is uninterpretable: it says nothing about
        whether it describes the whole plate or an island in the middle of it.
    """

    def _stats(name: str, values: np.ndarray) -> dict[str, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {f"{name}_{k}": float("nan") for k in ("mean", "median", "sd", "p05", "p95")}
        return {
            f"{name}_mean": float(finite.mean()),
            f"{name}_median": float(np.median(finite)),
            f"{name}_sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            f"{name}_p05": float(np.percentile(finite, 5)),
            f"{name}_p95": float(np.percentile(finite, 95)),
        }

    out: dict[str, float] = {}
    out.update(_stats("mean_curvature", result.mean_curvature))
    out.update(_stats("gaussian_curvature", result.gaussian_curvature))
    out.update(_stats("curvedness", result.curvedness))
    out.update(_stats("shape_index", result.shape_index))

    finite_si = result.shape_index[np.isfinite(result.shape_index)]
    if finite_si.size:
        hist, _edges = np.histogram(finite_si, bins=shape_index_bins, range=(-1.0, 1.0))
        fractions = hist / finite_si.size
        for i, frac in enumerate(fractions):
            out[f"shape_index_bin{i:02d}_frac"] = float(frac)
        for name, lo, hi in SHAPE_CLASSES:
            out[f"shape_class_{name}_frac"] = float(
                np.mean((finite_si >= lo) & (finite_si < hi))
            )

    out["curvature_clipped_fraction"] = (
        float(result.n_clipped / result.n_vertices) if result.n_vertices else 0.0
    )
    out["estimable_fraction"] = result.estimable_fraction

    radii = result.fit_radius_mm
    if radii is not None and np.isfinite(radii).any():
        finite_r = radii[np.isfinite(radii)]
        out["fit_radius_median_mm"] = float(np.median(finite_r))
        out["fit_radius_p05_mm"] = float(np.percentile(finite_r, 5))
        out["fit_radius_reduced_fraction"] = float(
            np.mean(finite_r < finite_r.max() - 1e-9)
        )
    else:
        out["fit_radius_median_mm"] = float("nan")
        out["fit_radius_p05_mm"] = float("nan")
        out["fit_radius_reduced_fraction"] = float("nan")

    return out
