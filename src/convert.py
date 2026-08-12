"""Mesh -> alternative 3D representations.

This module is the scientific core of the project. Every function here takes a
*normalized* trimesh and produces one representation. Because all representations
start from the same normalized mesh, downstream timing differences are
attributable to the REPRESENTATION, not the source data.

Representations implemented:
  - sample_points()    -> rep A: point cloud (N, 3)
  - voxelize_dense()   -> rep B: dense occupancy grid (R, R, R)
TODO (filled in later steps):
  - voxelize_sparse()  -> rep C: occupied cells only (coords + features)
  - new_representation()-> rep D: TBD (the thing we invent)
"""
from __future__ import annotations
import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load an .off/.obj/.ply mesh file."""
    return trimesh.load(path, force="mesh")


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center to centroid and scale so the farthest vertex is at radius 1
    (unit-sphere normalization, the PointNet convention)."""
    m = mesh.copy()
    m.apply_translation(-(m.vertices.mean(axis=0)))
    max_r = float(np.linalg.norm(m.vertices, axis=1).max())
    if max_r > 0:
        m.apply_scale(1.0 / max_r)
    return m


# ---------------------------------------------------------------------------
# Rep A: point cloud
# ---------------------------------------------------------------------------
def sample_points(mesh: trimesh.Trimesh, n: int = 1024) -> np.ndarray:
    """Uniformly sample n points from the mesh surface -> (n, 3) float32."""
    pts = mesh.sample(n)
    return pts.astype(np.float32)


# ---------------------------------------------------------------------------
# Rep B: dense voxel grid
# ---------------------------------------------------------------------------
def voxelize_dense(mesh: trimesh.Trimesh, resolution: int = 32,
                   n_surface: int = 8192) -> np.ndarray:
    """Build a centered dense occupancy grid from surface samples.

    We sample points on the mesh surface and mark the voxels they fall into
    (cube [-1, 1]^3 mapped onto [0, R)^3). Surface-sampling is *robust*: it
    works on any mesh, including the many non-watertight models in ModelNet,
    where solid voxelization (mesh.voxelized().fill()) raises. Using one
    consistent method across the whole dataset keeps the experiment controlled.

    Returns float32 array of shape (R, R, R) in {0.0, 1.0} (a surface shell).
    """
    pts = mesh.sample(n_surface)
    idx = np.floor((pts + 1.0) * 0.5 * resolution).astype(np.int64)
    idx = np.clip(idx, 0, resolution - 1)
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return grid


# ---------------------------------------------------------------------------
# Rep C: sparse voxel  (TODO — implement in the rep-C step)
# ---------------------------------------------------------------------------
def to_sparse_voxel_points(mesh: trimesh.Trimesh, resolution: int = 32,
                           n_points: int = 1024, n_surface: int = 8192,
                           seed: int = 0) -> np.ndarray:
    """Rep C - sparse voxel: occupied cell CENTERS as a point cloud.

    The dense grid (rep B) stores all resolution^3 cells. The sparse / COO
    representation stores only the ~occupied ones (a few % of the cube). Here we
    take those occupied cells' centers and resample to a fixed n_points so the
    set can be read by a PointNet - making the comparison to rep A (continuous
    surface points) apples-to-apples, while demonstrating the sparse-storage
    representation. Returns (n_points, 3) float32 in [-1, 1].
    """
    pts = mesh.sample(n_surface)
    idx = np.floor((pts + 1.0) * 0.5 * resolution).astype(np.int64)
    idx = np.clip(idx, 0, resolution - 1)
    occ = np.unique(idx, axis=0)                   # (K, 3) unique occupied cells
    centers = (occ + 0.5) / resolution * 2.0 - 1.0  # back to [-1, 1], (K, 3)
    rng = np.random.default_rng(seed)
    k = len(centers)
    sel = rng.choice(k, n_points, replace=(k < n_points))
    return centers[sel].astype(np.float32)


# ---------------------------------------------------------------------------
# Rep D: new representation  (TODO — design + implement)
# ---------------------------------------------------------------------------
def to_multiview(mesh: trimesh.Trimesh, n_views: int = 8, img_size: int = 64,
                 n_surface: int = 8192) -> np.ndarray:
    """Rep D - multi-view 2D: render the mesh to n_views silhouette images.

    For each view we rotate the (unit-sphere-normalized) surface samples about
    the Y axis and orthographically project onto an image plane, marking hit
    pixels. This is point-splat silhouette rendering - no triangle rasterizer or
    GPU renderer needed, so it runs anywhere. The speed bet: 2D conv on the
    resulting image stack is very GPU-friendly. Returns (n_views, img_size,
    img_size) float32 in {0, 1}.
    """
    pts = mesh.sample(n_surface)                   # (n_surface, 3), mesh in [-1,1]
    views = np.zeros((n_views, img_size, img_size), dtype=np.float32)
    half = 0.5 * img_size
    for vi in range(n_views):
        a = 2.0 * np.pi * vi / n_views
        c, s = np.cos(a), np.sin(a)
        x = pts[:, 0] * c + pts[:, 2] * s           # rotate about Y, keep (x, y)
        y = pts[:, 1]
        ix = np.clip(((x + 1.0) * half).astype(np.int64), 0, img_size - 1)
        iy = np.clip(((y + 1.0) * half).astype(np.int64), 0, img_size - 1)
        views[vi, iy, ix] = 1.0
    return views
