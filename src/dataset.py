"""ModelNet mesh dataset -> chosen representation, with on-disk cache.

Why a cache: converting a mesh to a representation is *preprocessing* cost,
distinct from *training* cost. By converting once and caching the array, the
first pass measures preprocessing (a metric we care about and report), and later
epochs measure pure model training time. This separation is methodologically
important for a project whose focus is data representation & processing.

Per-rep tensor shapes returned by __getitem__:
  point  -> (N, 3)            surface-sampled points            [PointNet]
  sparse -> (N, 3)            occupied voxel-cell centers       [PointNet]
  voxel  -> (1, R, R, R)      dense occupancy grid (+channel)   [VoxNet]
  mview  -> (V, H, W)         multi-view silhouette images      [MultiViewCNN]
"""
import os
import glob
import json
import hashlib
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .convert import (
    load_mesh, normalize_mesh, sample_points, voxelize_dense,
    to_sparse_voxel_points, to_multiview,
)

REPS = ("point", "sparse", "voxel", "mview")


class ModelNetDataset(Dataset):
    def __init__(self, root, split, classes=None, rep="point",
                 num_points=1024, vox_res=32, n_surface=8192,
                 mv_views=8, mv_img_size=64, seed=0, cache_dir=None):
        assert rep in REPS, f"rep must be one of {REPS}"
        assert split in ("train", "test")
        self.rep = rep
        self.num_points = num_points
        self.vox_res = vox_res
        self.n_surface = n_surface
        self.mv_views = mv_views
        self.mv_img_size = mv_img_size
        self.seed = seed
        self.cache_dir = cache_dir

        avail = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
        wanted = set(classes) if classes is not None else None
        self.classes = [c for c in avail if (wanted is None or c in wanted)]
        assert self.classes, f"no matching class dirs under {root} (wanted {wanted})"
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.files = []
        for c in self.classes:
            idx = self.class_to_idx[c]
            for f in glob.glob(os.path.join(root, c, split, "*.off")):
                self.files.append((f, idx))
        self.files.sort(key=lambda fl: fl[0])

    def __len__(self):
        return len(self.files)

    def _tag(self):
        """Cache tag encoding every parameter that changes the output array,
        so different reps / settings never share cache files or markers."""
        s = self.n_surface
        if self.rep == "point":
            return f"point_p{self.num_points}_s{s}"
        if self.rep == "sparse":
            return f"sparse_v{self.vox_res}_p{self.num_points}_s{s}"
        if self.rep == "voxel":
            return f"voxel_v{self.vox_res}_s{s}"
        if self.rep == "mview":
            return f"mview_v{self.mv_views}_{self.mv_img_size}_s{s}"
        raise ValueError(self.rep)

    def _cache_marker(self):
        """Marker per (tag, split, class-set). The class-set hash is essential:
        a 3-class run and a 10-class run share split=train but must NOT share a
        marker, or the larger run skips building its extra classes."""
        split = (os.path.basename(os.path.dirname(self.files[0][0]))
                 if self.files else "unk")
        cls_hash = hashlib.md5(",".join(self.classes).encode()).hexdigest()[:10]
        return self.cache_dir / self._tag() / f"{split}_{cls_hash}" / ".built.json"

    def build_cache(self, workers=4, batch=256):
        """Materialize on-disk cache if cold; return (seconds, was_cold).
        Idempotent: a marker records the cold-conversion time so warm runs
        skip the work and reuse the number."""
        if self.cache_dir is None:
            return 0.0, True
        marker = self._cache_marker()
        if marker.exists():
            try:
                return float(json.loads(marker.read_text())["sec"]), False
            except Exception:
                pass
        loader = DataLoader(self, batch_size=batch, shuffle=False, num_workers=workers)
        t0, n = time.perf_counter(), 0
        for _, y in loader:
            n += y.size(0)
        sec = time.perf_counter() - t0
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"sec": sec, "n": n}))
        return sec, True

    def _convert(self, path):
        m = normalize_mesh(load_mesh(path))
        if self.rep == "point":
            return sample_points(m, self.num_points)                       # (N, 3)
        if self.rep == "sparse":
            return to_sparse_voxel_points(m, self.vox_res, self.num_points,
                                           self.n_surface, seed=self.seed)  # (N, 3)
        if self.rep == "voxel":
            return voxelize_dense(m, self.vox_res, self.n_surface)[None]   # (1, R, R, R)
        if self.rep == "mview":
            return to_multiview(m, self.mv_views, self.mv_img_size,
                                self.n_surface)                            # (V, H, W)
        raise ValueError(self.rep)

    def _cache_path(self, mesh_path):
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        cls = os.path.basename(os.path.dirname(os.path.dirname(mesh_path)))
        split = os.path.basename(os.path.dirname(mesh_path))
        return self.cache_dir / self._tag() / split / cls / f"{name}.npy"

    def __getitem__(self, i):
        path, label = self.files[i]
        if self.cache_dir is not None:
            cp = self._cache_path(path)
            if cp.exists():
                arr = np.load(cp)
            else:
                arr = self._convert(path)
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.save(cp, arr)
        else:
            arr = self._convert(path)
        return torch.from_numpy(arr), torch.tensor(label, dtype=torch.long)
