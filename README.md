# Does 3D data *format* drive training & inference time?

**Research question:** Holding the task and the 3D objects fixed, does the **data
representation** (point cloud vs. dense voxel vs. sparse voxel vs. a new format)
change **training time** and **inference time** — and by how much?

Accuracy is *recorded* but is **not** the focus. The focus is **data
representation and data processing**.

> **Headline claim:** representation choice is a *data-ingestion* decision — it
> decides whether the pipeline pays to encode, or the model pays on every
> forward pass. See [`CLAIM.md`](CLAIM.md) for exactly how much of that is
> backed and what is still a hypothesis.

## Why this is a fair test

Every representation is derived from the **same `.off` mesh**, normalized to the
unit sphere. So the *information* is identical across conditions; any difference
in compute is attributable to the **representation**, not the data. That is the
whole point of the experiment.

## Related work — and the gap this fills

**Each representation has its champions, not its arbiters.** The 3D-representation
literature is written largely by *authors of a representation*, each showing their
rep beats a chosen baseline:

- **PointNet** [Qi 2017] — raw point sets beat voxelization by skipping the grid
  entirely (O(N), >1M pts/s).
- **OctNet** [Riegler 2017] — a hybrid grid-octree concentrates compute on
  occupied cells, making high-resolution 3D conv feasible.
- **PVCNN** [Liu 2019] — a *hybrid* (points for memory, voxels for conv) beats
  either pure form — the strongest evidence that no single rep wins outright.
- **Volumetric & Multi-View CNNs** [Qi 2016] — the closest precedent: directly
  compared volumetric vs. multi-view and found multi-view ahead.
- **PointPillars** [Lang 2019] — collapsed one spatial dimension (3D voxel → 2D
  pillar) to swap expensive 3D conv for fast 2D conv (62 fps).

What none of these does is hold task, data, and hardware fixed and vary *only* the
representation across all of them at once. Each champions one rep against a
baseline it can beat. That **controlled cross-representation comparison** is the
gap this benchmark fills — which is why the headline result (multi-view fastest by
~9×, dense voxel slowest but most accurate, parameter count not predicting speed)
is not redundant with any single prior paper.

**The counterpoint that makes the gap worth filling.** "Sparse is faster" is not a
law: sparse convolution can run **~10× slower** than dense when the input is small
or dense, because hash-table and rule-book overhead dominates the saved emptiness
[MinkowskiEngine #158]. *Which* representation wins is therefore genuinely
hardware- and scene-dependent — not something any one champion paper can settle.
Only a controlled cross-rep benchmark can.

**The bigger frame this points to.** Stepping back from 3D, every representation —
geometric or tabular — is a choice of two things: an **origin** (what counts as
zero) and a set of **axes** (how deviation from zero is measured). The 3D reps
above are axis choices (3D grid vs. 2D camera planes vs. point coordinates). The
identical move exists in business/tabular data: *deviation coding* sets the most
common category as the reference (zero) and measures the rest as deviation from it.
This project demonstrates the principle concretely in one domain; a natural
generalization is a **representation-routing pipeline** that selects the
cheapest-fitting representation per data type. That is named here as future work,
not claimed as done.

**References**
- Qi et al., *PointNet*, CVPR 2017 — https://arxiv.org/abs/1612.00593
- Riegler et al., *OctNet*, CVPR 2017 — https://arxiv.org/abs/1611.05009
- Liu et al., *Point-Voxel CNN (PVCNN)*, NeurIPS 2019 — https://arxiv.org/abs/1907.03739
- Qi et al., *Volumetric and Multi-View CNNs*, ECCV 2016 — https://arxiv.org/abs/1604.03265
- Lang et al., *PointPillars*, CVPR 2019 — https://arxiv.org/abs/1812.05784
- MinkowskiEngine #158 (sparse slower than dense) — https://github.com/NVIDIA/MinkowskiEngine/issues/158

## Representations

| ID | Representation | Derived from mesh by | Industry use | Net |
|----|----------------|----------------------|--------------|-----|
| A  | Point cloud (N×3)  | uniform surface sampling (N=1024) | LiDAR / robotics perception | PointNet |
| B  | Dense voxel grid   | surface-voxelized to 32³ occupancy (robust to non-watertight meshes) | occupancy mapping / VoxNet | 3D-CNN (VoxNet) |
| C  | Sparse voxel (COO) | occupied cell centers only | autonomous driving (spconv) | PointNet on occupied centers* |
| D  | Multi-view 2D      | point-splat silhouette render, 8 views × 64² | multi-view CNNs (MVCNN) | 2D-CNN |

\* Production sparse voxel uses sparse 3D-conv (spconv), unavailable on MPS; we
read occupied cells as a point set through PointNet — keeping the representation
comparison fair while staying honest about the net.

**B vs C is the thesis in miniature:** identical information, different layout,
very different compute. It gives representation D a clear target (go
sparser/compact).

## Dataset

- **ModelNet40** (Princeton, `.off` meshes, 12,311 models / 40 classes).
- Develop + debug on a **10-class subset**; report final numbers on **all 40**.

## Controls (so the conclusion is believable)

- Same mesh source, same normalization, same train/test split across all reps.
- Same device (MPS), same seeds, same batch size, same epoch budget.
- Timing: warm-up passes excluded; measure median over many batches.
- One variable changed at a time: the representation (+ its native net).

## Metrics

- **Primary:** seconds/epoch (train), µs/sample (inference), peak memory.
- **Secondary (noted, not optimized):** top-1 accuracy.

## Layout

```
data/raw/           # ModelNet40.zip (gitignored)
data/ModelNet40/    # extracted meshes (gitignored)
src/convert.py      # mesh -> {pointcloud, voxel, sparse, new}
src/models/         # PointNet, VoxNet, SparseConvNet, new net
src/benchmark.py    # train + timed eval harness
scripts/prepare_data.py   # unzip + sanity check + pick 10-class dev subset
```

## Status

- [x] Dataset — ModelNet40 (Princeton mirror, 12,311 meshes / 40 classes)
- [x] Env — Python 3.13, torch 2.13, MPS
- [x] Rep A — point cloud → PointNet
- [x] Rep B — dense voxel → VoxNet
- [x] Rep C — sparse voxel → PointNet
- [x] Rep D — multi-view 2D → CNN (the "new" faster representation)
- [x] Timed comparison — dev (10-class) **and** full ModelNet40 (40-class)
- [x] Analysis — see [results/ANALYSIS.md](results/ANALYSIS.md)

**Headline (full ModelNet40, edge-class MPS hardware):** total training time
spans ~10× across representations — multi-view 2D ~2.7 min vs dense voxel
~25–29 min. Multi-view is fastest (2D-conv fits the GPU) despite having the most
parameters; dense voxel is slowest but most accurate. Dense reps self-throttle
under sustained load — a real edge-hardware cost. Parameter count does not
predict speed. See [results/ANALYSIS.md](results/ANALYSIS.md).
