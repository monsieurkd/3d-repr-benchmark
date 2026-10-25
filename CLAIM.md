# The claim, and exactly how far it is backed

This file exists so the limits live next to the numbers. Read it before quoting
any result from this repo.

## The claim

**Representation choice is a data-ingestion decision, not a model decision.**

Every 3D representation is a choice about *who pays the encoding cost*:

| Representation | Who pays | What the payment is |
|---|---|---|
| Point cloud | **the model** | an irregular set to interpret, every forward pass |
| Dense / sparse voxel | **ingestion** | quantisation into a grid; cost grows cubically |
| Multi-view 2D | **nobody, here** | the object is a mesh, so rendering is trivially cheap |

Two consequences follow, and only one of them is currently proven:

1. **FLOPs do not tell you which side of that line you are on.** ✔ backed
2. **Where the line sits is hardware-dependent.** ✘ hypothesis only

## What is backed (Apple M3 Pro, MPS, ModelNet40, 15 epochs, 40 classes)

Wall-clock, identical information across all four conditions:

```
mview   2.7 min   <- cheapest
sparse 11.4 min
point  13.4 min
voxel  29.2 min   <- ~10x mview
```

FLOPs ordering is **point/sparse 295M > voxel 170M > mview 96M** — which
contradicts the wall-clock ordering. Dense voxel does not have the most FLOPs
and is still by far the slowest. That divergence is the result: **the bottleneck
is the encoding, not the model.**

`scripts/run_flops.py` computes MACs rather than asserting them, so this
comparison is against measured counts, not a paper's table.

## What is NOT backed

- **"Hardware-dependent."** One accelerator. No CUDA column yet. Until
  `results/cuda_vs_mps.md` exists, this is a hypothesis and must be stated as one.
- **A swap map / threshold / optimum.** This experiment supports a *ranking and a
  magnitude*, not a curve. Locating a crossover needs a resolution sweep, which
  is not in this repo.
- **"Cloud GPU."** The CUDA run is on a local RTX 3060. Write **GPU (CUDA)**.
  Nothing in this repo runs in a cloud environment.
- **Generality beyond ModelNet40.** Synthetic CAD, single task. Real sensor
  streams are noisy and incomplete in ways this data is not.

## The honest caveat on `mview`

`mview` wins here partly because **the input is a mesh**, so rendering to 2D is
nearly free. In deployment, RGB is usually already being captured — so the
transferable form of the claim is:

> *Given that the 3D data is already arriving (depth, LiDAR, point cloud), what
> does it cost to use it directly instead of flattening it?*

Not "3D is slower than 2D". That framing would be a different, weaker, and
partly false claim.

## Why the guards are the point

A benchmark that names its own limits is more useful than one that overstates,
and cheaper to defend in an interview. The ranking above can be defended today.
The hardware clause becomes defensible the moment the CUDA column is committed —
and not one minute earlier.
