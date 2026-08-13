# Next Steps: 3D Representation Benchmark

This document tracks planned improvements to the 3d-repr-benchmark project. The blog post at `~/dev/personal-website` uses the current results; future work will strengthen the technical rigor and reproducibility.

## Current Status (August 13, 2026)

**Blog post live**: "Why FLOPs mislead on real 3D hardware" at `~/dev/personal-website/app/blog/posts.js`

**Experiment completed**: 4 representations (point cloud, dense voxel, sparse voxel, multi-view 2D) on ModelNet40, trained on Apple M3 Pro (MPS), 15 epochs, batch 32.

**Key finding**: FLOPs ordering (point/sparse 295M > voxel 170M > mview 96M) disagrees with wall-clock ordering (voxel 29.2 min slowest > point 13.4 > sparse 11.4 > mview 2.7 fastest). Dense voxel self-throttles due to thermal coupling.

**Known limitations**:
- Sparse voxel uses PointNet workaround (spconv not available on MPS)
- One dataset (ModelNet40 CAD objects)
- Point estimates only (no variance across seeds)
- No resolution sweep (cubic-cost claim is speculative)
- Thermal evidence indirect (inferred from per-epoch times, no temperature logs)

---

## Priority Improvements

### 1. Add variance estimates (HIGH PRIORITY)

Run 3 seeds per representation (12 runs total) and report mean ± std for:
- Total training time
- Throughput (samples/sec)
- Inference time
- Accuracy

**Why**: Turns "roughly tenfold spread" into a statistically defensible claim with confidence intervals.

**Action**:
```bash
# Update scripts/run_baseline.py to accept --seed argument
# Run all 12 combinations:
for rep in point voxel sparse mview; do
  for seed in 0 1 2; do
    python scripts/run_baseline.py --rep $rep --seed $seed
  done
done
# Aggregate results/results/seeds_summary.json with mean/std
```

**Expected output**: Table like:

| rep       | mean train time | std  | mean throughput | std  |
|-----------|-----------------|------|-----------------|------|
| mview     | 2.7 min         | 0.1  | 61.7 s/s        | 1.2  |
| sparse    | 11.4 min        | 0.3  | 14.4 s/s        | 0.4  |
| point     | 13.4 min        | 0.5  | 12.2 s/s        | 0.6  |
| voxel     | 29.2 min        | 2.1  | 5.6 s/s         | 0.4  |

---

### 2. Add one more dataset (HIGH PRIORITY)

Run the same 4 representations on a non-CAD dataset to show the finding generalises.

**Options**:
- **ShapeNet** (easiest): Same format as ModelNet40, diverse object categories
- **ScanNet** subset: Real RGB-D scans, not CAD
- **ModelNet10**: Smaller version of ModelNet40 as a quick sanity check

**Why**: If the ranking holds on messy real data, the claim is robust. If it flips, that's interesting too.

**Action**:
```bash
# Download ShapeNet (or use existing if available)
# Update src/data.py to load new dataset
# Re-run all 4 reps on new dataset
# Compare ordering with ModelNet40
```

**Expected output**: A sentence in the blog: "The same ordering appears on ShapeNet, suggesting the FLOPs-vs-wall-clock mismatch is not CAD-specific."

---

### 3. Explain FLOPs methodology (ZERO EFFORT, HIGH IMPACT)

Add one sentence to the blog explaining the FLOPs counting convention.

**Why**: Different tools (thop, ptflops, fvcore) report different numbers due to conventions. Be explicit about which one you used.

**Action**: Add to the blog post's "FLOPs gets the ranking backwards" section:

> "FLOPs were counted using PyTorch's built-in `torch.utils.flop_counter.FlopCounterMode`, which reports one multiply-accumulate as one FLOP — the standard convention in the 3D literature. The count is device-independent; it reflects the network architecture and input shape, not the hardware."

---

### 4. Run resolution sweep (MEDIUM PRIORITY)

Run dense voxel at multiple resolutions (16³, 32³, 64³) to show the cubic-cost curve directly.

**Why**: The blog claims "dense is cubic" but doesn't prove it. This sweep makes it concrete.

**Action**:
```bash
# Update scripts/run_baseline.py to accept --res argument
for res in 16 32 64; do
  python scripts/run_baseline.py --rep voxel --res $res
done
# Plot wall-clock vs resolution on log-log axes
```

**Expected output**: A figure showing wall-clock scaling as O(res³). If super-cubic due to throttling, that's even more interesting.

---

### 5. Add preprocessing cost (LOW EFFORT)

Mention preprocessing time in the blog text or add it to the results table.

**Current preprocessing times** (from `preprocess_train_sec` in JSON):
- Point cloud: ~254s (4.2 min)
- Multi-view: ~83s (1.4 min)
- Dense voxel: similar to point
- Sparse voxel: similar to point

**Why**: Honesty about overhead makes the finding more credible. Preprocessing is real cost.

**Action**: Add one sentence to the blog:

> "Preprocessing took 4.2 minutes for point cloud and 1.4 minutes for multi-view — included in totals but dwarfed by training."

---

## Hardware Transition Notes (CURRENT → FUTURE)

### Current setup (Mac M3 Pro, MPS)
- Device: `mps` (Metal Performance Shaders)
- CPU: Apple Silicon
- GPU: Integrated GPU (no fan, passive cooling)
- Thermal: Severe throttling on dense workloads
- Limitations: `spconv` (sparse 3D convolution) not available on MPS

### Future setup (Ubuntu, RTX 3060, CUDA)
- Device: `cuda`
- GPU: NVIDIA RTX 3060 (active cooling)
- OS: Ubuntu Linux
- Advantages: `spconv` available, thermal logging via `nvidia-smi`

### Changes required for CUDA

1. **Sparse voxel with real spconv**
   - Install: `pip install spconv-cu117` (or appropriate CUDA version)
   - Update `src/models.py` to import `spconv`
   - Replace sparse PointNet workaround with SparseConvNet
   - Expected: Faster than current sparse baseline, better accuracy

2. **Thermal logging**
   - Monitor GPU temperature during training:
     ```python
     import subprocess
     temp = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=,noheader'])
     ```
   - Log temperature per epoch to JSON
   - Plot temperature vs epoch alongside seconds-per-epoch

3. **Device compatibility**
   - Update `scripts/run_baseline.py` to detect device:
     ```python
     device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
     ```
   - Update model `.to(device)` calls
   - Update FlopCounterMode to run on same device (or keep on CPU, device-agnostic)

4. **FLOPs re-count**
   - FLOPs are device-independent, but verify on CUDA setup
   - Re-run `scripts/run_flops.py` to confirm counts match MPS version

5. **Path updates**
   - Data paths may differ on Linux; update `src/data.py` if needed
   - `results/` paths should stay the same

### Expected differences on RTX 3060

- **Absolute times**: Much faster overall (RTX 3060 > M3 Pro for dense 3D)
- **Ranking**: Should preserve the relative ordering (mview fastest, voxel slowest)
- **Throttling**: Minimal (active cooling)
- **Sparse voxel**: Real spconv will be faster than PointNet workaround
- **Thermal coupling**: Will be greatly reduced; may not observe climb in per-epoch times

---

## Lower-Priority Improvements

### 6. Add convergence metrics

Current: 15 epochs fixed budget. Some networks still climbing in accuracy at epoch 15.

**Action**:
- Run until convergence (train_acc plateaus)
- Report "epochs to 80% of final accuracy"
- Or explicitly state "training stopped at 15 epochs for comparison; full convergence requires more"

---

### 7. Tighten README for reproducibility

**Ensure repo has**:
- `README.md` with setup:
  - Python version (3.10+)
  - `pip install -r requirements.txt`
  - How to download ModelNet40 (link + md5)
- `scripts/run_all.py`: One command to run all 4 reps
- `results/` populated with all JSONs (or script to regenerate)
- `.gitignore` excluding data but keeping code

**Goal**: Stranger can `git clone && pip install && python run_all.py` and regenerate numbers.

---

### 8. Engage with literature

**Current**: Further reading cites 4 papers, but doesn't engage with their findings.

**Action**: Add one sentence:

> "The closest precedent is Qi et al. (ECCV 2016), which compared volumetric and multi-view CNNs on ModelNet40. They report a similar accuracy edge for multi-view but do not compare FLOPs or wall-clock. This work extends that comparison to include thermal effects on constrained hardware."

---

### 9. Add ModelNet10 as sanity check

Quick validation run on smaller dataset (4,899 shapes, 10 classes).

**Why**: Faster iteration, sanity check that ordering holds.

---

### 10. Segment/detection task

Current: Classification only. Extending to segmentation (ModelNet40 has meshes) or detection would strengthen generality.

**Why**: Classification is simple; real 3D tasks often require segmentation.

---

## Implementation Order

Do these in order:

1. **FLOPs methodology sentence** (5 min, blog edit)
2. **Variance estimates** (3 seeds × 4 reps, ~6-8 hours compute, high impact)
3. **One more dataset** (ShapeNet, ~4-6 hours, high impact)
4. **Resolution sweep** (3 dense voxel runs, ~2-3 hours, concrete proof of cubic cost)
5. **Hardware transition** (switch to RTX 3060, enable real spconv)
6. **Thermal logging** (add temperature logging, one run to validate)
7. **Preprocessing mention** (blog edit, 5 min)
8. **Convergence metrics** (extend runs, optional)
9. **README tighten** (documentation polish)
10. **Literature engagement** (blog edit, 5 min)

---

## Files to Update

### Blog post
- `~/dev/personal-website/app/blog/posts.js`: Add FLOPs methodology, preprocessing mention, literature engagement, eventual update with variance/dataset numbers

### Repo (current: MPS)
- `scripts/run_baseline.py`: Add `--seed` argument, device detection
- `scripts/run_flops.py`: Keep device-agnostic (FLOPs are device-independent)
- `src/models.py`: Add sparse spconv model (when on CUDA)
- `src/data.py`: Add ShapeNet/ScanNet loaders
- `README.md`: Setup instructions, reproducibility notes
- `requirements.txt`: Add `spconv-cu117` (when on CUDA)

### Repo (future: CUDA)
- `scripts/run_thermal.py`: New script for temperature logging
- `results/thermal_*.json`: Store temperature per epoch

---

## Notes for Future Self

- When switching to RTX 3060, re-run FLOPs counting to verify (should match)
- Thermal logging on CUDA: use `nvidia-smi dmon -s p -c 1` or `pynvml` library
- spconv install: check CUDA version first (`nvidia-smi`), then `pip install spconv-cuXX`
- ShapeNet download: requires request access at https://shapenet.org/
- ScanNet download: requires request access at http://www.scan-net.org/

---

## Contact / Context

- Original run: August 12-13, 2026
- Hardware: Mac M3 Pro (MPS), passive cooling
- Blog: ~/dev/personal-website/app/blog/posts.js (slug: 3d-representation-training-speed)
- Repo: ~/dev/3d-repr-benchmark
- GitHub: https://github.com/monsieurkd/3d-repr-benchmark

Last updated: August 13, 2026
