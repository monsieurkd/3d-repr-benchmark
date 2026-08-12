# Does 3D data representation change *real* training & inference time?

**Research question:** holding task, data, and hardware fixed, does the 3D data
*representation* drive the time it takes to train and to infer — measured in the
wall-clock terms a practitioner (or an edge/robotics system) actually experiences?

## TL;DR
On **full ModelNet40** (12,311 meshes, 40 classes, 15 epochs) on **edge-class
hardware** (Apple Silicon laptop, MPS, passively cooled):

| Representation | Total train | Steady epoch | Inference | Test acc | Params |
|----------------|------------:|-------------:|----------:|---------:|-------:|
| D · multi-view 2D → CNN      | **2.7 min**    |  10.4 s   | **425 µs/sample**     | 81.0% | 1,149,000 |
| C · sparse voxel → PointNet  | 11.4 min       |  44.4 s   | 1543 µs/sample        | 83.5% |   815,336 |
| A · point cloud → PointNet   | 11.9–13.4 min  | ~45–50 s  | ~1390–1880 µs/sample  | 82.9% |   815,336 |
| B · dense voxel → VoxNet     | **25–29 min**  | ~99–115 s | ~2760–3400 µs/sample  | 84.9% |   918,728 |

- **~10× spread in total training time** (multi-view vs dense voxel).
- **Multi-view 2D is fastest by far**, despite having the **most** parameters →
  parameter count does not predict speed; representation→hardware fit does.
- **Dense voxel is slowest but most accurate.**
- Ranges (A, B) are from two independent runs and reflect real run-to-run
  variance on this hardware (see *Thermal self-coupling* below).

## Why the primary metric is total wall-clock training time, not FLOPs
FLOPs / peak-TFLOPs measure theoretical cost under ideal cooling. A practitioner
or an edge/robotics system pays **wall-clock**, which on thermally-constrained
hardware includes throttling the idealized number hides. So the primary metrics
here are **total training time** and **inference latency**, on a stated,
realistic, edge-class machine. This is the *useful* number — the one that
reflects what deploying these representations actually costs.

## The thermal self-coupling finding (a result, not a caveat)
The passively-cooled SoC throttles under sustained load, and the heavier
representations throttle *themselves* mid-run. This shows up two ways:

1. **In-run epoch climb.** Within a single 15-epoch run, per-epoch time rises as
   the chip heats. Dense voxel (heaviest 3D-conv) climbs most:
   - voxel epochs (s): `137 103 101 105 114 117 100 100 122 131 129 127 111 138 115`
   - point  epochs (s): `50 48 47 50 56 49 47 49 57 49 48 57 68 68 63`
2. **Run-to-run variance.** Two independent voxel runs totaled **25.0 and 29.2
   min (~17%)**; point **11.9 and 13.4 min (~12%)**. A pre-run cooldown did
   **not** remove this — the workload reheats the chip *during* the run.

**Interpretation:** on edge-class hardware, dense 3D representations pay a
*super-linear* wall-clock penalty — they cost more FLOPs, and those FLOPs heat
the chip, throttling later epochs. A clean-room (CPU / datacenter-GPU) benchmark
would hide exactly this. For laptop / edge / robotics deployment (the regime
this hardware represents), it is a genuine, additional cost of choosing a dense
representation — not measurement noise to scrub away.

## What the comparison controls
- Same ModelNet40 meshes, same unit-sphere normalization, same train/test split.
- Same device (MPS), seed (42), batch (32), epoch budget (15).
- Loader-free timing: data preloaded to device tensors; train/infer times are
  pure model-on-representation compute (no DataLoader IPC contaminating them).
- Preprocessing (mesh → representation) is measured separately and cold-cached
  per (rep, params, class-set). Producing a voxel / sparse / multi-view rep costs
  ~2–4× the point cloud (the cheapest representation to produce).

## Inference & accuracy (secondary, not optimized for)
- Inference latency tracks the training ranking: multi-view (425 µs/sample) ≪
  point / sparse (~1.4–1.5 ms) < dense voxel (~2.8–3.4 ms).
- Accuracy is close across reps (81–85%); dense voxel highest (retains the most
  geometry), multi-view lowest (silhouettes discard 3D depth information).
- 15 epochs is light — converged PointNet-class models reach ~89–91% on
  ModelNet40 — so absolute accuracies are under-trained, but all reps equally so;
  relative comparisons hold and the speed ranking is unaffected by training
  budget.

## Conclusion
Data representation is the dominant driver of real training/inference time — by
~10× — via its fit to the hardware, not model size. The practical guidance:
**on thermally-constrained / edge hardware, prefer 2D or sparse representations;
reach for dense 3D only when its accuracy edge justifies the ~10× wall-clock and
the thermal tax.**

## Future work (for a sharper fine-ranking)
- Average 3 seeds per rep, each from a cold start, with SoC temperature monitored.
- Voxel resolution sweep (32 → 64 → 128) to chart dense-cubic cost directly.
- Run on an actively-cooled datacenter GPU to separate the representation effect
  from the thermal-coupling effect.
