"""Run one representation through the full pipeline and record timings.

Flow (methodology):
  1. BUILD cache for train + test  -> this is PREPROCESSING (mesh -> rep).
     Timed and reported separately, because the project focus is data
     representation & processing, not just model speed.
  2. TRAIN on cached arrays         -> pure model training time.
  3. EVALUATE on cached arrays       -> pure inference time (warmup-excluded).

Usage:
  python scripts/run_baseline.py --rep point --first-n 3 --epochs 2   # quick smoke
  python scripts/run_baseline.py --rep point --dev --epochs 20        # 10-class dev
  python scripts/run_baseline.py --rep voxel --dev --epochs 20
"""
import argparse, json, os, sys, time, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader

from src import config as C
from src.dataset import ModelNetDataset
from src.models import PointNet, VoxNet, MultiViewCNN
from src.benchmark import train_model, evaluate, load_to_device


def select_classes(args):
    avail = sorted(d for d in os.listdir(C.MODELNET_DIR)
                   if os.path.isdir(C.MODELNET_DIR / d))
    if args.first_n:
        return avail[:args.first_n]
    return C.DEV_CLASSES if args.dev else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["point", "voxel", "sparse", "mview"], default="point")
    ap.add_argument("--dev", action="store_true", help="10-class dev subset")
    ap.add_argument("--first-n", type=int, default=0, help="first N classes (quick test)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default=C.DEVICE, choices=["cpu", "mps", "cuda"],
                    help="benchmark device (CPU gives the cleanest cross-rep timing)")
    args = ap.parse_args()
    C.DEVICE = args.device  # override for this run; flows into model + harness calls

    torch.manual_seed(C.SEED)
    classes = select_classes(args)
    cache = None if args.no_cache else C.PROCESSED

    ds_tr = ModelNetDataset(C.MODELNET_DIR, "train", classes=classes, rep=args.rep,
                            cache_dir=cache, num_points=C.NUM_POINTS, vox_res=C.VOX_RES,
                            mv_views=C.MV_VIEWS, mv_img_size=C.MV_IMG_SIZE)
    ds_te = ModelNetDataset(C.MODELNET_DIR, "test", classes=classes, rep=args.rep,
                            cache_dir=cache, num_points=C.NUM_POINTS, vox_res=C.VOX_RES,
                            mv_views=C.MV_VIEWS, mv_img_size=C.MV_IMG_SIZE)
    ncls = len(ds_tr.classes)

    if args.rep == "mview":
        model = MultiViewCNN(ncls, in_channels=C.MV_VIEWS, img_size=C.MV_IMG_SIZE).to(C.DEVICE)
    elif args.rep == "voxel":
        model = VoxNet(ncls, res=C.VOX_RES).to(C.DEVICE)
    else:  # point or sparse -> PointNet
        model = PointNet(ncls).to(C.DEVICE)
    nparams = sum(p.numel() for p in model.parameters())

    print(f"[{args.rep}] classes={ncls} train={len(ds_tr)} test={len(ds_te)} "
          f"params={nparams:,} device={C.DEVICE} cache={cache is not None}")

    # 1) preprocessing / cache build (cold-conversion time, remembered via marker)
    pre_tr, cold_tr = ds_tr.build_cache(args.workers, args.batch)
    pre_te, cold_te = ds_te.build_cache(args.workers, args.batch)
    print(f"  preprocess: train {pre_tr:.1f}s (cold={cold_tr}), "
          f"test {pre_te:.1f}s (cold={cold_te})")

    # 1b) load to device tensors (amortized setup, timed separately)
    t0 = time.perf_counter()
    Xtr, Ytr = load_to_device(ds_tr, C.DEVICE)
    Xte, Yte = load_to_device(ds_te, C.DEVICE)
    load_sec = time.perf_counter() - t0
    print(f"  load-to-device: {C.DEVICE} in {load_sec:.1f}s "
          f"(train {tuple(Xtr.shape)}, test {tuple(Xte.shape)})")

    # 2) train (timed, loader-free -> pure model-on-representation cost)
    per_t, per_acc, total = train_model(model, Xtr, Ytr, args.batch, args.epochs, C.LR, C.DEVICE)
    steady = statistics.median(per_t[1:]) if len(per_t) > 1 else per_t[0]

    # 3) evaluate (timed, loader-free)
    acc, infer_sec, n = evaluate(model, Xte, Yte, args.batch, C.DEVICE)

    result = {
        "rep": args.rep, "classes": ncls, "train_n": len(ds_tr), "test_n": len(ds_te),
        "params": nparams, "epochs": args.epochs, "device": C.DEVICE,
        "preprocess_train_sec": pre_tr, "preprocess_test_sec": pre_te,
        "load_to_device_sec": load_sec,
        "steady_epoch_sec": steady, "total_train_sec": total,
        "train_acc_final": per_acc[-1], "test_acc": acc,
        "infer_sec": infer_sec, "infer_us_per_sample": infer_sec / max(n, 1) * 1e6,
        "per_epoch_sec": [round(x, 3) for x in per_t],
        "per_epoch_acc": [round(x, 4) for x in per_acc],
    }
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_epoch_sec", "per_epoch_acc")}, indent=2))

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    scope = "dev" if args.dev else ("all" if not args.first_n else f"n{args.first_n}")
    out = C.RESULTS / f"{args.rep}_{scope}_c{ncls}_e{args.epochs}.json"
    out.write_text(json.dumps(result, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
