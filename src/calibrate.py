#!/usr/bin/env python3
"""
calibrate.py  –  применяет пост-калибровку (Temperature / Platt / Isotonic / BBQ)
к уже обученному чекпойнту и пишет метрики в общий CSV.

Пример:
python src/calibrate.py \
  --ckpt model/cifar10_resnet50_ce_ls0.0_mx0.0_noaug/best.pth \
  --method temperature \
  --dataset cifar10 --model resnet50 \
  --loss cross_entropy --ls 0.0 --mixup 0.0 \
  --output calib_master.csv
"""
import argparse, pathlib, csv, torch, numpy as np, pandas as pd
from pathlib import Path
import argparse, os, csv, torch, torch.nn as nn
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Calibration.temperature import TemperatureScaler
from Calibration.platt        import platt_scale
from Calibration.isotonic     import isotonic_calibrate, apply_isotonic
from Calibration.bbq          import bbq_calibrate
from Metrics import (expected_calibration_error as ece,
                     adaptive_ece, classwise_ece, brier_score)

# ────────────────────────────────────────────────────────────────────
from Data import cifar10, cifar100, otto
from Net.resnet import ResNet50Wrapper
from Net.mlp     import OttoMLP
# ────────────────────────────────────────────────────────────────────

DATASETS = {
    "cifar10":  cifar10.get_cifar10_loaders,
    "cifar100": cifar100.get_cifar100_loaders,
    "otto":     otto.get_otto_loaders,
}
MODELS = {
    "resnet50": lambda c: ResNet50Wrapper(c),
    "mlp":      lambda c, in_f=93: OttoMLP(in_features=in_f, num_classes=c),
}
CAL_METHODS = ["temperature","platt","isotonic","bbq"]

# ────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="путь к .pth чекпойнту")
    p.add_argument("--method", required=True, choices=CAL_METHODS)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model",   required=True, choices=MODELS)
    p.add_argument("--loss",    required=True)          # чтобы писать в CSV
    p.add_argument("--ls",      type=float, default=0.0)
    p.add_argument("--mixup",   type=float, default=0.0)
    p.add_argument("--augmix",  action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--output", default="calib_master.csv")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def collect_logits(model, loader):
    logits, labels = [], []
    for x,y in loader:
        x = x.cuda(); out = model(x).cpu()
        logits.append(out); labels.append(y)
    return torch.cat(logits), torch.cat(labels)

def main():
    args = parse()
    out_csv = Path(args.output)

    # loaders
    if args.dataset == "otto":
        tr, val, test = DATASETS[args.dataset](batch_size=args.batch_size)
        feat_dim = 93
    else:
        tr, val, test = DATASETS[args.dataset](batch_size=args.batch_size)
        feat_dim = None

    num_cls = {"cifar10":10,"cifar100":100,"otto":9}[args.dataset]
    model = ( MODELS[args.model](num_cls, in_f=feat_dim)
              if args.dataset=="otto" else MODELS[args.model](num_cls) ).cuda().eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # собираем logits
    val_logits,  val_labels  = collect_logits(model, val)
    test_logits, test_labels = collect_logits(model, test)

    # numpy версии
    val_np  = val_logits.detach().cpu().numpy()
    test_np = test_logits.detach().cpu().numpy()
    y_val   = val_labels.detach().cpu().numpy()
    y_test  = test_labels.detach().cpu().numpy()

    # ─── выбираем метод калибровки ────────────────────────────────
    if args.method == "temperature":
        scaler = TemperatureScaler().cuda()
        scaler.fit(val_logits, val_labels)
        test_logits_cal = scaler(test_logits.cuda()).detach().cpu()
        probs_post = torch.softmax(test_logits_cal,1).numpy()
    elif args.method == "platt":
        pipe = platt_scale(val_np, y_val)
        probs_post = pipe.predict_proba(test_np)
    elif args.method == "isotonic":
        calibs = isotonic_calibrate(torch.softmax(val_logits,1).numpy(), y_val)
        probs_post = apply_isotonic(calibs, torch.softmax(test_logits,1).numpy())
    else:  # BBQ
        bbq = bbq_calibrate(val_np, y_val)
        probs_post = bbq.predict_proba(test_np)

    # back to torch logits to reuse metric fns
    logits_post = torch.from_numpy(np.log(probs_post + 1e-12))

    # ─── метрики ──────────────────────────────────────────────────
    pred_post = probs_post.argmax(1)
    acc_post  = (pred_post == y_test).mean()
    ece_v   = ece(logits_post, test_labels)
    ada_v   = adaptive_ece(logits_post, test_labels).numpy()
    cls_v   = classwise_ece(logits_post, test_labels)
    brier_v = brier_score(logits_post, test_labels)

    # ─── пишем строку ─────────────────────────────────────────────
    header = ["dataset","model","method","loss","ls","mixup","augmix",
            "accuracy","ece","adaece","classece","brier"]
    row = [args.dataset, args.model, args.method,
           args.loss, args.ls, args.mixup, args.augmix,
           acc_post,ece_v, ada_v, cls_v, brier_v]

    if not out_csv.exists():
        with out_csv.open("w",newline="") as f: csv.writer(f).writerow(header)
    with out_csv.open("a",newline="") as f: csv.writer(f).writerow(row)

    print(pd.Series(row, index=header))

if __name__ == "__main__":
    main()
