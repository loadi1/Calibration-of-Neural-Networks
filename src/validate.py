#!/usr/bin/env python3
"""
validate.py  – оценка СЫРЫХ (до пост-калибровки) выходов модели
на тестовом наборе. Записывает accuracy, ECE, AdaECE, Classwise-ECE
и Brier Score в отдельный CSV, чтобы можно было сравнивать модели
между собой без влияния калибратора.

Пример:
python src/validate.py \
  --ckpt model/cifar10_resnet50_ce_ls0.0_mx0.0_noaug/best.pth \
  --dataset cifar10 --model resnet50 \
  --loss cross_entropy --ls 0.0 --mixup 0.0
"""
import argparse, csv, pathlib, torch
from pathlib import Path
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Data import cifar10, cifar100, otto
from Net.resnet import ResNet50Wrapper
from Net.mlp     import OttoMLP
from Metrics import (expected_calibration_error as ece,
                     adaptive_ece, classwise_ece, brier_score)

DATASETS = {
    "cifar10":  cifar10.get_cifar10_loaders,
    "cifar100": cifar100.get_cifar100_loaders,
    "otto":     otto.get_otto_loaders,
}
MODELS = {
    "resnet50": lambda c: ResNet50Wrapper(c),
    "mlp":      lambda c, in_f=93: OttoMLP(in_features=in_f, num_classes=c),
}

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model",   required=True, choices=MODELS)
    p.add_argument("--loss",    required=True)
    p.add_argument("--ls",      type=float, default=0.0)
    p.add_argument("--mixup",   type=float, default=0.0)
    p.add_argument("--augmix",  action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--output", default="test_master.csv")
    return p.parse_args()

@torch.inference_mode()
def collect_logits(model, loader):
    logits, labels = [], []
    for x,y in loader:
        x = x.cuda(); logits.append(model(x).cpu()); labels.append(y)
    return torch.cat(logits), torch.cat(labels)

def main():
    args = parse()
    out_csv = Path(args.output)

    # loaders ─ только test
    if args.dataset=="otto":
        _, _, test_loader= DATASETS[args.dataset](batch_size=args.batch_size)
        feat_dim =93
    else:
        _, _, test_loader = DATASETS[args.dataset](batch_size=args.batch_size)
        feat_dim = None

    num_cls = dict(cifar10=10,cifar100=100,otto=9)[args.dataset]
    model = ( MODELS[args.model](num_cls, in_f=feat_dim)
              if args.dataset=="otto" else MODELS[args.model](num_cls) ).cuda().eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    logits, labels = collect_logits(model, test_loader)
    probs  = torch.softmax(logits,1)
    pred   = probs.argmax(1)
    acc    = (pred==labels).float().mean().item()*100

    ece_v   = ece(logits, labels)
    ada_v   = adaptive_ece(logits, labels).numpy()
    cls_v   = classwise_ece(logits, labels).numpy()
    brier_v = brier_score(logits, labels)

    header = ["dataset","model","loss","ls","mixup","augmix",
              "accuracy","ece","adaece","classece","brier"]
    row    = [args.dataset, args.model, args.loss, args.ls, args.mixup,
              args.augmix,
              acc,ece_v, ada_v, cls_v, brier_v]

    if not out_csv.exists():
        with out_csv.open("w",newline="") as f: csv.writer(f).writerow(header)
    with out_csv.open("a",newline="") as f: csv.writer(f).writerow(row)
    print(dict(zip(header,row)))

if __name__ == "__main__":
    main()
