# src/train.py  ─ единый скрипт обучения + история всех метрик
import argparse, os, csv, json, torch, torch.nn as nn, pandas as pd
import argparse, os, csv, torch, torch.nn as nn
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.optim import SGD
from tqdm import tqdm
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
from Data import cifar10, cifar100, otto
from Net.resnet import ResNet50Wrapper
from Net.mlp import OttoMLP
from Losses import FocalLoss, DualFocalLoss, BSCELossGra
from Metrics import (expected_calibration_error as ece,
                     adaptive_ece, classwise_ece, brier_score)
from Regularization.mixup import mixup_data, mixup_criterion
from src.utils import save_checkpoint, load_checkpoint
# ────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "cifar10":  cifar10.get_cifar10_loaders,
    "cifar100": cifar100.get_cifar100_loaders,
    "otto":     otto.get_otto_loaders,
}
LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal":         FocalLoss,
    "dual_focal":    DualFocalLoss,
    "bsce_gra":      BSCELossGra,
}
MODELS = {
    "resnet50": lambda c: ResNet50Wrapper(c),
    "mlp":      lambda c, in_f=93: OttoMLP(in_features=in_f, num_classes=c),
}

# ────────────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model",   required=True, choices=MODELS)
    p.add_argument("--loss",    default="cross_entropy", choices=LOSSES)
    p.add_argument("--epochs",  type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--early_stop", type=int, default=0, help="patience; 0=off")
    p.add_argument("--output", type=str, default="./model")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_freq", type=int, default=20, help="Сколько эпох между чекпойнтами")
    p.add_argument("--label_smoothing", type=float, default=0.0, help="α для label smoothing; 0 = off")
    p.add_argument("--augmix", action="store_true", help="Включить AugMix")

    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
def main():
    args = parse()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─ dataloaders
    if args.dataset == "otto":
        train_loader, val_loader, test_loader = DATASETS[args.dataset](batch_size=args.batch_size)
        feat_dim = 93
    else:
        train_loader, val_loader, test_loader = DATASETS[args.dataset](batch_size=args.batch_size)
        feat_dim = None

    # ─ model
    num_classes = {"cifar10":10,"cifar100":100,"otto":9}[args.dataset]
    if args.dataset == "otto":
        model = MODELS[args.model](num_classes, in_f=feat_dim).cuda()
    else:
        model = MODELS[args.model](num_classes).cuda()

    # ─ loss
    if args.label_smoothing>0 and args.loss=="cross_entropy":
        from Regularization.label_smoothing import CELossWithLabelSmoothing
        criterion = CELossWithLabelSmoothing(alpha=args.label_smoothing)
    else:
        criterion = LOSSES[args.loss]()

    # ─ optim
    optim = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim,[150,250], gamma=0.1)

    # ─ resume
    start_ep = 0
    if args.resume:
        model, optim, start_ep = load_checkpoint(args.resume, model, optim)

    # ─ history csv
    hist_path = out_dir/"history.csv"
    if not hist_path.exists():
        with hist_path.open("w",newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss",
                                    "val_acc","val_ece","val_adaece",
                                    "val_classece","val_brier"])

    best_val_loss, patience = 1e9, 0

    # ─────────────────────────── training loop ────────────────────────────
    for epoch in range(start_ep, args.epochs):
        model.train(); running=0; n=0
        pbar = tqdm(train_loader, desc=f"Ep{epoch}")
        for x,y in pbar:
            x,y = x.cuda(), y.cuda()
            if args.mixup>0:
                x,y_a,y_b,lam = mixup_data(x,y,alpha=args.mixup)
            optim.zero_grad()
            out = model(x)
            loss = mixup_criterion(criterion,out,y_a,y_b,lam) if args.mixup>0 else criterion(out,y)
            loss.backward(); optim.step()
            running+=loss.item(); n+=1
            pbar.set_postfix(loss=running/n)
        sched.step()
        train_loss = running/n

        # ─ validation
        model.eval(); corr=tot=0; logits_lst=[]; labels_lst=[]; vloss=0; vb=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.cuda(), y.cuda()
                logits = model(x)
                vloss += criterion(logits,y).item(); vb+=1
                corr  += (logits.argmax(1)==y).sum().item(); tot+=y.size(0)
                logits_lst.append(logits.cpu()); labels_lst.append(y.cpu())
        val_loss = vloss/vb
        logits_all = torch.cat(logits_lst); labels_all=torch.cat(labels_lst)
        val_acc  = corr/tot*100
        val_ECE    = ece(logits_all, labels_all)
        val_ada  = adaptive_ece(logits_all, labels_all)
        val_cls  = classwise_ece(logits_all, labels_all)
        val_b    = brier_score(logits_all, labels_all)

        # ─ write history
        with hist_path.open("a",newline="") as f:
            csv.writer(f).writerow([epoch,train_loss,val_loss,
                                    val_acc,val_ECE,val_ada.numpy(),val_cls.numpy(),val_b])

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc {val_acc:.2f} | val_ECE {val_ECE:.4f}")
        # ─ early stop
        if args.early_stop:
            if val_loss < best_val_loss-1e-6:
                best_val_loss, patience = val_loss, 0
                save_checkpoint(model, optim, epoch, out_dir, tag="best")
            else:
                patience+=1
                if patience>=args.early_stop:
                    print(f"Early-stop at {epoch}")
                    break
        
        if epoch % args.save_freq == 0:
            save_checkpoint(model, optim, epoch, out_dir, tag="ckpt")  # step ckpt

    save_checkpoint(model, optim, epoch, out_dir, tag="final")

if __name__ == "__main__":
    main()
