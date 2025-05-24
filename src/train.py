import argparse, os, csv, torch, torch.nn as nn
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.optim import SGD
from tqdm import tqdm

from Data import cifar10, cifar100, otto
from Net.resnet import ResNet50Wrapper
from Net.mlp import OttoMLP
from Losses import FocalLoss, DualFocalLoss, BSCELossGra
from Metrics.ece import *
from Metrics.brier import *
from src.utils import save_checkpoint, load_checkpoint

DATASETS = {"cifar10": cifar10.get_cifar10_loaders,
            "cifar100": cifar100.get_cifar100_loaders,
            "otto": otto.get_otto_loaders}

MODELS = {"resnet50": lambda n: ResNet50Wrapper(n),
          "mlp": lambda n: OttoMLP(in_features=93, num_classes=n)}

LOSSES = {"cross_entropy": nn.CrossEntropyLoss,
          "focal": FocalLoss,
          "dual_focal": DualFocalLoss,
          "bsce_gra": BSCELossGra}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--loss", default="cross_entropy", choices=LOSSES)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output", type=str, default="./model")
    p.add_argument("--save_freq", type=int, default=50, help="Сколько эпох между чекпойнтами")
    p.add_argument("--mixup", type=float, default=0.0, help="alpha для MixUp; 0 = off")
    p.add_argument("--label_smoothing", type=float, default=0.0, help="α для label smoothing; 0 = off")
    p.add_argument("--augmix", action="store_true", help="Включить AugMix")
    return p.parse_args()

def log_writer(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f_exists = os.path.isfile(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if not f_exists:
        writer.writerow(header)
    return f, writer


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # loaders
    if args.augmix and args.dataset.startswith("cifar"):
        train_loader, val_loader, test_loader = DATASETS[args.dataset](batch_size=args.batch_size, use_augmix=True)
    else:
        train_loader, val_loader, test_loader = DATASETS[args.dataset](batch_size=args.batch_size,)
    
    num_classes = 100 if args.dataset == "cifar100" else 10 if args.dataset == "cifar10" else 9

    # model & optimisation
    model = MODELS[args.model](num_classes).cuda()
    
    if args.label_smoothing > 0 and args.loss == "cross_entropy":
        from Regularization.label_smoothing import CELossWithLabelSmoothing
        criterion = CELossWithLabelSmoothing(alpha=args.label_smoothing)
    else:
        criterion = LOSSES[args.loss]()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 250], gamma=0.1)

    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(args.resume, model, optimizer)

    # файл‑лог для динамики метрик
    log_path = os.path.join(args.output, "history.csv")
    header = ["epoch", "train_loss", "val_loss", "val_acc", "val_ece"]
    log_file, logger = log_writer(log_path, header)

    for epoch in range(start_epoch, args.epochs):
        model.train(); running_loss = 0.0; n_batch = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad(); 
            
            if args.mixup > 0:
                from Regularization.mixup import mixup_data, mixup_criterion
                x, y_a, y_b, lam = mixup_data(x, y, alpha=args.mixup)
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            
            
            loss.backward(); optimizer.step()
            running_loss += loss.item(); n_batch += 1
            pbar.set_postfix({"loss": running_loss / n_batch})
        scheduler.step()
        train_loss_epoch = running_loss / n_batch

        # validation
        model.eval(); corr = tot = 0; logits_all = []; labels_all = []; val_loss = 0.0; v_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda(); logits = model(x)
                loss_val = criterion(logits, y).item()
                val_loss += loss_val; v_batches += 1
                preds = logits.argmax(1); corr += (preds == y).sum().item(); tot += y.size(0)
                logits_all.append(logits.cpu()); labels_all.append(y.cpu())
        val_loss /= v_batches
        val_acc = corr / tot * 100
        val_ece = expected_calibration_error(torch.cat(logits_all), torch.cat(labels_all))

        # вывод и логирование
        print(f"Epoch {epoch}: train_loss={train_loss_epoch:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.2f}% | val_ECE={val_ece:.4f}")
        logger.writerow([epoch, f"{train_loss_epoch:.6f}", f"{val_loss:.6f}", f"{val_acc:.4f}", f"{val_ece:.6f}"])
        log_file.flush()

        # чекпойнт каждые 50 эпох
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, args.output)

    log_file.close()

    model.eval(); corr = tot = 0; logits_all = []; labels_all = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            preds = logits.argmax(1)
            corr += (preds == y).sum().item(); tot += y.size(0)
            logits_all.append(logits.cpu()); labels_all.append(y.cpu())
    acc = corr / tot * 100
    logits_all = torch.cat(logits_all); labels_all = torch.cat(labels_all)
    ece_test = expected_calibration_error(logits_all, labels_all)
    ada_test = adaptive_ece(logits_all, labels_all)
    cls_test = classwise_ece(logits_all, labels_all)
    brier = brier_score(logits_all, labels_all)

    # запись
    import pandas as pd
    res_path = os.path.join(args.output, "results.csv")
    row = pd.DataFrame([{"dataset": args.dataset, "model": args.model, "loss": args.loss,
                         "accuracy": acc, "ece": ece_test, "adaece": ada_test,
                         "classece": cls_test, "brier": brier}])
    if os.path.exists(res_path):
        row.to_csv(res_path, mode="a", header=False, index=False)
    else:
        row.to_csv(res_path, index=False)

if __name__ == "__main__":
    main()