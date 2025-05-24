import argparse, os, torch, numpy as np
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data import cifar10, cifar100, otto
from Net.resnet import ResNet50Wrapper
from Net.mlp import OttoMLP
from Metrics import expected_calibration_error as ece, adaptive_ece, classwise_ece, brier_score
from Calibration.temperature import TemperatureScaler
from Calibration.platt import platt_scale
from Calibration.isotonic import isotonic_calibrate, apply_isotonic
from Calibration.bbq import bbq_calibrate

DATASETS = {"cifar10": cifar10.get_cifar10_loaders,
            "cifar100": cifar100.get_cifar100_loaders,
            "otto": otto.get_otto_loaders}

MODELS = {"resnet50": lambda n: ResNet50Wrapper(n),
          "mlp": lambda n: OttoMLP(in_features=93, num_classes=n)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Путь к модели .pth")
    p.add_argument("--method", required=True, choices=["temperature", "platt", "isotonic", "bbq"])
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--output", type=str, default="./calib_results.csv")
    return p.parse_args()


def collect_logits(model, loader):
    model.eval(); logits_list = []; labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(); logits = model(x).cpu()
            logits_list.append(logits); labels_list.append(y)
    return torch.cat(logits_list), torch.cat(labels_list)


def main():
    args = parse_args()
    # loaders
    train_loader, val_loader, test_loader = DATASETS[args.dataset](batch_size=args.batch_size)
    num_classes = 100 if args.dataset == "cifar100" else 10 if args.dataset == "cifar10" else 9
    model = MODELS[args.model](num_classes).cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])

    # собираем логиты
    val_logits, val_labels = collect_logits(model, val_loader)
    test_logits, test_labels = collect_logits(model, test_loader)

    if args.method == "temperature":
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)
        test_logits_cal = scaler(test_logits.cuda()).detach().cpu()
        test_probs = torch.softmax(test_logits_cal, dim=1).numpy()
    elif args.method == "platt":
        pipe = platt_scale(val_logits.numpy(), val_labels.numpy())
        test_probs = pipe.predict_proba(test_logits.numpy())
    elif args.method == "isotonic":
        calibrators = isotonic_calibrate(torch.softmax(val_logits,1).numpy(), val_labels.numpy())
        test_probs = apply_isotonic(calibrators, torch.softmax(test_logits,1).numpy())
    else:  # bbq
        bbq = bbq_calibrate(val_logits.numpy(), val_labels.numpy())
        test_probs = bbq.predict_proba(test_logits.numpy())

    # метрики
    test_logits_post = torch.from_numpy(np.log(test_probs + 1e-12))  # для ECE удобнее логиты → softmax даст probs_postс ≈ probs
    ece_val = ece(test_logits_post, test_labels)
    ada_val = adaptive_ece(test_logits_post, test_labels)
    cls_val = classwise_ece(test_logits_post, test_labels)
    brier_val = brier_score(test_logits_post, test_labels)

    import pandas as pd
    row = pd.DataFrame([[args.dataset, args.model, args.method, ece_val, ada_val, cls_val, brier_val]],
                       columns=["dataset", "model", "method", "ece", "adaece", "classece", "brier"])
    if os.path.exists(args.output):
        row.to_csv(args.output, mode="a", header=False, index=False)
    else:
        row.to_csv(args.output, index=False)
    print(row)

if __name__ == "__main__":
    main()