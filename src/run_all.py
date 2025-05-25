"""Запуск полного набора обучений + пост‑калибровок.
Сценарий построен как очередь задач; при падении одной — пишет ошибку и идёт дальше.
Логи суммируются в run_all.log; результаты собираются в results_master.csv и calib_master.csv.
"""
import subprocess, itertools, json, os, time, logging, csv, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent  # корень проекта
MODEL_DIR = ROOT / "model"
RES_CSV = ROOT / "results_master.csv"
CAL_CSV = ROOT / "calib_master.csv"
LOG = ROOT / "run_all.log"

logging.basicConfig(filename=LOG, level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
print = lambda *a, **k: (logging.info(" ".join(map(str, a))), __builtins__.print(*a, **k))

DATASETS = [
    ("cifar10", "resnet50", 10),
    ("cifar100", "resnet50", 100),
    ("otto", "mlp", 9),
]
LOSSES = ["cross_entropy", "focal", "dual_focal", "bsce_gra"]
REGS = [  # (label_smoothing, mixup, augmix)
    (0.0, 0.0, False),      # none
    (0.1, 0.0, False),      # label smoothing
    (0.0, 0.2, False),      # mixup
    (0.0, 0.0, True),       # augmix
]
POST_METHODS = ["temperature", "platt", "isotonic", "bbq"]

EPOCHS = 200
BATCH = 128
PATIENCE = 10

def find_ckpt(out_dir):
    for pattern in ('best.pth', 'final.pth', 'ckpt_*.pth'):
        files = sorted(out_dir.glob(pattern))
        if files:
            return files[-1]
    return None

def run_cmd(cmd, env=None):
    print("RUN", " ".join(cmd))
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with code {e.returncode}: {' '.join(cmd)}")
        
        with open(RES_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([ds, mdl, loss, ls, mix, aug, 'ERROR', '', '', '', ''])


def ensure_csv(path, header):
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def main():
    ensure_csv(RES_CSV, ["dataset", "model", "loss", "ls", "mixup", "augmix",
                         "accuracy", "ece", "adaece", "classece", "brier"])
    ensure_csv(CAL_CSV, ["dataset", "model", "method", "loss", "ls", "mixup", "augmix",
                         "ece", "adaece", "classece", "brier"])

    for ds, mdl, _ in DATASETS:
        for loss in LOSSES:
            for ls, mix, aug in REGS:
                tag = f"{ds}_{mdl}_{loss}_ls{ls}_mx{mix}_{'aug' if aug else 'noaug'}"
                out_dir = MODEL_DIR / tag
                if (out_dir / f"ckpt_final.pth").exists():
                    print("Skip existing", tag); continue
                cmd = [sys.executable, "-m", "src.train",
                       "--dataset", ds, "--model", mdl,
                       "--loss", loss,
                       "--epochs", str(EPOCHS),
                       "--batch_size", str(BATCH),
                       "--output", str(out_dir)]
                cmd += ['--early_stop', str(PATIENCE)]
                if ls > 0: cmd += ["--label_smoothing", str(ls)]
                if mix > 0: cmd += ["--mixup", str(mix)]
                if aug: cmd += ["--augmix"]
                run_cmd(cmd)

                # пост‑калибровки ТОЛЬКО для базового CrossEntropy без регов
                if loss == "cross_entropy" and ls == 0 and mix == 0 and not aug:
                    ckpt = find_ckpt(out_dir)
                    for method in POST_METHODS:
                        cmd_c = [sys.executable, "src/calibrate.py",
                                 "--ckpt", str(ckpt),
                                 "--method", method,
                                 "--dataset", ds, "--model", mdl,
                                 "--output", str(CAL_CSV)]
                        run_cmd(cmd_c)

if __name__ == "__main__":
    main()