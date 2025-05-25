#!/usr/bin/env python3
"""
Автоматический перебор обучений и пост-калибровок.
• one GPU, single-process; перезапуск безопасен.
• Sentinel-файлы <method>.done не позволят повторно калибровать.
• .out/.err файлов много? – складываем их в ./logs/.
"""
import subprocess, csv, sys, logging, pathlib, os

ROOT   = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
LOG_DIR   = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)
RES_CSV   = ROOT / "results_master.csv"
CAL_CSV   = ROOT / "calib_master.csv"
logging.basicConfig(filename=ROOT/"run_all.log",
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
prt = lambda *a: (logging.info(" ".join(map(str,a))), print(*a, flush=True))

# ────────────────────────────────────────────────────────────────────────────
DATASETS = [("cifar10","resnet50"),("cifar100","resnet50"),("otto","mlp")]
LOSSES   = ["cross_entropy","focal","dual_focal","bsce_gra"]
REGS     = [(0,0,False), (0.1,0,False), (0,0.2,False), (0,0,True)]  # ls,mix,aug
POST_METHODS = ["temperature","platt","isotonic","bbq"]

EPOCHS=200; BATCH=128; PATIENCE=10; NEED_VALIDATE=True
# ────────────────────────────────────────────────────────────────────────────
def ensure_csv(path, header):
    if not path.exists():
        with open(path,"w",newline="") as f:
            csv.writer(f).writerow(header)

def run(cmd, tag):
    """stdout/err → logs/<tag>.out|err   ; returncode"""
    out = open(LOG_DIR/f"{tag}.out","w")
    err = open(LOG_DIR/f"{tag}.err","w")
    prt("RUN", *cmd)
    res = subprocess.run(cmd, stdout=out, stderr=err)
    if res.returncode: prt("FAIL", tag, res.returncode)
    return res.returncode

def find_ckpt(d: pathlib.Path):
    for pat in ("best.pth","final.pth","ckpt_*.pth"):
        f = next((p for p in d.glob(pat)), None)
        if f: return f
    return None

def sentinel(dir_: pathlib.Path, method: str): return dir_/f"{method}.done"

# ────────────────────────────────────────────────────────────────────────────
def main():
    ensure_csv(RES_CSV, ["dataset","model","loss","ls","mixup","augmix",
                         "accuracy","ece","adaece","classece","brier"])
    ensure_csv(CAL_CSV, ["dataset","model","method","loss","ls","mixup","augmix",
                        "accuracy","ece","adaece","classece","brier"])

    for ds, mdl in DATASETS:
        for loss in LOSSES:
            for ls, mix, aug in REGS:
                tag = f"{ds}_{mdl}_{loss}_ls{ls}_mx{mix}_{'aug' if aug else 'noaug'}"
                out_dir = MODEL_DIR/tag
                ckpt = find_ckpt(out_dir)

                # ---------- обучение (если нет ckpt) ----------
                if ckpt is None:
                    cmd = [sys.executable,"-m","src.train",
                           "--dataset",ds,"--model",mdl,
                           "--loss",loss,"--epochs",str(EPOCHS),
                           "--batch_size",str(BATCH),
                           "--output",str(out_dir),
                           "--early_stop",str(PATIENCE)]
                    if ls:  cmd += ["--label_smoothing",str(ls)]
                    if mix: cmd += ["--mixup",str(mix)]
                    if aug: cmd += ["--augmix"]
                    if run(cmd,tag):        # ошибка обучения
                        with open(RES_CSV,"a",newline="") as f:
                            csv.writer(f).writerow([ds,mdl,loss,ls,mix,aug,
                                                    "ERROR","","","",""])
                        continue
                    ckpt = find_ckpt(out_dir)

                # ---------- пост-калибровки для всех моделей без MixUp/AugMix
                if True: #mix==0 and not aug:
                    for method in POST_METHODS:
                        if sentinel(out_dir,method).exists(): continue
                        tag_cal = f"{tag}_{method}"
                        cmdc = [sys.executable,"src.calibrate.py",
                                "--ckpt",str(ckpt),
                                "--method",method,
                                "--dataset",ds,"--model",mdl,
                                "--loss",loss,"--ls",str(ls),
                                "--mixup",str(mix),
                                *(["--augmix"] if aug else []),
                                "--output",str(CAL_CSV)]
                        if run(cmdc, tag_cal)==0:
                            sentinel(out_dir,method).touch()
                        else:
                            with open(CAL_CSV,"a",newline="") as f:
                                csv.writer(f).writerow([ds,mdl,method,loss,ls,mix,aug,
                                                        "ERROR","","",""])
                                
                if NEED_VALIDATE:               
                    cmd_val = [sys.executable, "src/validate.py",
                            "--ckpt", str(ckpt),
                            "--dataset", ds, "--model", mdl,
                            "--loss", loss, "--ls", str(ls),
                            "--mixup", str(mix), *(["--augmix"] if aug else [])]
                    run(cmd_val, f"{tag}_val")

if __name__ == "__main__":
    main()
