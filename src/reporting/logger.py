import csv
import os
import json
import datetime

FIELDS = [
    "run_id", "model", "dedup", "split", "seed",
    "macro_f1", "f1_false", "pr_auc_false", "roc_auc",
    "brier", "ece", "threshold", "notes",
]


def log_run(path: str, row: dict):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def save_manifest(run_id: str, config: dict, hashes: dict,
                  outdir: str = "runs/manifests"):
    os.makedirs(outdir, exist_ok=True)
    manifest = {
        "run_id":    run_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hashes":    hashes,
        "config":    config,
    }
    with open(f"{outdir}/{run_id}.json", "w") as f:
        json.dump(manifest, f, indent=2)
