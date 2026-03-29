import os
import datetime

BLIND_FILENAMES = {"blind.json", "blind_test.json"}


def check_path(path: str):
    if any(b in str(path) for b in BLIND_FILENAMES):
        if not os.environ.get("ALLOW_BLIND_EVAL"):
            raise RuntimeError(
                f"BLIND ACCESS BLOCKED: {path}\n"
                "Set ALLOW_BLIND_EVAL=1 only for the final submission run."
            )
        else:
            _audit_log(path)


def _audit_log(path: str):
    with open("blind_access_audit.log", "a") as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()} BLIND_READ {path}\n")
