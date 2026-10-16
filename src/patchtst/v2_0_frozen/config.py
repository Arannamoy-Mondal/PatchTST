"""Load config/base.yaml. A missing key is an error, never a silent default."""

from pathlib import Path

import yaml

REQUIRED = {
    "data": ("root", "groups", "target_sr", "tasks"),
    "splits": ("type", "n_splits", "seed", "group_by"),
    "features": (
        "n_mels",
        "n_fft",
        "win_length",
        "hop_length",
        "window_sec",
        "hop_sec",
        "max_windows_per_file",
        "db_ref",
    ),
    "model": ("patch_len", "stride", "d_model", "n_heads", "n_layers", "dropout"),
    "train": ("epochs", "batch_size", "lr", "patience"),
}


def load_config(path: str | Path = "config/base.yaml") -> dict:
    path = Path(path)
    if not path.is_file():
        print("error: config file was not found")
        print("path:", path)
        return None

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        print("error: config file must contain a mapping at the top")
        print("path:", path)
        return None

    missing: list[str] = []
    for section, keys in REQUIRED.items():
        block = loaded.get(section)
        if not isinstance(block, dict):
            missing.append(section)
            continue
        for key in keys:
            if key not in block or block[key] is None:
                missing.append(f"{section}.{key}")
    if missing:
        print("error: missing config keys:", ", ".join(missing))
        return None
    return loaded


load_cfg = load_config
