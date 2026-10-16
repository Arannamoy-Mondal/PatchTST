"""Read the experiment settings and reject incomplete settings."""

from pathlib import Path
import yaml

# Every trained run must explicitly define these settings. Missing values are
# never replaced with hidden defaults.
REQUIRED_SETTINGS = {
    "data": ("root", "groups", "target_sr", "tasks"),
    "splits": ("type", "n_splits", "seed", "group_by"),
    "features": (
        "n_mels", "n_fft", "win_length", "hop_length", "window_sec",
        "hop_sec", "max_windows_per_file", "db_ref",
    ),
    "model": ("patch_len", "stride", "d_model", "n_heads", "n_layers", "dropout"),
    "train": ("epochs", "batch_size", "lr", "patience"),
}


def load_config(path="config/base.yaml"):
    """Return the settings dictionary, or None after printing the problem."""
    settings_path = Path(path)
    if not settings_path.is_file():
        print("ERROR: settings file was not found:", settings_path)
        return None

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        print("ERROR: the top of the settings file must be named sections.")
        return None

    missing_names = []
    for section_name, setting_names in REQUIRED_SETTINGS.items():
        section = settings.get(section_name)
        if not isinstance(section, dict):
            missing_names.append(section_name)
            continue
        for setting_name in setting_names:
            if setting_name not in section or section[setting_name] is None:
                missing_names.append(section_name + "." + setting_name)

    if missing_names:
        print("ERROR: missing settings:", ", ".join(missing_names))
        return None
    return settings


# Older scripts use this shorter name.
load_cfg = load_config
