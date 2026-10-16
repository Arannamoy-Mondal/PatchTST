"""Create the log-mel feature cache from data/manifest.csv."""

from pathlib import Path

import pandas as pd

from patchtst.config import load_config
from patchtst.data.audio import cache_features


def main():
    settings = load_config("config/base.yaml")
    if settings is None:
        return
    manifest_path = Path("data/manifest.csv")
    if not manifest_path.exists():
        print("ERROR: data/manifest.csv is missing. Run scripts/build_manifest.py first.")
        return

    recordings = pd.read_csv(manifest_path)
    recordings_with_features = cache_features(recordings, settings, Path("data/features"))
    recordings_with_features.to_csv(manifest_path, index=False)
    print("Cached feature arrays:", len(recordings_with_features))


if __name__ == "__main__":
    main()
