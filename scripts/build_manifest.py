"""Create the recording index and person folds."""

from pathlib import Path

from patchtst.config import load_config
from patchtst.data.index import build_manifest
from patchtst.data.splits import assign_folds


def main():
    settings = load_config("config/base.yaml")
    if settings is None:
        return
    recordings, speakers = build_manifest(settings)
    if recordings is None or speakers is None:
        return
    person_folds = assign_folds(recordings, settings)
    if person_folds is None:
        return

    recordings = recordings.merge(person_folds[["speaker_id", "fold"]], on="speaker_id", how="left")
    speakers = speakers.merge(person_folds[["speaker_id", "fold"]], on="speaker_id", how="left")
    output_directory = Path("data")
    output_directory.mkdir(exist_ok=True)
    recordings.to_csv(output_directory / "manifest.csv", index=False)
    speakers.to_csv(output_directory / "speakers.csv", index=False)
    person_folds.to_csv(output_directory / "folds.csv", index=False)
    print("Wrote data/manifest.csv, data/speakers.csv, and data/folds.csv.")


if __name__ == "__main__":
    main()
