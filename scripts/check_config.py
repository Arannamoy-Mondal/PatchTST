"""Check that settings load and that a missing required setting is rejected."""

from pathlib import Path
import tempfile

import yaml

from patchtst.config import load_config


def main():
    settings = load_config()
    if settings is None:
        print("ERROR: the project settings did not load.")
        return
    print("PASS" if settings["data"]["target_sr"] == 16000 else "ERROR", "sample rate is 16000")
    print("PASS" if len(settings["data"]["groups"]) == 2 else "ERROR", "exactly two cohorts")

    del settings["splits"]["seed"]
    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary_directory:
        broken_settings_path = Path(temporary_directory) / "missing-key.yaml"
        broken_settings_path.write_text(yaml.safe_dump(settings))
        broken_settings = load_config(broken_settings_path)
        if broken_settings is None:
            print("PASS missing seed is rejected")
        else:
            print("ERROR: missing seed was accepted")


if __name__ == "__main__":
    main()
