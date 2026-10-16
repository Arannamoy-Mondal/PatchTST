"""Print a completed comparison summary without retraining."""

import argparse
from pathlib import Path


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--run", required=True)
    arguments = argument_parser.parse_args()
    summary_path = Path(arguments.run) / "summary.json"
    if not summary_path.is_file():
        print("ERROR: summary is missing:", summary_path)
        print("Finish all three members, then use --summarize.")
        return
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
