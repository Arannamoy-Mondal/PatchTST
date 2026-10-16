"""Run the readable three-member comparison.

Examples:
    python scripts/compare_members.py --prepare
    python scripts/compare_members.py --run RUN --member arannamoy
    python scripts/compare_members.py --run RUN --summarize
"""

from patchtst.comparison import main


if __name__ == "__main__":
    main()
