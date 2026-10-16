"""Check printed file rows, ZIP exports, offline filters and PDF layout."""

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pymupdf as fitz
import numpy as np
import pandas as pd
from PIL import Image
from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--pdf", type=Path, default=PROJECT / "docs/wavlm_v4_0_technical_report.pdf")
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    source = arguments.source.resolve()
    generated = source / "generated"
    catalogue = pd.read_csv(generated / "file_catalogue.csv")
    failures = []
    text = subprocess.run(["pdftotext", "-layout", str(arguments.pdf), "-"], capture_output=True, text=True)
    pattern = re.compile(r"^\s*(F\d{4})(\*?)\s+(D\d{2})\s+(.+?)\s+([TVE]{3})\s+([TVE]{3})\s+([TVE]{3})\s+([TVE]{3})\s+([TVE]{3})\s*$")
    printed = {}
    for line in text.stdout.splitlines():
        match = pattern.match(line)
        if match:
            file_id, mark, directory, filename, *roles = match.groups()
            if file_id in printed:
                failures.append("duplicate printed file ID " + file_id)
            printed[file_id] = {"mark": mark, "directory_id": directory, "filename": filename, "roles": roles}
    for row in catalogue.itertuples(index=False):
        if row.file_id not in printed:
            failures.append("missing printed file " + row.file_id)
            continue
        actual = printed[row.file_id]
        roles = [getattr(row, "fold_" + str(fold) + "_roles") for fold in range(5)]
        if actual["filename"] != row.file_name or actual["directory_id"] != row.directory_id or actual["roles"] != roles:
            failures.append("printed filename or role differs " + row.file_id)
        if bool(actual["mark"]) != bool(row.native16k_eligible):
            failures.append("native-rate printed mark differs " + row.file_id)
    if len(printed) != len(catalogue):
        failures.append("expected all 773 printed recording rows")
    csv_count = 0
    with zipfile.ZipFile(generated / "fold_file_lists.zip") as archive:
        for scenario in ["main", "native16k"]:
            for path in sorted((run / scenario / "splits").rglob("*.csv")):
                name = path.relative_to(run).as_posix()
                if archive.read(name) != path.read_bytes():
                    failures.append("ZIP CSV differs " + name)
                csv_count += 1
    browser_errors = []
    filter_count = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(viewport={"width": 1100, "height": 850})
        context.set_offline(True)
        page = context.new_page()
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.goto((generated / "statistical_dashboard.html").as_uri(), wait_until="load")
        for scenario in ["main", "native16k"]:
            page.select_option("#scenario", scenario)
            for fold in range(5):
                page.select_option("#fold", str(fold))
                choices = [("refit", 0, "outer_train"), ("test", 0, "outer_test")]
                for inner in range(3):
                    choices += [("train", inner, "inner_" + str(inner) + "_train"),
                                ("validation", inner, "inner_" + str(inner) + "_validation")]
                for role, inner, name in choices:
                    page.select_option("#inner", str(inner))
                    page.select_option("#role", role)
                    actual_paths = page.locator("#files tr td:nth-child(2)").all_text_contents()
                    expected = pd.read_csv(run / scenario / "splits" / ("fold_" + str(fold)) / (name + ".csv"))
                    if set(actual_paths) != set(expected.path) or len(actual_paths) != len(expected):
                        failures.append("offline filter differs " + scenario + "/" + str(fold) + "/" + name)
                    filter_count += 1
        figure_count = page.locator(".js-plotly-plot").count()
        if figure_count != 1:
            failures.append("expected one rendered offline interactive score plot")
        page.set_viewport_size({"width": 390, "height": 844})
        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
        if overflow:
            failures.append("interactive page overflows the narrow-screen viewport")
        context.close()
        browser.close()
    failures += browser_errors
    pdf = fitz.open(arguments.pdf)
    outside_words = 0
    non_gray_pages = []
    for number, page in enumerate(pdf):
        for word in page.get_text("words"):
            x0, y0, x1, y1 = word[:4]
            if x0 < 15 or y0 < 15 or x1 > page.rect.width - 15 or y1 > page.rect.height - 15:
                outside_words += 1
        rendered = subprocess.run(["pdftoppm", "-f", str(number + 1), "-l", str(number + 1),
                                   "-scale-to", "700", "-singlefile", str(arguments.pdf)], capture_output=True)
        if rendered.returncode != 0:
            failures.append("Poppler RGB rendering failed on page " + str(number + 1))
            continue
        image = Image.open(io.BytesIO(rendered.stdout)).convert("RGB")
        values = np.asarray(image)
        if not np.array_equal(values[:, :, 0], values[:, :, 1]) or not np.array_equal(values[:, :, 1], values[:, :, 2]):
            non_gray_pages.append(number + 1)
    if outside_words:
        failures.append("PDF words outside checked page-edge bounds")
    if non_gray_pages:
        failures.append("PDF pages with non-gray pixels: " + str(non_gray_pages))
    result = {"printed_recordings_checked": len(printed), "zip_partition_csvs": csv_count,
              "offline_filter_combinations": filter_count, "browser_errors": browser_errors,
              "pdf_pages": len(pdf), "outside_words": outside_words, "non_gray_pages": non_gray_pages,
              "rgb_renderer": "Poppler pdftoppm; exact channel equality, no relaxed tolerance",
              "failures": failures, "passed": len(failures) == 0}
    pdf.close()
    (source / "quality_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return len(failures) == 0


if __name__ == "__main__":
    if not main():
        sys.exit(1)
