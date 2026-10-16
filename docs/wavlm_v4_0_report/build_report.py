"""Build a proposal-experiment report exclusively from completed saved records."""

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
sys.path.insert(0, str(PROJECT / "experiments/v4_0_wavlm_classifiers"))
from common import METHODS, METRICS, fingerprint, score_people, summarize, write_json

LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}


def tex(value):
    text = str(value)
    for old, new in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")]:
        text = text.replace(old, new)
    return text


def percent(value):
    return "%.2f" % (100 * value)


def check_saved_run(run):
    if not (run / "status.json").is_file() or not (run / "replay_checks.json").is_file():
        print("ERROR: finish training and saved-model replay before building the report.")
        return None
    status = json.loads((run / "status.json").read_text())
    replay_path = PROJECT / status["verification_file"] if "verification_file" in status else run / "replay_checks.json"
    replay = json.loads(replay_path.read_text())
    if not status["complete"] or not replay["all_passed"]:
        print("ERROR: the experiment is not completed and replay-checked.")
        return None
    settings = json.loads((run / "settings.json").read_text())
    saved = json.loads((run / "summary.json").read_text())
    tables = {}
    for scenario in ["main", "native16k"]:
        tables[scenario] = {}
        for method in METHODS:
            people = pd.read_csv(run / scenario / method / "speaker_predictions.csv")
            recordings = pd.read_csv(run / scenario / "manifest.csv")
            metadata = recordings.groupby("speaker_id", as_index=False).agg(label=("label", "first"), fold=("fold", "first"))
            checked_people = people.merge(metadata, on="speaker_id", suffixes=("", "_manifest"), validate="one_to_one")
            expected = 46 if scenario == "main" else 39
            if len(people) != expected or not people.speaker_id.is_unique:
                print("ERROR: expected", expected, "unique OOF speaker rows.")
                return None
            if (len(checked_people) != expected or not checked_people.label.eq(checked_people.label_manifest).all()
                or not checked_people.fold.eq(checked_people.fold_manifest).all()
                or not people.threshold.eq(settings["train"]["threshold"]).all()
                or not people.prediction.eq((people.probability >= settings["train"]["threshold"]).astype(int)).all()):
                print("ERROR: speaker metadata, fixed threshold or decision differs from frozen records.")
                return None
            recomputed = summarize(people, settings["train"]["threshold"])
            for metric in METRICS:
                if abs(recomputed["pooled"][metric] - saved[scenario][method]["pooled"][metric]) > 1e-12:
                    print("ERROR: saved pooled metric differs:", scenario, method, metric)
                    return None
                for fold in range(5):
                    if abs(recomputed["folds"][fold][metric] - saved[scenario][method]["folds"][fold][metric]) > 1e-12:
                        print("ERROR: saved fold metric differs:", scenario, method, fold, metric)
                        return None
                for statistic in ["mean", "sd"]:
                    if abs(recomputed["mean_sd"][metric][statistic] - saved[scenario][method]["mean_sd"][metric][statistic]) > 1e-12:
                        print("ERROR: fold variation differs:", scenario, method, metric, statistic)
                        return None
            without_flags = summarize(people[~people.sheet_flag], settings["train"]["threshold"])
            for metric in METRICS:
                if abs(without_flags["pooled"][metric] - saved[scenario][method]["without_sheet_flag"]["pooled"][metric]) > 1e-12:
                    print("ERROR: flag-excluded metric differs:", scenario, method, metric)
                    return None
            tables[scenario][method] = people
    return settings, saved, tables, replay


def export_partitions(run):
    manifest = pd.read_csv(run / "manifest.csv")
    catalogue = manifest.copy()
    counts = []
    csv_paths = []
    for scenario in ["main", "native16k"]:
        subset = manifest if scenario == "main" else manifest[manifest.native_sr == 16000]
        tested = []
        for fold in range(5):
            directory = run / scenario / "splits" / ("fold_" + str(fold))
            split = json.loads((directory / "split.json").read_text())
            train_ids, test_ids = set(split["outer_train"]), set(split["outer_test"])
            if train_ids & test_ids or train_ids | test_ids != set(subset.speaker_id):
                print("ERROR: inconsistent outer partition:", scenario, fold)
                return None
            tested.extend(split["outer_test"])
            for inner in split["inner"]:
                fit_ids, validation_ids = set(inner["train"]), set(inner["validation"])
                if fit_ids & validation_ids or fit_ids | validation_ids != train_ids:
                    print("ERROR: inconsistent inner partition:", scenario, fold)
                    return None
                column = scenario + "_fold_" + str(fold) + "_inner_" + str(inner["inner_fold"])
                roles = []
                for person in catalogue.speaker_id:
                    if person in test_ids:
                        roles.append("E")
                    elif person in validation_ids:
                        roles.append("V")
                    elif person in fit_ids:
                        roles.append("T")
                    else:
                        roles.append("-")
                catalogue[column] = roles
            for path in sorted(directory.glob("*.csv")):
                rows = pd.read_csv(path)
                role = path.stem
                if role == "outer_train":
                    ids = train_ids
                elif role == "outer_test":
                    ids = test_ids
                else:
                    inner_number = int(role.split("_")[1])
                    key = "train" if role.endswith("train") else "validation"
                    ids = set(split["inner"][inner_number][key])
                expected_paths = set(subset.loc[subset.speaker_id.isin(ids), "path"])
                if rows.path.duplicated().any() or set(rows.path) != expected_paths:
                    print("ERROR: partition CSV differs from the speaker split:", path)
                    return None
                csv_paths.append(path)
                people = rows[["speaker_id", "label"]].drop_duplicates()
                counts.append({"scenario": scenario, "fold": fold, "role": role, "speakers": len(people), "recordings": len(rows),
                               "pd_speakers": int(people.label.sum()), "control_speakers": int((people.label == 0).sum())})
        if len(tested) != subset.speaker_id.nunique() or len(set(tested)) != len(tested):
            print("ERROR: each person must be outer-test data exactly once.")
            return None
    if len(csv_paths) != 80:
        print("ERROR: expected 80 complete outer/inner partition CSVs; found", len(csv_paths))
        return None
    catalogue["file_id"] = ["F%04d" % number for number in range(1, len(catalogue) + 1)]
    folders = sorted(catalogue.folder.unique())
    folder_ids = {folder: "D%02d" % number for number, folder in enumerate(folders, start=1)}
    catalogue["directory_id"] = catalogue.folder.map(folder_ids)
    catalogue["native16k_eligible"] = catalogue.native_sr == 16000
    for scenario in ["main", "native16k"]:
        for fold in range(5):
            column = scenario + "_fold_" + str(fold) + "_roles"
            catalogue[column] = catalogue[[scenario + "_fold_" + str(fold) + "_inner_" + str(inner) for inner in range(3)]].agg("".join, axis=1)
    for fold in range(5):
        catalogue["fold_" + str(fold) + "_roles"] = catalogue["main_fold_" + str(fold) + "_roles"]
    catalogue.to_csv(GENERATED / "file_catalogue.csv", index=False)
    pd.DataFrame(counts).to_csv(GENERATED / "partition_counts.csv", index=False)
    folder_key = catalogue[["directory_id", "folder", "speaker_id"]].drop_duplicates().sort_values("directory_id")
    folder_key.to_csv(GENERATED / "folder_key.csv", index=False)
    with zipfile.ZipFile(GENERATED / "fold_file_lists.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in csv_paths:
            archive.write(path, path.relative_to(run).as_posix())
        for name in ["file_catalogue.csv", "partition_counts.csv", "folder_key.csv"]:
            archive.write(GENERATED / name, name)
    return catalogue, pd.DataFrame(counts), folder_key, csv_paths


def save_figure(name, figure):
    figure.tight_layout()
    for extension in ["pdf", "png"]:
        figure.savefig(GENERATED / (name + "." + extension), dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(run, tables, saved, counts, manifest):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    roles = ["inner_0_train", "inner_0_validation", "outer_test"]
    for axis, column in zip(axes, ["speakers", "recordings"]):
        for number, role in enumerate(roles):
            values = counts[(counts.scenario == "main") & (counts.role == role)].sort_values("fold")
            axis.bar(values.fold + (number - 1) * 0.24, values[column], width=0.24, label=role.replace("_", " "), color="white", edgecolor="black", hatch=["", "//", "xx"][number])
        axis.set_xlabel("Outer fold")
        axis.set_ylabel(column.title())
        axis.set_xticks(range(5))
    axes[0].legend(fontsize=7)
    save_figure("partition_sizes", figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    core = METRICS[:5]
    for axis, method in zip(axes, METHODS):
        result = saved["main"][method]
        for fold in result["folds"]:
            axis.scatter([100 * fold[name] for name in core], np.arange(5) + (fold["fold"] - 2) * 0.07,
                         marker=["o", "s", "^", "v", "x"][fold["fold"]], color="black", facecolors="none" if fold["fold"] < 4 else None, s=25)
        axis.scatter([100 * result["pooled"][name] for name in core], np.arange(5), marker="D", color="black", s=30)
        axis.set_xlim(0, 103)
        axis.set_yticks(range(5), ["F1", "Recall", "Specificity", "Precision", "Accuracy"])
        axis.set_xlabel("Speaker score (%)")
        axis.set_title(LABELS[method])
    axes[0].invert_yaxis()
    save_figure("fold_metrics", figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for axis, method in zip(axes, METHODS):
        people = tables["main"][method]
        groups = [people[people.label == label].probability.to_numpy() for label in [0, 1]]
        axis.boxplot(groups, tick_labels=["Control", "PD"], showfliers=False, medianprops={"color": "black"})
        for label in [0, 1]:
            rows = people[people.label == label]
            jitter = np.linspace(-0.14, 0.14, len(rows))
            axis.scatter(label + 1 + jitter, rows.probability, s=15, facecolors="white", edgecolors="black")
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(LABELS[method])
        axis.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("OOF speaker probability")
    save_figure("score_distributions", figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for method, style in zip(METHODS, ["-", "--", ":"]):
        people = tables["main"][method]
        fpr, tpr, unused = roc_curve(people.label, people.probability)
        precision, recall, unused = precision_recall_curve(people.label, people.probability)
        axes[0].plot(fpr, tpr, style, color="black", label=LABELS[method] + " %.3f" % saved["main"][method]["pooled"]["auroc"])
        axes[1].plot(recall, precision, style, color="black", label=LABELS[method] + " %.3f" % saved["main"][method]["pooled"]["auprc"])
    for axis in axes:
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.legend(fontsize=7)
    axes[0].set(xlabel="False-positive rate", ylabel="Recall", title="ROC / AUROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall / trapezoidal AUPRC")
    save_figure("ranking_curves", figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    selection_rows = []
    for axis, method in zip(axes, METHODS):
        for fold in range(5):
            result = json.loads((run / "main" / method / ("fold_" + str(fold)) / "selection.json").read_text())
            x = [row["candidate"] + 1 for row in result["candidates"]]
            y = [row["mean_inner_speaker_auroc"] for row in result["candidates"]]
            axis.plot(x, y, color=str(0.12 + fold * 0.16), marker=["o", "s", "^", "v", "x"][fold], label="Fold " + str(fold), linewidth=0.8)
            for row in result["candidates"]:
                selection_rows.append({"method": method, "outer_fold": fold, "candidate": row["candidate"],
                    "parameters": json.dumps(row["parameters"], sort_keys=True), "mean_inner_auroc": row["mean_inner_speaker_auroc"],
                    "selected": row["candidate"] == result["selected"]["candidate"]})
        axis.set_title(LABELS[method])
        axis.set_xlabel("Predeclared grid candidate")
        axis.set_ylim(0, 1.03)
    axes[0].set_ylabel("Mean inner speaker AUROC")
    axes[0].legend(fontsize=6)
    pd.DataFrame(selection_rows).to_csv(GENERATED / "selection_records.csv", index=False)
    save_figure("selection_curves", figure)
    burden = manifest.groupby(["speaker_id", "label"], as_index=False).agg(recordings=("path", "size"), seconds=("duration_s", "sum"))
    burden["minutes"] = burden.seconds / 60
    burden.to_csv(GENERATED / "speaker_audio_summary.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for axis, column in zip(axes, ["recordings", "minutes"]):
        groups = [burden[burden.label == label][column].to_numpy() for label in [0, 1]]
        axis.boxplot(groups, tick_labels=["Control", "PD"], showfliers=False, medianprops={"color": "black"})
        for label in [0, 1]:
            values = groups[label]
            axis.scatter(label + 1 + np.linspace(-0.14, 0.14, len(values)), values, s=18, facecolors="white", edgecolors="black")
        axis.set_ylabel(column.title() + " per person")
    save_figure("recording_burden", figure)


def make_dashboard(catalogue, tables):
    figure = go.Figure()
    for method in METHODS:
        people = tables["main"][method].sort_values(["label", "speaker_id"])
        figure.add_trace(go.Scatter(x=people.speaker_id.tolist(), y=people.probability.tolist(), mode="markers", name=LABELS[method],
                                  customdata=people[["label", "fold"]].to_numpy(), hovertemplate="%{x}<br>score %{y:.5f}<br>label %{customdata[0]}<br>fold %{customdata[1]}<extra>%{fullData.name}</extra>"))
    figure.add_hline(y=0.5, line_dash="dash")
    figure.update_layout(title="Saved main OOF speaker scores — v4.0", xaxis_title="Speaker", yaxis_title="Model probability", height=550)
    role_columns = []
    for scenario in ["main", "native16k"]:
        role_columns.extend([scenario + "_fold_" + str(fold) + "_roles" for fold in range(5)])
    rows = catalogue[["file_id", "path", "speaker_id", "label", "native16k_eligible"] + role_columns].to_dict("records")
    script = r"""
const rows = RECORDS;
function refresh() {
 const scenario=document.getElementById('scenario').value;
 const fold=document.getElementById('fold').value;
 const inner=Number(document.getElementById('inner').value);
 const role=document.getElementById('role').value;
 const query=document.getElementById('query').value.toLowerCase();
 const selected=rows.filter(row => {
  if(scenario==='native16k' && !row.native16k_eligible) return false;
  const roles=row[scenario+'_fold_'+fold+'_roles'];
  if(role==='test' && roles[0]!=='E') return false;
  if(role==='refit' && roles[0]==='E') return false;
  if(role==='train' && roles[inner]!=='T') return false;
  if(role==='validation' && roles[inner]!=='V') return false;
  return (row.path+' '+row.speaker_id).toLowerCase().includes(query);
 });
 document.getElementById('count').textContent=selected.length+' exact recordings';
 const body=document.getElementById('files'); body.replaceChildren();
 selected.forEach(row => {
  const tr=document.createElement('tr');
  [row.file_id,row.path,row.speaker_id,row.label,row[scenario+'_fold_'+fold+'_roles']].forEach(value => {
   const td=document.createElement('td'); td.textContent=value; tr.appendChild(td);
  }); body.appendChild(tr);
 });
}
['scenario','fold','inner','role','query'].forEach(id=>document.getElementById(id).addEventListener('input',refresh)); refresh();
""".replace("RECORDS", json.dumps(rows).replace("</", "<\\/"))
    html = "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>v4.0 WavLM saved records</title><style>body{font:15px Georgia,serif;margin:24px}select,input{margin:5px;padding:6px}.scroll{overflow:auto}table{border-collapse:collapse;font:12px monospace}td,th{border:1px solid #ccc;padding:5px;white-space:nowrap}</style><script>" + get_plotlyjs() + "</script></head><body><h1>WavLM proposal experiment — v4.0</h1><p>Internal speaker-disjoint, leakage-checked results. Acquisition confounding remains unresolved. Plots and filters use saved records only.</p><div class='scroll'>" + figure.to_html(full_html=False, include_plotlyjs=False) + "</div><h2>Exact fold files</h2><p>T/V/E triples show roles in inner splits 0/1/2. Final refit contains T and V; E is outer test.</p><select id='scenario'><option value='main'>Main</option><option value='native16k'>Native 16 kHz</option></select><select id='fold'>" + "".join("<option>" + str(fold) + "</option>" for fold in range(5)) + "</select><select id='inner'><option>0</option><option>1</option><option>2</option></select><select id='role'><option value='train'>Inner training</option><option value='validation'>Validation</option><option value='refit'>Final refit</option><option value='test'>Test</option></select><input id='query' placeholder='Search filename or speaker'><p id='count'></p><div class='scroll'><table><thead><tr><th>ID</th><th>Exact path</th><th>Speaker</th><th>Label</th><th>Roles</th></tr></thead><tbody id='files'></tbody></table></div><script>" + script + "</script></body></html>"
    (GENERATED / "statistical_dashboard.html").write_text(html, encoding="utf-8")


def recording_error_debug(run, threshold):
    rows = []
    for scenario in ["main", "native16k"]:
        for method in METHODS:
            tables = []
            for fold in range(5):
                tables.append(pd.read_csv(run / scenario / method / ("fold_" + str(fold)) / "recording_predictions.csv"))
            files = pd.concat(tables, ignore_index=True)
            files["error"] = (files.probability >= threshold).astype(int) != files.label
            summary = files.groupby("task", as_index=False).agg(recordings=("path", "size"), misclassified_recordings=("error", "sum"))
            summary["scenario"] = scenario
            summary["method"] = method
            rows.append(summary)
    pd.concat(rows, ignore_index=True).to_csv(GENERATED / "task_error_debug.csv", index=False)


def core_table(result, include_folds=False):
    lines = [r"\begin{center}\small\begin{tabular}{lrrrrrr}\toprule Fold & People & F1 & Recall & Specificity & Precision & Accuracy\\\midrule"]
    if include_folds:
        for row in result["folds"]:
            lines.append(str(row["fold"]) + " & " + str(row["speakers"]) + " & " + " & ".join(percent(row[name]) for name in METRICS[:5]) + r"\\")
        values = []
        for name in METRICS[:5]:
            variation = result["mean_sd"][name]
            values.append(percent(variation["mean"]) + r" $\pm$ " + percent(variation["sd"]))
        lines.append(r"Mean/SD & -- & " + " & ".join(values) + r"\\\midrule")
    row = result["pooled"]
    lines.append("Pooled & " + str(row["speakers"]) + " & " + " & ".join(percent(row[name]) for name in METRICS[:5]) + r"\\\bottomrule\end{tabular}\end{center}")
    return "\n".join(lines)


def make_latex(run, settings, saved, tables, replay, catalogue, counts, folder_key, decisions):
    date = datetime.now(timezone.utc).strftime("%d %B %Y")
    baseline = saved["main"]["logistic_regression"]["pooled"]
    pieces = [r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{mathptmx,courier,microtype}
\usepackage[margin=24mm]{geometry}\usepackage{amsmath,amssymb,booktabs,tabularx,longtable,graphicx}
\usepackage[obeyspaces]{url}\usepackage{hyperref}\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}\hypersetup{hidelinks}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\emergencystretch}{3em}
\newcommand{\plot}[2]{\begin{center}\includegraphics[width=\linewidth]{generated/#1.pdf}\end{center}\noindent\textbf{Figure:} #2}
\begin{document}\begin{center}{\LARGE\bfseries WavLM-Large for Parkinson's\\Speech Classification}\par\vspace{10pt}
{\large Proposal Experiment: Verified Speaker-disjoint OOF Evaluation}\par
Arannamoy Mondal\par DATE\quad Report 1.0\quad Experiment v4.0\end{center}\hrule
\section*{Abstract}
This separate experiment implements the WavLM-based proposal using the latest
PatchTST technical report's evidence structure. It asks whether Random Forest
or XGBoost improves on logistic regression when all three receive exactly the
same frozen WavLM-large recording embeddings, speech tasks and speaker folds.
It uses 773 recordings from 46 canonical people, 25 with Parkinson's disease
and 21 elderly controls. It does not replace the official v2.0 PatchTST result.
BASELINE
All comparisons are revised internal evaluations on a previously studied corpus.
Speaker separation and training-side selection are checked, but recording-condition
confounding and external clinical generalization remain unresolved.
\section{One question, one input, three matched classifiers}
The input is a 1,024-value representation of each recording, extracted by a pinned,
frozen WavLM-large encoder. Logistic regression is the predeclared baseline;
Random Forest and XGBoost are extensions, not replacements chosen after looking
at a test score. All three use the same cohort, tasks, outer folds, inner splits,
training weights and threshold. The question concerns corpus-label classification,
not a new clinical diagnosis.
\clearpage\section{The whole pipeline}
\begin{center}\begin{tikzpicture}[node distance=0.35cm,
box/.style={draw,align=center,text width=12cm,inner sep=6pt},
arr/.style={-{Stealth},draw=black}]
\node[box](a){1. Keep all raw WAVs unchanged; exclude young controls.};
\node[box,below=of a](b){2. Reuse canonical people and frozen v2.0 outer folds.};
\node[box,below=of b](c){3. Resample to 16 kHz; split into at-most-ten-second windows.};
\node[box,below=of c](d){4. Apply the pinned WavLM frontend and frozen encoder.};
\node[box,below=of d](e){5. Average hidden frames, then windows, into one recording vector.};
\node[box,below=of e](f){6. Search classifier settings using three inner speaker folds.};
\node[box,below=of f](g){7. Refit a fresh classifier on all outer-training people.};
\node[box,below=of g](h){8. Predict held-out recordings; average scores by speaker.};
\node[box,below=of h](i){9. Apply 0.5 once per speaker; pool the five OOF tables.};
\foreach \x/\y in {a/b,b/c,c/d,d/e,e/f,f/g,g/h,h/i}\draw[arr](\x)--(\y);
\end{tikzpicture}\end{center}
Fixed feature extraction learns nothing from this corpus. Classifier fitting,
scaling, parameter selection and the optional fine-tuning decision remain inside
training people. Every number reported below is generated from saved predictions.
\clearpage\section{The recordings and canonical people}
The unmodified corpus contains 831 WAV files. Removing 45 young-control files
leaves an eligible 786-recording index with 47 working person identities.
Known repeat sessions for Vito S, Nicola S and Roberto R are linked before splitting.
The study deliberately retains the latest report's common cohort: 773 files and
46 people. Vitantonio D lacks passage recordings and remains excluded from that
shared cohort. This is a design choice, not a requirement of WavLM itself.
Age, sex, CPS, native sample rate, dates, names, tasks and paths are metadata;
none is appended to the classifier feature vector. Task names only describe errors.
Young controls remain outside both training and reported evaluation.
The native-rate sensitivity experiment retains 613 recordings from 39 people:
18 patients and 21 controls. It is retrained separately using only recordings
originally sampled at 16 kHz. Three flagged controls are removed only in the
separate main-OOF sensitivity evaluation; those models are not retrained.
\clearpage\section{From waveform to WavLM representation}
Mono PCM is read as float32 and resampled to 16 kHz with librosa's soxr\_hq.
No extra peak normalization, task trimming or metadata-derived feature is applied.
The pinned pretrained frontend then performs its standard mean/variance
normalization within each waveform window. This is not peak normalization,
but it removes absolute waveform level cues, including some potential voice-level
and recording-gain information. It differs from the level-preserving PatchTST input.
Windows cover at most ten seconds. Long recordings use consecutive ten-second
windows, plus a final overlapping full window when needed to cover the tail.
Short recordings remain short; only inputs below the convolutional minimum of
400 samples are edge-padded. A ten-second window produces 499 hidden frames.
\[
e_{jk}=\frac{1}{T_{jk}}\sum_t h_{jkt}\in\mathbb R^{1024},\qquad
e_j=\frac{1}{K_j}\sum_k e_{jk}.
\]
The encoder has 315,453,120 parameters, all frozen during this extraction.
It has 24 transformer layers, 1,024 hidden values and 16 attention heads.
It is a pretrained speech representation, not 1,024 handcrafted measurements.
Revision: \path{c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c}.
The exact checkpoint, frontend and configuration files are fingerprinted.
WavLM was pretrained for general speech representation; Italian-corpus success
does not demonstrate language-independent or clinical generalization.
\clearpage\section{What each classifier learns}
\textbf{Logistic regression:} a training-fitted weighted standard scaler followed
by regularized linear classification. C is selected from 0.01, 0.1, 1 and 10.
\textbf{Random Forest:} 300 trees; depths 4 or unrestricted; minimum leaves
1 or 2; square-root feature subsampling. No learned scaler is required.
\textbf{XGBoost:} 200 histogram-based boosting rounds; depths 1, 2 or 3;
learning rates 0.03 or 0.1; half of embedding columns considered per tree;
L2 regularization 1. These six settings are predeclared.
The grids contain 4, 4 and 6 candidates respectively. The first candidate in
the frozen grid order wins exact validation-score ties. There is no outer-test
early stopping, threshold search, feature selection or best-fold selection.
Feature values must be finite. Invalid cache files stop the run rather than
silently being imputed or dropped.
\clearpage\section{Nested speaker folds and final refit}
The five outer folds are exactly those of v2.0. Each held-out person receives
a prediction from a model trained without that person. Repeat sessions and
all recordings from a person stay together.
Inside each outer-training group, three stratified inner speaker folds evaluate
every candidate. Each inner model fits only its training people, predicts its
validation recordings, and averages those probabilities to speakers before AUROC
is calculated. The average of the three validation AUROCs chooses the setting.
The selected model is then fitted again from scratch on all outer-training people.
The outer test group never chooses the classifier, hyperparameters or threshold.
Unlike PatchTST's single validation holdout, this experiment has three rotating
training/validation pairs inside each outer fold. The exact 80 partition CSVs
cover both main and native-rate scenarios. Final refit contains all development
people, not only an inner training subset.
\clearpage\section{Equal training influence and speaker decisions}
Each training class receives equal total weight. Within each class, every person
receives equal weight; within a person, every recording receives an equal share.
For recording j from person i in class c, raw weight is
\[
w_{ij}=1/(N_c F_i).
\]
Weights are rescaled to have mean one. The scaler and classifier receive only
current-training rows and weights. Test people never determine these counts.
The classifiers operate on complete recording vectors. A person's final score is
the equal average of that person's recording probabilities, not a pooled average
of every window and not an average of hard class labels.
\[
p_i=\frac{1}{F_i}\sum_j f(e_j),\qquad \hat y_i=\mathbf 1[p_i\geq0.5].
\]
The first averaging stage here combines embeddings before the recording classifier;
the second combines recording probabilities. It is not identical to averaging
PatchTST window probabilities, and the distinction is intentional and recorded.
Each person contributes one OOF row, regardless of recording count or duration.
\clearpage\section{The conditional fine-tuning rule}
The proposal suggests partial fine-tuning when baseline AUROC is below 0.80.
This implementation applies that rule to the selected logistic baseline's mean
inner speaker AUROC separately in each outer fold. Pooled outer-test AUROC never
chooses whether a model is trained.
If triggered, a training-side holdout compares two or four upper layers and learning
rates of $10^{-5}$ or $3\cdot10^{-5}$. Binary cross-entropy selects an epoch count,
with at most six epochs and patience two. Validation AUROC selects the candidate.
The lower encoder stays frozen and deterministic. Training uses at most two windows
per recording; evaluation uses all windows. A fresh pretrained model is refitted
on all outer-training people for the selected epoch count before testing.
CONDITIONAL
\clearpage\section{Five operating-point metrics and ranking}
F1 compares precision and recall but ignores true negatives. Recall counts detected
patients; specificity counts correctly excluded controls. Precision concerns the
people called positive in this nearly balanced sample. Accuracy counts all correct
decisions. The five are always shown together, in percentages.
Precision in this nearly balanced speaker sample is not community-screening
positive predictive value. Model scores are not established clinical risk probabilities.
AUROC describes ranking across thresholds. Trapezoidal AUPRC integrates the
precision--recall curve; average precision uses a different step-weighted definition
and is reported separately. Neither selects a new threshold after testing.
Fold mean and sample SD describe variation in these five small folds, not a
confidence interval for deployment. The experimental sample is 46 people,
not 773 independent patients. The proposal's above-85\% pooled-F1 target is
assessed as an internal numerical target, not clinical validation.
""".replace("DATE", date).replace("BASELINE", "The frozen logistic baseline has pooled speaker F1 " + percent(baseline["f1"]) + r"\%, recall " + percent(baseline["recall"]) + r"\%, specificity " + percent(baseline["specificity"]) + r"\%, precision " + percent(baseline["precision"]) + r"\%, and accuracy " + percent(baseline["accuracy"]) + r"\%.")]
    triggered = sum(row["triggered"] for row in decisions)
    conditional = str(triggered) + " of the ten main/native outer folds triggered fine-tuning. "
    if triggered == 0:
        conditional += "No upper-layer model was trained in this experiment. The conditional pipeline reuses the frozen logistic baseline and must not be presented as an independent fourth model."
    inner_values = [row["baseline_inner_auroc"] for row in decisions]
    conditional += " The ten training-side gate values range from %.4f to %.4f." % (min(inner_values), max(inner_values))
    pieces[0] = pieces[0].replace("CONDITIONAL", conditional)
    for method in METHODS:
        result = saved["main"][method]
        pieces.append(r"\clearpage\section{Main result: " + LABELS[method] + "}\n" + core_table(result, True))
        pooled = result["pooled"]
        pieces.append("Pooled confusion counts: TP=" + str(pooled["tp"]) + ", FN=" + str(pooled["fn"]) + ", TN=" + str(pooled["tn"]) + ", FP=" + str(pooled["fp"]) + ".\n")
        pieces.append("AUROC=" + "%.4f" % pooled["auroc"] + "; trapezoidal AUPRC=" + "%.4f" % pooled["auprc"] + "; average precision=" + "%.4f" % pooled["average_precision"] + ".\n")
        errors = tables["main"][method][tables["main"][method].label != tables["main"][method].prediction]
        pieces.append("The pooled F1 " + ("exceeds" if pooled["f1"] > 0.85 else "does not exceed") + r" the proposal's 85\% numerical target. ")
        if len(errors) == 0:
            pieces.append("No main OOF speaker was misclassified at threshold 0.5. This is not evidence of perfect external detection.\n")
        else:
            pieces.append("Misclassified people: " + "; ".join(tex(row.speaker_id) + " (true label " + str(row.label) + ", score %.5f)" % row.probability for row in errors.itertuples(index=False)) + ".\n")
    baseline_metrics = saved["main"]["logistic_regression"]["pooled"]
    pieces.append(r"\subsection*{Answer to the matched-classifier question}")
    for method in ["random_forest", "xgboost"]:
        differences = []
        for name in ["f1", "auroc", "auprc"]:
            value = saved["main"][method]["pooled"][name]
            reference = baseline_metrics[name]
            if abs(value - reference) < 1e-12:
                relation = "ties"
            elif value > reference:
                relation = "exceeds"
            else:
                relation = "is below"
            differences.append(name.upper() + " " + relation + " the logistic baseline")
        pieces.append(LABELS[method] + ": " + "; ".join(differences) + ".\n")
    pieces.append("These are descriptive comparisons on the same 46 people, not evidence of a statistically established advantage. A more complex classifier is not automatically an improvement.\n")
    native_ids = set(tables["native16k"]["logistic_regression"].speaker_id)
    excluded_errors = []
    for method in METHODS:
        people = tables["main"][method]
        errors = people[(people.label != people.prediction) & (~people.speaker_id.isin(native_ids))]
        for row in errors.itertuples(index=False):
            excluded_errors.append(row.speaker_id)
    if excluded_errors:
        names = ", ".join(tex(name) for name in sorted(set(excluded_errors)))
        pieces.append("The native-rate cohort excludes main-error speaker(s) " + names + ". Consequently, its perfect score changes the people evaluated as well as the training data; it is not an independent confirmation on the original cohort.\n")
    if triggered:
        pieces.append(r"\clearpage\section{Conditional pipeline results}")
        for scenario in ["main", "native16k"]:
            pieces.append(r"\subsection*{" + tex(scenario) + "}" + core_table(saved[scenario]["conditional_fine_tuning"], True))
        pieces.append("These rows evaluate the predeclared training-side branch rule, not a fourth classifier chosen using outer-test scores.\n")
    pieces.append(r"\clearpage\section{Flag exclusion and native-rate retraining}")
    for scenario, title in [("without_sheet_flag", "Main OOF without flagged controls"), ("native16k", "Native-16-kHz-only retraining")]:
        pieces.append(r"\subsection*{" + title + "}")
        for method in METHODS:
            result = saved["main"][method]["without_sheet_flag"] if scenario == "without_sheet_flag" else saved[scenario][method]
            pieces.append(r"\textbf{" + LABELS[method] + "}" + core_table(result))
    pieces.append("Flag exclusion changes evaluation people only, not fitted models. Native-rate restriction genuinely retrains all models, but also changes the patient cohort. It is not independent confirmation and does not match microphones, rooms or recruitment conditions.\n")
    pieces.append(r"\clearpage\section{Exact partition sizes}")
    pieces.append(r"\begin{center}\small\begin{tabular}{lrrrr}\toprule Scenario/fold & Inner train & Validation & Final refit & Test\\\midrule")
    for scenario in ["main", "native16k"]:
        for fold in range(5):
            subset = counts[(counts.scenario == scenario) & (counts.fold == fold)]
            values = []
            for role in ["inner_0_train", "inner_0_validation", "outer_train", "outer_test"]:
                row = subset[subset.role == role].iloc[0]
                values.append(str(row.speakers) + "/" + str(row.recordings))
            pieces.append(tex(scenario) + "/" + str(fold) + " & " + " & ".join(values) + r"\\")
    pieces.append(r"\bottomrule\end{tabular}\end{center}")
    pieces.append("Entries are people/files. The displayed training/validation pair is inner split 0; splits 1 and 2 are also used for selection and exported separately. Final refit is their common complete development set. No single global validation fold is reused as a shortcut.\n")
    plot_notes = [
        ("partition_sizes", "Main inner split 0 and outer test sizes. Final refit joins inner train and validation."),
        ("fold_metrics", "All five held-out folds and the pooled diamond, for all three matched classifiers. No best fold is selected."),
        ("score_distributions", "Every main OOF person and score distribution. The dashed line is the unchanged threshold 0.5."),
        ("ranking_curves", "Pooled ROC and precision--recall curves. AUPRC is trapezoidal; average precision is named separately."),
        ("selection_curves", "Training-side grid results only. Grid candidate IDs and exact selected parameters are saved in selection records."),
        ("recording_burden", "Number and duration of recordings per person. More recordings do not create more independent participants."),
    ]
    for name, note in plot_notes:
        pieces.append(r"\clearpage\section{Statistical view: " + tex(name.replace("_", " ")) + "}\n" + r"\plot{" + name + "}{" + note + "}")
        pieces.append("This is a descriptive plot of saved internal records. It does not establish disease-specific mechanisms, calibration or external clinical performance.\n")
    pieces.append(r"""\clearpage\section{Leakage checks, provenance and remaining limits}
The saved-model replay recomputes all thirty frozen-classifier test predictions,
checks their pooled source, and recomputes training-side candidate selection.
It checks outer/inner identities, weighted training rows and logistic-scaler means.
All raw input hashes are rechecked. The predeclared first path at each native
sample rate is freshly re-embedded as a limited feature-reproduction check.
This is not a full fresh reproduction of all model training or all 773 embeddings.
REPLAY
The initial v4.0 check marked nine logistic scaler fits inconsistent because its
expected mean used the original float64 weights. The installed scikit-learn
first casts weights to the float32 input dtype, then accumulates in float64.
Follow-up audit v4.0-audit.1 reconstructs that operation within the unchanged
$10^{-10}$ scaler-check tolerances and replays all thirty saved models again.
Every check passes. Models, predictions, selected settings and extracted features
were not changed. The original failed \path{replay_checks.json} remains preserved;
\path{status.json} points to the separately saved resolution and corrected replay.
Speaker separation controls known repeat-session overlap. It does not make
recording conditions identical. The prior v2.0 audits found class-separated
recording dates and predictive static/low-energy acoustic cues. Standard WavLM
normalization does not prove that microphone, room or other cohort cues disappear.
Earlier development used the same people and results; a revised run does not
erase that adaptation. External clinical generalization remains unverified.
The appropriate description is speaker-disjoint and leakage-checked, not
unconditionally leakage-free. The external benchmark remains deferred.
\section{Reproduction and the relation to the proposal}
The proposal's WavLM representation and three classifiers are implemented.
The report's corrected elderly-only cohort, canonical identities and exact folds
replace the proposal's nominal 65-person description. A training-side gate replaces
the proposal's potentially test-dependent fine-tuning trigger. Explicit grid loops
replace GridSearchCV/Optuna wrappers; their validation results are saved in full.
Whisper and handcrafted acoustic models already exist in v2.0 but use different
tasks. Their historical scores are not presented as new matched WavLM comparisons.
This experiment does not update the official PatchTST result or claim new architecture novelty.
Task-specific recording error counts are supplied in \path{generated/task_error_debug.csv}.
They are debug observations, not independent speaker trials or headline accuracy.
Frozen run: \path{RUNPATH}.
Code and commands: \path{experiments/v4_0_wavlm_classifiers/README.md}.
The run contains settings, source snapshots, input fingerprints, exact split files,
all candidate validation scores, selected settings, fresh refitted models, weighted
fit rows, recording and speaker predictions, pooled metrics, fold variation and
conditional-branch decisions. No training occurs when this document is rebuilt.
\section*{References and evidence}
Dimauro and Girardi, Italian Parkinson's Voice and Speech, IEEE DataPort, 2019,
DOI 10.21227/aw6b-tg17. Dimauro et al., VoxTester, MeMeA 2016,
DOI 10.1109/MeMeA.2016.7533761. Dimauro et al., IEEE Access 2017,
DOI 10.1109/ACCESS.2017.2762475.
Chen et al., WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech
Processing, arXiv:2110.13900; pinned Microsoft WavLM-large checkpoint.
Project proposal: \path{Detecting Parkinson's Disease from Speech PP.pdf}.
Report template: \path{docs/patchtst_technical_report_v1_2.pdf}.
Prior interpretation: \path{docs/publication_readiness_audit.md}.
\clearpage\appendix\section{Directory key}
Each directory ID below reconstructs the exact project-relative file path in the
recording appendix. Repeat-session directories share the canonical person ID.
""".replace("RUNPATH", run.relative_to(PROJECT).as_posix()).replace("REPLAY", "All " + str(len(replay["saved_model_replays"])) + " saved classifier replays passed; " + str(replay["raw_recordings_rehashed"]) + " raw-file hashes matched. Conditional gates and their actual prediction branches were also replayed."))
    for row in folder_key.itertuples(index=False):
        first, remainder = row.folder.split("/", 1)
        pieces.append(r"\textbf{" + row.directory_id + "}: " + tex(row.speaker_id)
                      + r"\par\begingroup\footnotesize\raggedright\path{" + first + r"/}\\\path{" + remainder + r"}\par\endgroup")
    pieces.append(r"""\clearpage\section{Every exact recording and all five fold roles}
T = inner training, V = validation, E = outer test. Each three-character entry
lists the roles in inner splits 0/1/2 for that outer fold. For example, VTT means
validation in inner split 0 and training in inner splits 1 and 2. EEE means outer
test and exclusion from every inner split. Final refit includes every T/V person.
Every file is EEE exactly once. The native-rate scenario retains only the rows
marked with an asterisk; its separately exported inner splits may differ.
""")
    # Balanced explicit pages avoid oversized tables and sparse overflow pages.
    page_count = (len(catalogue) + 53) // 54
    rows_per_page = len(catalogue) // page_count
    extra_rows = len(catalogue) % page_count
    start = 0
    for page in range(page_count):
        count = rows_per_page + (1 if page < extra_rows else 0)
        if page:
            pieces.append(r"\clearpage\section*{Exact recordings and fold roles (continued)}")
        pieces.append(r"\begingroup\fontsize{8}{10}\selectfont\setlength{\tabcolsep}{3pt}\setlength{\parskip}{0pt}\begin{center}\begin{tabular}{lllrrrrr}")
        pieces.append(r"\toprule File ID & Dir & Exact filename & F0 & F1 & F2 & F3 & F4\\\midrule")
        for row in catalogue.iloc[start:start + count].itertuples(index=False):
            mark = "*" if row.native16k_eligible else ""
            roles = [getattr(row, "fold_" + str(fold) + "_roles") for fold in range(5)]
            pieces.append(row.file_id + mark + " & " + row.directory_id + r" & \path{" + row.file_name + "} & " + " & ".join(roles) + r"\\")
        pieces.append(r"\bottomrule\end{tabular}\end{center}\endgroup")
        start += count
    pieces.append(r"\end{document}")
    (HERE / "patchtst_style_wavlm.tex").write_text("\n".join(pieces), encoding="utf-8")


def make_markdown(run, saved, decisions):
    lines = ["# v4.0 WavLM proposal experiment results", "", "Completed run: `" + run.relative_to(PROJECT).as_posix() + "/`.", "",
        "This separate, revised internal experiment does not replace the official v2.0 PatchTST result. All classifiers use the same frozen WavLM-large recording embeddings and the same speech tasks and speaker folds.", "",
        "## Pooled speaker metrics", "", "All values are percentages. AUPRC is trapezoidal; AP is average precision.", "",
        "| Scenario | Classifier | People | F1 | Recall | Specificity | Precision | Accuracy | AUROC | AUPRC | AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scenario in ["main", "without_sheet_flag", "native16k"]:
        for method in METHODS:
            result = saved["main"][method]["without_sheet_flag"]["pooled"] if scenario == "without_sheet_flag" else saved[scenario][method]["pooled"]
            lines.append("| " + scenario + " | " + LABELS[method] + " | " + str(result["speakers"]) + " | " + " | ".join(percent(result[name]) for name in METRICS) + " |")
    for scenario in ["main", "native16k"]:
        lines += ["", "## " + scenario + " fold results", "", "| Classifier | Fold | People | F1 | Recall | Specificity | Precision | Accuracy |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            result = saved[scenario][method]
            for row in result["folds"]:
                lines.append("| " + LABELS[method] + " | " + str(row["fold"]) + " | " + str(row["speakers"]) + " | " + " | ".join(percent(row[name]) for name in METRICS[:5]) + " |")
            values = [percent(result["mean_sd"][name]["mean"]) + " +/- " + percent(result["mean_sd"][name]["sd"]) for name in METRICS[:5]]
            lines.append("| " + LABELS[method] + " | Mean/sample SD | — | " + " | ".join(values) + " |")
    lines += ["", "## Answer to the proposal question", ""]
    reference = saved["main"]["logistic_regression"]["pooled"]
    for method in ["random_forest", "xgboost"]:
        comparisons = []
        for name in ["f1", "auroc", "auprc"]:
            value = saved["main"][method]["pooled"][name]
            if abs(value - reference[name]) < 1e-12:
                relation = "ties"
            elif value > reference[name]:
                relation = "exceeds"
            else:
                relation = "is below"
            comparisons.append(name.upper() + " " + relation + " the logistic baseline")
        lines.append("- " + LABELS[method] + ": " + "; ".join(comparisons) + ".")
    lines += ["", "The sole missed patient in the main experiment, `PD_CFAERLEI54F`, is excluded from the native-rate cohort. The smaller-cohort perfect results are not independent confirmation on the same people. No classifier superiority is established by these descriptive comparisons."]
    lines += ["", "## Conditional fine-tuning", "", "Training-side baseline AUROC below 0.80 triggered " + str(sum(row["triggered"] for row in decisions)) + " of ten outer folds. Test scores never choose this branch.", "",
        "If no folds trigger, the conditional pipeline equals the frozen logistic baseline; it is not an independently trained fourth model.", "",
        "## Evidence and limitations", "", "All 30 saved classifier models, their inner selection and actual conditional branches passed replay. Every main raw WAV fingerprint matched. Fresh embedding replay covers the first path at each native sample rate, not all recordings.", "",
        "The original checker falsely flagged nine weighted scaler means because it ignored sklearn's float32 input-weight cast. Separately versioned v4.0-audit.1 reproduces that cast with unchanged tolerances; all checks pass and all models/predictions remain unchanged. The initial failure and its resolution are preserved separately.", "",
        "The main cohort contains 46 people and 773 recordings. Native-only retraining contains 39 people and 613 recordings. Flag exclusion evaluates 43 people without retraining. Young controls stay excluded. Recording-condition confounding and adaptation to previously inspected data remain unresolved; this is not external clinical validation.", "",
        "[Technical report](wavlm_v4_0_technical_report.pdf) · [Editable source](wavlm_v4_0_report/README.md) · [Interactive dashboard](wavlm_v4_0_report/generated/statistical_dashboard.html) · [All 80 partition CSVs](wavlm_v4_0_report/generated/fold_file_lists.zip)"]
    (PROJECT / "docs/v4_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    checked = check_saved_run(run)
    if checked is None:
        return False
    settings, saved, tables, replay = checked
    GENERATED.mkdir(parents=True, exist_ok=True)
    partitions = export_partitions(run)
    if partitions is None:
        return False
    catalogue, counts, folder_key, partition_paths = partitions
    make_plots(run, tables, saved, counts, catalogue)
    make_dashboard(catalogue, tables)
    recording_error_debug(run, settings["train"]["threshold"])
    decisions = json.loads((run / "fine_tuning_decisions.json").read_text())
    make_latex(run, settings, saved, tables, replay, catalogue, counts, folder_key, decisions)
    build = Path("/tmp/opencode/wavlm-v4-report-build")
    build.mkdir(parents=True, exist_ok=True)
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=" + str(build), "patchtst_style_wavlm.tex"]
    for pass_number in range(2):
        result = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
        (build / ("pass_" + str(pass_number) + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            print("ERROR: report compilation failed; inspect", build / ("pass_" + str(pass_number) + ".log"))
            return False
    output = PROJECT / "docs/wavlm_v4_0_technical_report.pdf"
    # Recolor vector objects to DeviceGray without rasterizing the text or plots.
    grayscale = build / "grayscale_report.pdf"
    pdf = pymupdf.open(build / "patchtst_style_wavlm.pdf")
    pdf.recolor(components=1)
    pdf.save(grayscale, garbage=4, deflate=True)
    pdf.close()
    shutil.copy2(grayscale, output)
    make_markdown(run, saved, decisions)
    paths = list(HERE.glob("*.py")) + [HERE / "patchtst_style_wavlm.tex", HERE / "README.md", PROJECT / "docs/v4_results.md"]
    paths += list(GENERATED.glob("*"))
    paths += list(run.glob("*.json")) + [run / "manifest.csv", run / "pooled_metrics.csv", run / "fold_mean_sd.csv"]
    status = json.loads((run / "status.json").read_text())
    if "verification_file" in status:
        paths += [PROJECT / status["verification_file"], PROJECT / status["resolution_file"]]
    paths += partition_paths
    for scenario in ["main", "native16k"]:
        for method in METHODS:
            paths += list((run / scenario / method).rglob("*.csv")) + list((run / scenario / method).rglob("*.json"))
            paths += list((run / scenario / method).rglob("*.joblib")) + list((run / scenario / method).rglob("*.ubj"))
        paths += list((run / scenario / "conditional_fine_tuning").rglob("*.json"))
        paths += list((run / scenario / "conditional_fine_tuning").rglob("*.csv"))
        paths += list((run / scenario / "conditional_fine_tuning").rglob("*.pt"))
    protocol = json.loads((run / "protocol.json").read_text())
    paths += [run / "source" / name for name in protocol["source_sha256"]]
    write_json(HERE / "report_manifest.json", {"document_version": "1.0", "experiment": "v4.0", "primary_run": run.relative_to(PROJECT).as_posix(),
        "pdf": output.relative_to(PROJECT).as_posix(), "pdf_sha256": fingerprint(output),
        "inputs_sha256": {path.relative_to(PROJECT).as_posix(): fingerprint(path) for path in sorted(set(paths)) if path.is_file()},
        "recordings": len(catalogue), "partition_csvs": len(partition_paths), "statistical_figures": 6, "pipeline_diagrams": 1,
        "interpretation": "Speaker-disjoint and leakage-checked; unresolved recording-condition confounding."})
    print("REPORT=" + output.relative_to(PROJECT).as_posix())
    return True


if __name__ == "__main__":
    if not main():
        sys.exit(1)
