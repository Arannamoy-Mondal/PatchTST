"""Create exact fold-file lists and descriptive plots from the frozen run."""

import html
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PARTITIONS = ["inner_train", "inner_validation", "outer_train", "outer_test"]
METRICS = ["f1", "recall", "specificity", "precision", "accuracy"]
METRIC_LABELS = ["F1", "Recall", "Specificity", "Precision", "Accuracy"]
FOLD_MARKERS = ["o", "s", "^", "v", "P"]
PLOTLY_MARKERS = ["circle", "square", "triangle-up", "triangle-down", "cross"]


def write_file_lists(run, generated):
    """Use saved inner splits, because validation membership varies by fold."""
    manifest = pd.read_csv(run / "main/arannamoy/manifest.csv")
    catalogue = manifest.sort_values(["folder", "file_name"]).reset_index(drop=True)
    catalogue.insert(0, "file_id", ["F%04d" % number for number in range(1, len(catalogue) + 1)])
    folders = catalogue[["folder", "speaker_id", "label"]].drop_duplicates().reset_index(drop=True)
    folders.insert(0, "folder_id", ["D%02d" % number for number in range(1, len(folders) + 1)])
    folder_ids = folders.set_index("folder").folder_id
    catalogue.insert(1, "folder_id", catalogue.folder.map(folder_ids))
    file_directory = generated / "fold_files"
    file_directory.mkdir(exist_ok=True)
    counts = []
    for fold in range(5):
        split = json.loads((run / "main/arannamoy" / ("fold_" + str(fold)) / "split.json").read_text())
        role_column = "fold_" + str(fold) + "_role"
        catalogue[role_column] = ""
        for partition, role in (("inner_train", "T"), ("inner_validation", "V"), ("outer_test", "E")):
            catalogue.loc[catalogue.speaker_id.isin(split[partition]), role_column] = role
        for partition in PARTITIONS:
            selected = catalogue[catalogue.speaker_id.isin(split[partition])]
            counts.append({
                "fold": fold,
                "partition": partition,
                "speakers": selected.speaker_id.nunique(),
                "pd_speakers": selected.loc[selected.label == 1, "speaker_id"].nunique(),
                "control_speakers": selected.loc[selected.label == 0, "speaker_id"].nunique(),
                "recordings": len(selected),
                "pd_recordings": int((selected.label == 1).sum()),
                "control_recordings": int((selected.label == 0).sum()),
            })
            columns = ["file_id", "folder_id", "path", "file_name", "speaker_id", "label", "task", "duration_s", "native_sr"]
            exported = selected[columns].copy()
            exported.insert(0, "evaluation_fold", fold)
            exported.insert(1, "partition", partition)
            exported.to_csv(file_directory / ("fold_" + str(fold) + "_" + partition + ".csv"), index=False)
        train_paths = set(catalogue.loc[catalogue[role_column] == "T", "path"])
        validation_paths = set(catalogue.loc[catalogue[role_column] == "V", "path"])
        test_paths = set(catalogue.loc[catalogue[role_column] == "E", "path"])
        refit_paths = set(catalogue.loc[catalogue.speaker_id.isin(split["outer_train"]), "path"])
        if (train_paths & validation_paths) or (train_paths & test_paths) or (validation_paths & test_paths):
            print("ERROR: expected disjoint train/validation/test file sets in fold", fold)
            return
        if (train_paths | validation_paths) != refit_paths or (refit_paths | test_paths) != set(catalogue.path):
            print("ERROR: expected refit = train + validation, and all 773 files covered in fold", fold)
            return
        if (catalogue[role_column] == "").any():
            print("ERROR: every recording needs a role in fold", fold)
            return
    counts = pd.DataFrame(counts)
    counts.to_csv(generated / "partition_counts.csv", index=False)
    folders.to_csv(generated / "folder_key.csv", index=False)
    columns = ["file_id", "folder_id", "path", "file_name", "speaker_id", "label", "task", "duration_s", "native_sr"]
    columns += ["fold_" + str(fold) + "_role" for fold in range(5)]
    catalogue = catalogue[columns]
    catalogue.to_csv(generated / "file_catalogue.csv", index=False)
    rows = []
    for fold in range(5):
        parts = counts[counts.fold == fold].set_index("partition")
        values = [str(fold)]
        for partition in PARTITIONS:
            values.extend([str(parts.loc[partition, "speakers"]), str(parts.loc[partition, "recordings"])])
        rows.append(" & ".join(values) + r" \\")
    (generated / "partition_rows.tex").write_text("\n".join(rows) + "\n")
    write_catalogue_appendix(catalogue, folders, generated)
    with ZipFile(generated / "fold_file_lists.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(file_directory.glob("*.csv")):
            archive.write(path, str(path.relative_to(generated)))
        for name in ("file_catalogue.csv", "folder_key.csv", "partition_counts.csv"):
            archive.write(generated / name, name)
        archive.writestr("README.txt", "Report 1.2 / v2.0 / Arannamoy / main\n\n"
                         "Each fold has four exact file lists.\n"
                         "inner_train: weight fitting during epoch selection.\n"
                         "inner_validation: speaker-level epoch selection.\n"
                         "outer_train: final refit, the union of inner train and validation.\n"
                         "outer_test: held-out OOF evaluation.\n\n"
                         "All paths are relative to the project root. No audio is included.\n"
                         "Catalogue roles: T = inner train, V = validation, E = test.\n"
                         "Final refit = T + V. Each of 773 files is test data once.\n")
    return catalogue, counts


def write_catalogue_appendix(catalogue, folders, generated):
    lines = [
        r"\section{Folder key for the exact recording lists}\label{app:folders}",
        r"Each directory below is relative to this corpus root:\\",
        r"\sourcepath{Italian Parkinson's Voice and speech/}\par",
        r"Join the root, the listed directory, a slash and the filename in Appendix~\ref{app:files} to obtain the exact WAV path. File IDs are report lookup labels; speaker IDs are the frozen evaluation identities.\par",
        r"{\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}",
        r"\begin{longtable}{@{}lL{12.8cm}@{}}",
        r"\toprule Folder & Corpus-relative directory and speaker identity\\\midrule\endfirsthead",
        r"\toprule Folder & Corpus-relative directory and speaker identity (continued)\\\midrule\endhead",
        r"\midrule\multicolumn{2}{r}{Continued on the next page}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    for row in folders.itertuples():
        label = "PD" if row.label == 1 else "Elderly control"
        directory = str(Path(row.folder).relative_to("Italian Parkinson's Voice and speech"))
        lines.append(row.folder_id + r" & \path{" + directory + r"}\newline " + label + r"; \path{" + row.speaker_id + r"} \\")
    lines.extend([
        r"\end{longtable}}\clearpage",
        r"\section{Every recording's role in every fold}\label{app:files}",
        r"This is the complete 773-file catalogue. Read down a fold column to find its exact training, validation and test files. \textbf{T = inner training; V = inner validation; E = held-out test.} Final refit training is the union of T and V in that column. The same file can have a different role in another fold, but is tested exactly once. Folder IDs are resolved in Appendix~\ref{app:folders}.",
        r"{\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.05}",
        r"\begin{longtable}{@{}llL{6.6cm}ccccc@{}}",
        r"\toprule File ID & Folder & Exact filename & F0 & F1 & F2 & F3 & F4\\\midrule\endfirsthead",
        r"\multicolumn{8}{l}{\textbf{T: inner train\quad V: validation\quad E: test\quad Final refit: T + V}}\\[3pt]",
        r"\toprule File ID & Folder & Exact filename & F0 & F1 & F2 & F3 & F4\\\midrule\endhead",
        r"\midrule\multicolumn{8}{r}{Continued on the next page}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ])
    for row in catalogue.to_dict("records"):
        values = [row["file_id"], row["folder_id"], r"\path{" + row["file_name"] + "}"]
        values += [row["fold_" + str(fold) + "_role"] for fold in range(5)]
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\end{longtable}}")
    (generated / "file_appendix.tex").write_text("\n".join(lines) + "\n")


def finish_plot(figure, name, generated):
    figure.savefig(generated / "figures" / (name + ".pdf"), bbox_inches="tight")
    figure.savefig(generated / "figures" / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_statistical_plots(run, generated, predictions, counts, recorded):
    """Plot descriptive summaries; people, not recordings, are the scoring unit."""
    (generated / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "0.25", "axes.linewidth": 0.7,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "grid.color": "0.88", "grid.linewidth": 0.6, "pdf.fonttype": 42,
    })
    interactive = []
    x = np.arange(5)
    partitions = ["inner_train", "inner_validation", "outer_test"]
    labels = ["Inner train", "Validation", "Test"]
    fills = ["0.25", "0.68", "white"]
    hatches = ["", "//", ".."]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    web = make_subplots(rows=1, cols=2, subplot_titles=["People per partition", "Recordings per partition"])
    for panel, (measure, title) in enumerate((("speakers", "People"), ("recordings", "Recordings"))):
        ax = axes[panel]
        for index, partition in enumerate(partitions):
            values = counts[counts.partition == partition].sort_values("fold")[measure].to_numpy()
            bars = ax.bar(x + (index - 1) * 0.25, values, width=0.23, color=fills[index], edgecolor="black", hatch=hatches[index], linewidth=0.6, label=labels[index])
            ax.bar_label(bars, fontsize=7, padding=3, rotation=90 if panel == 1 else 0)
            web.add_trace(go.Bar(x=x, y=values, name=labels[index], legendgroup=partition,
                                showlegend=panel == 0, marker=dict(color=fills_for_web(index), line=dict(color="black", width=1)),
                                text=values, textposition="outside"), row=1, col=panel + 1)
        ax.set(xticks=x, xlabel="Outer fold", ylabel=title, ylim=(0, counts[counts.partition.isin(partitions)][measure].max() * 1.22))
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.set_title(title + " per partition", loc="left")
        web.update_yaxes(range=[0, counts[counts.partition.isin(partitions)][measure].max() * 1.22], title_text=title, row=1, col=panel + 1)
        web.update_xaxes(title_text="Outer fold", tickvals=x, row=1, col=panel + 1)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, fontsize=8, loc="outside upper center", ncol=3)
    finish_plot(figure, "partition_sizes", generated)
    web.update_layout(barmode="group", title="Exact split sizes · people and recordings")
    interactive.append(("partition-sizes", web))

    # Show every fold rather than selecting the most successful one.
    figure, ax = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
    web = go.Figure()
    positions = np.arange(5)
    for fold in range(5):
        values = [100 * recorded["folds"][fold][name] for name in METRICS]
        offset = (fold - 2) * 0.12
        ax.scatter(values, positions + offset, marker=FOLD_MARKERS[fold], s=35, facecolor="white", edgecolor="black", linewidth=0.9, label="Fold " + str(fold), zorder=3)
        web.add_trace(go.Scatter(x=values, y=positions + offset, mode="markers", name="Fold " + str(fold), marker=dict(symbol=PLOTLY_MARKERS[fold], color="black", size=9), customdata=METRIC_LABELS, hovertemplate="%{customdata}: %{x:.2f}%<extra>%{fullData.name}</extra>"))
    pooled = [100 * recorded["pooled"][name] for name in METRICS]
    ax.scatter(pooled, positions + 0.40, marker="D", color="black", s=40, label="Pooled", zorder=4)
    web.add_trace(go.Scatter(x=pooled, y=positions + 0.4, mode="markers", name="Pooled", marker=dict(symbol="diamond", color="black", size=11), customdata=METRIC_LABELS, hovertemplate="%{customdata}: %{x:.2f}%<extra>Pooled</extra>"))
    ax.set(yticks=positions, yticklabels=METRIC_LABELS, xlim=(0, 105), xlabel="Speaker-level score (%)")
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.legend(ncol=3, loc="lower left", frameon=False, fontsize=8)
    ax.set_title("All five folds and the pooled result", loc="left")
    finish_plot(figure, "fold_metric_distribution", generated)
    web.update_layout(title="Five speaker-level metrics · all folds and pooled", xaxis=dict(range=[0, 105], title="Score (%)"), yaxis=dict(tickvals=positions, ticktext=METRIC_LABELS, autorange="reversed"))
    interactive.append(("fold-metrics", web))

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), layout="constrained")
    web = make_subplots(rows=1, cols=2, subplot_titles=["Individual OOF speaker scores", "Empirical cumulative distributions"])
    probability_summary = []
    for label, title in ((0, "Control"), (1, "PD")):
        group = predictions[predictions.label == label].sort_values(["fold", "speaker_id"])
        scores = group.probability.to_numpy()
        axes[0].boxplot([scores], positions=[label], widths=0.4, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5), boxprops=dict(color="black"),
                        whiskerprops=dict(color="black"), capprops=dict(color="black"))
        for fold in range(5):
            people = group[group.fold == fold]
            jitter = np.linspace(-0.13, 0.13, len(people)) + (fold - 2) * 0.012
            axes[0].scatter(label + jitter, people.probability, marker=FOLD_MARKERS[fold], s=25, edgecolor="black", facecolor="white" if label == 0 else "black", linewidth=0.8, zorder=3)
            web.add_trace(go.Scatter(x=label + jitter, y=people.probability, mode="markers", name=title + ", fold " + str(fold),
                                    showlegend=False, marker=dict(symbol=PLOTLY_MARKERS[fold] + ("-open" if label == 0 else ""), color="black", size=8), customdata=people[["speaker_id", "fold", "prediction"]].to_numpy(),
                                    hovertemplate="%{customdata[0]}<br>Fold %{customdata[1]}<br>OOF score %{y:.7f}<br>Prediction %{customdata[2]}<extra>%{fullData.name}</extra>"), row=1, col=1)
        ordered = np.sort(scores)
        cumulative = np.arange(1, len(scores) + 1) / len(scores)
        line_style = "--" if label == 0 else "-"
        axes[1].step(np.r_[0, ordered, 1], np.r_[0, cumulative, 1], where="post", linestyle=line_style, color="black", linewidth=1.3, label=title + " (n=" + str(len(scores)) + ")")
        web.add_trace(go.Scatter(x=np.r_[0, ordered, 1], y=np.r_[0, cumulative, 1], mode="lines", name=title + " ECDF", line=dict(color="black", dash="dash" if label == 0 else "solid", shape="hv")), row=1, col=2)
        probability_summary.append({"class": title, "speakers": len(scores), "minimum": scores.min(), "q1": np.quantile(scores, 0.25), "median": np.median(scores), "q3": np.quantile(scores, 0.75), "maximum": scores.max()})
    axes[0].axhline(0.5, linestyle="--", color="0.35", linewidth=0.9)
    axes[0].set(xticks=[0, 1], xticklabels=["Control\n(n=21)", "PD\n(n=25)"], ylabel="OOF speaker score", ylim=(-0.04, 1.04), xlim=(-0.55, 1.55))
    axes[0].set_title("Scores by true class", loc="left")
    axes[1].set(xlabel="OOF speaker score", ylabel="Fraction at or below score", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[1].legend(frameon=False, fontsize=8, loc="center left")
    axes[1].set_title("Cumulative distribution", loc="left")
    for ax in axes:
        ax.grid(axis="y")
    finish_plot(figure, "score_distributions", generated)
    web.add_hline(y=0.5, line_dash="dash", line_color="#555555", row=1, col=1)
    web.update_xaxes(tickvals=[0, 1], ticktext=["Control", "PD"], range=[-0.55, 1.55], row=1, col=1)
    web.update_yaxes(title_text="OOF speaker score", range=[-0.04, 1.04], row=1, col=1)
    web.update_xaxes(title_text="OOF speaker score", row=1, col=2)
    web.update_yaxes(title_text="Fraction at or below score", row=1, col=2)
    web.update_layout(title="OOF score distributions · 46 people, shared CV training sets")
    interactive.append(("score-distributions", web))
    pd.DataFrame(probability_summary).to_csv(generated / "score_summary.csv", index=False)
    rows = []
    for row in probability_summary:
        values = [row["class"], str(row["speakers"])] + ["%.4g" % row[name] for name in ("minimum", "q1", "median", "q3", "maximum")]
        rows.append(" & ".join(values) + r" \\")
    (generated / "score_summary_rows.tex").write_text("\n".join(rows) + "\n")

    false_positive_rate, true_positive_rate, _ = roc_curve(predictions.label, predictions.probability)
    precision, recall, _ = precision_recall_curve(predictions.label, predictions.probability)
    auc = roc_auc_score(predictions.label, predictions.probability)
    average_precision = average_precision_score(predictions.label, predictions.probability)
    pooled_metrics = recorded["pooled"]
    prevalence = predictions.label.mean()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
    axes[0].plot(false_positive_rate, true_positive_rate, color="black", linewidth=1.5)
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="0.5", linewidth=1)
    axes[0].scatter([1 - pooled_metrics["specificity"]], [pooled_metrics["recall"]], marker="*", s=110, color="black", zorder=4)
    axes[0].annotate("Fixed 0.5 threshold", (0, 0.96), xytext=(0.16, 0.77), arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    axes[0].set(xlabel="False-positive rate (1 − specificity)", ylabel="Recall", title="ROC · AUROC = %.3f" % auc)
    axes[1].step(recall, precision, where="post", color="black", linewidth=1.5)
    axes[1].axhline(prevalence, linestyle=":", color="0.5", linewidth=1)
    axes[1].scatter([pooled_metrics["recall"]], [pooled_metrics["precision"]], marker="*", s=110, color="black", zorder=4)
    axes[1].annotate("Fixed 0.5 threshold", (0.96, 1), xytext=(0.34, 0.79), arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    axes[1].text(0.06, prevalence + 0.04, "Sample PD fraction: 25/46", fontsize=8)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall · AP = %.3f" % average_precision)
    for ax in axes:
        ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04))
        ax.grid(alpha=0.7)
    finish_plot(figure, "roc_precision_recall", generated)
    web = make_subplots(rows=1, cols=2, subplot_titles=["ROC · AUROC = %.3f" % auc, "Precision–recall · AP = %.3f" % average_precision])
    web.add_trace(go.Scatter(x=false_positive_rate, y=true_positive_rate, mode="lines", name="ROC", line=dict(color="black")), row=1, col=1)
    web.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision–recall", line=dict(color="black", shape="hv")), row=1, col=2)
    for column, px, py in ((1, 1 - pooled_metrics["specificity"], pooled_metrics["recall"]), (2, pooled_metrics["recall"], pooled_metrics["precision"])):
        web.add_trace(go.Scatter(x=[px], y=[py], mode="markers", name="Fixed threshold 0.5", showlegend=column == 1, marker=dict(symbol="star", color="black", size=15)), row=1, col=column)
    web.update_xaxes(title_text="False-positive rate", row=1, col=1)
    web.update_xaxes(title_text="Recall", row=1, col=2)
    web.update_yaxes(title_text="Recall", row=1, col=1)
    web.update_yaxes(title_text="Precision", row=1, col=2)
    web.update_xaxes(range=[-0.04, 1.04])
    web.update_yaxes(range=[-0.04, 1.04])
    web.update_layout(title="Pooled OOF ranking · the operating threshold stays at 0.5")
    interactive.append(("ranking-curves", web))
    (generated / "ranking_summary.json").write_text(json.dumps({"auroc": auc, "average_precision": average_precision, "pd_fraction": prevalence, "threshold": 0.5, "interpretation": "Descriptive pooled OOF ranking; no threshold selection or external confidence interval."}, indent=2) + "\n")

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), layout="constrained")
    web = make_subplots(rows=2, cols=3, subplot_titles=["Fold " + str(fold) for fold in range(5)] + [""])
    learning_rows = []
    for fold in range(5):
        directory = run / "main/arannamoy" / ("fold_" + str(fold))
        history = pd.read_json(directory / "selection_history.json")
        selected = json.loads((directory / "selection.json").read_text())["selected_epochs"]
        web.layout.annotations[fold].text = "Fold %d · chosen epoch %d" % (fold, selected)
        history["fold"] = fold
        history["selected"] = history.epoch == selected
        learning_rows.append(history)
        ax = axes.flat[fold]
        for column, label, style in (("train_loss", "Weighted train", "--"), ("val_speaker_loss", "Speaker validation", "-")):
            ax.plot(history.epoch, history[column], linestyle=style, color="black", linewidth=1.1, label=label)
            web.add_trace(go.Scatter(x=history.epoch, y=history[column], mode="lines+markers", name=label, legendgroup=column, showlegend=fold == 0, line=dict(color="black", dash="dash" if style == "--" else "solid"), marker=dict(size=4)), row=fold // 3 + 1, col=fold % 3 + 1)
        chosen_loss = float(history.loc[history.epoch == selected, "val_speaker_loss"].iloc[0])
        ax.scatter([selected], [chosen_loss], marker="o", s=45, facecolor="white", edgecolor="black", zorder=4)
        ax.axvline(selected, color="0.5", linestyle=":", linewidth=0.9)
        ax.set(title="Fold %d · chosen epoch %d" % (fold, selected), xlabel="Epoch", ylabel="Loss (log scale)", yscale="log")
        ax.grid(axis="y", which="major")
        web.add_trace(go.Scatter(x=[selected], y=[chosen_loss], mode="markers", name="Chosen epoch", legendgroup="chosen", showlegend=fold == 0, marker=dict(symbol="circle-open", color="black", size=13)), row=fold // 3 + 1, col=fold % 3 + 1)
    axes.flat[5].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, legend_labels, frameon=False, loc="upper left", fontsize=9)
    axes.flat[5].text(0.02, 0.54, "Circle: selected validation epoch\n\nSelection uses training-side people.\nOuter-test loss is never used.\n\nDifferent units and weighting:\ntrain windows vs validation people.", transform=axes.flat[5].transAxes, fontsize=8, va="top", linespacing=1.6)
    finish_plot(figure, "epoch_selection_curves", generated)
    web.update_yaxes(type="log", title_text="Loss (log scale)", dtick=1, tickformat=".0e")
    web.update_xaxes(title_text="Epoch")
    web.update_layout(title="Saved inner-selection histories · five folds", height=640)
    interactive.append(("learning-curves", web))
    pd.concat(learning_rows, ignore_index=True).to_csv(generated / "selection_histories.csv", index=False)

    manifest = pd.read_csv(run / "main/arannamoy/manifest.csv")
    speaker_audio = manifest.groupby(["speaker_id", "label"], as_index=False).agg(recordings=("path", "size"), duration_s=("duration_s", "sum"))
    speaker_audio["audio_minutes"] = speaker_audio.duration_s / 60
    speaker_audio.to_csv(generated / "speaker_audio_summary.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), layout="constrained")
    web = make_subplots(rows=1, cols=2, subplot_titles=["Recordings per speaker", "Audio minutes per speaker"])
    for panel, (column, title) in enumerate((("recordings", "Recordings per speaker"), ("audio_minutes", "Audio minutes per speaker"))):
        ax = axes[panel]
        for label, name in ((0, "Control"), (1, "PD")):
            people = speaker_audio[speaker_audio.label == label].sort_values([column, "speaker_id"])
            jitter = np.random.default_rng(42 + label).uniform(-0.16, 0.16, len(people))
            ax.boxplot([people[column]], positions=[label], widths=0.5, showfliers=False, medianprops=dict(color="black", linewidth=1.5))
            ax.scatter(label + jitter, people[column], s=20, edgecolor="black", facecolor="white" if label == 0 else "black", linewidth=0.6, zorder=3)
            web.add_trace(go.Box(y=people[column], name=name, legendgroup=name, showlegend=panel == 0, boxpoints="all", jitter=0.3, pointpos=0, line=dict(color="black"), fillcolor="white" if label == 0 else "#d0d0d0", customdata=people.speaker_id, hovertemplate="%{customdata}<br>%{y:.3f}<extra>" + name + "</extra>"), row=1, col=panel + 1)
        ax.set(xticks=[0, 1], xticklabels=["Control (n=21)", "PD (n=25)"], ylabel=title, ylim=(0, speaker_audio[column].max() * 1.15), xlim=(-0.55, 1.55))
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
    finish_plot(figure, "recording_burden", generated)
    web.update_layout(title="Recording availability · one point per person")
    interactive.append(("recording-availability", web))
    return interactive


def fills_for_web(index):
    return ["#404040", "#adadad", "#ffffff"][index]


def write_interactive_dashboard(interactive, catalogue, generated):
    sections = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PatchTST — OOF statistics and exact fold files</title>
<style>body{max-width:1160px;margin:40px auto;padding:0 22px;color:#171717;background:white;font:16px/1.6 Georgia,serif}h1{line-height:1.2}h2{border-top:1px solid #aaa;padding-top:24px}a{color:#222}nav{display:flex;flex-wrap:wrap;gap:8px 22px}.note{border-left:3px solid #444;padding-left:16px}section{margin:32px 0}.chart-scroll{overflow-x:auto}.plot-frame{min-width:960px;width:100%}.plotly-graph-div{width:100%}label{display:inline-block;margin:8px 16px 8px 0}select,input{font:inherit;padding:6px;max-width:95%}table{border-collapse:collapse;width:100%;min-width:750px;font:12px/1.5 system-ui,sans-serif}th,td{border-bottom:1px solid #ddd;padding:7px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#eee}td.path{overflow-wrap:anywhere}#table-wrap{max-height:680px;overflow:auto}code{font-size:.9em;overflow-wrap:anywhere}.caption{font-size:14px;color:#444}</style></head><body>
<h1>PatchTST: OOF statistics and exact fold files</h1>
<p>Report 1.2 · v2.0 / Arannamoy / main · 46 people · 773 recordings</p>
<p class="note">All figures use saved records. Each speaker score comes from that person's held-out model. Descriptive fold spread and ranking curves are not external-generalization confidence intervals. Recording-condition confounding remains unresolved.</p>
<p><a href="../../patchtst_technical_report_v1_2.pdf">Read the extended PDF report</a> · <a href="fold_file_lists.zip" download>Download all exact fold-file CSVs (ZIP)</a></p>
<p class="caption">Hover for exact values. On a small screen, scroll each chart or file table sideways to view the full panel.</p>
<nav>"""]
    for section_id, figure in interactive:
        sections.append('<a href="#' + section_id + '">' + html.escape(section_id.replace("-", " ").title()) + '</a>')
    sections.append('<a href="#files">Exact fold files</a></nav>')
    captions = {
        "partition-sizes": "Final refit training is inner train plus validation. File counts describe recording availability; classification metrics still use one row per person.",
        "fold-metrics": "Each marker is one fold, and the diamond is the pooled result. Small vertical offsets reveal equal scores. Fold spread is descriptive, not an external-generalization confidence interval.",
        "score-distributions": "The dashed threshold stays at 0.5. Open markers are controls; filled markers are patients. Horizontal jitter only separates points. The cumulative curves summarize all scores without selecting a new threshold.",
        "ranking-curves": "The stars mark the fixed 0.5 threshold. AUROC and average precision are both 1.000, but the fixed threshold misses one patient. These are pooled OOF ranking diagnostics, not clinical calibration.",
        "learning-curves": "Circles identify the selected epochs. Training loss is weighted over windows; validation loss is averaged over speakers. Different units and weights make the gap unsuitable as a direct overfitting measure. No outer-test curve is used for selection.",
        "recording-availability": "Every dot is one person. Duration includes the full recording, not verified speech-only time. Extra recordings do not create extra independent test participants.",
    }
    for number, (section_id, figure) in enumerate(interactive):
        figure.update_layout(template="plotly_white", font=dict(family="Georgia, serif", size=13, color="#171717"), paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=65, r=30, t=85, b=65), legend=dict(orientation="h", y=-0.2), height=640 if section_id == "learning-curves" else 440)
        figure.update_xaxes(gridcolor="#e5e5e5", zerolinecolor="#bbbbbb")
        figure.update_yaxes(gridcolor="#e5e5e5", zerolinecolor="#bbbbbb")
        sections.append('<section id="' + section_id + '">')
        sections.append('<div class="chart-scroll"><div class="plot-frame">')
        sections.append(pio.to_html(figure, full_html=False, include_plotlyjs=number == 0, div_id=section_id + "-plot", config={"responsive": True, "displaylogo": False, "toImageButtonOptions": {"format": "svg"}}))
        sections.append('</div></div><p class="caption">' + captions[section_id] + '</p>')
        sections.append('</section>')
    sections.append("""<section id="files"><h2>Search exact train, validation and test files</h2>
<p>T = inner train; V = validation; E = held-out test. Final refit = T + V. Select one evaluation fold, then filter the role. Paths are relative to the project root.</p>
<label>Fold <select id="fold"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></label>
<label>Role <select id="role"><option value="all">All recordings</option><option value="T">Inner train</option><option value="V">Validation</option><option value="E">Test</option><option value="refit">Final refit train</option></select></label>
<label>Search <input id="search" type="search" placeholder="Speaker ID, task or filename"></label>
<p id="count"></p><div id="table-wrap"><table><thead><tr><th>File</th><th>Role</th><th>Speaker</th><th>Class</th><th>Exact path</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<p>Source: <code>artifacts/runs/20261012T183328Z_f-all_cead14cc32f1/main/arannamoy/</code>. Full CSV lists accompany the report under <code>generated/fold_files/</code>. This page includes Plotly locally and works offline.</p>
<script id="file-data" type="application/json">""")
    columns = ["file_id", "speaker_id", "label", "path", "task"] + ["fold_" + str(fold) + "_role" for fold in range(5)]
    sections.append(catalogue[columns].to_json(orient="records").replace("<", "\\u003c"))
    sections.append("""</script><script>
const records = JSON.parse(document.getElementById('file-data').textContent);
function updateFiles() {
  const fold = document.getElementById('fold').value;
  const role = document.getElementById('role').value;
  const search = document.getElementById('search').value.toLowerCase();
  const selected = records.filter(row => {
    const actual = row['fold_' + fold + '_role'];
    const matchesRole = role === 'all' || actual === role || (role === 'refit' && actual !== 'E');
    return matchesRole && (row.path + ' ' + row.speaker_id + ' ' + row.task + ' ' + row.file_id).toLowerCase().includes(search);
  });
  const body = document.getElementById('rows');
  body.replaceChildren();
  for (const row of selected) {
    const line = document.createElement('tr');
    const cells = [row.file_id, row['fold_' + fold + '_role'], row.speaker_id, row.label === 1 ? 'PD' : 'Control', row.path];
    for (let i = 0; i < cells.length; i++) {
      const cell = document.createElement('td');
      cell.textContent = cells[i];
      if (i === 4) cell.className = 'path';
      line.appendChild(cell);
    }
    body.appendChild(line);
  }
  const people = new Set(selected.map(row => row.speaker_id));
  document.getElementById('count').textContent = selected.length + ' recordings · ' + people.size + (people.size === 1 ? ' speaker' : ' speakers');
}
for (const id of ['fold', 'role', 'search']) document.getElementById(id).addEventListener('input', updateFiles);
updateFiles();
</script></body></html>""")
    (generated / "statistical_dashboard.html").write_text("\n".join(sections))


def build_assets(run, generated, predictions, recorded):
    file_lists = write_file_lists(run, generated)
    if file_lists is None:
        return False
    catalogue, counts = file_lists
    interactive = build_statistical_plots(run, generated, predictions, counts, recorded)
    write_interactive_dashboard(interactive, catalogue, generated)
    print("PASS: exact file lists cover all five folds; wrote 20 partition CSVs and six statistical figures.")
    return True
