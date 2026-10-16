# High-score recheck and publication assessment

Audit version: **v2.0-audit.2**, parent **v2.0**. Reviewed 12 October 2026.

Evidence directory:
`artifacts/runs/20261012T215648Z_v2.0-audit.2_2167119cc167/`.

## Assessment

The reported predictions reproduce. The strongest explanation supported by the
evidence is that the two cohorts are acoustically easy to separate, including
through static spectra and recording-related cues. The evidence does not identify
how much of that separation is caused by Parkinson's disease versus acquisition
conditions. A transformer is not necessary to obtain the high main score.

A publication is possible in principle. The present evidence is better suited
to a carefully developed evaluation/shortcut-learning study than to a claim of
a clinically validated new Parkinson's detector. Acceptance and journal quartile
cannot be inferred from the classification score.

The earlier categorical statement that publication was impossible without a
second dataset was too strong. The necessary evidence depends on the claim.
A bounded audit of one corpus can be informative; a disease-generalization claim
requires evidence that the current experiment does not provide. The external
benchmark remains the user's deferred phase.

## What was checked in this revision

- Replayed all **30 saved outer models**, covering all three members and both
  scenarios. Every saved speaker probability reproduced within `1e-5`.
  The largest PatchTST discrepancy was approximately `2.9e-8`.
- Rechecked outer/inner speaker separation, fixed thresholds, native-rate
  filtering, logistic-model input names and training-scaler row counts.
- Independently recomputed all six main/native pooled metric sets from saved
  speaker predictions and checked all ten PatchTST epoch selections/refit lengths.
- Verified all registered v2.0, v2.0-audit.1 and v3.0 artifact fingerprints, and
  the archived v2.0 source against its original protocol.
- Rehashed all **773 main-cohort WAVs** and their decoded integer samples.
  All matched the prior evidence; no identical decoded waveform was assigned
  to different canonical people. This does not rule out all near-duplicates
  or unknown identity errors.
- Refit six existing static/low-energy controls on their original training
  folds; their saved predictions reproduced.
- Traced the change from the 46-person to the 39-person cohort, compared
  PatchTST with the static baseline person by person, and checked recording dates.

`model_replay.json`, `control_replays.json` and `findings.json` contain these
checks. Checkpoint replay is not a fresh reproduction of the complete training
process. The original v2.0 trained result remains the primary result.

### Feature-recomputation follow-up

The initial check found differences up to `0.0000152588` dB between newly
computed spectrograms and cached arrays, exceeding the audit's initial absolute
tolerance of `0.00001` dB. This initial failure is retained in `findings.json`.

A focused investigation reproduced the difference by changing only the number
of BLAS arithmetic threads from one to four. The active and archived extraction
functions agreed exactly in that comparison. The follow-up evidence is recorded
separately in `feature_rounding_resolution.json`, together with full raw-audio
feature checks and checkpoint replays. **All 773 feature arrays matched the
cache bit for bit with one BLAS thread, and all ten PatchTST models reproduced
their saved decisions from the newly computed features.** The initial discrepancy
is resolved as arithmetic-backend rounding. This numerical check concerns
floating-point reproducibility, not an identified label-dependent preprocessing step.

## Why the score can be so high

### 1. Static acoustic information is sufficient on this split

All results below are pooled **speaker-level** percentages on the same 46 people.
The perturbation rows come from v2.0-audit.1, whose archived code and predictions
were inspected; the static control was refitted again in this revision.

| Method or diagnostic | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| v2.0 / Arannamoy / main | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 |
| Saved PatchTST, shuffled frames within each window | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 |
| Saved PatchTST, constant time-averaged windows | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 |
| Static-spectrum logistic control | 98.04 | 100.00 | 95.24 | 96.15 | 97.83 |
| Lowest-energy-frame spectrum logistic control | 95.83 | 92.00 | 100.00 | 100.00 | 95.65 |

The static classifier and PatchTST each classify **45/46 people correctly**.
They agree correctly on 44 people; each gets one person right whom the other
gets wrong. The small F1 difference reflects false-positive versus false-negative
errors, not an established performance advantage.

All 46 PatchTST decisions survive removal of within-window temporal variation;
probabilities do change. This supports a limited inference: original temporal
detail inside those windows is unnecessary for these decisions. It does not
prove the network never uses time or that static features cannot contain genuine
disease information.

The v2.0 changes preserved absolute digital levels and mel-band identity that
v1.0 suppressed. Those changes make both genuine voice cues and recording
signatures more accessible. Several things changed together, including the
cohort and evaluation procedure, so the v1-to-v2 score increase cannot be
attributed to a single architectural improvement.

### 2. Speaker separation does not imply recording-condition separation

The native-16-kHz data have only two PD recording dates, 24 January and
10 February 2017. Controls were recorded on five dates in March and April 2017.
Every date contains only one class, and **all five speaker folds share recording
dates between training and testing**.

Dates are metadata, not network inputs. Their importance is that microphone
response, gain, environment or recruitment could be class-correlated and audible.
The model can recognize an acoustic recording signature without reading a date.
The recordings alone do not establish which of those possible mechanisms occurred.

On the native subset, the median speaker-level quietest-frame measurement is
`-66.77 dBFS` for controls and `-50.34 dBFS` for PD, a difference of **16.43 dB**.
A fixed-C logistic classifier using only this one feature achieves:

| Diagnostic / native16k / 39 speakers | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| Quietest-frame digital level only | 91.89 | 94.44 | 90.48 | 89.47 | 92.31 |
| Lowest-energy-frame spectrum | 94.44 | 94.44 | 95.24 | 94.44 | 94.87 |

These frames have not been verified as speech-free silence. They can contain
soft speech, breathing and noise. This is evidence of simple predictive cues,
not proof that background noise causes all of the score.

### 3. The 100% native-rate result also changes the people being evaluated

The main model misses one patient, `PD_CFAERLEI54F`, at probability `0.4159119`.
That person is among the seven patients removed by the native-rate restriction.

If the original main OOF predictions are simply restricted to the retained
39 people, **F1, recall, specificity, precision and accuracy are already all
100%**, before any native-only retraining. Those original predictions still use
the main training protocol and are only a descriptive cohort check.

The separate native-only retraining genuinely excludes all 44.1-kHz audio and
also produces perfect decisions. However, its perfect score is not evidence of
an improvement on the same people, and it is not an independent replication.
Matching sample rates addresses one acquisition difference; it does not match
recording conditions or establish the causal source of the original score.

### 4. Aggregation stabilizes decisions but does not increase participant count

The main model averages probabilities within each recording and then equally
across a person's recordings. Each person contributes 10-32 recordings, median
16. Aggregation can reduce random variation while preserving any consistently
class-correlated disease or acquisition cues.

The experimental sample is still **46 people**, with only 25 patients and 21
controls. There are not 773 independent patients, regardless of window count.

For perspective, an ordinary independent-binomial 95% exact interval for
24/25 sensitivity is approximately **79.65-99.90%**; for 21/21 specificity it is
**83.89-100%**. Even 18/18 native-rate sensitivity corresponds to approximately
**81.47-100%** under that idealized calculation.

These are sample-size illustrations, **not confidence intervals for this adaptive
cross-validation procedure or for external deployment**. Shared CV training
sets, earlier development decisions and dataset shift are not captured. A naive
bootstrap of all-correct predictions can return a misleading zero-width interval.

Main PatchTST fold F1 remains `97.78 +/- 4.97%` (sample SD). Full five-metric fold
variation is in [the original report](corrected_results.md#main-fold-mean-and-sample-standard-deviation).
Fold SD is not an external-generalization confidence interval.

### 5. Internal development and random-label controls answer different questions

Earlier results informed later design decisions on this same corpus. Inner
validation protects epoch/C selection within the rerun; it does not erase
researcher adaptation to previously inspected results.

The existing full PatchTST person-label permutation yielded F1 53.57%, recall
60.00%, specificity 23.81%, precision 48.39% and accuracy 43.48%, with AUROC 0.44.
This makes an implementation that automatically obtains high scores less likely.
It cannot separate disease cues from recording cues in the real labels. One
permutation is not a formal significance test, and F1 has no universal 50% null.

## Publication routes

| Target | Assessment of the current evidence | What would strengthen the contribution |
|---|---|---|
| Specialist speech/biomedical conference or relevant workshop | A plausible target for a rigorous corpus audit; submission readiness still requires a focused manuscript and adequate novelty | Clear question, reproducible protocol, controlled comparisons, uncertainty and carefully bounded conclusions |
| Relevant Q1 journal | Possible after substantive strengthening; the current 46-person score is insufficient as a clinical-method claim | A deeper, demonstrably novel methodological audit or a method with convincing validation and benefit over simple controls |
| ICLR/ICML main track | Current evidence does not establish a sufficiently strong general ML contribution | Broadly useful methodological/empirical insight or a justified new method, positioned against prior work and supported beyond one favorable split |

A completely new architecture is not an acceptance requirement. ICLR explicitly
states that state-of-the-art performance is not required; ICML explicitly values
new insight from evaluating existing methods. Conversely, combining attention,
CNNs and a transformer does not establish novelty by itself.

The most defensible current research question is:

> How much evidence of disease-specific temporal learning is provided by high
> speaker-disjoint scores when static and acquisition-related cues are predictive?

A possible manuscript framing is **“When Speaker-Disjoint Validation Is Not
Enough: An Audit of Static and Acquisition-Related Cues in Parkinsonian Speech.”**
This is a working framing, not a verified claim of first-of-its-kind novelty.

### Relevant precedents and venue information

1. **Hireš et al., International Journal of Medical Informatics, 2023**,
   [“On the inter-dataset generalization of machine learning approaches to
   Parkinson's disease detection from voice”](https://doi.org/10.1016/j.ijmedinf.2023.105237).
   Its [repository abstract](https://repository.ubaya.ac.id/45580/) describes
   four datasets, including Italian data, and degradation outside the development
   dataset. Generalization failure is already known; merely rediscovering it is
   not enough to establish novelty.
2. **Shukla et al., Journal of Medical Internet Research, August 2026**,
   [“Demographic Confounding in Voice-Based Parkinson Disease Screening”](https://pmc.ncbi.nlm.nih.gov/articles/PMC13492483/).
   This is direct precedent for a methodological audit on one primary corpus.
   It evaluates demographic/site baselines and restricted/retrained cohorts;
   its PD analysis includes 253 participants. It also acknowledges evaluation
   limitations. It is evidence that audit studies can be publishable, not a
   validation of our recording-condition hypothesis or a protocol to copy blindly.
3. **Ponchard and Serrano, BUCC workshop, 2026**,
   [“A Comparative Study of Parkinsonian Speech Corpora for Deep Learning-Based
   Detection of Dysarthria”](https://aclanthology.org/2026.bucc-1.2/).
   Four-corpus intra/cross/out-of-domain evaluation provides a relevant workshop
   precedent. This is a workshop paper, not an ACL main-conference paper.
4. Official [ICLR 2026 reviewer guidance](https://iclr.cc/Conferences/2026/ReviewerGuide)
   and [ICML 2026 reviewer instructions](https://icml.cc/Conferences/2026/ReviewerInstructions)
   describe soundness, significance and originality beyond leaderboard scores.
5. Topic-relevant journals to investigate include **Biomedical Signal Processing
   and Control** and **International Journal of Medical Informatics**. Their
   [SCImago listing](https://www.scimagojr.com/journalsearch.php?q=4700152237&tip=sid)
   and [SCImago listing](https://www.scimagojr.com/journalsearch.php?q=23689&tip=sid&clean=0)
   were indexed as **2025 best quartile Q1** when searched for this assessment.
   Full SCImago access was blocked for direct retrieval. This is not a verified
   JCR-category ranking; quartile depends on database, category and year.

## Improvements worth doing before the deferred external benchmark

1. **Freeze one manuscript question and protocol.** Use the main pooled speaker
   table, report all five metrics and fold variation, and document every explored
   version rather than selecting a favorable seed, fold or model after inspection.
2. **Match speech tasks for architectural comparisons.** The three members use
   different tasks and representations. Their current comparison is of pipelines.
   A model-effect claim needs matched tasks, cohorts and training-side selection.
3. **Verify what the quiet segments contain.** Use an explicit speech/non-speech
   annotation procedure, blinded to diagnosis where practical, and document its
   limitations. Lowest-energy frames alone cannot justify a “noise-only” claim.
4. **Use controlled perturbations and retraining ablations.** Evaluate predefined
   gain/channel/noise changes and static-versus-temporal controls with the same
   training protocol. Treat loudness removal as a diagnostic intervention: it can
   remove genuine hypophonia cues as well as gain differences. No raw WAV needs
   to be overwritten. A centered model still has variance, spectral covariance,
   task cues and possible recording artifacts; centering does not prove temporal
   order is essential.
5. **Assess stability and paired uncertainty.** Predeclare repeated speaker splits
   and training seeds, report the full distribution, and keep all repeated
   predictions from one person together in statistical analysis. Repeated CV
   assesses sensitivity; it does not create new independent people or undo
   prior model development on this dataset.
6. **Check session-transfer feasibility explicitly.** With only two native PD
   dates, five two-class folds with completely disjoint recording dates cannot
   be formed. Grouped analyses holding out PD/control date groups may provide
   an exploratory stress test if person separation is maintained, but do not
   identify the causal disease signal because dates are class-specific.

For a subsequent paper claiming robust disease detection or a generally useful
new method, complete the deferred benchmark phase with a frozen pipeline and
an appropriate independent/matched-condition evaluation. These internal
improvements build the evidence and manuscript now; they do not substitute for
the additional data needed to support that broader claim.
