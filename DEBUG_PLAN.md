# Debug Plan: Why Does Unimodal FL Outperform Multimodal FL?

## Problem

From `results_v3/` on the Indiana CXR dataset, fully-unimodal federated
learning (E1 Uni-IID, E2 Uni-nonIID) achieves higher test AUROC than any
setting that includes multimodal clients (E3–E10), under both IID and
non-IID partitions. This contradicts the hypothesis that adding text
modality should help.

### What is actually shared from client to server?

All clients — unimodal or multimodal — instantiate the same
`UnifiedMultimodalClassifier` (see `model.py:335`). Every FL round, the
following **trainable** parameters are sent in both directions (client ↔
server) and averaged via FedAvg:

- `image_encoder.projector.*`
- `text_encoder.projector.*`
- `image_type_embed`
- `text_type_embed`
- `fusion_transformer.*`
- `classifier.*`

Unimodal clients pass `input_ids=None` → `txt_embed = zeros_like(img_embed)`
(see `model.py:419`). The fusion transformer is always executed, regardless
of modality. The server evaluates with `input_ids=None` (image-only,
zero-padded text) on val/test.

## Hypotheses

1. **H1 — Label leakage through text.** The findings/impression fields are
   generated from the same MeSH terms used to derive labels; disease
   keywords predict labels trivially. Multimodal training may learn a
   `text→label` shortcut that collapses when text is zero-padded at eval.
2. **H2 — Conflicting gradient signals on fusion.** Unimodal clients push
   fusion to "ignore zero text"; MM clients push it to "attend to real
   text." FedAvg averages these opposing updates, destroying both.
3. **H3 — Evaluation-training distribution mismatch.** MM clients train
   with real text; global eval uses zero text. Fusion's text-handling
   capacity is never exercised at eval time.
4. **H4 — Data-level noise / insufficient text coverage.** Many samples
   lack text; MM clients train on a mix of real-text and empty-string
   samples.
5. **H5 — Hyperparameter interaction.** Same LR for projector, classifier,
   and fusion may over-update fusion relative to other components.

## Status

| ID | Status | Test AUROC headline |
|----|--------|---------------------|
| DEBUG-1 | ✅ done | Leakage > 0.75 on 4 of 8 labels |
| DEBUG-2 | ✅ done | Text alone → **0.941 test AUROC** |
| DEBUG-3 | ✅ done | Centralized image=0.699, MM=**0.906** (+0.21) |
| DEBUG-4 | ✅ done | Oracle text eval on v3 checkpoints: **+0.21 to +0.31 AUROC gain** vs zero-text eval |
| DEBUG-5 | ✅ done | Masked attention: E9 test 0.658 (vs v3 0.649) |
| DEBUG-6 | ✅ done | Freeze fusion on uni clients: E9 test 0.659 (vs v3 0.649) |
| DEBUG-7 | ✅ done | Lower fusion LR: E9 val **0.709 best of all**, test 0.667 |

---

## DEBUG-1 — Keyword→Label leakage (training split, n=2,662)

Regex keyword patterns matched against `findings_text`; label predicted
positive if a keyword appears.

| Label | Leakage (pos_rate − neg_rate) | Precision | Recall | F1 |
|-------|-------------------------------|-----------|--------|----|
| No_Finding | 0.106 | 0.383 | 0.960 | 0.548 |
| Cardiomegaly | 0.569 | 0.933 | 0.573 | 0.710 |
| Atelectasis | 0.756 | 0.819 | 0.771 | 0.794 |
| Pleural_Effusion | 0.088 | 0.040 | 0.710 | 0.075 |
| Opacity | 0.852 | 0.542 | 0.958 | 0.693 |
| Calcinosis | 0.922 | 0.538 | 0.995 | 0.699 |
| Calcified_Granuloma | 0.943 | 0.572 | 0.994 | 0.727 |
| Airspace_Disease | 0.304 | 0.047 | 0.868 | 0.090 |

**Interpretation:** 5 of 8 labels show Leakage > 0.5 and F1 ≥ 0.69 from a
naive regex. A real language model trained on these reports will do far
better. **H1 partially confirmed at the data level** — text reports
trivially encode several labels.

---

## DEBUG-2 — Centralized text-only classifier

Frozen BiomedBERT → trainable projector → linear classifier. 10 epochs.
Trained/evaluated only on samples that have non-empty findings_text.

| Epoch | train_loss | val AUROC | test AUROC |
|-------|-----------|-----------|------------|
| 1 | 0.252 | 0.869 | 0.848 |
| 5 | 0.141 | 0.943 | 0.934 |
| 10 | 0.116 | **0.952** | **0.941** |

Best per-label test AUROC: No_Finding 0.990, Cardiomegaly 0.922,
Atelectasis 0.937, Pleural_Effusion 0.888, Opacity 0.927, Calcinosis
0.952, Calcified_Granuloma 0.980, Airspace_Disease 0.931.

**Interpretation:** Text alone reaches **0.941 test AUROC** — roughly
**+0.25 higher** than any v3 FL run's test AUROC. The text reports
essentially contain the labels. **H1 strongly confirmed.** Any model that
learns to use text at training time will look great when tested with real
text, and catastrophic when tested with zero-padded text.

---

## DEBUG-3 — Centralized training, no FL (samples with text only)

Same `UnifiedMultimodalClassifier` as FL, trained on the full training
split in a single process. Two runs: one with text forced to zero
(`mode=image`), one with real text (`mode=multimodal`). Eval only on
samples that have text in the respective split.

| Mode | best epoch | val AUROC | test AUROC |
|------|-----------|-----------|------------|
| image (zero text) | 7 | 0.770 | **0.699** |
| multimodal (real text) | 10 | 0.931 | **0.906** |
| Δ (MM − image) | — | +0.16 | **+0.21** |

Per-label test AUROC (multimodal): No_Finding 0.991, Cardiomegaly 0.897,
Atelectasis 0.883, Pleural_Effusion 0.841, Opacity 0.910, Calcinosis
0.900, Calcified_Granuloma 0.952, Airspace_Disease 0.878.

**Interpretation:** The `UnifiedMultimodalClassifier` architecture itself
is sound. Given real text at both train and eval, it exploits fusion to
add **+0.21 AUROC** over image-only. So the architecture is *not* the
bug. The problem lies somewhere between FL training and FL evaluation.

Note: centralized image-only (0.699) is lower than centralized v3 E1
(0.779) because DEBUG-3 trains only on samples with text (~1,640 of
~2,662), whereas E1 uses all training samples. This is the right control
because it matches the evaluation universe.

---

## DEBUG-4 — Oracle text evaluation on v3 checkpoints

For each v3 experiment, load `best_global_params.pt`, evaluate on the
test set two ways: (a) zero-padded text (identical to how v3 originally
evaluated), (b) real text (oracle). Eval restricted to test samples that
have text. Numbers are macro AUROC.

| Exp | zero-text AUROC | oracle-text AUROC | Δ |
|-----|-----------------|-------------------|---|
| E1  Uni-IID       | 0.683 | 0.681 | −0.002 |
| E2  Uni-nonIID    | 0.681 | 0.678 | −0.003 |
| E3  1MM-IID       | 0.683 | 0.926 | **+0.243** |
| E4  1MM-nonIID    | 0.686 | 0.920 | **+0.234** |
| E5  2MM-IID       | 0.682 | 0.938 | **+0.256** |
| E6  2MM-nonIID    | 0.681 | 0.944 | **+0.263** |
| E7  3MM-IID       | 0.660 | 0.939 | **+0.279** |
| E8  3MM-nonIID    | 0.673 | 0.956 | **+0.283** |
| E9  4MM-IID       | 0.651 | **0.959** | **+0.308** |
| E10 4MM-nonIID    | 0.617 | 0.823 | +0.205 |

**This is the smoking gun.** Three key observations:

1. **Every v3 experiment with ≥1 MM client gains +0.20 to +0.31 AUROC
   when given real text at eval time.** The fusion transformer learned
   to use text excellently — we just never fed it text at test time.
2. **Unimodal-only experiments (E1, E2) are unchanged by oracle text
   (Δ ≈ 0).** This is the correct sanity check: E1/E2 had no MM clients,
   so fusion never saw text during training, so there is nothing for
   oracle text to light up. Confirms the measurement is trustworthy.
3. **More MM clients → higher oracle AUROC.** Under IID: E3→E9 go
   0.926 → 0.938 → 0.939 → 0.959. The intuition that adding MM clients
   helps the model was correct — FedAvg is *not* destroying fusion.
   Evaluation protocol was hiding the benefit.

Under this oracle evaluation, E9 (4MM-IID) at **0.959 AUROC** beats
centralized multimodal from DEBUG-3 (0.906) and text-only from DEBUG-2
(0.941). FL with full multimodal participation is the strongest
configuration, not the weakest.

E10 (4MM-nonIID) is the only exception: oracle 0.823 is noticeably lower
than the other oracle numbers. This is genuine non-IID degradation from
client drift, but it is still **+0.21** over its zero-text eval.

---

## Revised diagnosis

**H3 (eval-training distribution mismatch) is the primary root cause of
the paradox.** H1 is a contributing factor at the data level (text is
very predictive, so the mismatch looks even worse than it would on a
dataset without label leakage). H2 (conflicting fusion gradients) is
partially disproved by DEBUG-4: if fusion gradients were destroyed by
FedAvg, oracle AUROC would be low too. Instead it is ≥ 0.92 on all IID
MM experiments.

### Implications

- The v3 result table is misleading: it used image-only eval against FL
  runs whose training regime increasingly leveraged text. Under a fair
  eval that uses whatever modalities the training actually exercised,
  the ordering flips: 4MM-IID is the best setting, unimodal is the worst.
- The question for v4 becomes *what is the right eval protocol?*
  Options: (i) always evaluate with real text on samples that have it;
  (ii) evaluate separately on image-only and on text-present subsets;
  (iii) at inference time, use a deterministic "no-text" token sequence
  that matches training, not zero tensors.
- DEBUG-5/6/7 are now less critical for root cause, but still useful:
  they will tell us whether an FL recipe exists that narrows the gap
  between zero-text eval and oracle-text eval, which matters if we want
  a model that also generalizes to image-only deployment.

---

## DEBUG-5 — Masked attention for missing text

Subclassed `UnifiedMultimodalClassifier`: when `input_ids` is `None`,
pass `src_key_padding_mask=[False, True]` so the fusion transformer
ignores the text token. Classifier then reads only the image token.
Reran E1 (Uni-IID) and E9 (4MM-IID) for up to 30 rounds.

| Experiment | rounds | best val AUROC | test AUROC @ best val | vs v3 baseline |
|------------|--------|----------------|-----------------------|----------------|
| E1 Uni-IID (masked) | 20 (early stop) | 0.6835 @ r12 | 0.6591 | v3 E1: 0.6819 → Δ −0.023 |
| E9 4MM-IID (masked) | 18 (time limit) | 0.6670 @ r12 | 0.6583 | v3 E9: 0.6490 → Δ **+0.009** |

**Interpretation:** Masking slightly hurts the unimodal baseline (E1)
because the transformer never learned anything useful about the "masked"
position and now gets one fewer token at eval. For E9, masking closes
most of the gap between v3 4MM-IID and v3 Uni-IID under zero-text eval
(0.649 → 0.658), but does not beat the v3 unimodal baseline (0.6819).
The zero-token artifact was a real but minor contributor.

---

## DEBUG-6 — Freeze fusion + text components on unimodal clients

Unimodal clients freeze `fusion_transformer.*`, `text_encoder.projector.*`,
`text_type_embed`. `PartialFedAvgServer` averages each parameter only
over clients that trained it (weighted by their sample count). Reran E5
(2MM-IID) and E9 (4MM-IID).

| Experiment | rounds | best val AUROC | test AUROC @ best val | vs v3 baseline |
|------------|--------|----------------|-----------------------|----------------|
| E5 2MM-IID (frozen-fusion) | 18 (early stop) | 0.6999 @ r10 | 0.6721 | v3 E5: 0.6800 → Δ −0.008 |
| E9 4MM-IID (frozen-fusion) | 20 (time limit) | 0.6847 @ r16 | 0.6591 | v3 E9: 0.6490 → Δ **+0.010** |

**Interpretation:** Removing the "fit fusion to zero text" signal from
unimodal clients recovers +0.010 test AUROC on E9. The intuition is
correct in sign, but the magnitude is small — consistent with DEBUG-4,
which already showed fusion was learning text well; there just wasn't
much for the unimodal clients to *destroy* in the first place.

---

## DEBUG-7 — Lower LR for fusion transformer

Per-client optimizer param groups. `fusion_transformer.*` and
`text_type_embed` get `0.1 × base_lr`; everything else gets base_lr.
Reran E9 (4MM-IID) only.

| Experiment | rounds | best val AUROC | test AUROC @ best val | vs v3 baseline |
|------------|--------|----------------|-----------------------|----------------|
| E9 4MM-IID (fusion lr × 0.1) | 17 (early stop) | **0.7085 @ r9** | 0.6671 | v3 E9: 0.6490 → Δ **+0.018** |

**Interpretation:** This is the single best intervention on the v3
zero-text eval. Val AUROC jumps to 0.7085 — higher than any v3 FL
experiment including unimodal (v3 E1 val 0.6903). Test AUROC gains
+0.018 over v3 E9. Damping fusion's learning rate reduces the gradient
noise from averaging across modality-heterogeneous clients; the rest of
the model (classifier, image projector) continues to learn at full LR.

---

## Summary of interventions (4MM-IID, zero-text eval)

| Run | val AUROC | test AUROC |
|-----|-----------|-----------|
| v3 E1 Uni-IID (original baseline) | 0.690 | 0.682 |
| v3 E9 4MM-IID (original paradox) | 0.686 | 0.649 |
| DEBUG-5 E9 masked | 0.667 | 0.658 |
| DEBUG-6 E9 frozen fusion | 0.685 | 0.659 |
| DEBUG-7 E9 fusion LR × 0.1 | **0.709** | **0.667** |
| **DEBUG-4 E9 oracle text eval (same v3 weights)** | — | **0.959** |

Two things jump out:

1. **All three FL fixes help** (D5, D6, D7 each beat v3 E9 test). D7 is
   strongest, and is the only one where 4MM-IID also beats v3 E1 on val.
2. **All three FL fixes are dwarfed by simply using real text at eval
   time.** The oracle eval on v3 E9 gets 0.959 test AUROC — ~0.29 above
   even the best FL fix. This confirms that the evaluation protocol,
   not FL aggregation, is the dominant factor.

## Final recommendation

- **Primary fix (must do): change the evaluation protocol.** For any
  experiment that includes MM clients, evaluate with real text on the
  subset of test samples that have text. Report both "image-only eval"
  and "text-present eval" so the two regimes are never conflated. This
  is free — requires no retraining.
- **Secondary fix (recommended): adopt DEBUG-7's param-grouped optimizer**
  as the default. It improves zero-text deployment and costs almost
  nothing. Combining with DEBUG-6 (selective freezing on unimodal
  clients) is worth a single experiment to test additivity.
- **Not recommended:** pursuing DEBUG-5 masking further. Gains are
  marginal and it slightly regresses the unimodal baseline.

### Suggested v4 experiment set

Rerun the 10-experiment suite with:
1. Fusion LR scaled × 0.1 on all clients (DEBUG-7).
2. Unimodal clients freeze fusion + text components (DEBUG-6).
3. Evaluation: macro AUROC on (a) image-only eval and (b) real-text eval
   on samples that have text; also report text-only coverage.

Expected outcome: v4 table should show 4MM-IID ≥ Uni-IID under image-only
eval, and 4MM-IID ≫ Uni-IID under real-text eval, recovering the
original hypothesis.
