# How we got here

111 scored files in 29 days (3 to 31 October 2026), from a public RMSE of 0.0976 to 0.0555 (private 0.0561). Every score below is in
[`submissions/ledger.csv`](../submissions/ledger.csv); the file names (E19, E134, ...) are the files in `submissions/`.

The one idea that ran through all of it: **a leaderboard score is a measurement.** The score is an RMSE, so two scored
files give the covariance of their difference with the hidden target exactly. Each idea about how the hidden reference
behaves was added to a scored file as one small change, submitted once, and read back as a single number, its
agreement with the target (about 0.02 is one standard error). Every prediction was written down before the upload.

## 1. First files (3 to 11 October): the target's shape

Two constant files gave the target's mean and spread exactly (0.0598 and 0.0890 on the public split), so any later
score could be turned into a correlation. The first structural files modelled each component's reference by region
(0.0976), then by density within region (0.0853), then with separate road and fire parts (0.0837).

## 2. The road reference is bookkept (12 to 14 October)

| File | What changed | Public RMSE |
|---|---|---|
| E19 | a highway on a tract boundary counted inside the tract | 0.0774 |
| E29 | a road level per region | 0.0678 |
| E34 | the width of the boundary band | 0.0664 |
| E36 | motorway ramps | 0.0646 |

E19 had an agreement of +0.46, the largest single effect of the competition, and the start of the finding in Part A
of the paper: tract lines are drawn on roads, and the reference counts a boundary highway in both tracts. The general
idea that the road gap reflects TIGER's data model was first posted publicly by another participant (challenge
discussion 35000), and a third participant later published the boundary mechanism as their own entry (discussion 35478).

## 3. Places and facilities (15 to 19 October)

| File | What changed | Public RMSE |
|---|---|---|
| E44 | the places component confirmed (agreement +0.06) | 0.0656 |
| E46 | two distinct places mechanisms | 0.0638 |
| E49 | fire stations Overture files under another category | 0.0626 |
| E52 | stations visible only as buildings | 0.0622 |
| E56 | schools Overture files under another category | 0.0615 |
| E62 | rural communities keeping a volunteer fire department | 0.0613 |
| E66 | record completeness for the business half | 0.0612 |

The facility rules were the second family of large gains (E49 +0.12, E56 +0.12): the score counts Overture places by
their primary category only, so a real station or school filed elsewhere is a gap. The lead came from another
participant's public post (challenge discussion 34925, which also published the 740 and 270 fire-station counts we
calibrate to).

## 4. A simulator of the hidden reference (20 to 24 October)

| File | What changed | Public RMSE |
|---|---|---|
| E73, E74 | the simulator: the hidden reference drawn per tract, the exact formula applied | 0.0610, 0.0605 |
| E85 | one reference record per route number | 0.0598 |
| E86 | one record per local street name | 0.0598 |
| E87 | the best combination of scored files | 0.0592 |
| E90 | the fitted simulator's part outside what the files had measured | 0.0584 |
| E92 | combination | 0.0578 |
| E97 | one record per honorific highway name | 0.0576 |

The duplicate-record rules (E85, E86, E97) are the second half of Part A's finding. Both the rule and the method that
drove the rest came from another participant's public method notes (22 October; cited in the paper's references): the
reference keeps one record per edge and name, and the way
to find more is to fit the simulator's mechanism weights to what the scored files reveal, then bet on its prediction in
the directions no file has measured (E90). We tested the rule on our own model (E85, E86) and extended it to honorific
highway names (E97).

## 5. Re-identifying the road vehicle (25 to 28 October)

| File | What changed | Public RMSE |
|---|---|---|
| E109 | ramps restored: the scorer filters Overture on class only | 0.0574 |
| E110 | the boundary width re-identified | 0.0572 |
| E115 | the counting rules re-applied to the corrected vehicle | 0.0562 |
| E120 | the vehicle re-identified, railway boundaries set aside | 0.0557 |

## 6. Compliance audit and the clean rebuild (28 to 30 October)

An audit of every file's lineage found that the combinations since E35 still carried content from two early
exploratory files: E33, which imposed another team's published table of mean scores, and E37, which adjusted a file
toward the scorecard printed beside the leaderboard's top entry. Rather than submit them, we rebuilt the finals from
package-only pieces.

| File | What changed | Public RMSE | Private RMSE |
|---|---|---|---|
| E123 | clean pieces weighted directly by their measured agreement | 0.0570 | 0.05773 |
| E124, E125 | clean pieces fitted to the geometry of our best file (63, then 77 pieces) | 0.0563, 0.0558 | 0.05624, 0.05637 |
| E127 | the simulator v9b refitted, its new part | 0.0558 | 0.05616 |
| E129 | clean pieces fitted to a combination of 69 scored files (the "teacher") | 0.05559 | 0.05617 |
| E131 | the same teacher, light regularisation | 0.05557 | 0.05615 |
| E132 | stronger regularisation, for the private board | 0.05563 | 0.05616 |

E123 is the lesson of the week: letting scores pick the pieces' weights directly fitted the noise of the 30% public
sample and scored 0.0016 worse than predicted. Fitting to the teacher's geometry, with no score in the weights,
recovered it. A simulation with synthetic targets then said a teacher regularised more strongly should do better on
the private 70%, so we kept one file of each kind.

## 7. The misfit learner (31 October)

| File | What changed | Public RMSE | Private RMSE |
|---|---|---|---|
| E133 | a social-vulnerability term in the business reference | 0.05567 (no signal) | 0.05627 |
| E134 | a LightGBM model of what the simulator misses, its part outside the scored span | **0.05557** (agreement +0.030) | 0.05613 |
| E135 | the learner bagged and given the simulator's own components | 0.05561 (nothing beyond E134) | 0.05610 |
| E136 | E131 plus E134's step at its measured optimum | **0.055506** (best) | 0.05613 |
| E137 | the simulator refitted on all 104 scored files | 0.05552 (agreement +0.010) | 0.05623 |

By the last day every hand-written mechanism we could test came back null. A learner trained on the simulator's
in-span misfit (out-of-fold R2 0.45, driven by the facility features) found a direction the hand-written rules had
not: its first step paid, and a better-fitting second version added nothing beyond it.

## What did not work

Some of the ideas that came back null, each tested on the leaderboard or refuted from the package first: county and
Texas farm-to-market routes in the reference; building footprints as a signal (the building gap is almost always 0);
a post-fire remapping lag; region-level and per-stratum corrections; colleges in the school reference; the business
reference allocated by land area, by housing, by BrightQuery's registry places or by social vulnerability; and
machine-learned refinements beyond one step.

## What we would keep

1. **Treat every score as a measurement, and write the prediction down first.** It made every decision checkable.
2. **Counting rules beat tuning.** The large gains were rules about which record the reference counts (E19, E49, E56,
   E85, E86), not better-tuned weights.
3. **A matched published average is a sanity check, not evidence.** Files that reproduced the published averages
   exactly scored no better; the error lives inside each stratum.
4. **Regularise for the private board.** Weights read from the public 30% fit some of its noise. We expected
   the stronger fit to gain on the private board; it did not (E132 0.056156, E131 0.056149).
5. **Keep the build path clean and rebuildable.** Every final rebuilds from the package to 3e-16, and the builder
   refuses to run if an excluded module is loaded.

## After the finals

The prize form lets a discovery draw on public data beyond the package, so Section 4.5 of the paper asks who lives in the
tracts the bookkeeping swaps, on twelve Census and FEMA measures fixed before any comparison was computed. Ten point the
same way and two do not, and all twelve are reported. No scored file reads this data.

Private scores, revealed after the close: E134 0.056132715 and E136 0.05613082 for the finals; our best private file was
E135 at 0.056099119.
