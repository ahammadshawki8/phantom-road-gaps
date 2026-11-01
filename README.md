# Phantom Road Gaps

Team Yesternity's solution and writeup for the **Bias Bounty Mapping Equity Challenge** (Humane Intelligence and Zindi,
2026). The challenge asks for a coverage-gap score for each of 9,794 census tracts in five US regions: how much of a
reference map (TIGER/Line roads, Microsoft building footprints, HIFLD facilities, County Business Patterns) Overture
Maps lacks. The reference layers were removed from the challenge package, so no tract-level labels exist. Both finals
are built from the challenge package alone; only Part A's outside check (below) adds public Census and FEMA data, and no
scored file reads it.

**Paper:** [`docs/paper/paper.pdf`](docs/paper/paper.pdf), "Phantom Road Gaps: reference bookkeeping makes up most of the
road-coverage gap, and it bends the scorecard built to reveal inequity". Part A is our Best Bias Discovery entry, Part B
the methodology (Best Documentation).

**The finding in one paragraph.** A highway along a tract boundary is counted in full in both tracts, and a stretch
carrying two route numbers or a second name is counted again. Measured from the package, these two rules make up 72% of
the mean road gap of road-scored tracts (55 to 80% under any weighting). Because they weigh most in towns and cities,
they bend the organisers' equity scorecard: with each highway counted once, the rural-to-urban ratio of the score rises
from 2.34 to 2.73 and the wildfire-hazard ratio from 1.67 to 1.81, while the social-vulnerability ratio falls from 1.16 to
1.01. Public Census and FEMA data then show who the bookkeeping pushes off a worst-tenth worklist: tracts with 1.8 times
the share of mobile homes, and three times as many rated high for wildfire risk.

## Check Part A in a minute

```bash
pip install pandas pyarrow geopandas
python src/download.py --strata        # a few MB: sample submissions and strata tables
python src/quick_check.py              # 33 of the paper's numbers, in under a second
```

It reads our published per-tract results (`docs/writeup/tract_results.csv`), our final submission, the package's strata
tables and the Census and FEMA extract (`docs/writeup/external_tracts.csv`), and prints `33 of 33 checks match the paper`.
Add `--geometry` (after `python src/download.py northern-ca`) to re-measure the Yuba City worked example from Overture's
roads, which makes it 34 of 34.

**How we got here:** [`docs/HISTORY.md`](docs/HISTORY.md), from the first file to the finals, with every milestone's score.

## Final submissions

| Final | Zindi ID | File | Public RMSE | Private RMSE | What it is |
|---|---|---|---|---|---|
| A (robust) | `EVGrZBHt` | `submissions/e134_learner.csv` | 0.05556589 | 0.056132715 | the clean geometric fit regularised for the private board (E132), plus the misfit learner's step |
| B (public-best) | `E3u62GkV` | `submissions/e136_finalB.csv` | 0.055506023 | 0.05613082 | the lightly regularised fit (E131), plus the same step at its measured optimum |

Private scores were revealed after the close and played no part in any decision. `docs/writeup/private_scores.csv` lists the
private RMSE of our scored files that we could read from Zindi. Our best private file was E135 at 0.056099119.

Both rebuild from the challenge package and the files in `params/` to within 3e-16 of the uploaded CSVs
(`python src/final_build.py e136_finalB`).

## Reproducing everything

Python 3.11, CPU only, about 16 GB of memory. Every random draw is seeded.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download.py                 # the challenge package into ../data (no credentials; ~40 min at 3 MB/s)
python src/build_features.py           # every per-tract feature table, in dependency order (~25 min)

python src/final_build.py e136_finalB  # Final B; rebuilds E131, E132 and E134 on the way (~10 min)
python src/final_build.py e134_learner # Final A
```

Each `final_build` run rebuilds the file from the package, compares it with the uploaded CSV in `submissions/` and
stops with an error if any excluded module was loaded. Expected output ends with
`max |rebuilt - uploaded| 3.1e-16 ... excluded modules loaded: none`.

The code expects the package next to the repository, at `../data` (that is where `download.py` writes it).

### The paper's numbers and figures

```bash
python src/writeup_numbers.py          # every number in Part A -> docs/writeup/numbers.json, tracts.csv (~2 min)
python src/writeup_numbers.py displaced  # the worklist comparison of Section 4
python src/writeup_numbers.py scorecard  # the scorecard as scored and counted once, and tract_results.csv
python src/writeup_numbers.py sensitivity  # the bookkeeping share under other weights
python src/external_context.py         # Section 4.5, from the published extract (add fetch to download it again, ~3 min)
python src/paper_figures.py            # Figures 1 to 9 -> docs/paper/*.pdf (~1 min)
cd docs/paper && pdflatex paper.tex && pdflatex paper.tex
```

`docs/writeup/numbers.json` holds, among others, `road_gap_mean = {as_scored: 0.166, counted_once: 0.047}` and
`added_ge_0.2_total.tracts = 1548`.

### Part A's outside check

The prize form lets a discovery draw on public data beyond the package, so Section 4.5 compares the 321 hidden tracts
with the 321 flagged tracts on twelve measures fixed before any comparison was computed. All twelve are reported,
including the two that do not follow the pattern. `src/external_context.py` holds the list; no scored file and no build
path reads this module or its output.

| Source | What we take | Licence | Retrieved |
|---|---|---|---|
| [Census ACS 2020-2024 5-year, table-based summary file](https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/) | tables B01001, B17001, B18101, B25024, B25044, B28002 | public domain | 31 Oct 2026, 16:27 UTC |
| [FEMA National Risk Index v1.20, December 2025](https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/National_Risk_Index_Census_Tracts/FeatureServer/0) | ratings, loss, social vulnerability and resilience | [U.S. Government Work on data.gov](https://catalog.data.gov/dataset/national-risk-index-nri-data), under FEMA's terms | 31 Oct 2026, 16:27 UTC |

This product uses the Federal Emergency Management Agency's National Risk Index dataset API or downloadable datasets but
is not endorsed by FEMA. The Federal Government or FEMA cannot vouch for the data or analyses derived from these data
after the data have been retrieved from the Agency's website(s).

## How it works, in five steps

1. **Features.** For each tract, lengths and counts measured from Overture roads, places and buildings, the ACS
   housing table and the strata tables, including the measures behind Part A: route-numbered highway inside the
   tract, highway on its boundary, extra route numbers and names (`src/features*.py`).
2. **A simulator of the hidden reference.** `src/physics9.py` (version 9b) draws each tract's reference road length,
   facilities and businesses many times from distributions built on Overture's own data, applies the organisers'
   scoring formula to each draw and averages. It is calibrated to published averages (listed in `src/published.py`).
3. **Measuring ideas with the leaderboard.** Each idea was submitted once as a small change to a scored file. Because
   the score is an RMSE, two scores give the covariance of the change with the hidden target exactly; we call it the
   idea's agreement (`clean_proto.harvest`).
4. **The clean finals.** 77 package-only pieces (the simulator's variants and the raw vector of every idea we tested)
   are fitted, without using any score, to the shape of a "teacher": a ridge-regularised combination of our scored
   files (`src/clean_geo.py`, `src/final_model.py`).
5. **The misfit learner.** A seeded LightGBM model learns, out of fold, the part of the target that the simulator gets
   wrong as the scored files reveal it; its prediction outside the scored span was the last step that paid
   (E134, agreement +0.030; `src/misfit_gbm.py`).

Some ideas came from other participants' public posts, and we credit each one, with its source, in the paper's prior
work and references: the boundary mechanism (challenge discussions 35000 and 35478), the duplicate-record rule and the
fit-then-bet method (a participant's public method notes, 22 October), and the misfiled fire stations (discussion 34925).

## Repository map

```
src/
  download.py, build_features.py, features*.py, table.py   the package and the per-tract feature tables
  physics*.py                                             the simulator (v9b is the vehicle; physics12 adds keys that are neutral at 0)
  cand_*.py, clean_proto.py                               the builder of every tested idea; the 77-piece clean registry
  clean_geo.py, final_model.py, validate.py               the teacher affine and the geometric fit
  misfit_gbm.py                                           the misfit learner
  final_build.py                                          rebuild and verify a final from the package
  writeup_numbers.py, writeup_figures.py, paper_figures.py  the paper's Part A numbers and figures
  external_context.py                                     Part A's outside check, from public Census and FEMA data
params/        frozen weights, simulator parameters and one <name>_final.json per final
submissions/   our scored files (the teacher and the projection span are built from them) and ledger.csv, every public score
docs/paper/    the paper, LaTeX source and PDF
docs/writeup/  Part A outputs
docs/HISTORY.md
```

## What is not here, and why

- No removed reference layer (TIGER/Line roads, Microsoft footprints, HIFLD, CBP) or any copy of one is used anywhere.
- Two early exploratory files are left out and no code path reads them: E33 imposed another team's published table of
  mean scores at full strength, and E37 adjusted a file toward the scorecard printed beside the leaderboard's top
  entry. Three later files that inherited their content (E38, E42, E45) are included only because the span every bet
  is projected out of contains them; their content enters no final, and the teacher excludes them.
- A few diagnostic functions compared our files against a card derived from another team's published aggregates.
  They were never on a build path and raise `NotImplementedError` in this release.
- The published averages we do use (16 region, urbanity and burn cell means; 740 and 270 fire-station counts; the
  0.052 places average; the fire-to-school ratio; the organisers' road-undefined counts) set a handful of constants
  and are listed with their sources in `src/published.py`. None is a per-tract value.

## Team and licence

Team Yesternity on Zindi: Shawki (`ahammadshawki8`), Ahnaf (`ahnafjamil`) and Ashfaq (`Ashfaqul_Hasan`). We used an LLM coding assistant for
code drafting, debugging and editing; every analysis, number and claim was checked by the team, and nothing in the
pipeline needs it.

Code: MIT licence (`LICENSE`). Data: the challenge package's own terms (Overture Maps under ODbL and CDLA-Permissive
2.0, Census TIGER/Line and ACS in the public domain); the files in `submissions/` are derived from it and shared for
reproducibility. `docs/writeup/external_tracts.csv` is an extract of the two public sources listed under Part A's outside
check, on their own terms.
