"""The paper's figures (docs/paper/), as vector PDFs sized for a two-column A4 page, from the package,
`writeup_numbers.py`'s output (run that first) and the submission ledger.

    p1_gap_by_region.pdf   the road gap as scored and with each highway counted once (column)
    p2_worked_example.pdf  Yuba City 505.03: SR 99 on the tract line, Overture's copy outside it (page width)
    p3_map_oklahoma.pdf    the road gap the bookkeeping adds, Oklahoma (column)
    p4_agreement.pdf       each counting rule's measured agreement with the hidden target, +-1 s.e. (column)
    p5_worklist.pdf        every road-scored tract, as scored against counted once; the worst tenth (column)
    p6_trajectory.pdf      public RMSE of every scored file over the competition (column)
    p7_architecture.pdf    Part B's data flow, one box per script (column)
    p8_scorecard.pdf       the scorecard's ratios on our final, as scored and counted once (column)
    p9_outside_check.pdf   the hidden tracts against the flagged ones on Census and FEMA data (column)

    python src/paper_figures.py          # ~1 min
"""
import json

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import lb  # noqa: E402
from writeup_figures import DATA, GREY, INK, INK2, LABEL, ONCE, SCORED, SEQ, _roads_near  # noqa: E402

OUT = lb.ROOT / "docs" / "paper"
COL, PAGE = 3.30, 6.85                       # inches: one column and the full text width of the paper
GRID = "#e6e5e0"
HIDDEN = "#1baf7a"                           # categorical slot 3 (aqua), validated with SCORED/ONCE for CVD
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300, "pdf.fonttype": 42})


def _save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def p1():
    N = json.loads((lb.ROOT / "docs" / "writeup" / "numbers.json").read_text())["attribution"]
    short = {"eastern-ok": "OK", "eastern-wa": "WA", "maricopa-az": "AZ", "northern-ca": "CA", "south-central-tx": "TX"}
    rows = sorted(((f"{short[k.split('|')[0]]} {k.split('|')[1]} ({v['n']:,})", v["gap_full"], v["gap_without"])
                   for k, v in N.items()), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    for i, (lab, full, once) in enumerate(rows):
        ax.plot([once, full], [i, i], color=GREY, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter([once], [i], s=22, color=ONCE, zorder=2, edgecolor="white", linewidth=1.0)
        ax.scatter([full], [i], s=22, color=SCORED, zorder=3, edgecolor="white", linewidth=1.0)
        ax.text(full + 0.012, i, f"{full:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean road gap of road-scored tracts")
    ax.set_xlim(0, 0.55)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=22, color=SCORED, label="as the reference counts roads")
    ax.scatter([], [], s=22, color=ONCE, label="each highway counted once")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    _save(fig, "p1_gap_by_region.pdf")


def p2(geoid="06101050503", region="northern-ca"):
    t = gpd.read_parquet(DATA / "strata" / region / f"{region}-census-tracts.parquet")
    tr = t[t.GEOID == geoid]
    x0, y0, x1, y1 = tr.total_bounds
    pad = 0.006
    box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    rd = _roads_near(region, box)
    rd = rd[rd["class"].isin(["motorway", "trunk", "primary", "secondary"])].copy()
    rd["route"] = rd.routes.apply(lambda r: bool(r is not None and len(r) > 0))
    tp, rp, nb = tr.to_crs(5070), rd.to_crs(5070), t.cx[box[0]:box[2], box[1]:box[3]].to_crs(5070)
    poly = tp.geometry.iloc[0]
    hw = rp[rp.route]
    near = hw[hw.distance(poly.boundary) < 10]
    fig, (ax, ia) = plt.subplots(1, 2, figsize=(PAGE, 2.75), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.05))
    for a_, lw in ((ax, 1.3), (ia, 2.0)):
        nb.boundary.plot(ax=a_, color=GREY, lw=0.7)
        tp.plot(ax=a_, color="#eef4fc", edgecolor=SCORED, lw=lw)
        rp[~rp.route].plot(ax=a_, color=INK2, lw=1.0)
        tp.boundary.plot(ax=a_, color=SCORED, lw=lw)
        hw.plot(ax=a_, color=ONCE, lw=lw + 0.2)
        a_.set_xlabel("")
        a_.set_ylabel("")
    gx0, gy0, gx1, gy1 = tp.total_bounds
    ax.set_xlim(gx0 - 300, gx1 + 300)
    ax.set_ylim(gy0 - 300, gy1 + 300)
    ax.set_axis_off()
    ax.plot([gx0 - 250, gx0 + 250], [gy0 - 250] * 2, color=INK, lw=1.6)
    ax.text(gx0, gy0 - 230, "500 m", ha="center", va="bottom", fontsize=7.5, color=INK)
    c = poly.centroid
    ax.text(c.x - 350, c.y - 60, "Census tract 505.03\nYuba City, Sutter County, CA\n2,040 homes", ha="center",
            va="center", fontsize=8, color=INK)
    along = near.union_all().intersection(poly.boundary.buffer(5))        # SR 99 where it runs on the tract line
    p = poly.boundary.interpolate(poly.boundary.project(along.centroid))
    hw_, hh_ = 20, 16
    ax.add_patch(plt.Rectangle((p.x - 90, p.y - 90), 180, 180, fill=False, ec=INK, lw=0.8))
    ax.text(p.x - 110, p.y + 110, "zoom", ha="right", fontsize=7, color=INK)
    ia.set_xlim(p.x - hw_, p.x + hw_)
    ia.set_ylim(p.y - hh_, p.y + hh_)
    ia.set_xticks([])
    ia.set_yticks([])
    ia.set_aspect("equal")
    for sp in ia.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(INK2)
        sp.set_linewidth(0.7)
    ia.plot([p.x - hw_ + 3, p.x - hw_ + 13], [p.y - hh_ + 3] * 2, color=INK, lw=1.3)
    ia.text(p.x - hw_ + 8, p.y - hh_ + 3.8, "10 m", ha="center", va="bottom", fontsize=7, color=INK)
    ia.set_title("zoom on the east edge", fontsize=7.5, color=INK2)
    h1, = ax.plot([], [], color=SCORED, lw=1.4, label="tract line, drawn on the highway's centreline")
    h2, = ax.plot([], [], color=ONCE, lw=1.6, label="Overture's State Route 99, a median 2.8 m outside the tract")
    h3, = ax.plot([], [], color=INK2, lw=1.0, label="Overture's other highway-class road (Franklin Road)")
    ia.legend(handles=[h1, h2, h3], loc="upper left", bbox_to_anchor=(0.0, -0.03), ncol=1, frameon=False,
              fontsize=7, handlelength=1.6, borderaxespad=0)
    _save(fig, "p2_worked_example.pdf")


def p3(region="eastern-ok"):
    t = gpd.read_parquet(DATA / "strata" / region / f"{region}-census-tracts.parquet").to_crs(5070)
    w = pd.read_parquet(DATA / "derived" / "writeup_tracts.parquet")
    t = t.merge(w[["GEOID", "att_total", "ref_m"]], on="GEOID", how="left")
    fig, ax = plt.subplots(figsize=(COL, 3.1))
    t[~(t.ref_m > 0)].plot(ax=ax, color="#f0efec", edgecolor="white", lw=0.15, rasterized=True)
    s = t[t.ref_m > 0]
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 1.01]
    s.assign(c=pd.cut(s.att_total.clip(0, 1), bins, labels=False, include_lowest=True)).plot(
        ax=ax, column="c", cmap=matplotlib.colors.ListedColormap(SEQ), vmin=0, vmax=6, edgecolor="white", lw=0.15,
        rasterized=True)
    ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEQ] + [plt.Rectangle((0, 0), 1, 1, color="#f0efec")]
    labs = [f"{a:.2f}–{min(b, 1):.2f}" for a, b in zip(bins[:-1], bins[1:])] + ["no road score"]
    ax.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.5, frameon=False,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8,
              title="Road gap added by the reference's bookkeeping", title_fontsize=7)
    _save(fig, "p3_map_oklahoma.pdf")


# Table A3's reads (LEDGER rows 19, 85, 86, 97, 109, 110): each rule's agreement c, one public-sample s.e. 0.019
AGREEMENT = [("Boundary highway counted inside\n(first test, E19)", 0.46),
             ("One record per route number (E85)", 0.08),
             ("One record per local street name (E86)", 0.09),
             ("One record per highway name (E97)", 0.04),
             ("Ramps restored to Overture's side (E109)", 0.04),
             ("Boundary distance re-measured (E110)", 0.06)]
SE = 0.019


def p4():
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    y = np.arange(len(AGREEMENT))[::-1]
    v = np.array([c for _, c in AGREEMENT])
    ax.barh(y, v, height=0.55, color=SCORED, zorder=2)
    ax.errorbar(v, y, xerr=SE, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    for yy, c in zip(y, v):
        ax.text(c + SE + 0.012, yy, f"+{c:.2f}", va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y, [a for a, _ in AGREEMENT])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.56)
    ax.set_xlabel("Agreement with the hidden target (bars: ±1 standard error)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _save(fig, "p4_agreement.pdf")


def p5():
    w = pd.read_parquet(DATA / "derived" / "writeup_tracts.parquet")
    w = w[w.ref_m > 0]
    top_full = w.full >= w.full.quantile(0.9)
    top_once = w.neither >= w.neither.quantile(0.9)
    leave, stay, enter = top_full & ~top_once, top_full & top_once, top_once & ~top_full
    assert (int(top_full.sum()), int(leave.sum()), int(enter.sum())) == (660, 321, 321)
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    rest = ~(top_full | top_once)
    ax.scatter(w.full[rest], w.neither[rest], s=3, color=GREY, lw=0, alpha=0.7, rasterized=True,
               label=f"other road-scored tracts ({int(rest.sum()):,})")
    ax.scatter(w.full[stay], w.neither[stay], s=5, color=SCORED, lw=0, rasterized=True,
               label=f"worst tenth either way ({int(stay.sum())})")
    ax.scatter(w.full[leave], w.neither[leave], s=5, color=ONCE, lw=0, rasterized=True,
               label=f"worst tenth only as scored ({int(leave.sum())})")
    ax.scatter(w.full[enter], w.neither[enter], s=5, color=HIDDEN, lw=0, rasterized=True,
               label=f"worst tenth only counted once ({int(enter.sum())})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.axvline(w.full.quantile(0.9), color=INK2, lw=0.6)
    ax.axhline(w.neither.quantile(0.9), color=INK2, lw=0.6)
    ax.text(w.full.quantile(0.9) + 0.01, 0.97, "worst tenth as scored →", fontsize=6.5, color=INK2, va="top")
    ax.text(0.01, w.neither.quantile(0.9) - 0.012, "worst tenth counted once ↑", fontsize=6.5, color=INK2, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Road gap as scored")
    ax.set_ylabel("Road gap with each highway counted once")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=True, facecolor="white", edgecolor="none", framealpha=1,
              borderpad=0.2, markerscale=2.2, handletextpad=0.1, fontsize=6.5)
    _save(fig, "p5_worklist.pdf")


MILESTONES = {"e19_road_boundary.csv": ("boundary rule (E19)", (16, 0.0800), "left", (0, 0.5)),   # label, position,
              "e49_fireblk.csv": ("misfiled fire stations (E49)", (33, 0.0745), "left", (0, 0)),   # alignment, leader start
              "e73_physics.csv": ("simulator (E73)", (46, 0.0712), "left", (0, 0)),
              "e85_multi.csv": ("duplicate records (E85)", (58, 0.0679), "left", (0, 0)),
              "e92_affine.csv": ("blend of scored files (E92)", (70, 0.0646), "left", (0, 0)),
              "e129_teacher.csv": ("teacher fit (E129)", (76, 0.0528), "right", (1, 1)),
              "e134_learner.csv": ("misfit learner (E134)", (113, 0.0528), "right", (1, 1))}


def p6():
    L = lb.ledger()
    L = L[L.kind != "const"].reset_index(drop=True)
    best = L.rmse.cummin()
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    x = np.arange(1, len(L) + 1)
    ax.scatter(x, L.rmse, s=6, color=GREY, lw=0, zorder=2, label="each scored file")
    ax.step(x, best, where="post", color=SCORED, lw=1.2, zorder=3, label="best so far")
    for f, (lab, at, ha, rel) in MILESTONES.items():
        i = int(np.flatnonzero(L.file.values == f)[0])
        ax.scatter([x[i]], [L.rmse[i]], s=14, color=ONCE, zorder=4, edgecolor="white", linewidth=0.6)
        ax.annotate(lab, (x[i], L.rmse[i]), xytext=at, textcoords="data", fontsize=6.3, color=INK, va="center",
                    ha=ha, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=2, relpos=rel))
    ax.set_xlabel("Scored file, in upload order")
    ax.set_ylabel("Public RMSE")
    ax.set_ylim(0.051, 0.100)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, markerscale=1.5)
    _save(fig, "p6_trajectory.pdf")


def p7():
    steps = [("Challenge package", "Overture roads, places, buildings; ACS; strata", "download.py"),
             ("Features per tract", "lengths, counts, boundary and record measures", "features*.py"),
             ("Simulator", "hidden reference drawn per tract; exact formula", "physics9.py, physics12.py"),
             ("Clean pieces", "77 package-only vectors", "clean_proto.py"),
             ("Weights", "pieces fitted to a teacher's geometry", "clean_geo.py"),
             ("Misfit learner", "LightGBM on what the simulator misses", "misfit_gbm.py"),
             ("Final file and checks", "pins, [0, 1] clip, rebuild and module audit", "final_build.py")]
    fig, ax = plt.subplots(figsize=(COL, 3.7))
    h, gap = 0.78, 0.30
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * (h + gap))
    ax.set_axis_off()
    for i, (head, body, script) in enumerate(steps):
        y = (len(steps) - 1 - i) * (h + gap) + 0.05
        ax.add_patch(FancyBboxPatch((0.15, y), 9.7, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc="#f6f9fe", ec=SCORED, lw=0.8))
        ax.text(0.45, y + h * 0.68, head, fontsize=8, fontweight="bold", color=INK, va="center")
        ax.text(0.45, y + h * 0.28, body, fontsize=7, color=INK2, va="center")
        ax.text(9.6, y + h * 0.68, script, fontsize=6.5, color=SCORED, va="center", ha="right", family="monospace")
        if i < len(steps) - 1:
            ax.annotate("", (5, y - 0.01), (5, y - gap + 0.03), arrowprops=dict(arrowstyle="<-", color=INK2, lw=0.8))
    _save(fig, "p7_architecture.pdf")


def p8():
    """The organisers' scorecard on our final, as scored and with each highway counted once (numbers.json, scorecard)."""
    S = json.loads((lb.ROOT / "docs" / "writeup" / "numbers.json").read_text())["scorecard"]
    rows = [("Rural vs urban", "rural_vs_urban"), ("Wildfire hazard", "wildfire_hazard"),
            ("Winter drought", "winter_drought"), ("Summer heat", "summer_heat"), ("Tribal vs non-tribal",
            "tribal_vs_nontribal"), ("Climate vulnerability", "high_cvi"), ("Social vulnerability", "high_svi")]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, k) in zip(y, rows):
        a, b = S[k]["as_scored"], S[k]["counted_once"]
        ax.plot([a, b], [yy, yy], color=GREY, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter([a], [yy], s=24, color=SCORED, zorder=3, edgecolor="white", linewidth=1.0)
        ax.scatter([b], [yy], s=24, color=ONCE, zorder=3, edgecolor="white", linewidth=1.0)
        ax.text(max(a, b) + 0.06, yy, f"{a:.2f} → {b:.2f}", va="center", fontsize=6.8, color=INK2,
                bbox=dict(fc="white", ec="none", pad=0.5))
    ax.axvline(1.0, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.text(1.04, -0.42, "parity", ha="left", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.4, 3.45)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Scorecard ratio: mean score of the group over the rest")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=24, color=SCORED, label="as scored (the official card)")
    ax.scatter([], [], s=24, color=ONCE, label="each highway counted once")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.25), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=1.0)
    _save(fig, "p8_scorecard.pdf")


def p9():
    M = json.loads((lb.ROOT / "docs" / "writeup" / "numbers.json").read_text())["external"]["measures"]
    rows = [("mobile_home", "Mobile homes"), ("no_internet", "No internet access"), ("disability", "Disability"),
            ("age_65", "Aged 65 or over"), ("poverty", "Below poverty line"), ("no_vehicle", "No vehicle"), None,
            ("risk_high", "FEMA overall risk high"), ("wildfire_high", "FEMA wildfire risk high"),
            ("flood_high", "FEMA flood risk high"), ("loss_per_person", "FEMA loss per person"),
            ("social_vuln", "FEMA social vulnerability"), ("resilience", "FEMA resilience")]
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    y = {r[0]: len(rows) - 1 - i for i, r in enumerate(rows) if r}
    ax.axvline(1, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for key, _ in filter(None, rows):
        r = M[key]
        for off, val, ci, c, mk, lab in ((0.17, r["ratio"], r["ratio_ci95"], HIDDEN, "o", "all hidden against all flagged"),
                                         (-0.17, r["ratio_within_region"], r["ratio_within_region_ci95"], INK2, "D",
                                          "within region")):
            ax.plot(ci, [y[key] + off] * 2, color=c, lw=0.9, zorder=2, solid_capstyle="round")
            ax.scatter([val], [y[key] + off], s=13, color=c, marker=mk, zorder=3, edgecolor="white", linewidth=0.5,
                       label=lab if key == "mobile_home" else None)
        ax.text(9.2, y[key], f"{r['ratio']:.2f}", va="center", ha="left", fontsize=6.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(0.5, 9)
    ax.set_yticks(list(y.values()), [r[1] for r in rows if r])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Hidden tracts as a multiple of flagged tracts (bars: 95% interval)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=1.2)
    _save(fig, "p9_outside_check.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
        f()
    print("figures written to", OUT)
