"""The writeup's figures, from the package and `writeup_numbers.py`'s output (run that first).

    fig1_gap_by_region.png   the road gap as scored and with each highway counted once, by region and urbanity
    fig2_worked_example.png  one tract: the highway on its edge, Overture's copy a few metres outside
    fig3_map_oklahoma.png    where the bookkeeping adds most to the road gap (Oklahoma, the strongest region)
    fig4_architecture.png    Part B's data flow, one box per script

    python src/writeup_figures.py          # ~1 min
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

DATA = lb.ROOT.parent / "data"
OUT = lb.ROOT / "docs" / "writeup"
SCORED, ONCE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#c9c8c3", "#0b0b0b", "#52514e"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABEL = {"eastern-ok": "Oklahoma", "eastern-wa": "Washington", "maricopa-az": "Arizona", "northern-ca": "California",
         "south-central-tx": "Texas"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def fig1():
    N = json.loads((OUT / "numbers.json").read_text())["attribution"]
    rows = []
    for key, v in N.items():
        r, u = key.split("|")
        rows.append((f"{LABEL[r]}, {u}", v["gap_full"], v["gap_without"], v["n"]))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (lab, full, once, n) in enumerate(rows):
        ax.plot([once, full], [i, i], color=GREY, lw=2, zorder=1)
        ax.scatter([once], [i], s=36, color=ONCE, zorder=2, edgecolor="white", linewidth=1.5)
        ax.scatter([full], [i], s=36, color=SCORED, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(full + 0.012, i, f"{full:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [f"{lab} (n={n:,})" for lab, _, _, n in rows])
    ax.set_xlabel("Mean road-coverage gap of road-scored tracts (0 = fully covered, 1 = none)")
    ax.set_xlim(0, 0.55)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=36, color=SCORED, label="as the reference counts roads")
    ax.scatter([], [], s=36, color=ONCE, label="each highway counted once, in the tract it runs through")
    ax.legend(loc="lower right", frameon=False, fontsize=8, handletextpad=0.3)
    fig.savefig(OUT / "fig1_gap_by_region.png")
    plt.close(fig)


def _roads_near(region, box):
    import pyarrow.parquet as pq
    cols = [c for c in ["id", "class", "names", "routes", "geometry"]
            if c in pq.ParquetFile(DATA / "reference" / region / f"{region}-overture-roads.parquet").schema_arrow.names]
    return gpd.read_parquet(DATA / "reference" / region / f"{region}-overture-roads.parquet", columns=cols, bbox=box)


def fig2(geoid="06101050503", region="northern-ca"):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
    fig, (ax, ia) = plt.subplots(1, 2, figsize=(6.6, 3.7), gridspec_kw=dict(width_ratios=[1.75, 1], wspace=0.06))
    for a_, lw in ((ax, 1.6), (ia, 2.2)):
        nb.boundary.plot(ax=a_, color=GREY, lw=0.8)
        tp.plot(ax=a_, color="#eef4fc", edgecolor=SCORED, lw=lw)
        rp[~rp.route].plot(ax=a_, color=INK2, lw=1.2)
        tp.boundary.plot(ax=a_, color=SCORED, lw=lw)
        hw.plot(ax=a_, color=ONCE, lw=lw + 0.2)
        a_.set_xlabel("")
        a_.set_ylabel("")
    gx0, gy0, gx1, gy1 = tp.total_bounds
    ax.set_xlim(gx0 - 300, gx1 + 300)
    ax.set_ylim(gy0 - 300, gy1 + 300)
    ax.set_axis_off()
    ax.plot([gx0 - 250, gx0 + 250], [gy0 - 250] * 2, color=INK, lw=2)
    ax.text(gx0, gy0 - 230, "500 m", ha="center", va="bottom", fontsize=8, color=INK)
    c = poly.centroid
    ax.text(c.x - 350, c.y - 60, "Census Tract 505.03\nYuba City, Sutter County, CA\n2,040 homes", ha="center",
            va="center", fontsize=8.5, color=INK)
    along = near.union_all().intersection(poly.boundary.buffer(5))        # SR 99 where it runs on the tract line
    p = poly.boundary.interpolate(poly.boundary.project(along.centroid))
    hw_, hh_ = 20, 16
    ax.add_patch(plt.Rectangle((p.x - 90, p.y - 90), 180, 180, fill=False, ec=INK, lw=0.9))
    ax.text(p.x - 110, p.y + 110, "zoom", ha="right", fontsize=7.5, color=INK)
    ia.set_xlim(p.x - hw_, p.x + hw_)
    ia.set_ylim(p.y - hh_, p.y + hh_)
    ia.set_xticks([])
    ia.set_yticks([])
    ia.set_aspect("equal")
    for sp in ia.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(INK2)
        sp.set_linewidth(0.8)
    ia.plot([p.x - hw_ + 3, p.x - hw_ + 13], [p.y - hh_ + 3] * 2, color=INK, lw=1.5)
    ia.text(p.x - hw_ + 8, p.y - hh_ + 3.8, "10 m", ha="center", va="bottom", fontsize=7.5, color=INK)
    ia.set_title("zoom on the east edge", fontsize=8, color=INK2)
    h1, = ax.plot([], [], color=SCORED, lw=1.6, label="tract line, drawn on the highway's centreline")
    h2, = ax.plot([], [], color=ONCE, lw=1.8, label="Overture's State Route 99, a median 2.8 m outside the tract")
    h3, = ax.plot([], [], color=INK2, lw=1.2, label="Overture's other highway-class roads (Franklin Road)")
    fig.legend(handles=[h1, h2, h3], loc="upper center", ncol=1, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.2))
    fig.savefig(OUT / "fig2_worked_example.png")
    plt.close(fig)
    pts = [along.interpolate(f, normalized=True) for f in np.linspace(0, 1, 201)] if along.geom_type == "LineString" \
        else [g.interpolate(f, normalized=True) for g in along.geoms for f in np.linspace(0, 1, 51)]
    off = np.array([poly.boundary.distance(q) for q in pts])
    ex = {"geoid": geoid, "route_highway_inside_m": round(float(hw.intersection(poly).length.sum()), 1),
          "route_highway_along_edge_m": round(float(along.length), 1),
          "offset_m_p10_median_p90": [round(float(np.percentile(off, q)), 1) for q in (10, 50, 90)],
          "other_highway_class_inside_m": round(float(rp[~rp.route].intersection(poly).length.sum()), 1)}
    (OUT / "worked_example.json").write_text(json.dumps(ex, indent=1))
    return ex


def fig3(region="eastern-ok"):
    t = gpd.read_parquet(DATA / "strata" / region / f"{region}-census-tracts.parquet").to_crs(5070)
    w = pd.read_parquet(DATA / "derived" / "writeup_tracts.parquet")
    t = t.merge(w[["GEOID", "att_total", "ref_m"]], on="GEOID", how="left")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    t[~(t.ref_m > 0)].plot(ax=ax, color="#f0efec", edgecolor="white", lw=0.2)
    s = t[t.ref_m > 0]
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 1.01]
    s.assign(c=pd.cut(s.att_total.clip(0, 1), bins, labels=False, include_lowest=True)).plot(
        ax=ax, column="c", cmap=matplotlib.colors.ListedColormap(SEQ), vmin=0, vmax=6, edgecolor="white", lw=0.2)
    ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEQ] + [plt.Rectangle((0, 0), 1, 1, color="#f0efec")]
    labs = [f"{a:.2f}–{min(b, 1):.2f}" for a, b in zip(bins[:-1], bins[1:])] + ["no road scored"]
    ax.legend(handles, labs, title="Road gap added by the\nreference's bookkeeping", loc="center left", fontsize=7,
              title_fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 0.5))
    fig.savefig(OUT / "fig3_map_oklahoma.png")
    plt.close(fig)


def fig4():
    steps = [("Challenge package", "Overture roads, places, buildings;\nACS housing; strata; tract polygons",
              "download.py"),
             ("Features per tract", "lengths, counts, boundary and record\nmeasures (one script per family)",
              "features*.py"),
             ("Simulator", "the hidden reference counts drawn\nper tract, calibrated to published totals",
              "physics9.py, physics10.py"),
             ("Clean pieces", "77 package-only vectors\n(no excluded file, no other team's table)", "clean_proto.py"),
             ("Weights", "pieces fitted to a teacher's geometry;\nmeasured mechanisms added by exact algebra",
              "clean_geo.py, final_build.py"),
             ("Final file and checks", "pins, [0, 1] clip, 9,794 rows; rebuilt\nfrom the package, module audit",
              "final_build.py, validate.py")]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.6)
    ax.set_axis_off()
    for i, (head, body, script) in enumerate(steps):
        y = (len(steps) - 1 - i) * 1.6 + 0.2
        ax.add_patch(FancyBboxPatch((0.4, y), 9.2, 1.15, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#f6f9fe", ec=SCORED, lw=1.0))
        ax.text(0.7, y + 0.82, head, fontsize=9.5, fontweight="bold", color=INK, va="center")
        ax.text(0.7, y + 0.38, body, fontsize=8, color=INK2, va="center")
        ax.text(9.35, y + 0.82, script, fontsize=7.5, color=SCORED, va="center", ha="right", family="monospace")
        if i < len(steps) - 1:
            ax.annotate("", (5, y - 0.02), (5, y - 0.43), arrowprops=dict(arrowstyle="<-", color=INK2, lw=1))
    fig.savefig(OUT / "fig4_architecture.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    print(json.dumps(fig2(), indent=1))
    fig3()
    fig4()
    print("figures written to", OUT)
