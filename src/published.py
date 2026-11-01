"""Every published number the clean builders use as a calibration target, in one place for code review.

All of them are AGGREGATES (region or stratum means, counts, ratios), the class the team accepted on 16 Oct
(CLAUDE.md §1); no per-tract value of anyone's appears here, and nothing here is derived from the removed
TIGER / Microsoft / HIFLD / CBP layers by us. Sources are the challenge discussion board (thread numbers below) and
the organisers' Info page (the challenge's Info page).
"""

# Mean coverage_gap_score per region x (rural burned, rural unburned, urban burned, urban unburned): a competitor's
# post, rural = ruca_primary >= 4, burned = MTBS ever-burned. Used ONLY to calibrate the physics vehicle's
# per-region road level k (physics3.calibrate), never imposed on tracts.
PUB16 = {
    "eastern-ok":       (0.176, 0.191, 0.176, 0.081),
    "maricopa-az":      (0.122, 0.136, 0.096, 0.029),
    "northern-ca":      (0.122, 0.120, 0.096, 0.039),
    "south-central-tx": (0.076, 0.111, 0.069, 0.040),
}
RURAL = 4                 # ruca_primary >= RURAL is "rural" in PUB16's post

# The other published targets, where each builder keeps its own copy (listed here for the audit):
#   transport mean gap 0.112 and burned/unburned 1.87 (a competitor's pooled 4-region component means)
#   POI mean 0.052 (physics3.calibrate's s) and burned/unburned 3.40 (the same post)
#   fire stations 8.7x worse than schools, all-tract (thread 34847; cand_round30.M_S_STAR)
#   740 fire-station blackouts, 270 of them misfiled places (a participant, thread 34925; cand_round30.BLACKOUTS)
#   road-undefined rates OK 0.21 / CA 0.37 / TX 0.28 / AZ 0.55 (the organisers' Info page)
PUBLISHED = dict(transport_mean=0.112, transport_burned_ratio=1.87, poi_mean=0.052, poi_burned_ratio=3.40,
                 fire_vs_school=8.7, fire_blackouts=740, fire_blackouts_misfiled=270,
                 road_undefined=dict(OK=0.21, CA=0.37, TX=0.28, AZ=0.55))

# The two water tracts with zero Overture roads, buildings and places that the sample submission still lists:
# every component is undefined from the reference side except by the scorer's convention, so y = 1.0 (LEDGER 111a,
# confirmed by E114's score).
PIN_GEOIDS = ("48039990000", "48057990000")
