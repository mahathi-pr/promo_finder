"""Validation that does not need human labels.

V1 (leave-one-out): writers tagged 26 beats with the literal word 'cliffhanger'. Turn that keyword
    (and the episode-end bonus) OFF, then check where those beats rank among ALL valid cut points.
V2 (hard constraints): every output satisfies length 20-60s, no sliced line, >=2 spoken lines,
    no overlaps, timestamps inside the episode.
"""
import re, statistics as st
from .parse import load_all
from .score import CFG, cliff_score, straddles, hook_score, gen_windows, overlap


def v1_cliffhanger_loo(eps):
    cfg = dict(CFG, use_cliffhanger_keyword=False, use_episode_end_bonus=False)
    pos, allv = [], []
    for ep in eps:
        marks = [e.end for e in ep.events if re.search("cliffhanger", e.text, re.I) and e.kind in ("sfx", "camera")]
        bounds = sorted({round(c.end, 2) for c in ep.events if c.kind == "camera" and c.end > 20 and not straddles(ep, c.end)})
        for b in bounds:
            s = cliff_score(ep, b, 0, cfg)[0]
            allv.append(s)
            if any(abs(b - m) < 0.6 for m in marks):
                pos.append(s)
    allv.sort()
    pct = [100.0 * sum(1 for x in allv if x < p) / len(allv) for p in pos]
    return dict(n_marked=len(pos), n_cut_points=len(allv), median_percentile=st.median(pct),
                share_above_median=sum(p >= 50 for p in pct) / len(pct),
                share_top_quartile=sum(p >= 75 for p in pct) / len(pct),
                mean_marked=st.mean(pos), mean_all=st.mean(allv))


def v2_constraints(eps, picks):
    by = {e.num: e for e in eps}
    errs = []
    for c in picks:
        ep = by[c.ep]
        if not (20 <= c.length <= 60.05): errs.append((c.ep, "length", c.length))
        for a, b in c.segs:
            if a < 0 or b > ep.duration + 0.01: errs.append((c.ep, "outside episode", (a, b)))
            if straddles(ep, a) or straddles(ep, b): errs.append((c.ep, "sliced a spoken line", (a, b)))
        if c.n_lines < 2: errs.append((c.ep, "fewer than 2 lines", c.n_lines))
    for i, c1 in enumerate(picks):
        for c2 in picks[i + 1:]:
            if overlap(c1, c2): errs.append((c1.ep, "overlap", None))
    return errs
