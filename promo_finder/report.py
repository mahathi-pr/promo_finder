import csv, json
from pathlib import Path
from .parse import fmt


def quote(s, n=110):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def to_record(rank, c):
    h_why = [w.strip() for w in c.hook_why.split(";") if w.strip()][:2]
    c_why = [w.strip() for w in c.cliff_why.split(";") if w.strip()][:2]
    if c.llm and c.llm.get("why"):
        hook, cliff, why = c.llm["hook"], c.llm["cliffhanger"], c.llm["why"]
    else:
        hook = f'“{quote(c.hook_text)}”'
        cliff = (quote(c.cliff_text) if c.cliff_text.startswith('[action]') else f'“{quote(c.cliff_text)}”') if c.cliff_text else "(ends on a sound/camera beat)"
        why = f"Hook signals: {'; '.join(h_why) or 'immediate action'}. Cliffhanger signals: {'; '.join(c_why) or 'cut mid-scene'}."
    tier = "A (post first)" if rank <= 10 else ("B (post if slots free)" if rank <= 25 else "C (editor must review)")
    drivers = []
    if c.ep <= 10: drivers.append("early episode (low spoiler/context cost)")
    if c.scores["standalone"] >= 0.7: drivers.append("reads cold")
    if c.scores["hook"] >= 0.6: drivers.append("strong hook")
    if c.scores["cliff"] >= 0.65: drivers.append("strong cliffhanger")
    return dict(
        rank=rank, tier=tier, episode=c.ep,
        segments=[dict(start=fmt(a), end=fmt(b), start_s=round(a, 1), end_s=round(b, 1)) for a, b in c.segs],
        length_s=round(c.length, 1), hook=hook, cliffhanger=cliff, why_it_works=why,
        why_ranked_here=", ".join(drivers) or "meets both gates but no standout strength",
        scores={k: round(v, 2) for k, v in c.scores.items()}, final=round(c.final, 3),
        rank_score=round(c.rank_score, 3), flags=c.flags, speakers=c.speakers, source=c.source)


def write_all(picks, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    recs = [to_record(i, c) for i, c in enumerate(picks, 1)]
    (out / "promos.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False))
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "tier", "episode", "segments(start-end; ...)", "length_s", "hook", "cliffhanger",
                    "why_it_works", "why_ranked_here", "hook_score", "cliff_score", "standalone_score", "flags", "source"])
        for r in recs:
            w.writerow([r["rank"], r["tier"], r["episode"],
                        "; ".join(f'{s["start"]}-{s["end"]}' for s in r["segments"]), r["length_s"], r["hook"],
                        r["cliffhanger"], r["why_it_works"], r["why_ranked_here"], r["scores"]["hook"],
                        r["scores"]["cliff"], r["scores"]["standalone"], " | ".join(r["flags"]), r["source"]])
    L = ["# Promo shortlist — ranked\n", f"{len(recs)} promos. Tier A = post first; B = fine if slots free; C = weak, an editor must look first.\n"]
    for r in recs:
        segs = ", ".join(f'{s["start"]}–{s["end"]}' for s in r["segments"])
        L += [f'## #{r["rank"]} · Episode {r["episode"]} · {segs} ({r["length_s"]}s) · Tier {r["tier"]}',
              f'- **Hook:** {r["hook"]}', f'- **Cliffhanger:** {r["cliffhanger"]}',
              f'- **Why it works:** {r["why_it_works"]}', f'- **Why ranked here:** {r["why_ranked_here"]}',
              f'- Scores: hook {r["scores"]["hook"]} · cliff {r["scores"]["cliff"]} · standalone {r["scores"]["standalone"]}'
              + (f' · flags: {"; ".join(r["flags"])}' if r["flags"] else ""), ""]
    (out / "report.md").write_text("\n".join(L))
    return recs
