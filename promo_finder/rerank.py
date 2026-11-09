"""LLM re-ranking. The model only SCORES and EXPLAINS windows the code already cut on valid shot
boundaries; it never emits timestamps, so it cannot invent a cut point. On any failure we keep the
heuristic result for that episode and record why."""
from .parse import fmt
from .llm import parse_json_list

SYSTEM = (
    "You are a senior short-drama (microdrama) promo editor for TikTok/Reels/Shorts. A promo is a 20-60s clip "
    "that (1) HOOKS in the first 3 seconds (a slap, reveal, shocking or cruel line), (2) ENDS on an unresolved "
    "CLIFFHANGER, and (3) makes sense to someone who has never seen the show. Judge ONLY from the transcript given. "
    "Be harsh: most windows are mediocre. Return ONLY a JSON array, one object per window: "
    '{"id","hook_strength"(1-10),"cliffhanger_strength"(1-10),"standalone"(1-10),'
    '"hook"(<=15 words, what happens in first 3s),"cliffhanger"(<=15 words, the unresolved moment),'
    '"why"(one sentence, max 25 words)}. Do not invent events that are not in the transcript.'
)


def window_transcript(ep, segs):
    lines = []
    for a, b in segs:
        for e in ep.events:
            if e.kind in ("dialogue", "action", "sfx") and e.end > a and e.start < b:
                who = e.label + (f" ({e.tone})" if e.tone else "")
                lines.append(f"[{fmt(e.start - a)}] {who}: {e.text[:220]}")
        lines.append("--- cut ---")
    return "\n".join(lines[:-1])


def build_prompt(ep, cands):
    out = [f"SHOW CONTEXT — Episode {ep.num}. Logline: {ep.logline}\nContinuity: {ep.continuity}\n"]
    for i, c in enumerate(cands):
        out.append(f"### WINDOW {ep.num}-{i}  ({c.length:.0f}s)\n{window_transcript(ep, c.segs)}\n")
    return "\n".join(out)


def rerank_episode(ep, cands, backend, w_llm=0.6):
    """cands: heuristic-ranked list for ONE episode. Mutates .llm / .final and returns a log string."""
    try:
        raw = backend(SYSTEM, build_prompt(ep, cands))
        rows = {r["id"]: r for r in parse_json_list(raw)}
    except Exception as e:  # keep heuristic result, never crash the batch
        for c in cands: c.final = c.total; c.llm = None
        return f"EP{ep.num}: LLM failed ({type(e).__name__}: {e}); kept heuristic"
    for i, c in enumerate(cands):
        r = rows.get(f"{ep.num}-{i}")
        if not r:
            c.final, c.llm = c.total, None
            continue
        c.llm = r
        l = (0.4 * r["hook_strength"] + 0.4 * r["cliffhanger_strength"] + 0.2 * r["standalone"]) / 10
        c.final = w_llm * l + (1 - w_llm) * c.total
    return f"EP{ep.num}: scored {len(cands)} windows"
