"""Candidate generation + heuristic scoring (no LLM, no network, deterministic).

A candidate = one or more segments [(ep, start, end), ...] cut on camera-shot boundaries so
no spoken line is ever sliced. Scores are 0..1 for: hook (first 3s), cliffhanger (last ~6s),
standalone (makes sense cold), pace (length + density).
"""
import re
from dataclasses import dataclass, field
from .parse import fmt

CFG = dict(
    min_len=20.0, max_len=60.0, ideal_lo=30.0, ideal_hi=50.0,
    hook_window=3.0, cliff_window=6.0,
    w_hook=0.35, w_cliff=0.35, w_stand=0.15, w_pace=0.15,
    use_cliffhanger_keyword=True,   # set False for the leave-one-out validation
    use_episode_end_bonus=True,     # set False for the leave-one-out validation
    gate_hook=0.40, gate_cliff=0.45,  # a promo needs BOTH a hook and a cliffhanger
)

# ---- lexicons (tuned on this show's vocabulary; swap per genre) --------------------------------
SHOCK = r"slap|slaps|slapped|strike|strikes|struck|shove|shoves|hurl|hurls|throws|slam|slams|drag|drags|dragged|" \
        r"scream|screams|shriek|gasp|gasps|collapse|collapses|blood|bleed|dagger|knife|sword|poison|" \
        r"fire|burn|burning|flames|gunshot|choke|chokes|kneel|kneels|tears off|rips|shatters|smash"
REVEAL = r"truth|secret|lied|lie|liar|betray|betrayed|traitor|murder|murderer|dead|dies|kill|killed|" \
         r"arrest|arrested|guilty|innocent|impostor|thief|stole|steal|stolen|forged|forgery|frame|framed|" \
         r"wife|husband|marry|married|wedding|duke|duchess|heir|bastard|lawful|illegitimate|annul|" \
         r"pregnan\w*|child|son|daughter|father|mother|mine|never|always|you are not|you're not|" \
         r"who are you|how dare|get out|leave|divorce|trial|execute|hang|prison|dungeon|ruin|revenge|" \
         r"proof|evidence|confess|admit|expose|reveal"
INSULT = r"beggar|servant|worthless|pathetic|plain|beneath|disgrace|fool|wretch|filth|scum|witch|penniless|peasant|commoner|unworthy|stupid|ugly|cripple|scarred|liar|cheat|poisoner|murderess|slave|trash|dirt|vermin|beast"
HI_TONE = r"arrogant|callous|derisive|sneering|biting|boastful|conceited|gleeful|furious|screaming|shouting|shrieking|venomous|threatening|cruel|mocking|incredulous|shocked|" \
          r"horrified|terrified|desperate|menacing|vicious|spiteful|taunting|scornful|triumphant|gloating|" \
          r"chilling|ominous|sinister|panicked|frantic|enraged|seething|condemning|accusing|stunned|" \
          r"fierce|cold|defiant|pleading|sobbing|weeping|devastated|heartbroken|smug"
CALM_TONE = r"gentle|soft|reassuring|sincere|warm|tender|calm|peaceful|content|relieved|grateful|loving|" \
            r"earnest|polite|solemn"
CLOSURE = r"thank you|thanks|goodnight|good night|farewell|goodbye|very well|as you wish|of course|" \
          r"i understand|amen|sleep well|rest well|you're welcome|welcome home|i promise|together"
SFX_CLIFF = r"swell|sting|stinger|abrupt|crash|slam|gunshot|scream|thud|shatter|bang|hold|peak|" \
            r"ominous|menacing|suspense|dramatic"
CAM_CLIFF = r"extreme close|push-in|push in|snap|whip|freeze|cut to black|smash|reveal|zoom|dolly"
BAD_OPEN = re.compile(r"^(yes|no|yeah|because|thank|thanks|well|and|but|so|then|very well|i see|"
                      r"oh|hmm|indeed|of course|please|sir|my lady|my lord)\b", re.I)
HAS = lambda pat, s: len(re.findall(r"\b(?:" + pat + r")\b", s.lower()))


@dataclass
class Cand:
    ep: int
    segs: list                      # [(start, end), ...] seconds, in order
    scores: dict = field(default_factory=dict)
    total: float = 0.0
    hook_text: str = ""
    cliff_text: str = ""
    flags: list = field(default_factory=list)
    n_lines: int = 0
    speakers: list = field(default_factory=list)

    @property
    def length(self):
        return sum(e - s for s, e in self.segs)


def _in(ev, a, b):
    return ev.end > a and ev.start < b


def _clip(x):
    return max(0.0, min(1.0, x))


def window_events(ep, a, b):
    return [e for e in ep.events if _in(e, a, b)]


def straddles(ep, t):
    return any(e.kind == "dialogue" and e.start < t < e.end for e in ep.events)


def gen_windows(ep, cfg=CFG):
    cams = [e for e in ep.events if e.kind == "camera"]
    starts = sorted({round(c.start, 2) for c in cams})
    ends = sorted({round(c.end, 2) for c in cams})
    out = []
    for s in starts:
        if straddles(ep, s):
            continue
        for e in ends:
            L = e - s
            if L < cfg["min_len"]:
                continue
            if L > cfg["max_len"]:
                break
            if straddles(ep, e):
                continue
            dl = [x for x in ep.events if x.kind == "dialogue" and x.start >= s - 0.01 and x.end <= e + 0.01]
            if len(dl) >= 2:
                out.append((s, e))
    return out


# --- feature scorers -----------------------------------------------------------------------------
def hook_score(ep, a, cfg=CFG):
    evs = [e for e in ep.events if _in(e, a, a + cfg["hook_window"]) and e.start >= a - 0.25]
    spoken = [e for e in evs if e.kind == "dialogue" and e.start < a + cfg["hook_window"]]
    acts = [e for e in evs if e.kind in ("action", "sfx")]
    text = " ".join(e.text for e in evs if e.kind != "camera")
    camtxt = " ".join(e.text for e in evs if e.kind == "camera")
    s, why = 0.0, []
    if not spoken and not acts:
        return 0.05, "slow open (no line/action in first 3s)", ""
    n_shock = HAS(SHOCK, text) + 0.5 * HAS(SHOCK, camtxt)
    n_rev = HAS(REVEAL, " ".join(e.text for e in spoken))
    s += min(0.35, 0.18 * n_shock)
    s += min(0.30, 0.12 * n_rev)
    n_ins = HAS(INSULT, " ".join(e.text for e in spoken))
    s += min(0.20, 0.10 * n_ins)
    if n_ins: why.append("insult / class contempt")
    if spoken and re.search(r"\byou(r|rs)?\b", spoken[0].text.lower()): s += 0.05; why.append("direct address")
    if n_shock: why.append("physical/shock action")
    if n_rev: why.append("reveal/accusation language")
    if spoken:
        f = spoken[0]
        words = len(f.text.split())
        if HAS(HI_TONE, f.tone): s += 0.15; why.append(f"high-arousal delivery ({f.tone.lower()})")
        if f.text.rstrip().endswith(("!", "?")): s += 0.10; why.append("opens on exclamation/question")
        if BAD_OPEN.match(f.text.strip()) or words < 3:
            s -= 0.25; why.append("opener is a reply — needs context")
        if re.match(r"^\w+[?!]\s", f.text.strip()) and len(f.text.split()) > 1 and len(f.text.split(" ")[0]) < 9 and f.text.split(" ")[0].endswith("?"):
            s -= 0.15; why.append("opens by echoing a line the viewer never hears")
        if "V.O" in f.label.upper():
            s -= 0.05
    elif acts:
        s += 0.05
    s += 0.10                       # baseline: something is happening in the first 3s
    first = (spoken[0].text if spoken else acts[0].text)
    return _clip(s), "; ".join(why) or "something happens", first


def cliff_score(ep, b, a_start, cfg=CFG):
    w0 = b - cfg["cliff_window"]
    evs = [e for e in ep.events if _in(e, w0, b + 0.01)]
    spoken = [e for e in evs if e.kind == "dialogue"]
    sfx = " ".join(e.text for e in evs if e.kind == "sfx")
    cam = " ".join(e.text for e in evs if e.kind == "camera")
    act = " ".join(e.text for e in evs if e.kind == "action")
    s, why = 0.0, []
    last = spoken[-1] if spoken else None
    if last:
        t = last.text.strip()
        if t.endswith("?"): s += 0.20; why.append("ends on a question")
        if t.endswith("..."): s += 0.12; why.append("trailing line")
        if t.endswith("!"): s += 0.08
        if HAS(REVEAL, t): s += 0.15; why.append("final line carries a reveal/threat")
        if HAS(HI_TONE, last.tone): s += 0.08
        if HAS(CALM_TONE, last.tone): s -= 0.15; why.append("calm final delivery")
        if HAS(CLOSURE, t): s -= 0.25; why.append("closure language")
        if len(t.split()) <= 3 and not t.endswith(("?", "!")): s -= 0.08
    kw = SFX_CLIFF + (r"|cliffhanger" if cfg["use_cliffhanger_keyword"] else "")
    if HAS(kw, sfx): s += 0.25; why.append("music/sound sting at the cut")
    if HAS(CAM_CLIFF + (r"|cliffhanger" if cfg["use_cliffhanger_keyword"] else ""), cam):
        s += 0.15; why.append("emphatic final shot")
    if HAS(SHOCK, act): s += 0.10
    # unresolved: window ends before its scene ends  (cut mid-scene)
    sc = max((e.scene for e in ep.events if _in(e, b - 1.0, b + 0.01)), default=0)
    s_end = ep.scenes.get(sc, ("", 0, 0))[2]
    if s_end - b > 3.0:
        s += 0.10; why.append("cuts mid-scene (unresolved)")
    if cfg["use_episode_end_bonus"] and b >= ep.duration - 1.0:
        s += 0.10; why.append("episode's own closing beat")
    s += 0.12
    fin = [e for e in evs if e.kind in ("dialogue", "action")]
    ctext = ""
    if fin:
        z = max(fin, key=lambda e: e.end)
        ctext = z.text if z.kind == "dialogue" else "[action] " + z.text
    return _clip(s), "; ".join(why) or "no strong cliff signal", ctext


def standalone_score(ep, a, b):
    dl = [e for e in ep.events if e.kind == "dialogue" and e.start >= a - .01 and e.end <= b + .01]
    spk = sorted({e.label for e in dl if "V.O" not in e.label.upper()})
    scenes = {e.scene for e in ep.events if _in(e, a, b)}
    s, flags = 0.6, []
    if len(spk) > 4: s -= 0.25; flags.append(f"{len(spk)} speakers (crowded)")
    elif len(spk) <= 1: s -= 0.2; flags.append("single speaker")
    if len(scenes) > 2: s -= 0.2; flags.append("spans 3+ scenes")
    vo = sum(1 for e in dl if "V.O" in e.label.upper())
    if vo / max(1, len(dl)) > .3: s -= 0.15; flags.append("heavy voice-over (context-dependent)")
    if len(spk) in (2, 3): s += 0.2
    # needs earlier-episode knowledge? continuity note says 'picks up' and window is in first 25s
    if ep.continuity.lower().startswith("picks up") and a < 25: s -= 0.12; flags.append("opens inside previous episode's scene")
    return _clip(s), flags, spk, len(dl)


def pace_score(ep, a, b, n_lines, cfg=CFG):
    L = b - a
    if cfg["ideal_lo"] <= L <= cfg["ideal_hi"]: d = 1.0
    elif L < cfg["ideal_lo"]: d = 0.6 + 0.4 * (L - cfg["min_len"]) / (cfg["ideal_lo"] - cfg["min_len"])
    else: d = 0.6 + 0.4 * (cfg["max_len"] - L) / (cfg["max_len"] - cfg["ideal_hi"])
    dens = min(1.0, (n_lines / L) / 0.25)         # ~1 line / 4s is "dense enough"
    return _clip(0.5 * d + 0.5 * dens)


def score_window(ep, a, b, cfg=CFG):
    h, hw, htxt = hook_score(ep, a, cfg)
    c, cw, ctxt = cliff_score(ep, b, a, cfg)
    st, flags, spk, n = standalone_score(ep, a, b)
    p = pace_score(ep, a, b, n, cfg)
    total = cfg["w_hook"] * h + cfg["w_cliff"] * c + cfg["w_stand"] * st + cfg["w_pace"] * p
    cand = Cand(ep.num, [(a, b)], dict(hook=h, cliff=c, standalone=st, pace=p), total,
                htxt, ctxt, flags, n, spk)
    cand.hook_why, cand.cliff_why = hw, cw
    return cand


def overlap(c1, c2):
    if c1.ep != c2.ep: return False
    return any(a1 < b2 and a2 < b1 for a1, b1 in c1.segs for a2, b2 in c2.segs)


def find_stitched(ep, singles, cfg=CFG):
    """2-segment promos: a strong hook beat (8-20s) + the best closing beat (12-30s) of the same
    episode, separated by >=10s. Lower standalone score by design (cuts across the story)."""
    out = []
    cams = [e for e in ep.events if e.kind == "camera"]
    starts = sorted({round(c.start, 2) for c in cams}); ends = sorted({round(c.end, 2) for c in cams})
    A = []
    for s in starts:
        if straddles(ep, s): continue
        for e in ends:
            L = e - s
            if L < 8: continue
            if L > 20: break
            if straddles(ep, e): continue
            h, hw, ht = hook_score(ep, s, cfg)
            if h >= 0.6: A.append((h, s, e, hw, ht))
    B = []
    for s in starts:
        if straddles(ep, s): continue
        for e in ends:
            L = e - s
            if L < 12: continue
            if L > 30: break
            if straddles(ep, e): continue
            c, cw, ct = cliff_score(ep, e, s, cfg)
            if c >= 0.6: B.append((c, s, e, cw, ct))
    A.sort(reverse=True); B.sort(reverse=True)
    for h, sa, ea, hw, ht in A[:4]:
        for c, sb, eb, cw, ct in B[:4]:
            if sb - ea < 10: continue
            L = (ea - sa) + (eb - sb)
            if not (cfg["min_len"] <= L <= cfg["max_len"]): continue
            n = len([x for x in ep.events if x.kind == "dialogue" and
                     ((x.start >= sa and x.end <= ea) or (x.start >= sb and x.end <= eb))])
            if n < 2: continue
            total = cfg["w_hook"] * h + cfg["w_cliff"] * c + cfg["w_stand"] * 0.35 + cfg["w_pace"] * 0.7
            cd = Cand(ep.num, [(sa, ea), (sb, eb)], dict(hook=h, cliff=c, standalone=0.35, pace=0.7),
                      total, ht, ct, ["stitched: story jump between segments — verify it reads cold"], n, [])
            cd.hook_why, cd.cliff_why = hw, cw
            out.append(cd)
    return out
