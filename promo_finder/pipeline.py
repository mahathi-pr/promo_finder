from .parse import load_all, fmt
from .score import CFG, gen_windows, score_window, find_stitched, overlap

MAX_PER_EP = 2


def episode_weight(num, n_total):
    """Promos for NEW viewers should lean on early episodes: less spoiler, less context debt."""
    if num <= 10: return 1.0
    return 1.0 - 0.15 * (num - 10) / max(1, n_total - 10)


def all_candidates(ep, cfg=CFG):
    cands = [score_window(ep, a, b, cfg) for a, b in gen_windows(ep, cfg)]
    cands += find_stitched(ep, cands, cfg)
    cands.sort(key=lambda c: -c.total)
    return cands


def pick_non_overlapping(cands, k, cfg=CFG, gate=True):
    chosen = []
    for c in cands:
        if gate and (c.scores['hook'] < cfg['gate_hook'] or c.scores['cliff'] < cfg['gate_cliff']): continue
        if any(overlap(c, x) for x in chosen): continue
        if len(c.segs) > 1 and any(len(x.segs) > 1 for x in chosen): continue
        chosen.append(c)
        if len(chosen) >= k: break
    return chosen


def run(folder, cfg=CFG, llm=None, shortlist_k=6, log=print):
    """llm: None (heuristic only) or a backend callable. Returns (episodes, ranked picks, n_candidates)."""
    from .rerank import rerank_episode
    eps = load_all(folder)
    n = len(eps)
    allp, ncand = [], 0
    for ep in eps:
        cands = all_candidates(ep, cfg)
        ncand += len(cands)
        if llm is None:
            ch = pick_non_overlapping(cands, MAX_PER_EP, cfg, gate=True)
            for c in ch: c.final, c.llm, c.source = c.total, None, "heuristic"
        else:
            short = pick_non_overlapping(cands, shortlist_k, cfg, gate=False)
            log(rerank_episode(ep, short, llm))
            ok = [c for c in short if c.llm is None and c.scores['hook'] >= cfg['gate_hook'] and c.scores['cliff'] >= cfg['gate_cliff']
                  or (c.llm and min(c.llm["hook_strength"], c.llm["cliffhanger_strength"]) >= 6)]
            ok.sort(key=lambda c: -c.final)
            ch = ok[:MAX_PER_EP]
            for c in ch: c.source = "heuristic+llm" if c.llm else "heuristic"
        for c in ch:
            c.rank_score = c.final * episode_weight(ep.num, n)
            c.episode = ep
            allp.append(c)
    allp = diversify(allp)
    return eps, allp, ncand


def diversify(picks, decay=0.93):
    """Greedy re-rank: a pick whose cast overlaps (Jaccard>=0.5) already-ranked picks is discounted, so the
    top of the list is not 5 near-identical 'family humiliates Juliana' scenes."""
    left = sorted(picks, key=lambda c: -c.rank_score)
    out = []
    while left:
        def adj(c):
            sp = set(c.speakers)
            n = sum(1 for k in out if sp and set(k.speakers) and len(sp & set(k.speakers)) / len(sp | set(k.speakers)) >= 0.5)
            return c.rank_score * (decay ** n)
        best = max(left, key=adj)
        best.rank_score = adj(best)
        out.append(best); left.remove(best)
    return out
